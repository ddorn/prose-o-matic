
from collections import Counter
import re
from io import TextIOWrapper, StringIO
from time import ctime
from typing import Generator, Iterator, TextIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import FILE_LIKE

from torchtyping import TensorType as TT, patch_typeguard  # type: ignore
from typeguard import typechecked

patch_typeguard()

batch = 'batch'
token = 'token'

class BPE:
    """
    A Byte Pair Encoding tokenizer that does not cross word boundaries.
    Digits are not merged. Works at the byte level to allow the whole range
    of UTF-8 characters.


    This might not be an exact implementation of the original BPE algorithm,
    but it corresponds to what I think BPE does.
    """

    def __init__(self, token_frequencies: dict[bytes, int]) -> None:
        self.token_frequencies = token_frequencies
        self.tokens = sorted(token_frequencies,
            key=lambda t: token_frequencies[t], reverse=True)
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}

    def __str__(self) -> str:
        return f'<BPE({len(self.tokens)} tokens)>'

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @classmethod
    def train_from_text(cls, text: str | Iterator[str], vocab_size: int, verbose: int = 0) -> 'BPE':
        """
        Trains a BPE model on the given text.
        :param text: The text to train on.
        :param vocab_size: The size of the vocabulary.
        :param verbose: The verbosity level (0, 1 or 2).
        :return: A list of tuples (symbol, count) sorted by count.
        """

        # We split the text into words so that the merging of byte pairs
        # does not cross word boundaries (and thus be more chaotic).
        # And we keep only one copy of each word with its frequency
        # for efficiency.
        word_counts = Counter(cls.stream_words(text))

        data = tuple(
            (cls._encode(word), count)
            for word, count in word_counts.items()
        )
        tokens = {bytes([b]) for b in range(256)}

        if verbose > 1: print(ctime(), f'Split text into {len(data)} words and {len(tokens)} base tokens.')

        # Compute the frequency of each adjacent pair.
        counts: dict[tuple[bytes, bytes], int] = Counter()
        for word, count in data:
            for i in range(len(word) - 1):
                counts[word[i], word[i + 1]] += count
        if verbose > 1: print(ctime(), f'Computed pair frequencies for {len(counts)} pairs.')

        while len(tokens) < vocab_size:

            # Find the most frequent pair that is not already a token.
            p1, p2 = max(counts, key=lambda p: counts[p] * (p[0] + p[1] not in tokens))
            count = counts[p1, p2]
            if count == 1:
                print('No more pairs with count > 1')
                break

            tokens.add(p1 + p2)

            if verbose: print(ctime(), f'Added token {p1 + p2!r} with frequency {count} (nb_of_tokens: {len(tokens)})')

            # Merge every occurrence of the pair and update the counts.
            pair = [p1, p2]
            for word, count in data:
                if len(word) == 1:
                    continue  # Opti: everything has been merged already.
                write_head = 0
                read_head = 0
                # We modify the word in place.
                while read_head < len(word):
                    # If the current token and the next one are the pair:
                    if word[read_head : read_head + 2] == pair:
                        # Update the counts.
                        if read_head > 0:
                            counts[word[read_head - 1], p1] -= count
                            counts[word[read_head - 1], p1 + p2] += count
                        if read_head + 2 < len(word):
                            counts[p2, word[read_head + 2]] -= count
                            counts[p1 + p2, word[read_head + 2]] += count
                        # Write the new token and skip p2.
                        word[write_head] = p1 + p2
                        read_head += 2
                    else:
                        # Copy the current token to the current position.
                        word[write_head] = word[read_head]
                        read_head += 1
                    # Move the write head forward as we have written one token.
                    write_head += 1
                # Remove the extra tokens that are now after the end of the word.
                del word[write_head:]

        # We return a frequency of 0 for individual characters.
        # I don't know if this is what I want, but it's what I have for now.
        token_with_freq = {tok: 0 for tok in tokens}
        token_with_freq.update({
            tok: count
            for token, count in counts.items()
            if (tok := token[0] + token[1]) in tokens
        })
        return cls(token_with_freq)

    @staticmethod
    def stream_words(stream: str | Iterator[str]) -> Generator[str, None, None]:
        if isinstance(stream, str):
            stream = iter([stream])

        # We make sure that digits are only by themselves,
        # otherwise the tokenisation of numbers is not very natural.
        part_regex = re.compile(r'^ ?(\d|[a-zA-Z]+|[^\da-zA-Z]+)')
        chunk = 32

        while True:
            text = next(stream, None)
            if text is None: return

            pos = 0
            while pos < len(text):
                match = part_regex.search(text[pos:pos+chunk])
                assert match is not None
                part = match.group()
                if len(part) > 1 and part[-1] == ' ':
                    yield part[:-1]
                    pos += len(part) - 1
                else:
                    yield part
                    pos += len(part)

    @typechecked
    def tokenize(self, texts: list[str], pad: bytes=b' ', pad_length: int | None=None) -> TT['batch', 'token', torch.long]:
        """
        Tokenizes a list of texts.
        :param texts: The texts to tokenize.
        :return: A tensor of shape (batch, token).
        """

        assert len(texts) > 0
        assert pad in self.tokens, f'Pad token {pad!r} is not in the vocabulary.'

        result = []
        for text in texts:
            tokens: list[bytes] = []
            for word_ in self.stream_words(text):
                word = self._encode(word_)
                # We merge the tokens until there is no more pair.
                while True:
                    # Find the most frequent pair that is a token.
                    best = None
                    best_count = 0
                    for p1, p2 in zip(word, word[1:]):
                        count = self.token_frequencies.get(p1 + p2, 0)
                        if count > best_count:  # We found a better pair!
                            best = p1, p2
                            best_count = count

                    # If We did not find any pair, we are done for this word.
                    if best is None:
                        break

                    # Merge all the pairs
                    p1, p2 = best
                    write_head = 0
                    read_head = 0
                    while read_head < len(word):
                        if word[read_head : read_head + 2] == [p1, p2]:
                            word[write_head] = p1 + p2
                            read_head += 2
                        else:
                            word[write_head] = word[read_head]
                            read_head += 1
                        write_head += 1
                    del word[write_head:]

                tokens.extend(word)
            result.append(tokens)

        if pad_length is None:
            pad_length = max(len(tokens) for tokens in result)

        tensor = torch.tensor([], dtype=torch.long)
        for line in result:
            padding = [self.token_to_id[pad]] * (pad_length - len(line))
            rest = [self.token_to_id[tok] for tok in line]
            tensor = torch.cat([tensor, torch.tensor(padding + rest)])

        return tensor.view(len(result), pad_length)

    def detokenize(self, tokens: TT['batch', 'token', torch.long]) -> list[str]:
        """
        Detokenizes a tensor of tokens.
        Note that not all sequences of tokens are valid in utf-8,
        so the result may contains some �.
        :param tokens: A tensor of shape (batch, token).
        :return: The texts.
        """
        return [
            b''.join(self.tokens[tok] for tok in line).decode('utf-8', errors='replace')
            for line in tokens
        ]

    @staticmethod
    def _encode(s: str) -> list[bytes]:
        return [bytes([b]) for b in s.encode('utf-8')]

    def save(self, file: FILE_LIKE) -> None:
        torch.save(self.token_frequencies, file)

    @classmethod
    def load(cls, file: FILE_LIKE) -> 'BPE':
        return cls(torch.load(file))

class BigramModel(nn.Module):
    def __init__(self, voc_size: int, embedding_dim: int):
        super().__init__()

        self.voc_size = voc_size
        self.embedding = nn.Embedding(voc_size, embedding_dim)
        self.unembedding = nn.Linear(embedding_dim, voc_size)

    def forward(self, tokens: TT['batch', 'token', torch.long]) -> TT['batch', 'token', 'embedding', torch.long]:
        x = self.embedding(tokens)
        logits = self.unembedding(x)
        return logits

    def loss(self, tokens: TT['batch', 'token', torch.long],
                   targets: TT['batch', 'token', torch.long]) -> tuple[TT['batch', 'token', 'embedding', torch.long], torch.Tensor]:
        logits = self(tokens)[:, -1, :]
        print("loss", logits.shape, targets.shape)
        loss = F.cross_entropy(logits, targets.squeeze(-1))
        return logits, loss