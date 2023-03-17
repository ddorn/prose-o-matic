from collections import Counter
import re
from io import TextIOWrapper, StringIO
from typing import Generator
import torch
import torch.nn as nn
import torch.nn.functional as F


# My own BPE implementation

def stream_words(text: str) -> Generator[str, None, None]:
    # We make sure that digits are only by themselves,
    # otherwise the tokenisation of numbers is not very natural.
    part_regex = re.compile(r'^ ?(\d|[a-zA-Z]+|[^\da-zA-Z]+)')
    chunk = 32
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


def train_bpe(text: str, vocab_size: int):
    """
    Trains a BPE model on the given text.
    :param text: The text to train on.
    :param vocab_size: The size of the vocabulary.
    :return: A list of tuples (symbol, count) sorted by count.
    """

    data = tuple(list(w) for w in stream_words(text))
    tokens = set(text)

    while len(tokens) < vocab_size:
        counts = Counter()
        for word in data:
            for i in range(len(word) - 1):
                counts[word[i], word[i + 1]] += 1

        # find the most common adjacent pair
        p1, p2 = max(counts, key=counts.get)
        count = counts[p1, p2]
        if count == 1:
            print()
            break

        tokens.add(p1 + p2)
        print(f'Added token {p1 + p2} with count {count}')
        print('•'.join('·'.join(word) for word in data))

        # replace the pair with the new token
        pair = [p1, p2]
        for word in data:
            write_head = 0
            read_head = 0
            while read_head < len(word):
                if word[read_head : read_head + 2] == pair:
                    word[write_head] = p1 + p2
                    read_head += 1
                else:
                    word[write_head] = word[read_head]
                write_head += 1
                read_head += 1
            del word[write_head:]

    return tokens