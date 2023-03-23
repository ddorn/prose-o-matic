from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from torchtyping import TensorType as TT, patch_typeguard  # type: ignore
from typeguard import typechecked

patch_typeguard()

batch = 'batch'
token = 'token'
embedding = 'embedding'


class Attention(nn.Module):

    def __init__(self, embedding_dim: int, head_count: int):
        super().__init__()
        assert embedding_dim % head_count == 0, 'Embedding dimension must be a multiple of head count.'
        head_dim = embedding_dim // head_count
        self.query = nn.Parameter(torch.randn(head_count, embedding_dim, head_dim))
        self.key = nn.Parameter(torch.randn(head_count, embedding_dim, head_dim))
        self.value = nn.Parameter(torch.randn(head_count, embedding_dim, head_dim))

        self.out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: TT['batch', 'token', 'embedding'],
                mask: Optional[TT['token', 'token', float]]) -> TT['batch', 'token', 'embedding']:
        q = torch.einsum('hei,bte->hbti', self.query, x)
        k = torch.einsum('hei,bte->hbti', self.key, x)
        v = torch.einsum('hei,bte->hbti', self.value, x)

        fit = torch.einsum('hbte,hbTe->hbtT', q, k) / x.shape[1]**0.5
        if mask is not None:
            fit = fit.masked_fill(mask, float('-inf'))
        fit = torch.softmax(fit, dim=-1)
        attended = torch.einsum('hbtT,hbTi->hbti', fit, v)
        attended = einops.rearrange(attended, 'h b t i -> b t (h i)')
        return self.out(attended)


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim: int, head_count: int):
        super().__init__()
        self.attention = Attention(embedding_dim, head_count)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.layernorm_2 = nn.LayerNorm(embedding_dim)

    def forward(
            self,
            x: TT['batch', 'token', 'embedding'],
            mask: Optional[TT['token', 'token', bool]] = None) -> TT['batch', 'token', 'embedding']:
        x = x + self.attention(self.layernorm_1(x), mask=mask)
        x = x + self.feed_forward(self.layernorm_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 depth: int,
                 head_count: int,
                 block_size: int,
                 voc_size: int = 256,
                 out_voc_size: Optional[int] = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.head_count = head_count
        self.block_size = block_size
        self.voc_size = voc_size
        self.out_voc_size = out_voc_size or voc_size

        self.embedding = nn.Embedding(voc_size, embedding_dim)
        self.positional_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, head_count) for _ in range(depth)], )
        self.last_layer_norm = nn.LayerNorm(embedding_dim)
        self.unembedding = nn.Linear(embedding_dim, self.out_voc_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @typechecked
    def forward(
            self,
            tokens: TT['batch', 'token', int],
            mask: Optional[TT['token', 'token', bool]] = None) -> TT['batch', 'token', 'voc_size']:
        assert tokens.shape[-1] <= self.block_size
        tokens = tokens.to(self.embedding.weight.device)

        emb: TT['batch', 'token', 'emb'] = self.embedding(tokens)
        pos_emb: TT['token', 'emb'] = self.positional_embedding(
            torch.arange(tokens.shape[1], device=tokens.device))
        x = emb + pos_emb
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.last_layer_norm(x)
        return self.unembedding(x)

    @typechecked
    def loss(self, prompt: TT['batch', 'token', int], expected: TT['batch', 'token',
                                                                   int]) -> TT[()]:
        logits = self(prompt)
        loss = F.cross_entropy(logits.view(-1, self.out_voc_size), expected.view(-1))

        return loss


class UpcaseTransformer(Transformer):

    def __init__(self,
                 embedding_dim: int,
                 depth: int,
                 head_count: int,
                 block_size: int,
                 voc_size: int = 256):
        # We only predict whether a character should be upcased or not.
        # So we only need to predict 2 classes.
        super().__init__(embedding_dim, depth, head_count, block_size, voc_size, out_voc_size=2)

    @torch.no_grad()
    def predict(self, prompt: str) -> str:
        enc = list(prompt.encode())
        for i in range(0, len(enc), self.block_size):
            block = torch.tensor(enc[i:i + self.block_size], device=self.embedding.weight.device)
            logits = self(block.unsqueeze(0))
            flip = logits.argmax(-1).squeeze(0).tolist()
            for j, f in enumerate(flip):
                if f:
                    enc[i + j] = enc[i + j] ^ 0b00100000
        return bytes(enc).decode(errors='replace')

    @typechecked
    def loss(self,
             prompt: TT['batch', 'token', int],
             expected: TT['batch', 'token', int],
             weights: Tuple[float, float] = (1, 25)) -> TT[()]:
        logits = self(prompt)
        loss = F.cross_entropy(logits.view(-1, self.out_voc_size),
                               expected.view(-1),
                               weight=torch.tensor(weights, dtype=torch.float).to(logits.device))
        return loss


class PromptOMaticV1(Transformer):

    def __init__(self, embedding_dim: int, depth: int, head_count: int, block_size: int):
        super().__init__(embedding_dim, depth, head_count, block_size, voc_size=256)

    def forward(self,
                tokens: TT['batch', 'token', int],
                mask: None = None) -> TT['batch', 'token', 'voc_size']:
        T = tokens.shape[1]
        attention_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), 1)
        return super().forward(tokens, mask=attention_mask)

    @torch.no_grad()
    @typechecked
    def generate(self, start: str, length: int, beams: int = 1, n: int = 1) -> Union[str, List[str]]:
        x = torch.tensor(list(start.encode()), device=self.embedding.weight.device).unsqueeze(0)
        log_probs = torch.zeros(1)

        for _ in range(length):
            logits: TT['batch', 'vocab'] = self(x[:, -100:])[:, -1].softmax(-1)
            chosen: TT['batch', 'beams'] = torch.multinomial(logits, beams)
            x = x.repeat_interleave(beams, dim=0)
            log_probs = log_probs.repeat_interleave(beams)
            # print(log_probs.shape, logits.shape, chosen.shape, chosen)
            # print("gather", logits.gather(-1, chosen).shape)
            log_probs += logits.gather(-1, chosen).log().view(-1)
            # print(chosen, log_probs)
            # print(x.shape, chosen.shape)
            x = torch.cat([x, chosen.view(-1, 1)], dim=-1)

            # Keeping only the `beams` best generations
            log_probs, indices = log_probs.topk(max(beams, n))
            x = x[indices]

        b = [
            bytes(b).decode(errors='replace')
            for b in x[:n].tolist()
        ]
        if n == 1:
            return b[0]
        return b