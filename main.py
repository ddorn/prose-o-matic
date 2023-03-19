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

    def forward(self, x: TT['batch', 'token', 'embedding']) -> TT['batch', 'token', 'embedding']:
        q = torch.einsum('hei,bte->hbti', self.query, x)
        k = torch.einsum('hei,bte->hbti', self.key, x)
        v = torch.einsum('hei,bte->hbti', self.value, x)

        fit = torch.einsum('hbte,hbse->hbts', q, k) / x.shape[1]**0.5
        fit = torch.softmax(fit, dim=-1)
        attended = torch.einsum('hbts,hbsi->hbti', fit, v)
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

    def forward(self, x: TT['batch', 'token', 'embedding']) -> TT['batch', 'token', 'embedding']:
        x = x + self.attention(self.layernorm_1(x))
        x = x + self.feed_forward(self.layernorm_2(x))
        return x


class UpcasingTransformer(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 depth: int,
                 head_count: int,
                 block_size: int,
                 voc_size: int = 256):
        super().__init__()
        self.voc_size = voc_size
        self.block_size = block_size
        self.depth = depth

        self.embedding = nn.Embedding(voc_size, embedding_dim)
        self.positional_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, head_count) for _ in range(depth)],
            nn.LayerNorm(embedding_dim),
        )
        self.unembedding = nn.Linear(embedding_dim, 2)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @typechecked
    def forward(self, tokens: TT['batch', 'token', int]) -> TT['batch', 'token', 2]:
        assert tokens.shape[-1] <= self.block_size

        tokens = tokens.to(self.embedding.weight.device)

        emb: TT['batch', 'token', 'emb'] = self.embedding(tokens)
        pos_emb: TT['token', 'emb'] = self.positional_embedding(
            torch.arange(tokens.shape[1], device=tokens.device))
        x = emb + pos_emb

        x = self.blocks(x)

        out = self.unembedding(x)

        return out

    @typechecked
    def loss(self, prompt: TT['batch', 'token', int], expected: TT['batch', 'token',
                                                                   int]) -> TT[()]:
        logits = self(prompt)
        loss = F.cross_entropy(logits.view(-1, 2),
                               expected.view(-1),
                               weight=torch.tensor([1, 25.0]).to(logits.device))
        return loss

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