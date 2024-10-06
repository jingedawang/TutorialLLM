import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    """
    Single head of self-attention.

    This module computes the self-attention for a sequence of vectors of shape (B, T, C),
    where B(batch) is the batch size, T(token) is the sequence length, and C(channel) is the dimension of each vector.
    The attention mechanism refers to the famous transformer paper "Attention is All You Need".
    """

    def __init__(self, dim_embedding, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(dim_embedding, head_size, bias=False)
        self.query = nn.Linear(dim_embedding, head_size, bias=False)
        self.value = nn.Linear(dim_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, dim_embedding, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(dim_embedding, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(dim_embedding, dim_embedding)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class TutorialLLM(nn.Module):

    def __init__(self, vocabulary_size, dim_embedding, block_size, num_head, num_layer, device):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.dim_embedding = dim_embedding
        self.block_size = block_size
        self.num_head = num_head
        self.num_layer = num_layer
        self.device = device
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocabulary_size, dim_embedding)
        self.position_embedding_table = nn.Embedding(block_size, dim_embedding)
        self.blocks = nn.Sequential(*[Block(dim_embedding, num_head, block_size) for _ in range(num_layer)])
        self.ln_f = nn.LayerNorm(dim_embedding) # final layer norm
        self.lm_head = nn.Linear(dim_embedding, vocabulary_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            if idx_next.item() == 0:
                break
        return idx

    @torch.no_grad()
    def estimate_loss_eval(self, iterations, dataset, stage='pretrain'):
        self.eval()
        if stage == 'pretrain':
            losses = torch.zeros(iterations)
            for k in range(iterations):
                X, Y = dataset.get_batch_pretrain('evaluate')
                logits, loss = self(X, Y)
                losses[k] = loss.item()
            loss = losses.mean()
        else:
            loss_sum = 0
            batch_generator = dataset.generate_batch_finetune('evaluate')
            for k, batch in enumerate(batch_generator):
                X, Y = batch
                logits, loss = self(X, Y)
                loss_sum += loss.item()
            loss = loss_sum / (k+1)
        self.train()
        return loss