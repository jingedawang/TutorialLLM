import json
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

print(f'------------------ Prepare Data ------------------')


"""
Configuration
"""
# The number of parallel items to process
batch_size = 16
# The maximum length of a text to be processed
block_size = 256
# The number of iterations to train (each iteration processes a batch)
train_iterations = 50000
# The interval of iterations to evaluate the model
evaluation_interval = 500
# The learning rate of the optimizer
learning_rate = 1e-3
# Run the model on GPU(cuda) if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# The number of iterations to evaluate the model
eval_iters = 200
# The dimension of the embedding vector in the transformer
dim_embedding = 64
# The number of heads in the multi-head attention
num_head = 4
# The number of layers in the transformer
num_layer = 4
# The dropout rate, 0.0 means no dropout
dropout = 0.0

# Set a seed for reproducibility
torch.manual_seed(1993)

"""
Prepare Data
"""
from dataset import Dataset
dataset = Dataset('data.json', batch_size, block_size, device)
print('Check a batch of pretrain data:')
print(dataset.get_batch_pretrain('train'))
print('Check a batch of finetune data:')
print(next(dataset.generate_batch_finetune('train')))

@torch.no_grad()
def estimate_loss_eval(stage='pretrain'):
    model.eval()
    if stage == 'pretrain':
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
          X, Y = get_batch_pretrain('val')
          logits, loss = model(X, Y)
          losses[k] = loss.item()
      loss = losses.mean()
    else:
      loss_sum = 0
      batch_generator = generate_batch_finetune('val')
      for k, batch in enumerate(batch_generator):
        X, Y = batch
        logits, loss = model(X, Y)
        loss_sum += loss.item()
      loss = loss_sum / (k+1)
    model.train()
    return loss

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(dim_embedding, head_size, bias=False)
        self.query = nn.Linear(dim_embedding, head_size, bias=False)
        self.value = nn.Linear(dim_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(dim_embedding, dim_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocabulary_size, dim_embedding)
        self.position_embedding_table = nn.Embedding(block_size, dim_embedding)
        self.blocks = nn.Sequential(*[Block(dim_embedding, n_head=num_head) for _ in range(num_layer)])
        self.ln_f = nn.LayerNorm(dim_embedding) # final layer norm
        self.lm_head = nn.Linear(dim_embedding, vocabulary_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
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
            idx_cond = idx[:, -block_size:]
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




print(f'------------------ Pretrain ------------------')

model = BigramLanguageModel()
model.train()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

loss_sum = 0
for iter in range(train_iterations):

    # every once in a while evaluate the loss on train and val sets
    if iter % evaluation_interval == 0 or iter == train_iterations - 1:
        mean_loss_train = loss_sum / evaluation_interval
        loss_sum = 0
        loss = estimate_loss_eval()
        print(f"step {iter}: train loss {mean_loss_train:.4f}, val loss {loss:.4f}")
        context = torch.tensor(encode('月'), dtype=torch.long, device=device).unsqueeze(0)
        print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))

    # sample a batch of data
    xb, yb = get_batch_pretrain('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    loss_sum += loss.item()

    # backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save model to file
torch.save(model, 'model.pth')



print(f'------------------ Finetune ------------------')

# Load model from file
model = torch.load('model.pth')
m = model.to(device)

# generate from the model
context = torch.tensor(encode('月'), dtype=torch.long, device=device).unsqueeze(0)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))

# Set optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)



loss_sum = 0
epochs = 20
test_input = '<INS>請用以下題目寫一首詩<INP>月色<RES>'
for epoch in range(epochs):
  for iter, (xb, yb) in enumerate(generate_batch_finetune('train')):
    # every once in a while evaluate the loss on train and val sets
    if iter % evaluation_interval == 0:
        mean_loss_train = loss_sum / evaluation_interval
        loss_sum = 0
        loss = estimate_loss_eval('finetune')
        print(f"epoch {epoch}, step {iter}, train loss {mean_loss_train:.4f}, val loss {loss:.4f}")
        context = torch.tensor(encode(test_input), dtype=torch.long, device=device).unsqueeze(0)
        output = decode(m.generate(context, max_new_tokens=100)[0].tolist())
        # Truncate the output to the '\0' character
        output = output[:output.find('\0')]
        print(output[len(test_input):])

    # evaluate the loss
    logits, loss = model(xb, yb)
    loss_sum += loss.item()

    # backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
