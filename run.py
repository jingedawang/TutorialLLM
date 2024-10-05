import json
import random
from model import BigramLanguageModel
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
train_iterations = 500
# The interval of iterations to evaluate the model
evaluation_interval = 50
# The learning rate of the optimizer
learning_rate = 1e-3
# Run the model on GPU(cuda) if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# The number of iterations to evaluate the model
eval_iters = 100


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
          X, Y = dataset.get_batch_pretrain('val')
          logits, loss = model(X, Y)
          losses[k] = loss.item()
      loss = losses.mean()
    else:
      loss_sum = 0
      batch_generator = dataset.generate_batch_finetune('val')
      for k, batch in enumerate(batch_generator):
        X, Y = batch
        logits, loss = model(X, Y)
        loss_sum += loss.item()
      loss = loss_sum / (k+1)
    model.train()
    return loss

print(f'------------------ Pretrain ------------------')
# The dimension of the embedding vector in the transformer
dim_embedding = 64
# The number of heads in the multi-head attention
num_head = 4
# The number of layers in the transformer
num_layer = 4
# The dropout rate, 0.0 means no dropout
dropout = 0.0
model = BigramLanguageModel(dataset.vocabulary_size, dim_embedding, block_size, num_head, num_layer, device)
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
        loss = estimate_loss_eval(stage='pretrain')
        print(f"step {iter}: train loss {mean_loss_train:.4f}, val loss {loss:.4f}")
        context = torch.tensor(dataset.encode('月'), dtype=torch.long, device=device).unsqueeze(0)
        print(dataset.decode(m.generate(context, max_new_tokens=100)[0].tolist()))

    # sample a batch of data
    xb, yb = dataset.get_batch_pretrain('train')

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
context = torch.tensor(dataset.encode('月'), dtype=torch.long, device=device).unsqueeze(0)
print(dataset.decode(m.generate(context, max_new_tokens=200)[0].tolist()))

# Set optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)



loss_sum = 0
epochs = 1
test_input = '<INS>請用以下題目寫一首詩<INP>月色<RES>'
for epoch in range(epochs):
  for iter, (xb, yb) in enumerate(dataset.generate_batch_finetune('train')):
    # every once in a while evaluate the loss on train and val sets
    if iter % evaluation_interval == 0:
        mean_loss_train = loss_sum / evaluation_interval
        loss_sum = 0
        loss = estimate_loss_eval('finetune')
        print(f"epoch {epoch}, step {iter}, train loss {mean_loss_train:.4f}, val loss {loss:.4f}")
        context = torch.tensor(dataset.encode(test_input), dtype=torch.long, device=device).unsqueeze(0)
        output = dataset.decode(m.generate(context, max_new_tokens=100)[0].tolist())
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
