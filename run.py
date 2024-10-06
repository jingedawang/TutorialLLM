import math
import torch

from model import TutorialLLM


print(f'{"-"*50}\nSTAGE 1: CONFIGURATION')
# The number of parallel items to process, known as the batch size
batch_size = 16
# The maximum length of a text to be processed
max_length = 256
# The number of iterations to train (each iteration processes a batch)
iterations_for_training = 500
# The number of iterations to evaluate the model (each iteration processes a batch)
iterations_for_evaluation = 100
# The interval of iterations to evaluate the model
evaluation_interval = 50
# The learning rate of the optimizer
learning_rate = 1e-3
# Run the model on GPU(cuda) if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set a seed for reproducibility
torch.manual_seed(2024)

print(f'{"-"*50}\nSTAGE 2: PREPARE THE DATA')
from dataset import Dataset
dataset = Dataset('data.json', batch_size, max_length, device)
print('Check a batch of pretrain data:')
print(dataset.get_batch_pretrain('train'))
print('Check a batch of finetune data:')
print(next(dataset.generate_batch_finetune('train')))


print(f'{"-"*50}\nSTAGE 3: CREATE THE MODEL')
# The dimension of the embedding vector in the transformer
dim_embedding = 64
# The number of heads in the multi-head attention
num_head = 4
# The number of layers in the transformer
num_layer = 4

# Create a TutorialLLM instance
model = TutorialLLM(dataset.vocabulary_size, dim_embedding, max_length, num_head, num_layer, device)
# Switch the model to training mode
model.train()
model.to(device)
# Show the model size
print(f'Our model has {sum(parameter.numel() for parameter in model.parameters())/1e6} M parameters')
# Use AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f'{"-"*50}\nSTAGE 4: PRETRAIN')
loss_sum = math.nan
for i in range(iterations_for_training):
    # Evaluate the model every evaluation_interval iterations
    if i % evaluation_interval == 0 or i == iterations_for_training - 1:
        # Calculate the average loss for this interval
        mean_loss_train = loss_sum / evaluation_interval
        loss_sum = 0
        loss = model.estimate_loss_eval(iterations_for_evaluation, dataset, 'pretrain')
        print(f"Step {i}, train loss {mean_loss_train:.4f}, evaluate loss {loss:.4f}")

        # Let's generate a poem starting with the word '月' to see how the model is doing
        test_tokens = torch.tensor(dataset.encode('月'), dtype=torch.long, device=device).unsqueeze(0)
        print('Generate first 100 characters of poems starting with 月:')
        print(dataset.decode(model.generate(test_tokens, max_new_tokens=100)[0].tolist()))

    # Get a batch of pretrain data
    xb, yb = dataset.get_batch_pretrain('train')

    # Forward pass and calculate the loss
    logits, loss = model(xb, yb)
    loss_sum += loss.item()

    # Backward pass and update the model
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print('Save the pretrained model...')
# Save model to file
torch.save(model, 'model_pretrain.pth')
# Reload the model from file
model.load_state_dict(torch.load('model_pretrain.pth').state_dict())
# Reset the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f'{"-"*50}\nSTAGE 5: FINETUNE')
loss_sum = math.nan
epochs = 1
test_input = '<INS>請用以下題目寫一首詩<INP>月色<RES>'
for epoch in range(epochs):
  for i, (xb, yb) in enumerate(dataset.generate_batch_finetune('train')):
    # Evaluate the model every evaluation_interval iterations
    if i % evaluation_interval == 0:
        # Calculate the average loss for this interval
        mean_loss_train = loss_sum / evaluation_interval
        loss_sum = 0
        loss = model.estimate_loss_eval(iterations_for_evaluation, dataset, 'finetune')
        print(f"Epoch {epoch}, step {i}, train loss {mean_loss_train:.4f}, evaluate loss {loss:.4f}")

        # Let's generate a poem with a given title to see how the model is doing
        test_tokens = torch.tensor(dataset.encode(test_input), dtype=torch.long, device=device).unsqueeze(0)
        output = dataset.decode(model.generate(test_tokens, max_new_tokens=100)[0].tolist())
        # Truncate the output to the '\0' character
        output = output[:output.find('\0')]
        print('Generate a complete poem for title 月色:')
        print(output[len(test_input):])

    # Forward pass and calculate the loss
    logits, loss = model(xb, yb)
    loss_sum += loss.item()

    # Backward pass and update the model
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print('Save the finetuned model...')
# Save model to file
torch.save(model, 'model_finetune.pth')