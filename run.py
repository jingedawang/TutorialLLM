import math
import torch
from torch.nn import functional as F

from dataset import Dataset
from model import TutorialLLM


print(f'{"-"*50}\nSTAGE 1: CONFIGURATION')
# The number of parallel items to process, known as the batch size
batch_size = 16
# The maximum length of a text to be processed
max_length = 256
# Run the model on GPU(cuda) if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set a seed for reproducibility
torch.manual_seed(2024)

print(f'{"-"*50}\nSTAGE 2: PREPARE THE DATA')
dataset = Dataset('data.json', batch_size, max_length, device)
print('Check a batch of pretrain data:')
print(dataset.get_batch_pretrain('train'))
print('Check a batch of finetune data:')
print(next(dataset.get_batch_generator_finetune('train')))

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
learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f'{"-"*50}\nSTAGE 4: PRETRAIN')
# The number of iterations for pretrain (each iteration processes a batch)
iterations_for_pretrain = 500
# The number of iterations to evaluate the pretrain process (each iteration processes a batch)
iterations_to_evaluate_pretrain = 100
# The interval of iterations to evaluate the pretrain process
interval_to_evaluate_pretrain = 50

loss_sum = math.nan
for i in range(iterations_for_pretrain):
    # Evaluate the model every evaluation_interval iterations
    if i % interval_to_evaluate_pretrain == 0:
        # Calculate the average loss for this interval
        mean_loss_train = loss_sum / interval_to_evaluate_pretrain
        loss_sum = 0
        loss = model.estimate_loss_eval(dataset, 'pretrain', iterations_to_evaluate_pretrain)
        print(f"Step {i}, train loss {mean_loss_train:.4f}, evaluate loss {loss:.4f}")

        # Let's generate a poem starting with the word '月' to see how the model is doing
        test_tokens = torch.tensor(dataset.encode('月'), dtype=torch.long, device=device).unsqueeze(0)
        print('Generate first 100 characters of poems starting with 月:')
        print(dataset.decode(model.generate(test_tokens, max_new_tokens=100)[0].tolist()))

    # Get a batch of pretrain data
    inputs, labels = dataset.get_batch_pretrain('train')

    # Forward pass and calculate the loss
    logits, loss = model(inputs, labels)
    loss_sum += loss.item()

    # Backward pass and update the model
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print('Save the pretrained model...')
# Save model to file
torch.save(model, 'model_pretrain.pth')

print(f'{"-"*50}\nSTAGE 5: FINETUNE')
# Reload the model from file
model.load_state_dict(torch.load('model_pretrain.pth').state_dict())
# Reset the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# The number of epochs to finetune the model
epochs_for_finetune = 1
# The interval of iterations to evaluate the finetune process
interval_to_evaluate_finetune = 50
test_input = '<INS>請用以下題目寫一首詩<INP>月色<RES>'
for epoch in range(epochs_for_finetune):
    loss_sum = math.nan
    for i, (inputs, labels) in enumerate(dataset.get_batch_generator_finetune('train')):
        # Evaluate the model every evaluation_interval iterations
        if i % interval_to_evaluate_finetune == 0:
            # Calculate the average loss for this interval
            mean_loss_train = loss_sum / interval_to_evaluate_finetune
            loss_sum = 0
            loss = model.estimate_loss_eval(dataset, 'finetune')
            print(f"Epoch {epoch}, step {i}, train loss {mean_loss_train:.4f}, evaluate loss {loss:.4f}")

            # Let's generate a poem with a given title to see how the model is doing
            test_tokens = torch.tensor(dataset.encode(test_input), dtype=torch.long, device=device).unsqueeze(0)
            output = dataset.decode(model.generate(test_tokens, max_new_tokens=100)[0].tolist())
            # Truncate the output to the '\0' character
            output = output[:output.find('\0')]
            print('Generate a complete poem for title 月色:')
            print(output[len(test_input):])

        # Forward pass and calculate the loss
        logits, loss = model(inputs, labels)
        loss_sum += loss.item()

        # Backward pass and update the model
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

print('Save the finetuned model...')
# Save model to file
torch.save(model, 'model_finetune.pth')

print(f'{"-"*50}\nSTAGE 6: ALIGNMENT')
# Reload the model from file
model.load_state_dict(torch.load('model_finetune.pth').state_dict())
# Clone the model as a reference model and switch it to evaluation mode
reference_model = model.copy().eval()
# Reset the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# A hyperparameter to control the strength of the alignment loss, larger beta means stronger alignment
beta = 0.1
# The number of epochs to align the model
epochs_for_alignment = 1
# The interval of iterations to evaluate the alignment process
interval_to_evaluate_alignment = 50
for epoch in range(epochs_for_alignment):
    loss_sum = positive_distance_sum = negative_distance_sum = math.nan
    for i, (positive_inputs, positive_labels, negative_inputs, negative_labels) in enumerate(dataset.get_batch_generator_alignment('train')):
        # Evaluate the model every evaluation_interval iterations
        if i % interval_to_evaluate_alignment == 0:
            # Calculate the average loss for this interval
            mean_loss_train = loss_sum / interval_to_evaluate_alignment
            loss_sum = positive_distance_sum = negative_distance_sum = 0
            loss = model.estimate_loss_eval(dataset, 'alignment')
            print(f"Epoch {epoch}, step {i}, train loss {mean_loss_train:.4f}, evaluate loss {loss:.4f}")

        # Forward pass the positive and negative samples on model and reference model
        positive_logits, positive_loss = model(positive_inputs, positive_labels, False)
        negative_logits, negative_loss = model(negative_inputs, negative_labels, False)
        reference_positive_logits, reference_positive_loss = reference_model(positive_inputs, positive_labels)
        reference_negative_logits, reference_negative_loss = reference_model(negative_inputs, negative_labels)

        # Implement the DPO(Direct Preference Optimiazation) loss
        loss = (positive_loss - reference_positive_loss) - (negative_loss - reference_negative_loss)
        loss = F.logsigmoid(beta * loss).mean()
        loss_sum += loss.item()

        # Calculate the distance between the model and the reference model
        positive_distance = (positive_loss - reference_positive_loss).mean().detach()
        negative_distance = (negative_loss - reference_negative_loss).mean().detach()
        positive_distance_sum += positive_distance
        negative_distance_sum += negative_distance

        # Backward pass and update the model
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()