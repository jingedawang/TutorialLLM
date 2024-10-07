import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from dataset import Dataset


class AttentionHead(nn.Module):
    """
    Single head of self-attention.

    This module computes the self-attention for a batch of sequences,
    where each sequence is maximum `max_length` tokens long
    and each token is a vector of dimension `dim_embed`.
    The attention mechanism refers to the famous transformer paper "Attention is All You Need".
    """

    def __init__(self, dim_embed: int, head_size: int, max_length: int):
        """
        Initialize the module with 3 linear layers and a mask buffer.

        Args:
            dim_embed: The dimension of each token vector in the input sequences.
            head_size: The dimension of the output vectors.
            max_length: The maximum length of any token sequence. Known as the maximum context length.
        """
        super().__init__()
        # Create linear layers to project the input tensor to key tensor, query tensor, and value tensor.
        # The 3 layers do the same transformation but they do not share weights.
        # After training, the layers will learn different aspects of the input vectors.
        self.project_to_key = nn.Linear(dim_embed, head_size, bias=False)
        self.project_to_query = nn.Linear(dim_embed, head_size, bias=False)
        self.project_to_value = nn.Linear(dim_embed, head_size, bias=False)
        # Create a matrix buffer to mask a square matrix to a lower triangular matrix.
        # This is used to add the causal constraint to the self-attention mechanism,
        # which means that each token can only see the previous tokens but not the future tokens.
        self.register_buffer('tril', torch.tril(torch.ones(max_length, max_length)))

    def forward(self, input: Tensor):
        """
        Compute the self-attention for the input tensor.

        Args:
            input: A tensor of shape (B, T, `dim_embed`) where B is the batch size,
                T is the token sequence length, and `dim_embed` is the dimension of each token vector.

        Returns:
            A tensor of shape (B, T, `head_size`). Each vector in the input tensor is transformed
            into a new vector of dimension `head_size` that captures the self-attention.
        """
        B, T, dim_embed = input.shape
        # Project the input tensor to key tensor, query tensor, and value tensor
        key = self.project_to_key(input)          # (B, T, dim_embed) -> (B, T, head_size)
        query = self.project_to_query(input)      # (B, T, dim_embed) -> (B, T, head_size)
        value = self.project_to_value(input)      # (B, T, dim_embed) -> (B, T, head_size)
        # Compute the self-attention weights
        weights = query @ key.transpose(-2, -1)   # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # Scale the attention weights to
        weights *= dim_embed ** -0.5
        # Mask the attention weights to respect the causal constraint
        # Slice the tril matrix to fit the size of the current input
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Turn the attention weights into probabilities
        weights = F.softmax(weights, dim=-1)
        # Apply the attention to the values
        output = weights @ value                  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return output

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention.

    The multi-head self-attention is composed by aggregating the outputs of several `AttentionHead` modules.
    """

    def __init__(self, dim_embed: int, num_heads: int, head_size: int, max_length: int):
        """
        Initialize the module with concatenated `AttentionHead`s and a projection layer.

        Args:
            dim_embed: The dimension of each token vector in the input tensor.
            num_heads: The number of heads included in a multi-head attention.
            head_size: The dimension of the output vectors for each head.
            max_length: The maximum length of any token sequence. Known as the maximum context length.
        """
        super().__init__()
        # Create a list of `num_heads` attention heads
        self.heads = nn.ModuleList([AttentionHead(dim_embed, head_size, max_length) for _ in range(num_heads)])
        # Create a linear layer to project the concatenated output of all heads to the original dimension.
        # In our case, the concatenated output is happen to be the same as the original dimension, so we can skip
        # this projection layer. But in general, the output of the heads may have different dimension than the input.
        self.project = nn.Linear(head_size * num_heads, dim_embed)

    def forward(self, input: Tensor):
        """
        Compute the multi-head self-attention for the input tensor.

        Args:
            input: A tensor of shape (B, T, `dim_embed`) where B is the batch size,
                T is the token sequence length, and `dim_embed` is the dimension of each token vector.

        Returns:
            A tensor of shape (B, T, `dim_embed`). Each vector in the input tensor is transformed
            into a new vector of same dimension that captures the multi-head self-attention.
        """
        # Send the input tensor to each attention head and concatenate the outputs
        output = torch.cat([head(input) for head in self.heads], dim=-1)    # (B, T, dim_embed) -> [(B, T, head_size)] * num_heads -> (B, T, head_size * num_heads)
        # Project the concatenated output to the original dimension
        output = self.project(output)                                       # (B, T, head_size * num_heads) -> (B, T, dim_embed)
        return output

class FeedFoward(nn.Module):
    """
    Feed-forward neural network.

    This module is a simple feed-forward neural network with 2 linear layers and
    a ReLU activation function. It scales up 4 times the input dimension and then
    scales down back to the original dimension to learn more complex patterns in the
    input tensor.
    """

    def __init__(self, dim_embed: int):
        """
        Initialize the module with 2 linear layers and a ReLU activation function.

        Args:
            dim_embed: The dimension of each token vector in the input tensor.
        """
        super().__init__()
        # Create a sequential module with 2 linear layers and a ReLU activation function.
        # The first layer scales up the input dimension by 4 times. Then the ReLU activation
        # function is applied to the output. Finally, the second layer scales down the
        # dimension back to the original size.
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_embed, 4 * dim_embed),
            nn.ReLU(),
            nn.Linear(4 * dim_embed, dim_embed)
        )

    def forward(self, input: Tensor):
        """
        Compute the output of the feed-forward neural network for the input tensor.

        Args:
            input: A tensor of shape (B, T, `dim_embed`) where B is the batch size,
                T is the token sequence length, and `dim_embed` is the dimension of each token vector.
        """
        return self.feed_forward(input)           # (B, T, dim_embed) -> (B, T, 4 * dim_embed) -> (B, T, dim_embed)

class TranformerBlock(nn.Module):
    """
    Transformer block.

    This module is a single transformer block that consists of a multi-head self-attention
    followed by a feed-forward neural network. Layer normalization is applied before each
    sub-module to stabilize the training process.
    """

    def __init__(self, dim_embed: int, num_heads: int, max_length: int):
        """
        Initialize the module with a multi-head self-attention, a feed-forward neural network,
        and 2 layer normalization layers.

        Args:
            dim_embed: The dimension of each token vector in the input tensor.
            num_heads: The number of heads included in a multi-head attention.
            max_length: The maximum length of any token sequence. Known as the maximum context length.
        """
        super().__init__()
        # We choose the `head_size` as a divisor of `dim_embed` for simplicity.
        head_size = dim_embed // num_heads
        # Create a multi-head self-attention module
        self.multi_head_attention = MultiHeadAttention(dim_embed, num_heads, head_size, max_length)
        # Create a feed-forward neural network module
        self.feed_forward = FeedFoward(dim_embed)
        # Create 2 layer normalization layers
        self.layer_norm1 = nn.LayerNorm(dim_embed)
        self.layer_norm2 = nn.LayerNorm(dim_embed)

    def forward(self, input: Tensor):
        """
        Compute the output of the transformer block for the input tensor.

        We treat the attention heads and the feed-forward neural network as residual
        steams.

        Args:
            input: A tensor of shape (B, T, `dim_embed`) where B is the batch size,
                T is the token sequence length, and `dim_embed` is the dimension of each token vector.

        Returns:
            A tensor of shape (B, T, `dim_embed`). Each vector in the input tensor is transformed
            into a new vector of same dimension that captures the transformer mechanism.
        """
        # Apply the multi-head self-attention and add to the input tensor as a residual stream
        output = input + self.multi_head_attention(self.layer_norm1(input)) # (B, T, dim_embed) + (B, T, dim_embed) -> (B, T, dim_embed)
        # Apply the feed-forward neural network and add to the output tensor as a residual stream
        output = output + self.feed_forward(self.layer_norm2(output))       # (B, T, dim_embed) + (B, T, dim_embed) -> (B, T, dim_embed)
        return output

class TutorialLLM(nn.Module):
    """
    Tutorial Large Language Model.

    This is a very simple language model built on top of the transformer architecture.
    It resembles the GPT-2 model but used for educational purposes only.
    """

    def __init__(self, vocabulary_size: int, dim_embed: int, max_length: int, num_head: int, num_layer: int, device: str):
        """
        Initialize the model with a token embedding table, a position embedding table,
        several transformer blocks, a final layer normalization layer, and a linear layer.

        Args:
            vocabulary_size: The number of unique tokens in the vocabulary.
            dim_embed: The dimension of the embedding vector throughout the model.
            max_length: The maximum length of a text to be processed. Known as the maximum context length.
            num_head: The number of heads in the multi-head attention.
            num_layer: The number of transformer blocks in the model.
            device: The device to run the model on, either 'cpu' or 'cuda'.
        """
        super().__init__()
        self.max_length = max_length
        self.device = device
        # Create a token embedding table to convert token ids to vectors
        self.token_embedding_table = nn.Embedding(vocabulary_size, dim_embed)
        # Create a position embedding table to add positional information to the token vectors
        self.position_embedding_table = nn.Embedding(max_length, dim_embed)
        # Create a series of transformer blocks
        self.transformer_blocks = nn.Sequential(*[TranformerBlock(dim_embed, num_head, max_length) for _ in range(num_layer)])
        # Create a layer normalization layer for the final output
        self.layer_norm_final = nn.LayerNorm(dim_embed)
        # Create a linear layer to project the output from embedding space to vocabulary space
        self.project = nn.Linear(dim_embed, vocabulary_size)

    def forward(self, token_ids: Tensor, labels: Tensor = None):
        """
        Compute the forward pass of the model.

        Args:
            token_ids: A tensor of shape (B, T) where B is the batch size and T is the token sequence length.
                The tensor contains the token ids of the input sequences.
            labels: A tensor of shape (B, T) where B is the batch size and T is the token sequence length.
                The tensor contains the groundtruth token ids of the target sequences. If None, the model
                will not compute the loss.
        """
        B, T = token_ids.shape
        # Get the token embedding and position embedding
        token_embedding = self.token_embedding_table(token_ids) # (B, T) -> (B, T, dim_embed)
        position_embedding = self.position_embedding_table(torch.arange(T, device=self.device)) # (T) -> (T, dim_embed)
        # Add the token embedding and position embedding in the last dimension
        embedding = token_embedding + position_embedding        # (B, T, dim_embed) + (T, dim_embed) -> (B, T, dim_embed)
        # Send the embedding through the transformer blocks
        embedding = self.transformer_blocks(embedding)          # (B, T, dim_embed) -> (B, T, dim_embed)
        # Apply layer normalization to the final output
        embedding = self.layer_norm_final(embedding)            # (B, T, dim_embed) -> (B, T, dim_embed)
        # Project the output to the vocabulary space
        logits = self.project(embedding)                        # (B, T, dim_embed) -> (B, T, vocabulary_size)

        if labels is None:
            loss = None
        else:
            B, T, vocabulary_size = logits.shape
            # Flatten the logits to a list of vectors in the vocabulary space
            logits = logits.view(B * T, vocabulary_size)
            # Flatten the labels to a list of token ids
            labels = labels.view(B * T)
            # Compute the cross-entropy loss between the logits and the labels
            loss = F.cross_entropy(logits, labels)

        return logits, loss

    def generate(self, token_ids: Tensor, max_new_tokens: int):
        """
        Generate subsequent tokens given the input tokens.

        Args:
            token_ids: A tensor of shape (B, T) where B is the batch size and T is the token sequence length.
                The tensor contains the token ids of the input sequences.
            max_new_tokens: The maximum number of new tokens to generate.
        """
        for _ in range(max_new_tokens):
            # Crop the input sequence to if it exceeds the maximum length
            token_ids_available = token_ids[:, -self.max_length:]   # (B, T) -> (B, T'), where T' = min(T, max_length)
            # Run the model to get the logits
            logits, loss = self(token_ids_available)                # (B, T') -> (B, T', vocabulary_size)
            # Pick the logits of the last token where the next token should be predicted
            logits = logits[:, -1, :]                               # (B, T', vocabulary_size) -> (B, vocabulary_size)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)                       # (B, vocabulary_size) -> (B, vocabulary_size)
            # Sample the next token from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)      # (B, vocabulary_size) -> (B, 1)
            # Append the next token to the input sequence for the next iteration
            token_ids = torch.cat((token_ids, idx_next), dim=1)     # (B, T) + (B, 1) -> (B, T+1)
            # Stop if the next token is the end-of-sequence token whose id is 0
            if idx_next.item() == 0:
                break
        return token_ids

    @torch.no_grad()
    def estimate_loss_eval(self, iterations: int, dataset: Dataset, stage='pretrain'):
        """
        Estimate the average loss of the model on the evaluation dataset.

        Args:
            iterations: The number of iterations to evaluate the model (each iteration processes a batch).
                This argument is only used for the pretrain stage. For the finetune stage, the number of
                iterations is determined by the evaluation dataset.
            dataset: The dataset where the evaluation data is stored.
            stage: The training stage of the model, either 'pretrain' or 'finetune'.

        Returns:
            The average loss of the model on the evaluation dataset.
        """
        if stage == 'pretrain':
            losses = torch.zeros(iterations)
            # Evaluate the model `iterations` times
            for k in range(iterations):
                # Get a batch of pretrain data and compute the loss
                token_ids, labels = dataset.get_batch_pretrain('evaluate')
                logits, loss = self(token_ids, labels)
                losses[k] = loss.item()
            loss = losses.mean()
        else:
            loss_sum = 0
            # Get a batch generator of finetune data
            batch_generator = dataset.get_batch_generator_finetune('evaluate')
            # Evaluate the model by processing all batches generated by the generator
            for k, batch in enumerate(batch_generator):
                token_ids, labels = batch
                logits, loss = self(token_ids, labels)
                loss_sum += loss.item()
            loss = loss_sum / (k + 1)
        return loss