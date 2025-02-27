import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from config import GPT2Config


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        """This module implements the causal self-attention mechanism.

        Args:
        -----
        config: GPT2Config
            The configuration of the GPT2 model.
        """
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size

        # Initialize the linear transformation layers for key, value, query.
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # This dropout is applied to normalized attention scores following the original
        # implementation of transformer. Although it is a bit unusual, we empirically
        # observe that it yields better performance.
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        causal_mask = torch.tril(
            torch.ones(config.max_position_embeddings, config.max_position_embeddings)
        )
        self.register_buffer("causal_mask", causal_mask)

    def transform(self, x: torch.Tensor, linear_layer: nn.Linear) -> torch.Tensor:
        """This function projects the hidden state to key, value, query for multi-head attention.

        Args:
        -----
        x: torch.Tensor
            The hidden state of the model.
        linear_layer: nn.Linear
            The linear transformation layer for key, value, query.

        Returns:
        --------
        proj: torch.Tensor
            The projected hidden state.
        """

        # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
        proj = linear_layer(x)
        # Next, we need to produce multiple heads for the proj. This is done by spliting the
        # hidden state to self.num_attention_heads, each of size self.attention_head_size.
        # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
        proj = rearrange(proj, "b t (h d) -> b h t d", h=self.num_attention_heads)

        return proj

    def attention(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """This function calculates the multi-head attention.

        Args:
        -----
        key: torch.Tensor
            The key tensor.
        query: torch.Tensor
            The query tensor.
        value: torch.Tensor
            The value tensor.
        attention_mask: torch.Tensor
            The attention mask tensor.

        Returns:
        --------
        attn_value: torch.Tensor
            The attention value tensor.
        """

        attn_scores = torch.einsum("b h i d, b h j d -> b h i j", query, key)
        attn_scores /= torch.tensor(self.attention_head_size).sqrt()
        attn_scores.masked_fill_(self.causal_mask[: attn_scores.size(-2), : attn_scores.size(-1)] == 0, float("-inf"))
        attn_scores += attention_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_value = torch.einsum("b h i j, b h j d -> b h i d", attn_probs, value)
        attn_value = rearrange(attn_value, "b h t d -> b t (h d)")

        return attn_value

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Performs the forward pass for multi-head attention.

        Args:
        -----
        hidden_states: torch.Tensor [batch_size, seq_len, hidden_state]
            The input tensor representing the hidden states for each token.
        attention_mask: torch.Tensor [batch_size, 1, 1, seq_len]
            The attention mask used to prevent attention to certain positions.

        Returns:
        --------
        attn_value: torch.Tensor [batch_size, seq_len, hidden_state]
            The output tensor after applying multi-head attention.
        """

        # First, we have to generate the key, value, query for each token for multi-head attention
        # using self.transform (more details inside the function).
        # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)

        # Calculate the multi-head attention.
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)

        return attn_value
