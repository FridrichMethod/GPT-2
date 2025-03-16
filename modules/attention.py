import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from config import GPT2Config


class LoRALinear(nn.Module):
    """
    A linear layer with a low-rank update.
    Given an input x, it computes:
         output = x @ W^T + scaling * (x @ A) @ B,
    where W is frozen, and A and B are trainable low-rank matrices.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0

        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features)

        # Low-rank factors (only if r > 0)
        if r > 0:
            # A: shape (in_features, r) and B: shape (r, out_features)
            self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
            self.lora_B = nn.Parameter(torch.randn(r, out_features) * 0.01)
            self.lora_dropout = (
                nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            )
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        if self.r > 0:
            # Compute low-rank update: (x @ A) @ B and scale it.
            lora_update = self.lora_dropout(x) @ self.lora_A @ self.lora_B
            result += self.scaling * lora_update

        return result


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

        # Determine if LoRA is enabled.
        self.use_lora = config.use_lora

        if self.use_lora:
            lora_r = config.lora_r
            lora_alpha = config.lora_alpha
            lora_dropout = config.lora_dropout

            self.query = LoRALinear(
                config.hidden_size,
                self.all_head_size,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.key = LoRALinear(
                config.hidden_size,
                self.all_head_size,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.value = LoRALinear(
                config.hidden_size,
                self.all_head_size,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        else:
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
        attn_scores.masked_fill_(
            self.causal_mask[: attn_scores.size(-2), : attn_scores.size(-1)] == 0,
            float("-inf"),
        )
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
