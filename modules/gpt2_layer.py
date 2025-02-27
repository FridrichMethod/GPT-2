import torch
import torch.nn.functional as F
from torch import nn

from config import GPT2Config
from modules.attention import CausalSelfAttention


class GPT2Layer(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()

        # Layer normalization for multi-head attention.
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        # Multi-head attention.
        self.self_attention = CausalSelfAttention(config)
        # Add-norm for multi-head attention.
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Layer normalization for feed forward.
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        # Feed forward.
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # Add-norm for feed forward.
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        dense_layer: nn.Linear,
        dropout: nn.Dropout,
    ) -> torch.Tensor:
        """This function implements the add-norm component of the transformer.

        Args:
        -----
        input_tensor: torch.Tensor
            The input tensor to the add-norm.
        output_tensor: torch.Tensor
            The output tensor from the sub-layer.
        dense_layer: nn.Linear
            The dense layer applied to the output tensor.
        dropout: nn.Dropout
            The dropout layer applied to the dense output.

        Returns:
        --------
        output_tensor: torch.Tensor
            The output tensor after applying the add-norm.


        Notes:
        ------
        - This function is applied after the multi-head attention layer as well as after the feed forward layer.
        - GPT-2 layer applies dropout to the transformed output of each sub-layer,
          before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
          IN THIS FUNCTION.
        """

        output_tensor = dense_layer(output_tensor)
        output_tensor = dropout(output_tensor)
        output_tensor += input_tensor

        return output_tensor

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through a GPT-2 transformer layer.

        Args:
        -----
        hidden_states: torch.Tensor
            The input tensor representing the hidden states for each token.
        attention_mask: torch.Tensor
            The attention mask used to prevent attention to certain positions.

        Returns:
        --------
        hidden_states: torch.Tensor
            The output tensor after applying multi-head attention and feed-forward layer.

        Notes:
        ------
        - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
        - Layer normalization applied *before* the attention layer and feed-forward layer.
        - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
        - A feed-forward layer that applies transformations to further refine the hidden states.
        """

        normed_states = self.attention_layer_norm(hidden_states)
        attn_output = self.self_attention(normed_states, attention_mask)
        hidden_states = self.add(
            hidden_states, attn_output, self.attention_dense, self.attention_dropout
        )

        normed_states = self.out_layer_norm(hidden_states)
        feed_forward = self.interm_af(self.interm_dense(normed_states))
        hidden_states = self.add(
            hidden_states, feed_forward, self.out_dense, self.out_dropout
        )

        return hidden_states
