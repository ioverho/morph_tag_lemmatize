import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
import torch.distributions as D

from utils.errors import ConfigurationError

class SequenceMask(nn.Module):
    """Masks a sequence tensor randomly.
    A form of regularization/data augmentation.
    Expects the tensor to be of form [B,T] or [T,B] (i.e. no dimensions) and of type torch.long.

    Args:
        mask_p (float): the probability of a single entry in the sequence gets masked. Defaults to 0.
        mask_idx (int): the mask index replacing the given values
        ign_idx (int): an index to be ignored, for example, padding
    """

    def __init__(self, mask_p: float = 0.0, mask_idx: int = 0, ign_idx: int = 1) -> None:
        super().__init__()

        self.mask_p = float(mask_p)
        self.register_buffer("mask_idx", torch.tensor(mask_idx, dtype=torch.long))
        self.ign_idx = ign_idx

    def forward(self, x: torch.Tensor):

        if self.training and self.mask_p > 0.0:
            x = torch.where(
                torch.logical_or(
                    torch.bernoulli(x, 1-self.mask_p), x == self.ign_idx
                ),
                x,
                self.mask_idx,
            )

        return x

class ResidualRNN(nn.Module):
    """An RNN with residual connections.

    Strongly inspired by UDIFY's implementation:
        https://github.com/Hyperparticle/udify/blob/master/udify/modules/residual_rnn.py

    """

    def __init__(
        self,
        input_size: int,
        h_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        batch_first: bool = True,
    ) -> None:
        super(ResidualRNN, self).__init__()

        self.input_size = input_size
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.residual = residual
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        rnn_type = rnn_type.lower()
        if rnn_type == "lstm":
            rnn_cell = nn.LSTM
        elif rnn_type == "gru":
            rnn_cell = nn.GRU
        else:
            raise ConfigurationError(f"Unknown RNN cell type {rnn_type}")

        layers = []
        for layer_index in range(num_layers):
            # Use hidden size on later layers so that the first layer projects and all other layers are residual
            input_ = input_size if layer_index == 0 else h_dim
            rnn = rnn_cell(
                input_,
                h_dim,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
            )
            layers.append(rnn)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for i, layer in enumerate(self.layers):
            h, _ = layer(x)
            if self.residual:
                # Sum the backward and forward states to allow residual connections
                h = h[:, :, : self.h_dim] + h[:, :, self.h_dim :]

            if self.residual and not (i == 0 and self.input_size != self.h_dim):
                x = x + self.dropout(h)
            else:
                # Skip residual connection on first layer (input size is different from hidden size)
                x = self.dropout(h)

        return x


class Char2Word(nn.Module):
    """Character to word embeddings.

    """

    def __init__(
        self,
        vocab_len: int,
        embedding_dim: int = 256,
        h_dim: int = 256,
        bidirectional: bool = True,
        out_dim: int = 256,
        padding_idx: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "lstm",
        **rnn_kwargs,
    ) -> None:
        super().__init__()

        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.bidirectional = bidirectional
        self.out_dim = out_dim
        self.padding_idx = padding_idx
        self.dropout = dropout
        self.rnn_type = rnn_type.lower()
        self.rnn_kwargs = rnn_kwargs

        if self.rnn_type == "lstm":
            rnn_cell = nn.LSTM
        elif self.rnn_type == "gru":
            rnn_cell = nn.GRU

        self.embed = nn.Embedding(
            num_embeddings=self.vocab_len,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
        )

        self.embed_dropout = nn.Dropout(p=self.dropout)

        self.rnn = rnn_cell(
            input_size=self.embedding_dim,
            hidden_size=self.h_dim,
            bidirectional=self.bidirectional,
            **self.rnn_kwargs,
        )

        self.rnn_dropout = nn.Dropout(p=self.dropout)

        if self.out_dim >= 0:
            self.out_project = nn.Linear(
                in_features=(2 if self.bidirectional else 1) * self.h_dim,
                out_features=self.out_dim,
            )

    def forward(self, chars: torch.Tensor, char_lens: torch.Tensor):

        c_embeds = self.embed(chars)

        c_embeds = self.embed_dropout(c_embeds)

        packed_c_embeds = pack_padded_sequence(
            c_embeds, char_lens, enforce_sorted=False
        )

        if self.rnn_type == "lstm":
            _, (h_T_out, _) = self.rnn(packed_c_embeds)
        elif self.rnn_type == "gru":
            _, h_T_out = self.rnn(packed_c_embeds)

        h_T_out = h_T_out.view(-1, (2 if self.bidirectional else 1) * self.h_dim)

        if self.out_dim > 0:
            h_T_out = self.rnn_dropout(h_T_out)

            c2w_embeds = self.out_project(h_T_out)

            return c2w_embeds

        else:
            return h_T_out


class LayerAttention(nn.Module):
    """A layer attention module.

    Args:
        L (int): number of layers to attend over
        u (float, optional): range for initiliazation. Defaults to 0.2.
        dropout (float, optional): probability of dropout. Defaults to 0.0.
    """

    def __init__(self, L: int, u: float = 0.2, dropout: float = 0.0) -> None:
        super().__init__()

        self.L = L
        self.u = u
        self.dropout = dropout

        self.h_w = nn.Parameter(torch.empty(self.L), requires_grad=True)
        self.c = nn.Parameter(torch.ones(1), requires_grad=True)
        init.uniform_(self.h_w, a=-self.u, b=self.u)

        if self.dropout > 0.0:
            self.register_buffer("mask_probs", self.dropout * torch.ones(L))
            self.register_buffer("mask_vals", torch.full((L,), -float(torch.inf)))

    def forward(self, h: torch.Tensor):
        """Attends on L layers of h.

        Args:
            h (torch.Tensor): takes a torch tensor representing the hidden layers
                of a transformer. Assumed shape of [B, T, L, D] or [T, B, L, D].
        """

        if self.dropout > 0.0 and self.training:
            # Layer dropout
            alpha = torch.softmax(
                torch.where(
                    torch.bernoulli(self.get_buffer("mask_probs")).bool(),
                    self.get_buffer("mask_vals"),
                    self.h_w,
                ),
                dim=0,
            )
        else:
            alpha = torch.softmax(self.h_w, dim=0)

        h_out = self.c * torch.sum((alpha.view(1, 1, -1, 1) * h), dim=2)

        return h_out


class ResidualMLP(nn.Module):
    """MLP layer with residual connection.

    """

    def __init__(self, act_fn: nn.Module = nn.ReLU(), **linear_kwargs) -> None:
        super().__init__()

        self.linear = nn.Linear(**linear_kwargs)
        self.act_fn = act_fn

    def forward(self, x):

        h = self.act_fn(self.linear(x))
        h = h + x

        return h


class MultiHeadSequenceAttention(nn.Module):
    """Multihead variant of scaled-dot product attention for sequence summarization.
    Converts a sequence of length T to 1 (e.g. character embeddings to token embeddings).

    The query is not input-dependent, but just a learned matrix.

    Args:
        d_in (int): input dimensionality
        d_out (int): desired output dimensionality. Must be a multiple of n_heads
        n_heads (int): number of attention heads
        dropout (float, optional): dropout applied within attention layer. Defaults to 0.0.
        batch_first: (bool, optional): in which order the dimensions appear [B, T] vs. [T, B]. Defaults to True.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.n_heads = n_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.mha = nn.MultiheadAttention(
            self.d_in, self.n_heads, dropout=self.dropout, batch_first=self.batch_first
        )

        self.Q = nn.Parameter(torch.empty((self.d_in,)), requires_grad=True)
        init.normal_(self.Q)

    def forward(self, x, attention_mask, require_attention_weights: bool = False):

        h, attn_weights = self.mha(
            self.Q.expand((x.size(0), 1, -1)),
            x,
            x,
            key_padding_mask=(1 - attention_mask).bool(),
        )

        if require_attention_weights:
            return h.squeeze(), attn_weights
        else:
            return h.squeeze()

