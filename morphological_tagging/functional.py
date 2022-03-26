import torch
from torch.nn.utils.rnn import pad_sequence
import opt_einsum as oe

USE_OPT_EINSUM = True

def einsum(formula, *tensors):

    if USE_OPT_EINSUM:
        return oe.contract(formula, *tensors, backend="torch")
    else:
        return torch.einsum(formula, *tensors)


def break_batch(x, char_lens, batch_first: bool = True, padding_value: int = 0):
    """Move from [B, L_s x L_t, D] output tensor to [B x L_s, L_t, D] tensor
    i.e. break batch of sentences into batch of tokens
    """

    output, mask = [], []
    for i, lens in enumerate(char_lens):
        seq_tokens, seq_mask = [], []

        beg = 1
        for l in lens:
            token_chars = x[i, beg : beg + l]
            seq_tokens.append(token_chars)
            seq_mask.append(torch.ones((token_chars.size(0),), device=x.device))
            beg += l

        output.extend(seq_tokens)
        mask.extend(seq_mask)

    output = pad_sequence(output, batch_first=batch_first, padding_value=padding_value,)
    mask = pad_sequence(mask, batch_first=batch_first, padding_value=0)

    return output, mask


def fuse_batch(x, token_lens, batch_first: bool = True, padding_value: int = 0):
    """Move from [BxL_s, D] output tensor to [B, L_s, D] tensor.
    i.e. fuse batch of tokens into batch of sentences
    """
    seqs = []
    beg = 0
    for l in token_lens:
        token_chars = x[beg : beg + l, :]
        seqs.append(token_chars)
        beg += l

    tokens_batch = pad_sequence(
        seqs, batch_first=batch_first, padding_value=padding_value
    )

    return tokens_batch
