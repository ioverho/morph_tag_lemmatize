import torch
from torch.nn.utils.rnn import pad_sequence


class TokenDataloader(object):
    """A mock dataloader that batches based on sequence length.
    Total sequence length, and hopefully memory used, should stay constant.

        Args:
            dataset (_type_): _description_
            max_tokens (int): _description_
            max_batch_size (int): _description_
            device (_type_, optional): _description_. Defaults to torch.device("cpu").
            collate_fn (_type_, optional): _description_. Defaults to lambdax:x.
        """

    def __init__(
        self,
        dataset,
        max_tokens: int,
        max_batch_size: int,
        collate_fn: callable = lambda x: x,
    ):

        self.dataset = dataset
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.collate_fn = collate_fn

        # Generate batches such that the number of tokens is roughly constant ==
        self._batches = []
        prev_i = 0
        while True:
            i, _ = self.get_n_tokens_batch(prev_i)
            self._batches.append((prev_i, i))

            prev_i = i

            if prev_i >= len(self.dataset):
                break

        self._index = 0

    def get_n_tokens_batch(self, i):

        i_ = i
        ii_ = 0
        n_tokens = 0
        while True:
            if i_ == len(self.dataset) or ii_ == self.max_batch_size:
                break

            elif n_tokens + len(self.dataset[i_].tokens) <= self.max_tokens or ii_ == 0:
                n_tokens += len(self.dataset[i_].tokens)
                i_ += 1

            else:
                break

            ii_ += 1
        return i_, n_tokens

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._batches)

    def __next__(self):

        try:
            beg_idx, end_idx = self._batches[self._index]

        except IndexError:
            self._index = 0
            raise StopIteration

        batch = [self.dataset[i] for i in range(beg_idx, end_idx)]

        self._index += 1

        return self.collate_fn(batch)


class InferenceDataloader(TokenDataloader):
    """_summary_

    Args:
        dataset (_type_): _description_
        max_tokens (int): _description_
        max_batch_size (int): _description_
        char_pad_idx (int): _description_
        token_pad_idx (int): _description_
        id_to_script (dict): _description_
        device (_type_, optional): _description_. Defaults to torch.device("cpu").
    """

    def __init__(
        self,
        dataset,
        max_tokens: int,
        max_batch_size: int,
        char_pad_idx: int,
        token_pad_idx: int,
        id_to_script: dict,
        device=torch.device("cpu"),
    ):
        super().__init__(
            dataset, max_tokens, max_batch_size, collate_fn=self.collate_batch
        )

        self.char_pad_idx = char_pad_idx
        self.token_pad_idx = token_pad_idx
        self.id_to_script = id_to_script
        self.device = device

    def collate_batch(self, batch):
        docs_subset = [
            [
                d.chars_tensor,
                d.tokens,
                d.tokens_tensor,
                d.morph_tags,
                d.lemma_tags_tensor,
                d.lemmas,
            ]
            for d in batch
        ]

        (chars, tokens_raw, tokens, morph_tags, lemma_scripts_idx, lemmas,) = list(
            map(list, zip(*docs_subset))
        )

        # Characters [T_c, B]
        char_lens = [c.size(0) for seq in chars for c in seq]

        chars = pad_sequence(
            [c for seq in chars for c in seq], padding_value=self.char_pad_idx,
        ).to(self.device)

        lemma_scripts = [
            [
                self.id_to_script[l_script_idx]
                for l_script_idx in list(idx_tensor.numpy())
            ]
            for idx_tensor in lemma_scripts_idx
        ]

        # Tokens [T_t, B]
        token_lens = [seq.size(0) for seq in tokens]

        tokens_raw = [[token for token in seq] for seq in tokens_raw]

        tokens = pad_sequence(tokens, padding_value=self.token_pad_idx,).to(self.device)

        return (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            tokens,
            morph_tags,
            lemmas,
            lemma_scripts,
        )
