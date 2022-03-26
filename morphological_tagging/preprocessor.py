import os
import re
from typing import Callable, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.vocab import Vectors
import nltk
from nltk.tokenize import word_tokenize

from transformers import AutoConfig, AutoTokenizer, AutoModel
from utils.errors import ConfigurationError

FASTTEXT_LANG_CONVERSION = {
    "Arabic": "ar",
    "Czech": "cs",
    "Dutch": "nl",
    "English": "en",
    "Finnish": "fi",
    "French": "fr",
    "Russian": "ru",
    "Turkish": "tr",
}


class FastText(Vectors):
    """Slightly rewritten FastText vector class to use multi-lingual embeddings.
    """

    url_base = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{}.300.vec.gz"
    dim = 300

    def __init__(self, language="en", **kwargs):

        if len(language) > 2:
            language = FASTTEXT_LANG_CONVERSION.get(language)
            if language is None:
                raise ConfigurationError(
                    f"Language acronym '{language}' not recognized."
                )

        url = self.url_base.format(language)
        name = os.path.basename(url)
        super(FastText, self).__init__(name, url=url, **kwargs)


class UDPipe2PreProcessor(nn.Module):
    """A module that pre-processes text to match the UDPipe2 input.
    Can take either a single text (str), a list of texts (List(str)) or a batch with tokens, token_lens (Tuple(List[int], List[str])).
    In the former two cases, the text is tokenized. In the latter, tokenization is already implied.

    Outputs a list of length [batch_size] with torch.Tensors with dimensions [seq_len, dim]. Device can be set.

    Args:
        word_embeddings (bool): [description]
        context_embeddings (bool): [description]
        tokenizer (Callable, optional): [description]. Defaults to None.
        language (str, optional): [description]. Defaults to "English".
        cache_location (str, optional): [description]. Defaults to "./morphological_tagging/data/pretrained_vectors".
        lower_case_backup (bool, optional): [description]. Defaults to False.
        transformer_name (str, optional): [description]. Defaults to "distilbert-base-multilingual-cased".
        transformer_dropout (float, optional): overwrites the transformer's various dropout layers. Defaults to pretrained value.
        layer_pooling (str, optional): [description]. Defaults to "average".
        n_layers_pooling (int, optional): [description]. Defaults to 4.
        wordpiece_pooling (str, optional): [description]. Defaults to "first".
    """

    def __init__(
        self,
        word_embeddings: bool,
        context_embeddings: bool,
        tokenizer: Callable = None,
        language: str = "English",
        lower_case_backup: bool = False,
        transformer_name: str = "distilbert-base-multilingual-cased",
        transformer_dropout: float = None,
        layer_pooling: str = "average",
        n_layers_pooling: int = 4,
        wordpiece_pooling: str = "first",
    ) -> None:
        super().__init__()

        self.word_embeddings = word_embeddings
        self.tokenizer = tokenizer
        self.context_embeddings = context_embeddings
        self.language = language
        self.lower_case_backup = lower_case_backup
        self.transformer_name = transformer_name
        self.transformer_dropout = transformer_dropout
        self.layer_pooling = layer_pooling
        self.n_layers_pooling = n_layers_pooling
        self.wordpiece_pooling = wordpiece_pooling

        #! DEPRECATED: tokenization is handled elsewhere
        if tokenizer is None:
            self.tokenizer = word_tokenize
        else:
            self.tokenizer = tokenizer

        if self.word_embeddings:
            self.vecs = FastText(language=FASTTEXT_LANG_CONVERSION[self.language])

        if self.context_embeddings:
            self.config = AutoConfig.from_pretrained(self.transformer_name)

            # Change all dropout parameters, regardless of model's naming convention
            if self.transformer_dropout is not None:
                dropouts = dict()
                for k, v in self.config.__dict__.items():
                    if "dropout" in k and isinstance(v, float):
                        dropouts[k] = self.transformer_dropout
                self.config.__dict__.update(dropouts)

            # Quick check if dimensionality is known
            self.dim

            self.transformer_tokenizer = AutoTokenizer.from_pretrained(
                self.transformer_name, use_fast=True
            )
            self.transformer = AutoModel.from_pretrained(self.transformer_name, config=self.config)

            self.layer_pooling = layer_pooling
            self.n_layers_pooling = n_layers_pooling
            self.wordpiece_pooling = wordpiece_pooling

        self.dummy_parameter = nn.Parameter(torch.zeros(1), requires_grad=False)

        # By default, this model is frozen
        # Can be trained though
        self.freeze_and_eval()

    @property
    def device(self):
        return self.dummy_parameter.device

    @property
    def dim(self):
        dim = 0

        if self.word_embeddings:
            dim += self.vecs.dim

        if self.context_embeddings:

            if re.search(r"^(distilbert)-", self.transformer_name) is not None:
                dim += self.config.dim

            elif re.search(r"^(bert)-", self.transformer_name) is not None:
                dim += self.config.hidden_size

            elif re.search(r"-(roberta)-", self.transformer_name) is not None:
                dim += self.config.hidden_size

            else:
                raise ConfigurationError(
                    "Preprocessor does not know how to get the transformer hidden dimensionality"
                    + " for a model of type {self.transformer_name}."
                )

        return dim

    def train(self):
        if self.context_embeddings:
            self.transformer.train()

    def eval(self):
        if self.context_embeddings:
            self.transformer.eval()

    def freeze_and_eval(self):

        if self.context_embeddings:
            self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_and_train(self):

        self.thaw_and_train()

    def thaw_and_train(self):

        if self.context_embeddings:
            self.transformer.train()
        for param in self.parameters():
            param.requires_grad = True

    def _unpack_input(self, batch):

        if isinstance(batch, list):
            tokens_raw = batch

        elif isinstance(batch, str):
            tokens_raw = [batch]

        tokens_raw = [self.tokenizer(seq) for seq in tokens_raw]
        char_lens = [[len(t) for t in seq] for seq in tokens_raw]
        token_lens = [len(seq) for seq in tokens_raw]

        return tokens_raw, char_lens, token_lens

    def _word_embeddings(self, token_lens, tokens_raw):

        word_embeddings_ = self.vecs.get_vecs_by_tokens(
            [t for seq in tokens_raw for t in seq], self.lower_case_backup
        )

        beg = 0
        word_embeddings = []
        for l in token_lens:
            word_embeddings.append(word_embeddings_[beg : beg + l, :].to(self.device))
            beg += l

        return word_embeddings

    def _context_embeddings(self, token_lens, tokens_raw):

        transformer_input = self.transformer_tokenizer(
            tokens_raw,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            is_split_into_words=True,
        )

        transformer_output = self.transformer(
            transformer_input["input_ids"].to(self.device),
            transformer_input["attention_mask"].to(self.device),
            output_hidden_states=True,
        )

        # Get the word-piece context embedding
        if self.layer_pooling == "average":
            # Pool using layer average
            wp_ce = torch.mean(
                torch.stack(
                    transformer_output.hidden_states[-self.n_layers_pooling :], dim=2
                ),
                dim=2,
            )

        # Pool the wordpiece embeddings into token embeddings
        if self.wordpiece_pooling == "first":
            # Pool using only the first wordpiece

            token_map = [
                torch.logical_and(
                    transformer_input["offset_mapping"][i, :, 0]
                    == 0,  # Only keep the first BPE, i.e. those with non-zero span start
                    transformer_input["offset_mapping"][i, :, 1]
                    != 0,  # Remove [CLS], [END], [PAD] tokens, i.e. those with zero span end
                )
                for i in range(len(tokens_raw))
            ]

            context_embeddings = [
                wp_ce[i, token_map[i], :] for i in range(len(tokens_raw))
            ]

        # In case of truncation, add 0 vectors at the end
        context_embeddings = [
            F.pad(c, (0, 0, 0, token_lens[i] - c.size(0)), mode="constant", value=0)
            for i, c in enumerate(context_embeddings)
        ]

        return context_embeddings

    def forward(
        self, batch: Union[tuple, List[List[str]], str], pre_tokenized: bool = True
    ):

        if pre_tokenized and isinstance(batch, tuple):
            if len(batch) == 2:
                (token_lens, tokens_raw,) = batch
            elif len(batch) == 9:
                (_, _, token_lens, tokens_raw, _, _, _, _, _,) = batch
            else:
                raise ValueError(
                    f"Batch of len {len(batch)} is not a recognized format."
                )

        else:
            tokens_raw, char_lens, token_lens = self._unpack_input(batch)

        embeddings_ = []

        if self.word_embeddings:
            word_embeddings = self._word_embeddings(token_lens, tokens_raw)
            embeddings_.append(word_embeddings)

        if self.context_embeddings:
            context_embeddings = self._context_embeddings(token_lens, tokens_raw)
            embeddings_.append(context_embeddings)

        if len(embeddings_) == 2:
            embeddings = pad_sequence(
                [
                    torch.cat([w_seq, c_seq], dim=-1)
                    for w_seq, c_seq in zip(*embeddings_)
                ],
                padding_value=0,
            )

        elif len(embeddings_) == 1:
            embeddings = pad_sequence(embeddings_[0], padding_value=0)

        elif len(embeddings_) == 0:
            return None

        return embeddings
