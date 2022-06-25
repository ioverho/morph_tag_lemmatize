import pickle
from copy import deepcopy
from typing import Optional, Callable, Union, Tuple, List, Dict, Any
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer, AutoModel

from morphological_tagging.data.corpus import TreebankDataModule
from morphological_tagging.data.lemma_script import apply_lemma_script
from morphological_tagging.preprocessor import UDPipe2PreProcessor
from morphological_tagging.modules import (
    Char2Word,
    SequenceMask,
    ResidualMLP,
    ResidualRNN,
    MultiHeadSequenceAttention,
)
from utils.errors import ConfigurationError


class TorchUDPipe2(nn.Module):
    """A PyTorch implementation of UDPipe2.0.
    For training, I highly recommend using the PyTorch Lightning variant in the `models' package.

    As described in:
        Straka, M., Straková, J., & Hajič, J. (2019). UDPipe at SIGMORPHON 2019: Contextualized embeddings, regularization with morphological categories, corpora merging. arXiv preprint arXiv:1908.06931.
        Straka, M. (2018, October). UDPipe 2.0 prototype at CoNLL 2018 UD shared task. In Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies (pp. 197-207).
        Straka, M., & Straková, J. (2020). UDPipe at EvaLatin 2020: Contextualized embeddings and treebank embeddings. arXiv preprint arXiv:2006.03687.

    Args:
        len_char_vocab (int): length of all character vocabulary in dataset
        char_unk_idx (int): int value of the unk character
        char_pad_idx (int): int value of the padding character
        len_token_vocab (int): length of all tokens vocabulary in the dataset
        token_unk_idx (int): int value of the unk token
        token_pad_idx (int): int value of the padding token
        c2w_kwargs (dict): dictionary with keyword arguments relevant to a Char2Word module
        preprocessor_kwargs (dict): dictionary with keyword arguments relevant to a UDPipe2PreProcessor module
        word_rnn_kwargs (dict): dictionary with keyword arguments relevant to a ResidualRNN module
        n_lemma_scripts (int): number of lemma script classes
        n_morph_tags (int): number of morphological tags classes
        n_morph_cats (int): number of morphological category classes
        w_embedding_dim (int, optional): embedding dimension of E2E trained word vectors. Defaults to 512.
        pretrained_embedding_dim (int, optional): embedding dimension of pre-trained word vectors. Defaults to 300.
        dropout (float, optional): dropout applied after every non-recurrent layer in network. Defaults to 0.5.
        char_mask_p (float, optional): probability of replacing a character with the unk character during training. Defaults to 0.0.
        token_mask_p (float, optional): probability of replacing a token with the unk token during training. Defaults to 0.2.
        label_smoothing (float, optional): strength of label-smoothing applied. Defaults to 0.03.
        reg_loss_weight (float, optional): weight of the moprh. cat prediction loss to serve as regularizer. Defaults to 1.0.
        lr (float, optional): model wide learning rate. Defaults to 1e-3.
        betas (Tuple[float], optional): Adam betas. Defaults to (0.9, 0.99).
        weight_decay (int, optional): AdamW weight decay. Defaults to 0.
        ignore_idx (int, optional): the class to ignore (for example, due to padding the targets). Defaults to -1.

    """

    def __init__(
        self,
        len_char_vocab: int,
        char_unk_idx: int,
        char_pad_idx: int,
        len_token_vocab: int,
        token_unk_idx: int,
        token_pad_idx: int,
        c2w_kwargs: dict,
        preprocessor_kwargs: dict,
        word_rnn_kwargs: dict,
        n_lemma_scripts: int,
        n_morph_tags: int,
        w_embedding_dim: int = 512,
        pretrained_embedding_dim: int = 300,
        dropout: float = 0.5,
        char_mask_p: float = 0.0,
        token_mask_p: float = 0.2,
        label_smoothing: float = 0.03,
        reg_loss_weight: float = 1.0,
        lr: float = 1e-3,
        betas: Optional[float] = (0.9, 0.99),
        weight_decay=0,
        ignore_idx: int = -1,
        **kwargs
    ) -> None:
        super().__init__()

        # ======================================================================
        # Model hyperparameters
        # ======================================================================
        # Module hyperparmeters ================================================
        self.c2w_kwargs = c2w_kwargs
        self.w_embedding_dim = w_embedding_dim
        self.pretrained_embedding_dim = pretrained_embedding_dim
        self.word_rnn_kwargs = word_rnn_kwargs

        # Number of classes ====================================================
        self.n_lemma_scripts = n_lemma_scripts
        self.n_morph_tags = n_morph_tags

        # Special tokens =======================================================
        self.char_unk_idx = char_unk_idx
        self.char_pad_idx = char_pad_idx

        self.token_unk_idx = token_unk_idx
        self.token_pad_idx = token_pad_idx

        # Preprocessor =========================================================
        self.preprocessor_kwargs = preprocessor_kwargs
        self.preprocessor = UDPipe2PreProcessor(**self.preprocessor_kwargs)

        # Embedding Modules ====================================================
        self.char_mask = SequenceMask(
            mask_p=char_mask_p, mask_idx=char_unk_idx, ign_idx=char_pad_idx,
        )

        self.c2w_embedder = Char2Word(
            vocab_len=len_char_vocab, padding_idx=char_pad_idx, **self.c2w_kwargs,
        )

        self.token_mask = SequenceMask(
            mask_p=token_mask_p,
            mask_idx=self.token_unk_idx,
            ign_idx=self.token_pad_idx,
        )

        self.w_embedder = nn.Embedding(
            num_embeddings=len_token_vocab,
            embedding_dim=self.w_embedding_dim,
            padding_idx=self.token_pad_idx,
            sparse=False,
        )

        self._total_embedding_size = (
            self.c2w_kwargs["embedding_dim"]
            + self.w_embedding_dim
            + self.preprocessor.dim
        )

        self.embed_dropout = nn.Dropout(p=dropout)

        # Word-level RNN =======================================================
        self.word_rnn = ResidualRNN(
            input_size=self._total_embedding_size, **self.word_rnn_kwargs,
        )

        # Lemma classification =================================================
        self._lemma_in_features = (
            self.word_rnn_kwargs["h_dim"] + self.c2w_kwargs["out_dim"]
        )

        self.lemma_clf = nn.Sequential(
            ResidualMLP(
                in_features=self._lemma_in_features,
                out_features=self._lemma_in_features,
            ),
            nn.Linear(
                in_features=self._lemma_in_features, out_features=self.n_lemma_scripts
            ),
        )

        # Morph classification =================================================
        self._morph_in_features = self.word_rnn_kwargs["h_dim"]

        self.morph_clf_unf = nn.Sequential(
            ResidualMLP(
                in_features=self._morph_in_features,
                out_features=self._morph_in_features,
            ),
            nn.Linear(
                in_features=self._morph_in_features, out_features=self.n_morph_tags
            ),
        )

        # ==========================================================================
        # Regularization
        # ==========================================================================
        self.dropout = dropout

        self.label_smoothing = label_smoothing
        self.reg_loss_weight = reg_loss_weight
        self.weight_decay = weight_decay

        # ======================================================================
        # Optimization
        # ======================================================================
        self.lr = lr
        self.betas = betas

        # ======================================================================
        # Misc (e.g. logging)
        # =======================================================================

        self.ignore_idx = ignore_idx
        self.unused_kwargs = kwargs

    @property
    def device(self):
        return next(self.parameters()).device

    def _trainable_modules(self):
        return [
            self.preprocessor,
            self.char_mask,
            self.c2w_embedder,
            self.token_mask,
            self.w_embedder,
            self.embed_dropout,
            self.word_rnn,
            self.lemma_clf,
            self.morph_clf_unf,
        ]

    def train(self):
        for mod in self._trainable_modules():
            mod.train()
        self.training = True

    def eval(self):
        for mod in self._trainable_modules():
            mod.eval()
        self.training = False

    def preprocess(self, token_lens, tokens_raw):
        return self.preprocessor((token_lens, tokens_raw), pre_tokenized=True)

    def forward(
        self,
        char_lens: Union[list, torch.Tensor],
        chars: torch.Tensor,
        token_lens: Union[list, torch.Tensor],
        tokens: torch.Tensor,
        pretrained_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor]:

        chars = self.char_mask(chars)

        c2w_e = self.c2w_embedder.forward(chars=chars, char_lens=char_lens)

        seqs = []
        beg = torch.tensor([0])
        for l in token_lens:
            seqs.append(c2w_e[beg : beg + l])
            beg += l

        c2w_e = pad_sequence(seqs, padding_value=self.token_pad_idx)

        tokens = self.token_mask(tokens)

        w_e = self.w_embedder.forward(tokens)

        if pretrained_embeddings is not None:
            e = torch.cat([c2w_e, w_e, pretrained_embeddings], dim=-1)
        else:
            e = torch.cat([c2w_e, w_e], dim=-1)

        e = self.embed_dropout(e)

        h = self.word_rnn(e)

        lemma_logits = self.lemma_clf(torch.cat([h, c2w_e], dim=-1))

        morph_logits_unf = self.morph_clf_unf(h)

        return lemma_logits, morph_logits_unf


class UDPipe2Pipeline(nn.Module):
    """A wrapper for UDPipe2 that makes it a pipeline.
    Text goes in, a list of lemmas, and morph_tags goes out.

    Args:
        tokenizer (Optional[Callable], optional): Any callable function that takes a list of
        text and returns a list of lists of strings, representing tokens. Defaults to None.
    """

    def __init__(self, tokenizer: Optional[Callable] = None):
        super().__init__()

        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.performance_stats = dict()

    def load_tagger(self, check_point_path: str, **load):
        """
        Takes a UDPipe2 model as a PyTorch Lighting module,
        and loads in only those aspects relevant for a PyTorch module to function within this pipeline.

        Args:
            check_point_path (str): _description_
            load: kwargs fed into the `torch.load' method
        """

        with open(check_point_path, "rb") as f:
            checkpoint = torch.load(f, **load)

        self._hparams = checkpoint["hyper_parameters"]

        self.tagger = TorchUDPipe2(**checkpoint["hyper_parameters"])
        self.tagger.load_state_dict(checkpoint["state_dict"], strict=False)

        self.tagger.eval()
        for param in self.tagger.parameters():
            param.requires_grad = False

    def load_vocabs_from_treebankdatamodule_checkpoint(self, fp: str):

        dm = TreebankDataModule.load(fp)

        self.id_to_lemma_script = deepcopy(dm.corpus.id_to_script)
        self.id_to_morph_tag = deepcopy(
            {v: k for k, v in dm.corpus.morph_tag_vocab.items()}
        )
        self.morph_tag_to_morph_cat = deepcopy(dm.corpus.morph_tag_cat_vocab)
        self.char_vocab = deepcopy(dm.corpus.char_vocab)
        self.token_vocab = deepcopy(dm.corpus.token_vocab)
        self.pad_token = deepcopy(dm.corpus.pad_token)
        self.unk_token = deepcopy(dm.corpus.unk_token)

        del dm

    @property
    def device(self):
        return next(self.parameters()).device

    def text_to_input(
        self, texts: Union[List[str], List[List[str]]], pre_tokenized: bool = False
    ):
        if pre_tokenized:
            tokens_raw = texts
        else:
            tokens_raw = self.tokenizer(texts)

        token_lens = [len(seq) for seq in tokens_raw]
        tokens = pad_sequence(
            [
                torch.tensor(
                    self.token_vocab.lookup_indices(text),
                    dtype=torch.long,
                    device=self.device,
                )
                for text in tokens_raw
            ],
            batch_first=False,
            padding_value=self.token_vocab[self.pad_token],
        )

        chars = [
            torch.tensor(
                self.char_vocab.lookup_indices([char for char in token]),
                dtype=torch.long,
                device=self.device,
            )
            for token_seq in tokens_raw
            for token in token_seq
        ]
        char_lens = [char_seq.size(0) for char_seq in chars]
        chars = pad_sequence(
            [char_seq for char_seq in chars],
            padding_value=self.char_vocab[self.pad_token],
            batch_first=False,
        )

        return char_lens, chars, token_lens, tokens_raw, tokens

    @torch.no_grad()
    def predict(self, char_lens, chars, token_lens, tokens_raw, tokens):

        pretrained_embeddings = self.tagger.preprocess(token_lens, tokens_raw)

        lemma_logits, morph_logits = self.tagger.forward(
            char_lens,
            chars.to(self.device),
            token_lens,
            tokens.to(self.device),
            pretrained_embeddings,
        )

        lemma_preds = torch.argmax(lemma_logits, dim=-1)
        morph_preds = torch.round(torch.sigmoid(morph_logits))

        return lemma_preds, morph_preds

    def preds_to_text(self, tokens_raw, lemma_preds, morph_preds):

        lemma_preds_ = lemma_preds.detach().cpu().permute(1, 0).numpy()
        morph_preds_ = morph_preds.detach().cpu().permute(1, 0, 2).numpy()

        lemma_scripts = [
            [self.id_to_lemma_script[ls] for _, ls in zip(tok_seq, ls_seq)]
            for tok_seq, ls_seq in zip(tokens_raw, lemma_preds_)
        ]

        lemmas = [
            [
                apply_lemma_script(token, ls, verbose=False)
                for token, ls in zip(tok_seq, ls_seq)
            ]
            for tok_seq, ls_seq in zip(tokens_raw, lemma_scripts)
        ]

        morph_tags = [
            [
                set(self.id_to_morph_tag[mt] for mt in np.where(mts)[0])
                for _, mts in zip(tok_seq, mt_seq)
            ]
            for tok_seq, mt_seq in zip(tokens_raw, morph_preds_)
        ]

        morph_cats = [
            [
                set(self.morph_tag_to_morph_cat[mt.lower()] for mt in list(mts))
                for _, mts in zip(tok_seq, mt_seq)
            ]
            for tok_seq, mt_seq in zip(tokens_raw, morph_tags)
        ]

        return lemmas, lemma_scripts, morph_tags, morph_cats

    def forward(
        self,
        inp: Union[List[str], List[List[str]], Tuple[torch.tensor]],
        is_pre_tokenized: bool = False,
        is_batch_input: bool = False,
        transpose: bool = False,
    ):
        """

        Args:
            inp (_type_): the input
            is_pre_tokenized (bool, optional): avoids passing the tokenizer over the input. Expects input to be List[List[str]]. Defaults to False.
            is_batch_input (bool, optional): treats input as batch for TreebankDataModule. Expects input to be iterable containing `char_lens', `chars', `token_lens', `tokens_raw', `tokens'. Defaults to False.
            transpose (bool, optional): whether or not to transpose the output. Defaults to False.

        Returns:
            Union[Tuple[List[List]], List[List[Tuple]]]: returns either a tuple of lists [4, N_sents, N_tokens] which are the lemmas, lemma scripts, morph_tag_sets and morp_cat_sets, with each element in the list corresponding to a token or a list of tuples, or its transpose [N_sents, N_tokens, 4]
        """

        if not is_batch_input:
            (char_lens, chars, token_lens, tokens_raw, tokens) = self.text_to_input(
                inp, pre_tokenized=is_pre_tokenized
            )
        else:
            (char_lens, chars, token_lens, tokens_raw, tokens) = inp

        lemma_preds, morph_preds = self.predict(
            char_lens, chars, token_lens, tokens_raw, tokens
        )

        lemmas, lemma_scripts, morph_tags, morph_cats = self.preds_to_text(
            tokens_raw, lemma_preds, morph_preds
        )

        if transpose:
            return [
                list(zip(*batch_output))
                for batch_output in zip(lemmas, lemma_scripts, morph_tags, morph_cats)
            ]
        else:
            return lemmas, lemma_scripts, morph_tags, morph_cats

    def add_tokenizer(self, tokenizer: callable):
        self.tokenizer = tokenizer

    def save(self, file_path):
        """Pickles a minimal subset of parameters to be loaded in later.

        Args:
            file_path (str): location of the save file.
        """

        pruned_state_dict = OrderedDict(
            [
                (k, self.state_dict()[k])
                for k in self.state_dict().keys()
                if "preprocessor" not in k
            ]
        )

        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "state_dict": pruned_state_dict,
                    "hparams": self._hparams,
                    "dicts": {
                        "id_to_lemma_script": self.id_to_lemma_script,
                        "id_to_morph_tag": self.id_to_morph_tag,
                        "morph_tag_to_morph_cat": self.morph_tag_to_morph_cat,
                        "char_vocab": self.char_vocab,
                        "token_vocab": self.token_vocab,
                        "pad_token": self.pad_token,
                        "unk_token": self.unk_token,
                    },
                    "performance_stats": self.performance_stats,
                },
                f,
            )

    @classmethod
    def load(cls, file_path, tokenizer: Optional[callable] = None):

        with open(file_path, "rb") as f:
            pipeline_state = pickle.load(f)

        pipeline = UDPipe2Pipeline()
        pipeline._hparams = pipeline_state["hparams"]
        pipeline.tagger = TorchUDPipe2(**pipeline._hparams)

        _ = pipeline.load_state_dict(pipeline_state["state_dict"], strict=False)
        pipeline.tagger.eval()
        for param in pipeline.tagger.parameters():
            param.requires_grad = False

        pipeline.id_to_lemma_script = pipeline_state["dicts"]["id_to_lemma_script"]
        pipeline.id_to_morph_tag = pipeline_state["dicts"]["id_to_morph_tag"]
        pipeline.morph_tag_to_morph_cat = pipeline_state["dicts"][
            "morph_tag_to_morph_cat"
        ]
        pipeline.char_vocab = pipeline_state["dicts"]["char_vocab"]
        pipeline.token_vocab = pipeline_state["dicts"]["token_vocab"]
        pipeline.pad_token = pipeline_state["dicts"]["pad_token"]
        pipeline.unk_token = pipeline_state["dicts"]["unk_token"]

        pipeline.performance_stats = pipeline_state["performance_stats"]

        if tokenizer is not None:
            pipeline.add_tokenizer(tokenizer)

        return pipeline


class TorchDogTag(nn.Module):
    """A PyTorch implementation of DogTag.
    For training, I highly recommend using the PyTorch Lightning variant in the `models' package.

    Args:
        len_char_vocab (int): length of all character vocabulary in dataset
        char_unk_idx (int): int value of the unk character
        char_pad_idx (int): int value of the padding character
        len_token_vocab (int): length of all tokens vocabulary in the dataset
        token_unk_idx (int): int value of the unk token
        token_pad_idx (int): int value of the padding token
        c2w_kwargs (dict): dictionary with keyword arguments relevant to a Char2Word module
        preprocessor_kwargs (dict): dictionary with keyword arguments relevant to a UDPipe2PreProcessor module
        word_rnn_kwargs (dict): dictionary with keyword arguments relevant to a ResidualRNN module
        n_lemma_scripts (int): number of lemma script classes
        n_morph_tags (int): number of morphological tags classes
        n_morph_cats (int): number of morphological category classes
        w_embedding_dim (int, optional): embedding dimension of E2E trained word vectors. Defaults to 512.
        pretrained_embedding_dim (int, optional): embedding dimension of pre-trained word vectors. Defaults to 300.
        dropout (float, optional): dropout applied after every non-recurrent layer in network. Defaults to 0.5.
        char_mask_p (float, optional): probability of replacing a character with the unk character during training. Defaults to 0.0.
        token_mask_p (float, optional): probability of replacing a token with the unk token during training. Defaults to 0.2.
        label_smoothing (float, optional): strength of label-smoothing applied. Defaults to 0.03.
        reg_loss_weight (float, optional): weight of the moprh. cat prediction loss to serve as regularizer. Defaults to 1.0.
        lr (float, optional): model wide learning rate. Defaults to 1e-3.
        betas (Tuple[float], optional): Adam betas. Defaults to (0.9, 0.99).
        weight_decay (int, optional): AdamW weight decay. Defaults to 0.
        ignore_idx (int, optional): the class to ignore (for example, due to padding the targets). Defaults to -1.

    """

    def __init__(
        self,
        transformer_dropout: float,
        rnn_kwargs: Dict[str, Any],
        mha_kwargs: Dict[str, Any],
        batch_first: bool,
        embedding_dropout: float,
        mask_p: float,
        idx_char_pad: int,
        idx_token_pad: int,
        n_lemma_scripts: int,
        n_morph_tags: int,
        n_morph_cats: int,
        ignore_idx: int = -1,
        **unused_kwargs
    ) -> None:
        super().__init__()

        # ======================================================================
        # Model hyperparameters
        # ======================================================================
        # Module hyperparmeters ================================================
        self.transformer_type = "canine"
        self.transformer_name = "google/canine-s"
        self.transformer_dropout = transformer_dropout
        self.mha_kwargs = mha_kwargs
        self.batch_first = batch_first
        self.embedding_dropout = embedding_dropout

        # Number of classes ====================================================
        self.n_lemma_scripts = n_lemma_scripts
        self.n_morph_tags = n_morph_tags
        self.n_morph_cats = n_morph_cats

        # ======================================================================
        # Initiatlization
        # ======================================================================
        # Contextual embeddings ================================================
        self.config = AutoConfig.from_pretrained(
            self.transformer_name,
            hidden_dropout_prob=transformer_dropout,
            attention_probs_dropout_prob=transformer_dropout,
            attention_dropout=transformer_dropout,
        )
        self.h_dim = self.config.hidden_size
        self.transformer = AutoModel.from_pretrained(
            self.transformer_name, config=self.config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.transformer_name, use_fast=True
        )

        self.token_mha = MultiHeadSequenceAttention(
            d_in=self.h_dim,
            d_out=self.h_dim,
            batch_first=self.batch_first,
            **self.mha_kwargs,
        )

        self.token_dropout = nn.Dropout(p=self.embedding_dropout)

        self.token_rnn = ResidualRNN(
            input_size=self.h_dim,
            h_dim=self.h_dim,
            batch_first=self.batch_first,
            **rnn_kwargs,
        )

        self.lemma_mha = MultiHeadSequenceAttention(
            d_in=self.h_dim,
            d_out=self.h_dim,
            batch_first=self.batch_first,
            **self.mha_kwargs,
        )

        self.lemma_dropout = nn.Dropout(p=self.embedding_dropout)

        # Classifiers ==================================================================
        self._lemma_input_dim = 2 * self.h_dim
        self.lemma_clf = nn.Sequential(
            ResidualMLP(
                in_features=self._lemma_input_dim, out_features=self._lemma_input_dim,
            ),
            nn.Linear(in_features=self._lemma_input_dim, out_features=n_lemma_scripts),
        )

        self._morph_input_dim = 2 * self.h_dim
        self.morph_unf_clf = nn.Sequential(
            ResidualMLP(in_features=self._morph_input_dim, out_features=self._morph_input_dim,),
            nn.Linear(in_features=self._morph_input_dim, out_features=self.n_morph_tags),
        )

        self.morph_fac_clf = nn.Sequential(
            ResidualMLP(in_features=self._morph_input_dim, out_features=self._morph_input_dim,),
            nn.Linear(in_features=self._morph_input_dim, out_features=self.n_morph_cats),
        )

        # ==========================================================================
        # Regularization
        # ==========================================================================
        self.mask_p = mask_p

        # ======================================================================
        # Misc (e.g. logging)
        # ======================================================================

        self.ignore_idx = ignore_idx

        # Special tokens =======================================================
        self.idx_char_pad = idx_char_pad
        self.idx_token_pad = idx_token_pad

        self.unused_kwargs = unused_kwargs

    @property
    def device(self):
        return next(self.parameters()).device

    def _trainable_modules(self):
        return [
            self.transformer,
            self.token_mha,
            self.token_dropout,
            self.token_rnn,
            self.lemma_mha,
            self.lemma_dropout,
            self.lemma_clf,
            self.morph_unf_clf,
            self.morph_fac_clf,
        ]

    def train(self):
        for mod in self._trainable_modules():
            mod.train()
        self.training = True

    def eval(self):
        for mod in self._trainable_modules():
            mod.eval()
        self.training = False

    def forward(
        self, tokens: torch.Tensor, skip_morph_reg: bool = False,
    ) -> Tuple[torch.Tensor]:

        n_chars_per_token = [[len(token) for token in sent] for sent in tokens]
        n_tokens_per_sent = [len(sent) for sent in tokens]

        tokenizer_output = self.tokenizer(
            tokens,
            padding=True,
            return_tensors="pt",
            is_split_into_words=True,
            return_length=True,
        )

        tokenizer_output["input_ids"] = torch.where(
            torch.logical_or(
                torch.bernoulli(tokenizer_output["input_ids"], 1 - self.mask_p).bool(),
                tokenizer_output["attention_mask"] == 0,
            ),
            tokenizer_output["input_ids"],
            self.tokenizer.mask_token_id,
        )

        # Generate the contextualized character embeddings
        cce = self.transformer(
            tokenizer_output["input_ids"].to(self.device),
            tokenizer_output["attention_mask"].to(self.device),
        ).last_hidden_state

        # Break the batch from a batch of sentence characters to a batch of token characters
        # Also generate an attention map for pooling operation
        # [B, L_t x L_c, D] -> [B x L_t, L_c, D]
        molecules_ = []
        for i, sent in enumerate(cce):

            atoms = sent[1 : tokenizer_output["length"][i] - 1]

            molecules = [
                [atoms[beg:end], torch.ones((end - beg), device=atoms.device)]
                for beg, end in zip(
                    np.cumsum([0] + n_chars_per_token[i][:-1]),
                    np.cumsum(n_chars_per_token[i]),
                )
            ]

            molecules_.extend(molecules)

        embeddings, attention_mask = list(map(list, zip(*molecules_)))

        # Pad the ragged list of character embeddings to a the longest token length
        embeddings = pad_sequence(
            embeddings, padding_value=self.tokenizer.pad_token_id, batch_first=True
        )
        attention_mask = pad_sequence(
            attention_mask, padding_value=self.tokenizer.pad_token_id, batch_first=True
        )

        # Morph Pooling ========================================================
        # Pool the character embeddings to token embeddings
        # [B x L_t, L_c, D] -> [B x L_t, D]
        pooled_token_embeddings = self.token_mha(embeddings, attention_mask)

        # Move the token embeddings back to batch, seq length, dim tensor
        # [B x L_t, D] -> [B, L_t, D]
        token_embeddings = [
            pooled_token_embeddings[beg:end]
            for beg, end in zip(
                np.cumsum([0] + n_tokens_per_sent[:-1]), np.cumsum(n_tokens_per_sent)
            )
        ]

        # Pad the ragged list of token embeddings to the longest sentence length
        token_embeddings = pad_sequence(
            token_embeddings,
            padding_value=self.tokenizer.pad_token_id,
            batch_first=True,
        )

        token_embeddings = self.token_dropout(token_embeddings)

        token_contextual_embeddings = self.token_rnn(token_embeddings)

        # Lemma Pooling =========================================================
        # Pool the character embeddings to token embeddings
        # [B x L_t, L_c, D] -> [B x L_t, D]
        pooled_lemma_embeddings = self.lemma_mha(embeddings, attention_mask)

        # Move the token embeddings back to batch, seq length, dim tensor
        # [B x L_t, D] -> [B, L_t, D]
        lemma_embeddings = [
            pooled_lemma_embeddings[beg:end]
            for beg, end in zip(
                np.cumsum([0] + n_tokens_per_sent[:-1]), np.cumsum(n_tokens_per_sent)
            )
        ]

        # Pad the ragged list of token embeddings to the longest sentence length
        lemma_embeddings = pad_sequence(
            lemma_embeddings,
            padding_value=self.tokenizer.pad_token_id,
            batch_first=True,
        )

        lemma_embeddings = self.lemma_dropout(lemma_embeddings)

        # Lemma classification =========================================================
        lemma_logits = self.lemma_clf(
            torch.cat([lemma_embeddings, token_contextual_embeddings], dim=-1)
        )

        # Morph classification =========================================================

        morph_unf_logits = self.morph_unf_clf(
            torch.cat([token_embeddings, token_contextual_embeddings], dim=-1)
        )

        if not skip_morph_reg:
            morph_fac_logits = self.morph_fac_clf(
                torch.cat([token_embeddings, token_contextual_embeddings], dim=-1)
            )

            return lemma_logits, morph_unf_logits, morph_fac_logits

        return lemma_logits, morph_unf_logits

class DogTagPipeline(nn.Module):
    """A wrapper for UDPipe2 that makes it a pipeline.
    Text goes in, a list of lemmas, and morph_tags goes out.

    Args:
        tokenizer (Optional[Callable], optional): Any callable function that takes a list of
        text and returns a list of lists of strings, representing tokens. Defaults to None.
    """

    def __init__(self, tokenizer: Optional[Callable] = None):
        super().__init__()

        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.performance_stats = dict()

    def load_tagger(self, check_point_path: str, **load):
        """
        Takes a UDPipe2 model as a PyTorch Lighting module,
        and loads in only those aspects relevant for a PyTorch module to function within this pipeline.

        Args:
            check_point_path (str): _description_
            load: kwargs fed into the `torch.load' method
        """

        with open(check_point_path, "rb") as f:
            checkpoint = torch.load(f, **load)

        self._hparams = checkpoint["hyper_parameters"]

        self.tagger = TorchDogTag(**checkpoint["hyper_parameters"])
        self.tagger.load_state_dict(checkpoint["state_dict"], strict=False)

        self.tagger.eval()
        for param in self.tagger.parameters():
            param.requires_grad = False

    def load_vocabs_from_treebankdatamodule_checkpoint(self, fp: str):

        dm = TreebankDataModule.load(fp)

        self.id_to_lemma_script = deepcopy(dm.corpus.id_to_script)
        self.id_to_morph_tag = deepcopy(
            {v: k for k, v in dm.corpus.morph_tag_vocab.items()}
        )
        self.morph_tag_to_morph_cat = deepcopy(dm.corpus.morph_tag_cat_vocab)
        self.pad_token = deepcopy(dm.corpus.pad_token)
        self.unk_token = deepcopy(dm.corpus.unk_token)

        del dm

    @property
    def device(self):
        return next(self.parameters()).device

    def text_to_input(
        self, texts: Union[List[str], List[List[str]]], pre_tokenized: bool = False
    ):
        if pre_tokenized:
            tokens_raw = texts
        else:
            tokens_raw = self.tokenizer(texts)

        return tokens_raw

    @torch.no_grad()
    def predict(self, tokens_raw):

        lemma_logits, morph_logits = self.tagger.forward(
            tokens_raw,
            True
        )

        lemma_preds = torch.argmax(lemma_logits, dim=-1)
        morph_preds = torch.round(torch.sigmoid(morph_logits))

        return lemma_preds, morph_preds

    def preds_to_text(self, tokens_raw, lemma_preds, morph_preds):

        lemma_preds_ = lemma_preds.detach().cpu().numpy()
        morph_preds_ = morph_preds.detach().cpu().numpy()

        lemma_scripts = [
            [self.id_to_lemma_script[ls] for _, ls in zip(tok_seq, ls_seq)]
            for tok_seq, ls_seq in zip(tokens_raw, lemma_preds_)
        ]

        lemmas = [
            [
                apply_lemma_script(token, ls, verbose=False)
                for token, ls in zip(tok_seq, ls_seq)
            ]
            for tok_seq, ls_seq in zip(tokens_raw, lemma_scripts)
        ]

        morph_tags = [
            [
                set(self.id_to_morph_tag[mt] for mt in np.where(mts)[0])
                for _, mts in zip(tok_seq, mt_seq)
            ]
            for tok_seq, mt_seq in zip(tokens_raw, morph_preds_)
        ]

        morph_cats = [
            [
                set(self.morph_tag_to_morph_cat[mt.lower()] for mt in list(mts))
                for _, mts in zip(tok_seq, mt_seq)
            ]
            for tok_seq, mt_seq in zip(tokens_raw, morph_tags)
        ]

        return lemmas, lemma_scripts, morph_tags, morph_cats

    def forward(
        self,
        inp: Union[List[str], List[List[str]], Tuple[torch.tensor]],
        is_pre_tokenized: bool = False,
        is_batch_input: bool = False,
        transpose: bool = False,
    ):
        """

        Args:
            inp (_type_): the input
            is_pre_tokenized (bool, optional): avoids passing the tokenizer over the input. Expects input to be List[List[str]]. Defaults to False.
            is_batch_input (bool, optional): treats input as batch for TreebankDataModule. Expects input to be iterable containing `char_lens', `chars', `token_lens', `tokens_raw', `tokens'. Defaults to False.
            transpose (bool, optional): whether or not to transpose the output. Defaults to False.

        Returns:
            Union[Tuple[List[List]], List[List[Tuple]]]: returns either a tuple of lists [4, N_sents, N_tokens] which are the lemmas, lemma scripts, morph_tag_sets and morp_cat_sets, with each element in the list corresponding to a token or a list of tuples, or its transpose [N_sents, N_tokens, 4]
        """

        if not is_batch_input:
            tokens_raw = self.text_to_input(
                inp, pre_tokenized=is_pre_tokenized
            )
        else:
            tokens_raw = inp

        lemma_preds, morph_preds = self.predict(tokens_raw)

        lemmas, lemma_scripts, morph_tags, morph_cats = self.preds_to_text(
            tokens_raw, lemma_preds, morph_preds
        )

        if transpose:
            return [
                list(zip(*batch_output))
                for batch_output in zip(lemmas, lemma_scripts, morph_tags, morph_cats)
            ]
        else:
            return lemmas, lemma_scripts, morph_tags, morph_cats

    def add_tokenizer(self, tokenizer: callable):
        self.tokenizer = tokenizer

    def save(self, file_path):
        """Pickles a minimal subset of parameters to be loaded in later.

        Args:
            file_path (str): location of the save file.
        """

        if self._hparams["transformer_lrs"] is None:
            pruned_state_dict = OrderedDict(
                [
                    (k, self.state_dict()[k])
                    for k in self.state_dict().keys()
                    if "transformer" not in k
                ]
            )

        else:
            pruned_state_dict = self.state_dict()

        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "state_dict": pruned_state_dict,
                    "hparams": self._hparams,
                    "dicts": {
                        "id_to_lemma_script": self.id_to_lemma_script,
                        "id_to_morph_tag": self.id_to_morph_tag,
                        "morph_tag_to_morph_cat": self.morph_tag_to_morph_cat,
                        "pad_token": self.pad_token,
                        "unk_token": self.unk_token,
                    },
                    "performance_stats": self.performance_stats,
                },
                f,
            )

    @classmethod
    def load(cls, file_path, tokenizer: Optional[callable] = None):

        with open(file_path, "rb") as f:
            pipeline_state = pickle.load(f)

        pipeline = DogTagPipeline()
        pipeline._hparams = pipeline_state["hparams"]
        pipeline.tagger = TorchDogTag(**pipeline._hparams)

        _ = pipeline.load_state_dict(pipeline_state["state_dict"], strict=False)
        pipeline.tagger.eval()
        for param in pipeline.tagger.parameters():
            param.requires_grad = False

        pipeline.id_to_lemma_script = pipeline_state["dicts"]["id_to_lemma_script"]
        pipeline.id_to_morph_tag = pipeline_state["dicts"]["id_to_morph_tag"]
        pipeline.morph_tag_to_morph_cat = pipeline_state["dicts"][
            "morph_tag_to_morph_cat"
        ]
        pipeline.pad_token = pipeline_state["dicts"]["pad_token"]
        pipeline.unk_token = pipeline_state["dicts"]["unk_token"]

        pipeline.performance_stats = pipeline_state["performance_stats"]

        if tokenizer is not None:
            pipeline.add_tokenizer(tokenizer)

        return pipeline
