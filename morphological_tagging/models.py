import warnings
import random
from typing import Tuple, Union, Dict, Any, Optional
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from transformers import AutoConfig, AutoTokenizer, AutoModel
import pytorch_lightning as pl

from morphological_tagging.metrics import RunningStats, RunningStatsBatch, RunningF1
from morphological_tagging.modules import (
    SequenceMask,
    Char2Word,
    ResidualRNN,
    ResidualMLP,
    LayerAttention,
    MultiHeadSequenceAttention,
)
from morphological_tagging.functional import break_batch, fuse_batch
from morphological_tagging.optim import InvSqrtWithLinearWarmupScheduler
from morphological_tagging.preprocessor import UDPipe2PreProcessor
from utils.common_operations import label_smooth


class JointTaggerLemmatizer(pl.LightningModule):
    """General class for a morphological tagger and lemmatizer.

    Args:
        pl ([type]): [description]
    """

    def __init__(self):
        super().__init__()

        self.configure_metrics()

    @property
    def device(self):
        return next(self.parameters()).device

    def _trainable_modules(self):
        raise NotImplementedError(
            "Model needs to implement a private method `_trainable_modules` which"
            + " returns an iterator over parameters which are set to train and eval."
        )

    def train(self):
        for mod in self._trainable_modules():
            mod.train()

    def eval(self):
        for mod in self._trainable_modules():
            mod.eval()

    def configure_metrics(self):

        self._metrics_dict = {
            "train": {
                "loss_total": RunningStats(),
                "loss_lemma": RunningStats(),
                "loss_morph": RunningStats(),
                "loss_morph_reg": RunningStats(),
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            },
            "valid": {
                "loss_total": RunningStats(),
                "loss_lemma": RunningStats(),
                "loss_morph": RunningStats(),
                "loss_morph_reg": RunningStats(),
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            },
            "test": {
                "loss_total": RunningStats(),
                "loss_lemma": RunningStats(),
                "loss_morph": RunningStats(),
                "loss_morph_reg": RunningStats(),
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            },
            "predict": {
                "loss_total": RunningStats(),
                "loss_lemma": RunningStats(),
                "loss_morph": RunningStats(),
                "loss_morph_reg": RunningStats(),
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            },
        }

    def clear_metrics(self, split: str):

        if split in self._metrics_dict.keys():
            self._metrics_dict[split] = {
                "loss_total": RunningStats(),
                "loss_lemma": RunningStats(),
                "loss_morph": RunningStats(),
                "loss_morph_reg": RunningStats(),
                "lemma_acc": RunningStatsBatch(),
                "lemma_lev_dist": RunningStats(),
                "morph_tag_acc": RunningStatsBatch(),
                "morph_set_acc": RunningStatsBatch(),
                "morph_f1": RunningF1(),
            }

        else:
            warnings.warn(
                f"{split} is not in the metrics_dict keys. Metrics are not cleared currently."
            )

    @torch.no_grad()
    def metrics(
        self,
        split: str,
        losses,
        lemma_logits,
        lemma_tags,
        morph_logits,
        morph_tags,
        token_lens=None,
        tokens_raw=None,
    ):

        self._metrics_dict[split]["loss_total"](losses["total"].detach().cpu().item())
        self._metrics_dict[split]["loss_lemma"](losses["lemma"].detach().cpu().item())
        self._metrics_dict[split]["loss_morph"](losses["morph"].detach().cpu().item())
        self._metrics_dict[split]["loss_morph_reg"](
            losses["morph_reg"].detach().cpu().item()
        )

        if (token_lens is not None) and (tokens_raw is not None):
            skip_lev_dist = False
        else:
            skip_lev_dist = True

        ## Lemma CLF metrics
        lemma_preds = torch.argmax(lemma_logits, dim=-1).detach().cpu().numpy()
        lemma_targets = lemma_tags.detach().cpu().numpy()
        lemma_mask = np.where((lemma_tags != -1).detach().cpu().numpy(), 1.0, np.nan)

        if not skip_lev_dist:
            # TODO (ivo): implement lev_distance reporting inside model
            # would require mapping back to string...
            raise NotImplementedError
            # for i, (preds_seq, target_seq, seq_len) in enumerate(
            #    zip(lemma_preds, lemma_targets, token_lens)
            # ):
            #    for pred, target, token in zip(
            #        preds_seq[:seq_len], target_seq[:seq_len], tokens_raw[i]
            #    ):
            #        pred_lemma_script = corpus.id_to_script[pred]
            #        pred_lemma = apply_lemma_script(token, pred_lemma_script)

            #        target_lemma_script = corpus.id_to_script[target]
            #        target_lemma = apply_lemma_script(token, target_lemma_script)

            #        lemma_lev_dist(distance(pred_lemma, target_lemma), output=False)

        self._metrics_dict[split]["lemma_acc"](lemma_preds == lemma_targets, lemma_mask)

        ## Morph. CLF metrics
        morph_preds = torch.round(torch.sigmoid(morph_logits)).detach().cpu().numpy()
        morph_targets = morph_tags.detach().cpu().numpy()
        morph_mask = np.where((morph_tags != -1).detach().cpu().numpy(), 1.0, np.nan)
        morph_set_mask = np.max(morph_mask, axis=-1)

        item_match = morph_preds == morph_targets
        set_match = np.all((morph_preds == morph_targets), axis=-1)

        # Morph. Acc
        self._metrics_dict[split]["morph_tag_acc"](item_match, morph_mask)
        self._metrics_dict[split]["morph_set_acc"](set_match, morph_set_mask)

        self._metrics_dict[split]["morph_f1"](
            morph_preds, morph_targets, morph_set_mask
        )

    def log_metrics(self, split):

        loss_total, _, loss_total_se, _, _ = self._metrics_dict[split][
            "loss_total"
        ]._return_stats()
        loss_lemma, _, loss_lemma_se, _, _ = self._metrics_dict[split][
            "loss_lemma"
        ]._return_stats()
        loss_morph, _, loss_morph_se, _, _ = self._metrics_dict[split][
            "loss_morph"
        ]._return_stats()
        loss_morph_reg, _, loss_morph_reg_se, _, _ = self._metrics_dict[split][
            "loss_morph_reg"
        ]._return_stats()

        lemma_acc, _, lemma_acc_se = self._metrics_dict[split][
            "lemma_acc"
        ]._return_stats()
        (lemma_dist, _, lemma_dist_se, _, _,) = self._metrics_dict[split][
            "lemma_lev_dist"
        ]._return_stats()

        morph_tag_acc, _, morph_tag_acc_se = self._metrics_dict[split][
            "morph_tag_acc"
        ]._return_stats()
        morph_set_acc, _, morph_set_acc_se = self._metrics_dict[split][
            "morph_set_acc"
        ]._return_stats()
        (morph_precision, morph_recall, morph_f1,) = self._metrics_dict[split][
            "morph_f1"
        ]._return_stats()

        # A metric to use for finetuning etc.
        # Since there are more lemma classes than morph classes,
        # loss tends to favour lemma performance
        # Harmonic mean between lemma acc and morph set acc better balances the two
        agg_metric = 2 * (lemma_acc * morph_set_acc) / (lemma_acc + morph_set_acc)

        metrics_dict = {
            f"{split}/loss/total": loss_total,
            f"{split}/loss/total_se": loss_total_se,
            f"{split}/loss/lemma": loss_lemma,
            f"{split}/loss/lemma_se": loss_lemma_se,
            f"{split}/loss/morph": loss_morph,
            f"{split}/loss/morph_se": loss_morph_se,
            f"{split}/loss/morph_reg": loss_morph_reg,
            f"{split}/loss/morph_reg_se": loss_morph_reg_se,
            f"{split}/lemma/acc": lemma_acc,
            f"{split}/lemma/acc_se": lemma_acc_se,
            f"{split}/morph/tag_acc": morph_tag_acc,
            f"{split}/morph/tag_acc_se": morph_tag_acc_se,
            f"{split}/morph/set_acc": morph_set_acc,
            f"{split}/morph/set_acc_se": morph_set_acc_se,
            f"{split}/morph/precision": morph_precision,
            f"{split}/morph/recall": morph_recall,
            f"{split}/morph/f1": morph_f1,
            f"{split}/lemma/dist": lemma_dist,
            f"{split}/lemma/dist_se": lemma_dist_se,
            f"{split}/clf_agg": agg_metric,
        }

        self.log_dict(metrics_dict)

        return metrics_dict

    def _unpack_input(self, batch):

        batch_ = []
        for x in batch:
            if isinstance(x, torch.Tensor) and x.device != self.device:
                batch_.append(x.to(self.device))
            else:
                batch_.append(x)

        if len(batch_) == 5:
            (char_lens, chars, token_lens, tokens_raw, tokens,) = batch_

            # The lens tensors need to be on CPU in case of packing
            if isinstance(char_lens, list):
                char_lens = torch.tensor(char_lens, dtype=torch.long, device="cpu")

            if isinstance(token_lens, list):
                token_lens = torch.tensor(token_lens, dtype=torch.long, device="cpu")

            return (char_lens, chars, token_lens, tokens_raw, tokens, None, None, None)

        else:
            (
                char_lens,
                chars,
                token_lens,
                tokens_raw,
                tokens,
                _,
                lemma_tags,
                morph_tags,
                morph_cats,
            ) = batch_

            # The lens tensors need to be on CPU in case of packing
            if isinstance(char_lens, list):
                char_lens = torch.tensor(char_lens, dtype=torch.long, device="cpu")

            if isinstance(token_lens, list):
                token_lens = torch.tensor(token_lens, dtype=torch.long, device="cpu")

            return (
                char_lens,
                chars,
                token_lens,
                tokens_raw,
                tokens,
                lemma_tags,
                morph_tags,
                morph_cats,
            )

    def on_train_epoch_end(self) -> None:
        self.log_metrics("train")
        self.clear_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self.log_metrics("valid")
        self.clear_metrics("valid")

    def on_test_epoch_end(self) -> None:
        self.log_metrics("test")
        self.clear_metrics("test")


class UDPipe2(JointTaggerLemmatizer):
    """A PyTorch Lightning implementation of UDPipe2.0.

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
        scheduler_name (Tuple[str, None], optional): name of scheduler methods to implement, either "step" or None. Defaults to None.
        scheduler_kwargs (Tuple[dict, None], optional): dictionary with keyword arguments relevant to a scheduler. Defaults to None.
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
        n_morph_cats: int,
        w_embedding_dim: int = 512,
        pretrained_embedding_dim: int = 300,
        dropout: float = 0.5,
        char_mask_p: float = 0.0,
        token_mask_p: float = 0.2,
        label_smoothing: float = 0.03,
        reg_loss_weight: float = 1.0,
        lr: float = 1e-3,
        betas: Tuple[float] = (0.9, 0.99),
        weight_decay=0,
        scheduler_name: Tuple[str, None] = None,
        scheduler_kwargs: Tuple[dict, None] = None,
        ignore_idx: int = -1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

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
        self.n_morph_cats = n_morph_cats

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

        self.morph_clf_fac = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualMLP(
                        in_features=self._morph_in_features,
                        out_features=self._morph_in_features,
                    ),
                    nn.Linear(in_features=self._morph_in_features, out_features=1),
                )
                for _ in range(self.n_morph_cats)
            ]
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
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs

        # ======================================================================
        # Misc (e.g. logging)
        # =======================================================================

        self.ignore_idx = ignore_idx

    def configure_optimizers(self):

        # Separate optimizers is a holdover from using sparse embeddings & lazy adam
        # Likely lead to optimization instability/issues
        optimizer_embeddings = optim.AdamW(
            [
                {"params": self.w_embedder.parameters()},
                {"params": self.c2w_embedder.embed.parameters()},
            ],
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        optimizer_rest = optim.AdamW(
            [
                *[
                    {"params": p}
                    for n, p in self.c2w_embedder.named_parameters()
                    if (not "embed" in n)
                ],
                {"params": self.word_rnn.parameters()},
                {"params": self.lemma_clf.parameters()},
                {"params": self.morph_clf_unf.parameters()},
                {"params": self.morph_clf_fac.parameters()},
            ],
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_name is None:
            optimizers = [optimizer_embeddings, optimizer_rest]
            schedulers = None

        elif self.scheduler_name.lower() == "step":
            scheduler_embeddings = MultiStepLR(
                optimizer_embeddings, **self.scheduler_kwargs
            )
            scheduler_rest = MultiStepLR(optimizer_rest, **self.scheduler_kwargs)

            optimizers = [optimizer_embeddings, optimizer_rest]
            schedulers = [scheduler_embeddings, scheduler_rest]

        return optimizers, schedulers

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
            self.morph_clf_fac,
        ]

    def preprocess(self, token_lens, tokens_raw):
        return self.preprocessor((token_lens, tokens_raw), pre_tokenized=True)

    def forward(
        self,
        char_lens: Union[list, torch.Tensor],
        chars: torch.Tensor,
        token_lens: Union[list, torch.Tensor],
        tokens: torch.Tensor,
        pretrained_embeddings: torch.Tensor,
        skip_morph_reg: bool = False,
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

        if not skip_morph_reg:
            morph_logits_fac = [fac(h) for fac in self.morph_clf_fac]

            return lemma_logits, morph_logits_unf, morph_logits_fac

        return lemma_logits, morph_logits_unf

    def loss(
        self,
        lemma_logits: torch.Tensor,
        lemma_tags: torch.Tensor,
        morph_logits_unf: torch.Tensor,
        morph_tags: torch.Tensor,
        morph_logits_fac: Union[torch.Tensor, None] = None,
        morph_cats: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:

        lemma_loss = F.cross_entropy(
            lemma_logits.permute(0, 2, 1),
            lemma_tags,
            ignore_index=-1,
            label_smoothing=self.label_smoothing,
        )

        morph_unf_loss = F.binary_cross_entropy_with_logits(
            morph_logits_unf,
            label_smooth(self.label_smoothing, morph_tags.float()),
            reduction="none",
        )
        morph_unf_loss = torch.mean(morph_unf_loss[morph_tags != -1])

        if (morph_logits_fac is not None) and (morph_cats is not None):
            morph_fac_loss = 0
            for i, fac_logits in enumerate(morph_logits_fac):
                cats_target = morph_cats[:, :, i].unsqueeze(-1)

                morph_fac_loss_ = F.binary_cross_entropy_with_logits(
                    fac_logits,
                    label_smooth(self.label_smoothing, cats_target.float()),
                    reduction="none",
                )
                morph_fac_loss_ = torch.mean(morph_fac_loss_[cats_target != -1])

                morph_fac_loss += morph_fac_loss_

            morph_fac_loss /= len(morph_logits_fac)

            loss = lemma_loss + morph_unf_loss + self.reg_loss_weight * morph_fac_loss
            losses = {
                "total": lemma_loss + morph_unf_loss,
                "lemma": lemma_loss,
                "morph": morph_unf_loss,
                "morph_reg": morph_fac_loss,
            }

        else:
            loss = lemma_loss + morph_unf_loss
            losses = {"total": loss, "lemma": lemma_loss, "morph": morph_unf_loss}

        return loss, losses

    def training_step(self, batch, batch_idx, optimizer_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            tokens,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        pretrained_embeddings = self.preprocess(token_lens, tokens_raw)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens, pretrained_embeddings
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics(
            "train", losses, lemma_logits, lemma_tags, morph_logits, morph_tags
        )

        return loss

    def validation_step(self, batch, batch_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            tokens,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        pretrained_embeddings = self.preprocess(token_lens, tokens_raw)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens, pretrained_embeddings
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics(
            "valid", losses, lemma_logits, lemma_tags, morph_logits, morph_tags
        )

        return loss

    def test_step(self, batch, batch_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            tokens,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        pretrained_embeddings = self.preprocess(token_lens, tokens_raw)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens, pretrained_embeddings
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics("test", losses, lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss


class UDIFY(JointTaggerLemmatizer):
    """A PyTorch Lightning implementation of UDIFY.

    As described in:
        Kondratyuk, D. (2019, August). Cross-lingual lemmatization and morphology tagging with two-stage multilingual BERT fine-tuning. In Proceedings of the 16th Workshop on Computational Research in Phonetics, Phonology, and Morphology (pp. 12-18).

    """

    def __init__(
        self,
        transformer_type: str,
        transformer_name: str,
        transformer_dropout: float,
        c2w_kwargs: Dict[str, Any],
        token_embeddings_dropout: float,
        layer_attn_kwargs: Dict[str, Any],
        rnn_kwargs: Dict[str, Any],
        label_smoothing: float,
        char_mask_p: float,
        mask_p: float,
        transformer_lrs: Dict[int, float],
        rnn_lr: float,
        clf_lr: float,
        n_warmup_steps: int,
        optim_kwargs: Dict[str, Any],
        len_char_vocab: int,
        idx_char_unk: int,
        idx_char_pad: int,
        idx_token_unk: int,
        idx_token_pad: int,
        n_lemma_scripts: int,
        n_morph_tags: int,
        n_morph_cats: int,
        unfreeze_transformer_epoch: int,
        ignore_idx: int = -1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # ======================================================================
        # Model hyperparameters
        # ======================================================================
        # Module hyperparmeters ================================================
        self.transformer_type = transformer_type
        self.transformer_name = transformer_name
        self.transformer_dropout = transformer_dropout
        self.c2w_kwargs = c2w_kwargs
        self.layer_attn_kwargs = layer_attn_kwargs
        self.rnn_kwargs = rnn_kwargs
        self.label_smoothing = label_smoothing

        # Number of classes ====================================================
        self.n_lemma_scripts = n_lemma_scripts
        self.n_morph_tags = n_morph_tags
        self.n_morph_cats = n_morph_cats

        # Special tokens =======================================================
        self.idx_char_unk = idx_char_unk
        self.idx_char_pad = idx_char_pad
        self.idx_token_unk = idx_token_unk
        self.idx_token_pad = idx_token_pad

        # Transformer & C2W Embeddings =========================================
        self.config = AutoConfig.from_pretrained(transformer_name)

        if transformer_dropout is not None:
            dropouts = dict()
            for k, v in self.config.__dict__.items():
                if "dropout" in k and isinstance(v, float):
                    dropouts[k] = transformer_dropout
            self.config.__dict__.update(dropouts)

        self.transformer = AutoModel.from_pretrained(
            transformer_name, config=self.config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True,)

        self.char_mask = SequenceMask(
            mask_p=char_mask_p, mask_idx=0, ign_idx=idx_char_pad,
        )

        self.c2w = Char2Word(
            vocab_len=len_char_vocab, padding_idx=idx_char_pad, **c2w_kwargs,
        )

        # Word-level RNNs ======================================================

        self.attend_last_L = layer_attn_kwargs["L"]

        self.lemma_layer_attn = LayerAttention(**layer_attn_kwargs)

        self.lemma_token_dropout = nn.Dropout(p=token_embeddings_dropout)

        self.lemma_lstm = ResidualRNN(**rnn_kwargs)

        self.morph_layer_attn = LayerAttention(**layer_attn_kwargs)

        self.morph_token_dropout = nn.Dropout(p=token_embeddings_dropout)

        self.morph_lstm = ResidualRNN(**rnn_kwargs)

        # Lemma classification =================================================
        # self._lemma_in_features = (
        #    self.word_rnn_kwargs["h_dim"] + self.c2w_kwargs["out_dim"]
        # )
        self._lemma_in_features = rnn_kwargs["h_dim"]

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
        self._morph_in_features = rnn_kwargs["h_dim"]

        self.morph_clf_unf = nn.Sequential(
            ResidualMLP(
                in_features=self._morph_in_features,
                out_features=self._morph_in_features,
            ),
            nn.Linear(
                in_features=self._morph_in_features, out_features=self.n_morph_tags
            ),
        )

        self.morph_clf_fac = nn.Sequential(
            ResidualMLP(
                in_features=self._morph_in_features,
                out_features=self._morph_in_features,
            ),
            nn.Linear(
                in_features=self._morph_in_features, out_features=self.n_morph_cats
            ),
        )

        # ==========================================================================
        # Regularization
        # ==========================================================================
        self.mask_p = mask_p

        self.label_smoothing = label_smoothing

        # ======================================================================
        # Optimization
        # ======================================================================
        self.transformer_lrs = transformer_lrs
        self.rnn_lr = rnn_lr
        self.clf_lr = clf_lr
        self.n_warmup_steps = n_warmup_steps
        self.optim_kwargs = optim_kwargs

        self.unfreeze_transformer_epoch = unfreeze_transformer_epoch

        # ======================================================================
        # Misc (e.g. logging)
        # ======================================================================

        self.ignore_idx = ignore_idx

    def _trainable_modules(self):
        return [
            self.transformer,
            self.c2w,
            self.lemma_layer_attn,
            self.lemma_token_dropout,
            self.lemma_lstm,
            self.morph_layer_attn,
            self.morph_token_dropout,
            self.morph_lstm,
            self.lemma_clf,
            self.morph_clf_unf,
            self.morph_clf_fac,
        ]

    def configure_optimizers(self):

        if self.transformer_type == "distilbert":
            transformer_layers = {
                l: layer
                for l, layer in enumerate(
                    self.transformer._modules["transformer"].layer
                )
            }

        elif self.transformer_type == "bert":
            transformer_layers = {
                l: layer
                for l, layer in enumerate(self.transformer._modules["encoder"].layer)
            }

        transformer_lrs = [
            {"params": v.parameters(), "lr": self.transformer_lrs[k]}
            for k, v in transformer_layers.items()
        ]

        transformer_optimizer = optim.AdamW(transformer_lrs, **self.optim_kwargs)

        transformer_scheduler = InvSqrtWithLinearWarmupScheduler(
            transformer_optimizer,
            default_lrs=transformer_lrs,
            n_warmup_steps=self.n_warmup_steps,
        )

        lrs = []
        lrs.append({"params": self.c2w.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.lemma_layer_attn.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.lemma_lstm.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.morph_layer_attn.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.morph_lstm.parameters(), "lr": self.rnn_lr})

        lrs.append({"params": self.lemma_clf.parameters(), "lr": self.clf_lr})
        lrs.append({"params": self.morph_clf_unf.parameters(), "lr": self.clf_lr})
        lrs.append({"params": self.morph_clf_fac.parameters(), "lr": self.clf_lr})

        rest_optimizer = optim.AdamW(lrs, **self.optim_kwargs)

        rest_scheduler = InvSqrtWithLinearWarmupScheduler(
            rest_optimizer, default_lrs=lrs, n_warmup_steps=self.n_warmup_steps
        )

        return (
            [transformer_optimizer, rest_optimizer],
            [
                {"scheduler": transformer_scheduler, "interval": "step"},
                {"scheduler": rest_scheduler, "interval": "step"},
            ],
        )

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def forward(
        self,
        char_lens: Union[list, torch.Tensor],
        chars: torch.Tensor,
        token_lens: Union[list, torch.Tensor],
        tokens_raw: torch.Tensor,
        skip_morph_reg: bool = False,
    ) -> Tuple[torch.Tensor]:

        # The lens tensors need to be on CPU in case of packing
        if isinstance(char_lens, list):
            char_lens = torch.tensor(char_lens, dtype=torch.long, device="cpu")

        if isinstance(token_lens, list):
            token_lens = torch.tensor(token_lens, dtype=torch.long, device="cpu")

        batch_size = len(tokens_raw)

        # ==============================================================================
        # Contextual Token Encoding
        # ==============================================================================
        if self.mask_p >= 0.0 and self.training:
            tokens_raw = [
                [
                    t if random.random() >= self.mask_p else self.tokenizer._mask_token
                    for t in seq
                ]
                for seq in tokens_raw
            ]

        transformer_input = self.tokenizer(
            tokens_raw,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            is_split_into_words=True,
        )

        # Concatenate the hidden layer together and chop of the <BOS> and <EOS> tokens
        context_embeddings = torch.stack(
            self.transformer(
                transformer_input["input_ids"].to(self.device),
                transformer_input["attention_mask"].to(self.device),
                output_hidden_states=True,
            ).hidden_states,
            dim=2,
        )

        # Deal with BPE and <BOS> and <EOS> tokens
        # MASK and UNK tokens are also kept
        token_map = [
            torch.logical_and(
                # Only keep the first BPE, i.e. those with non-zero span start
                transformer_input["offset_mapping"][i, :, 0] == 0,
                # Remove [CLS], [END], [PAD] tokens, i.e. those with zero span end
                transformer_input["offset_mapping"][i, :, 1] != 0,
            )
            for i in range(batch_size)
        ]

        context_embeddings = [
            context_embeddings[i, token_map[i], :] for i in range(batch_size)
        ]

        # Pad the tensors so they match the un-truncated input lengths
        context_embeddings = torch.stack(
            [
                F.pad(
                    cte,
                    # Add nothing to the top of the first three dimensions
                    # Pad the end of the 1st dimension (sequence length)
                    (0, 0, 0, 0, 0, max(token_lens) - cte.size(0)),
                    mode="constant",
                    value=0,
                )
                for cte in context_embeddings
            ],
            dim=0,
        )

        # Throw an error if transformer output does not match the desired output length
        assert context_embeddings.size(1) == max(
            token_lens
        ), f"Output of transformer, {context_embeddings.size(1)}, is not equal to maximum seq length, {max(token_lens)}"

        # ======================================================================
        # C2W embeddings
        # ======================================================================
        chars = self.char_mask(chars)

        c2w_embeds_ = self.c2w(chars, char_lens)

        seqs = []
        beg = torch.tensor([0])
        for l in token_lens:
            seqs.append(c2w_embeds_[beg : beg + l])
            beg += l

        c2w_embeds = pad_sequence(
            seqs, padding_value=self.idx_token_pad, batch_first=True
        )

        # ==============================================================================
        # Lemma decoder
        # ==============================================================================
        h_lemma = self.lemma_layer_attn(
            context_embeddings[:, :, -self.attend_last_L :, :]
        )

        h_lemma += c2w_embeds

        h_lemma = self.lemma_token_dropout(h_lemma)

        h_lemma = self.lemma_lstm(h_lemma)

        lemma_logits = self.lemma_clf(h_lemma)

        # ==============================================================================
        # Morph tag decoder
        # ==============================================================================
        h_morph = self.morph_layer_attn(
            context_embeddings[:, :, -self.attend_last_L :, :]
        )

        h_morph += c2w_embeds

        h_morph = self.morph_token_dropout(h_morph)

        h_morph = self.morph_lstm(h_morph)

        morph_logits_unf = self.morph_clf_unf(h_morph)

        if not skip_morph_reg:
            morph_logits_fac = self.morph_clf_fac(h_morph)

            return lemma_logits, morph_logits_unf, morph_logits_fac

        return lemma_logits, morph_logits_unf

    def loss(
        self,
        lemma_logits: torch.Tensor,
        lemma_tags: torch.Tensor,
        morph_logits_unf: torch.Tensor,
        morph_tags: torch.Tensor,
        morph_logits_fac: Union[torch.Tensor, None] = None,
        morph_cats: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:

        lemma_loss = F.cross_entropy(
            lemma_logits.permute(0, 2, 1),
            lemma_tags,
            ignore_index=self.ignore_idx,
            label_smoothing=self.label_smoothing,
        )

        morph_unf_loss = F.binary_cross_entropy_with_logits(
            morph_logits_unf,
            label_smooth(self.label_smoothing, morph_tags.float()),
            reduction="none",
        )
        morph_unf_loss = torch.mean(morph_unf_loss[morph_tags != self.ignore_idx])

        if (morph_logits_fac is not None) and (morph_cats is not None):
            morph_fac_loss = F.binary_cross_entropy_with_logits(
                morph_logits_fac,
                label_smooth(self.label_smoothing, morph_cats.float()),
                reduction="none",
            )
            morph_fac_loss = torch.mean(morph_fac_loss[morph_cats != self.ignore_idx])

            loss = lemma_loss + morph_unf_loss + morph_fac_loss
            losses = {
                # Total never includes the factored loss to avoid differences between models
                # Plus, not relevant for model choice anyway
                "total": lemma_loss + morph_unf_loss,
                "lemma": lemma_loss,
                "morph": morph_unf_loss,
                "morph_reg": morph_fac_loss,
            }

        else:
            loss = lemma_loss + morph_unf_loss
            losses = {"total": loss, "lemma": lemma_loss, "morph": morph_unf_loss}

        return loss, losses

    def on_train_epoch_start(self):
        transformer_scheduler = self.lr_schedulers()[0]
        if self.current_epoch < self.unfreeze_transformer_epoch:
            transformer_scheduler.freeze()
        else:
            transformer_scheduler.thaw()

    def training_step(self, batch, batch_idx, optimizer_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            tokens,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens_raw
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics(
            "train", losses, lemma_logits, lemma_tags, morph_logits, morph_tags
        )

        return loss

    def validation_step(self, batch, batch_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            tokens,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens_raw
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics(
            "valid", losses, lemma_logits, lemma_tags, morph_logits, morph_tags
        )

        return loss

    def test_step(self, batch, batch_idx):

        (
            char_lens,
            chars,
            token_lens,
            tokens_raw,
            tokens,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(
            char_lens, chars, token_lens, tokens_raw
        )

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics("test", losses, lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss


class UDIFYFineTune(UDIFY):
    def __init__(
        self,
        file_path: str,
        device,
        n_lemma_scripts: int,
        n_morph_tags: int,
        n_morph_cats: int,
        rnn_lr: float,
        clf_lr: float,
        optim_kwargs: dict,
        n_warmup_steps: int,
    ):
        self.save_hyperparameters()

        # Load in pre-trained model
        self.file_path = Path(file_path)

        model_state_dict = torch.load(file_path, map_location=device)

        super().__init__(**model_state_dict["hyper_parameters"])

        self.load_state_dict(model_state_dict["state_dict"], strict=True)

        # Overwrite inherited hyperparameters
        self.n_lemma_scripts = n_lemma_scripts
        self.n_morph_tags = n_morph_tags
        self.n_morph_cats = n_morph_cats
        self.rnn_lr = rnn_lr
        self.clf_lr = clf_lr
        self.optim_kwargs = optim_kwargs
        self.n_warmup_steps = n_warmup_steps

        # Freeze the transformer
        for param in self.transformer.parameters():
            param.requires_grad = False

        # Reset the RNNs

        self.lemma_layer_attn = LayerAttention(
            **model_state_dict["hyper_parameters"]["layer_attn_kwargs"]
        )

        self.lemma_token_dropout = nn.Dropout(
            p=model_state_dict["hyper_parameters"]["token_embeddings_dropout"]
        )

        self.lemma_lstm = ResidualRNN(
            **model_state_dict["hyper_parameters"]["rnn_kwargs"]
        )

        self.morph_layer_attn = LayerAttention(
            **model_state_dict["hyper_parameters"]["layer_attn_kwargs"]
        )

        self.morph_token_dropout = nn.Dropout(
            p=model_state_dict["hyper_parameters"]["token_embeddings_dropout"]
        )

        self.morph_lstm = ResidualRNN(
            **model_state_dict["hyper_parameters"]["rnn_kwargs"]
        )

        # Reset the classifiers

        self.lemma_clf = nn.Sequential(
            ResidualMLP(
                in_features=self._lemma_in_features,
                out_features=self._lemma_in_features,
            ),
            nn.Linear(
                in_features=self._lemma_in_features, out_features=self.n_lemma_scripts
            ),
        )

        self.morph_clf_unf = nn.Sequential(
            ResidualMLP(
                in_features=self._morph_in_features,
                out_features=self._morph_in_features,
            ),
            nn.Linear(
                in_features=self._morph_in_features, out_features=self.n_morph_tags
            ),
        )

        self.morph_clf_fac = nn.Sequential(
            ResidualMLP(
                in_features=self._morph_in_features,
                out_features=self._morph_in_features,
            ),
            nn.Linear(
                in_features=self._morph_in_features, out_features=self.n_morph_cats
            ),
        )

    def _trainable_modules(self):
        return [
            self.transformer,
            self.c2w,
            self.lemma_layer_attn,
            self.lemma_token_dropout,
            self.lemma_lstm,
            self.morph_layer_attn,
            self.morph_token_dropout,
            self.morph_lstm,
            self.lemma_clf,
            self.morph_clf_unf,
            self.morph_clf_fac,
        ]

    def configure_optimizers(self):

        lrs = []
        lrs.append({"params": self.c2w.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.lemma_layer_attn.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.lemma_lstm.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.morph_layer_attn.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.morph_lstm.parameters(), "lr": self.rnn_lr})

        lrs.append({"params": self.lemma_clf.parameters(), "lr": self.clf_lr})
        lrs.append({"params": self.morph_clf_unf.parameters(), "lr": self.clf_lr})
        lrs.append({"params": self.morph_clf_fac.parameters(), "lr": self.clf_lr})

        optimizer = optim.AdamW(lrs, **self.optim_kwargs)

        scheduler = InvSqrtWithLinearWarmupScheduler(
            optimizer, default_lrs=lrs, n_warmup_steps=self.n_warmup_steps
        )

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step"}],
        )

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def on_train_epoch_start(self):
        # Ignore the freezing/unfreezing of the transformer
        pass

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx, 0)

    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return super().test_step(batch, batch_idx)


class DogTagSmall(JointTaggerLemmatizer):
    """A CANINE based joint morphological tagger and lemmatizer.

    """

    def __init__(
        self,
        transformer_dropout: float,
        mha_kwargs: Dict[str, Any],
        batch_first: bool,
        label_smoothing: float,
        mask_p: float,
        embedding_dropout: float,
        transformer_lrs: Optional[Dict[int, float]],
        rnn_lr: float,
        clf_lr: float,
        n_warmup_steps: int,
        optim_kwargs: Dict[str, Any],
        scheduler_kwargs: Dict[str, Any],
        idx_char_pad: int,
        idx_token_pad: int,
        n_lemma_scripts: int,
        n_morph_tags: int,
        n_morph_cats: int,
        unfreeze_transformer_epoch: int,
        ignore_idx: int = -1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # ======================================================================
        # Model hyperparameters
        # ======================================================================
        # Module hyperparmeters ================================================
        self.transformer_type = "canine"
        self.transformer_name = "google/canine-c"
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

        self.morph_mha = MultiHeadSequenceAttention(
            d_in=self.h_dim,
            d_out=self.h_dim,
            batch_first=self.batch_first,
            **self.mha_kwargs,
        )

        self.morph_dropout = nn.Dropout(p=self.embedding_dropout)

        self.lemma_mha = MultiHeadSequenceAttention(
            d_in=self.h_dim,
            d_out=self.h_dim,
            batch_first=self.batch_first,
            **self.mha_kwargs,
        )

        self.lemma_dropout = nn.Dropout(p=self.embedding_dropout)

        # Classifiers ==================================================================
        self.lemma_clf = nn.Sequential(
            ResidualMLP(in_features=self.h_dim, out_features=self.h_dim,),
            nn.Linear(in_features=self.h_dim, out_features=n_lemma_scripts),
        )

        self.morph_unf_clf = nn.Sequential(
            ResidualMLP(in_features=self.h_dim, out_features=self.h_dim,),
            nn.Linear(in_features=self.h_dim, out_features=self.n_morph_tags),
        )

        self.morph_fac_clf = nn.Sequential(
            ResidualMLP(in_features=self.h_dim, out_features=self.h_dim,),
            nn.Linear(in_features=self.h_dim, out_features=self.n_morph_cats),
        )

        # ==========================================================================
        # Regularization
        # ==========================================================================
        self.mask_p = mask_p

        self.label_smoothing = label_smoothing

        # ======================================================================
        # Optimization
        # ======================================================================
        self.transformer_lrs = transformer_lrs
        if self.transformer_lrs is None:
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.rnn_lr = rnn_lr
        self.clf_lr = clf_lr
        self.n_warmup_steps = n_warmup_steps
        self.optim_kwargs = optim_kwargs
        self.scheduler_kwargs = scheduler_kwargs

        self.unfreeze_transformer_epoch = unfreeze_transformer_epoch

        # ======================================================================
        # Misc (e.g. logging)
        # ======================================================================

        self.ignore_idx = ignore_idx

        # Special tokens =======================================================
        self.idx_char_pad = idx_char_pad
        self.idx_token_pad = idx_token_pad

    def _trainable_modules(self):
        reg_params = [
            self.morph_mha,
            self.morph_dropout,
            self.lemma_mha,
            self.lemma_dropout,
            self.lemma_clf,
            self.morph_unf_clf,
            self.morph_fac_clf,
        ]

        if self.transformer_lrs is not None:
            return reg_params

        else:
            return [self.transformer] + reg_params

    def configure_optimizers(self):

        if self.transformer_lrs is not None:
            transformer_lrs = [
                {
                    "params": self.transformer._modules[mod_name].parameters(),
                    "lr": float(self.transformer_lrs[mod_name]),
                }
                for mod_name in self.transformer._modules
            ]

            transformer_optimizer = optim.AdamW(transformer_lrs, **self.optim_kwargs)

            transformer_scheduler = InvSqrtWithLinearWarmupScheduler(
                transformer_optimizer,
                default_lrs=transformer_lrs,
                n_warmup_steps=self.n_warmup_steps,
            )

        lrs = []
        lrs.append({"params": self.morph_mha.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.lemma_mha.parameters(), "lr": self.rnn_lr})

        lrs.append({"params": self.lemma_clf.parameters(), "lr": self.clf_lr})
        lrs.append({"params": self.morph_unf_clf.parameters(), "lr": self.clf_lr})
        lrs.append({"params": self.morph_fac_clf.parameters(), "lr": self.clf_lr})

        rest_optimizer = optim.AdamW(lrs, **self.optim_kwargs)

        rest_scheduler = InvSqrtWithLinearWarmupScheduler(
            rest_optimizer,
            default_lrs=transformer_lrs,
            n_warmup_steps=self.n_warmup_steps,
        )

        if self.transformer_lrs is not None:
            return (
                [transformer_optimizer, rest_optimizer],
                [
                    {"scheduler": transformer_scheduler, "interval": "step"},
                    {"scheduler": rest_scheduler, "interval": "step"},
                ],
            )
        else:
            return (
                [rest_optimizer],
                [{"scheduler": rest_scheduler, "interval": "step"}],
            )

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

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
        lemma_logits = self.lemma_clf(lemma_embeddings)

        # Morph Pooling ========================================================
        # Pool the character embeddings to token embeddings
        # [B x L_t, L_c, D] -> [B x L_t, D]
        pooled_morph_embeddings = self.morph_mha(embeddings, attention_mask)

        # Move the token embeddings back to batch, seq length, dim tensor
        # [B x L_t, D] -> [B, L_t, D]
        morph_embeddings = [
            pooled_morph_embeddings[beg:end]
            for beg, end in zip(
                np.cumsum([0] + n_tokens_per_sent[:-1]), np.cumsum(n_tokens_per_sent)
            )
        ]

        # Pad the ragged list of token embeddings to the longest sentence length
        morph_embeddings = pad_sequence(
            morph_embeddings,
            padding_value=self.tokenizer.pad_token_id,
            batch_first=True,
        )

        # Morph classification =========================================================

        morph_unf_logits = self.morph_unf_clf(morph_embeddings)

        if not skip_morph_reg:
            morph_fac_logits = self.morph_fac_clf(morph_embeddings)

            return lemma_logits, morph_unf_logits, morph_fac_logits

        return lemma_logits, morph_unf_logits

    def loss(
        self,
        lemma_logits: torch.Tensor,
        lemma_tags: torch.Tensor,
        morph_logits_unf: torch.Tensor,
        morph_tags: torch.Tensor,
        morph_logits_fac: Union[torch.Tensor, None] = None,
        morph_cats: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:

        lemma_loss = F.cross_entropy(
            lemma_logits.permute(0, 2, 1),
            lemma_tags,
            ignore_index=self.ignore_idx,
            label_smoothing=self.label_smoothing,
        )

        morph_unf_loss = F.binary_cross_entropy_with_logits(
            morph_logits_unf,
            label_smooth(self.label_smoothing, morph_tags.float()),
            reduction="none",
        )
        morph_unf_loss = torch.mean(morph_unf_loss[morph_tags != self.ignore_idx])

        if (morph_logits_fac is not None) and (morph_cats is not None):
            morph_fac_loss = F.binary_cross_entropy_with_logits(
                morph_logits_fac,
                label_smooth(self.label_smoothing, morph_cats.float()),
                reduction="none",
            )
            morph_fac_loss = torch.mean(morph_fac_loss[morph_cats != self.ignore_idx])

            loss = lemma_loss + morph_unf_loss + morph_fac_loss
            losses = {
                # Total never includes the factored loss to avoid differences between models
                # Plus, not relevant for model choice anyway
                "total": lemma_loss + morph_unf_loss,
                "lemma": lemma_loss,
                "morph": morph_unf_loss,
                "morph_reg": morph_fac_loss,
            }

        else:
            loss = lemma_loss + morph_unf_loss
            losses = {"total": loss, "lemma": lemma_loss, "morph": morph_unf_loss}

        return loss, losses

    def on_train_epoch_start(self):
        if self.transformer_lrs is not None:
            transformer_scheduler = self.lr_schedulers()[0]
            if self.current_epoch < self.unfreeze_transformer_epoch:
                transformer_scheduler.freeze()
            else:
                transformer_scheduler.thaw()

    def training_step(self, batch, batch_idx: int = 0, optimizer_idx: int = 0):

        (
            _,
            _,
            _,
            tokens_raw,
            _,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(tokens_raw)

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics(
            "train", losses, lemma_logits, lemma_tags, morph_logits, morph_tags
        )

        return loss

    def validation_step(self, batch, batch_idx):

        (
            _,
            _,
            _,
            tokens_raw,
            _,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(tokens_raw)

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics(
            "valid", losses, lemma_logits, lemma_tags, morph_logits, morph_tags
        )

        return loss

    def test_step(self, batch, batch_idx):

        (
            _,
            _,
            _,
            tokens_raw,
            _,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(tokens_raw)

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics("test", losses, lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss


class DogTag(JointTaggerLemmatizer):
    """A CANINE based joint morphological tagger and lemmatizer.

    """

    def __init__(
        self,
        transformer_dropout: float,
        rnn_kwargs: Dict[str, Any],
        mha_kwargs: Dict[str, Any],
        batch_first: bool,
        label_smoothing: float,
        mask_p: float,
        embedding_dropout: float,
        transformer_lrs: Optional[Dict[int, float]],
        rnn_lr: float,
        clf_lr: float,
        n_warmup_steps: int,
        optim_kwargs: Dict[str, Any],
        idx_char_pad: int,
        idx_token_pad: int,
        n_lemma_scripts: int,
        n_morph_tags: int,
        n_morph_cats: int,
        unfreeze_transformer_epoch: int,
        ignore_idx: int = -1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

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

        self.char_dropout = nn.Dropout(p=self.embedding_dropout)

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

        self.label_smoothing = label_smoothing

        # ======================================================================
        # Optimization
        # ======================================================================
        self.transformer_lrs = transformer_lrs
        if self.transformer_lrs is None:
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.rnn_lr = rnn_lr
        self.clf_lr = clf_lr
        self.n_warmup_steps = n_warmup_steps
        self.optim_kwargs = optim_kwargs

        self.unfreeze_transformer_epoch = unfreeze_transformer_epoch

        # ======================================================================
        # Misc (e.g. logging)
        # ======================================================================

        self.ignore_idx = ignore_idx

        # Special tokens =======================================================
        self.idx_char_pad = idx_char_pad
        self.idx_token_pad = idx_token_pad

    def _trainable_modules(self):
        reg_params = [
            self.char_dropout,
            self.token_mha,
            self.token_dropout,
            self.token_rnn,
            self.lemma_mha,
            self.lemma_dropout,
            self.lemma_clf,
            self.morph_unf_clf,
            self.morph_fac_clf,
        ]

        if self.transformer_lrs is None:
            return reg_params

        else:
            return [self.transformer] + reg_params

    def configure_optimizers(self):

        if self.transformer_lrs is not None:
            transformer_lrs = [
                {
                    "params": self.transformer._modules[mod_name].parameters(),
                    "lr": float(self.transformer_lrs[mod_name]),
                }
                for mod_name in self.transformer._modules
            ]

            transformer_optimizer = optim.AdamW(transformer_lrs, **self.optim_kwargs)

            transformer_scheduler = InvSqrtWithLinearWarmupScheduler(
                transformer_optimizer,
                default_lrs=transformer_lrs,
                n_warmup_steps=self.n_warmup_steps,
            )

        lrs = []
        lrs.append({"params": self.token_mha.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.token_rnn.parameters(), "lr": self.rnn_lr})
        lrs.append({"params": self.lemma_mha.parameters(), "lr": self.rnn_lr})

        lrs.append({"params": self.lemma_clf.parameters(), "lr": self.clf_lr})
        lrs.append({"params": self.morph_unf_clf.parameters(), "lr": self.clf_lr})
        lrs.append({"params": self.morph_fac_clf.parameters(), "lr": self.clf_lr})

        rest_optimizer = optim.AdamW(lrs, **self.optim_kwargs)

        rest_scheduler = InvSqrtWithLinearWarmupScheduler(
            rest_optimizer, default_lrs=lrs, n_warmup_steps=self.n_warmup_steps
        )

        if self.transformer_lrs is not None:
            return (
                [transformer_optimizer, rest_optimizer],
                [
                    {"scheduler": transformer_scheduler, "interval": "step"},
                    {"scheduler": rest_scheduler, "interval": "step"},
                ],
            )

        else:
            return (
                [rest_optimizer],
                [{"scheduler": rest_scheduler, "interval": "step"}],
            )

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

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

        # Dropout over the contextualized character embeddings
        cce = self.char_dropout(cce)

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

    def loss(
        self,
        lemma_logits: torch.Tensor,
        lemma_tags: torch.Tensor,
        morph_logits_unf: torch.Tensor,
        morph_tags: torch.Tensor,
        morph_logits_fac: Union[torch.Tensor, None] = None,
        morph_cats: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:

        lemma_loss = F.cross_entropy(
            lemma_logits.permute(0, 2, 1),
            lemma_tags,
            ignore_index=self.ignore_idx,
            label_smoothing=self.label_smoothing,
        )

        morph_unf_loss = F.binary_cross_entropy_with_logits(
            morph_logits_unf,
            label_smooth(self.label_smoothing, morph_tags.float()),
            reduction="none",
        )
        morph_unf_loss = torch.mean(morph_unf_loss[morph_tags != self.ignore_idx])

        if (morph_logits_fac is not None) and (morph_cats is not None):
            morph_fac_loss = F.binary_cross_entropy_with_logits(
                morph_logits_fac,
                label_smooth(self.label_smoothing, morph_cats.float()),
                reduction="none",
            )
            morph_fac_loss = torch.mean(morph_fac_loss[morph_cats != self.ignore_idx])

            loss = lemma_loss + morph_unf_loss + morph_fac_loss
            losses = {
                # Total never includes the factored loss to avoid differences between models
                # Plus, not relevant for model choice anyway
                "total": lemma_loss + morph_unf_loss,
                "lemma": lemma_loss,
                "morph": morph_unf_loss,
                "morph_reg": morph_fac_loss,
            }

        else:
            loss = lemma_loss + morph_unf_loss
            losses = {"total": loss, "lemma": lemma_loss, "morph": morph_unf_loss}

        return loss, losses

    def on_train_epoch_start(self):
        if self.transformer_lrs is not None:
            transformer_scheduler = self.lr_schedulers()[0]
            if self.current_epoch < self.unfreeze_transformer_epoch:
                transformer_scheduler.freeze()
            else:
                transformer_scheduler.thaw()

    def training_step(self, batch, batch_idx: int = 0, optimizer_idx: int = 0):

        (
            _,
            _,
            _,
            tokens_raw,
            _,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(tokens_raw)

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics(
            "train", losses, lemma_logits, lemma_tags, morph_logits, morph_tags
        )

        return loss

    def validation_step(self, batch, batch_idx):

        (
            _,
            _,
            _,
            tokens_raw,
            _,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(tokens_raw)

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics(
            "valid", losses, lemma_logits, lemma_tags, morph_logits, morph_tags
        )

        return loss

    def test_step(self, batch, batch_idx):

        (
            _,
            _,
            _,
            tokens_raw,
            _,
            lemma_tags,
            morph_tags,
            morph_cats,
        ) = self._unpack_input(batch)

        lemma_logits, morph_logits, morph_reg_logits = self.forward(tokens_raw)

        loss, losses = self.loss(
            lemma_logits,
            lemma_tags,
            morph_logits,
            morph_tags,
            morph_reg_logits,
            morph_cats,
        )

        self.metrics("test", losses, lemma_logits, lemma_tags, morph_logits, morph_tags)

        return loss

class DogTagFineTune(DogTag):

    def __init__(
        self,
        file_path: str,
        device = torch.device("cpu"),
        **dogtag_kwargs,
    ) -> None:

        self.save_hyperparameters()

        # Load in pre-trained model
        self.file_path = file_path

        state_dict = torch.load(self.file_path, map_location=device)

        self.hyper_params = state_dict["hyper_parameters"]
        self.hyper_params["transformer_lrs"] = None
        self.hyper_params.update(**dogtag_kwargs)

        pruned_state_dict = OrderedDict([(k, v ) for k, v in state_dict["state_dict"].items() if "transformer" in k])

        super().__init__(**state_dict["hyper_parameters"])

        self.load_state_dict(pruned_state_dict, strict=False)

        # Freeze the transformer
        for param in self.transformer.parameters():
            param.requires_grad = False

        for mod in self._trainable_modules():
            for param in mod.parameters():
                param.requires_grad = True
