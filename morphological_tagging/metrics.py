from typing import Union, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torchmetrics
from torchmetrics import MetricCollection


@dataclass
class RunningStats():
    # Inspired by:
    # https://www.johndcook.com/blog/standard_deviation/

    N: int = 0

    mean: int = 0.0
    pseudo_var: int = 0.0

    min: Union[int, None] = None
    max: Union[int, None] = None

    def clear(self):

        self.N = 0

        self.mean = 0.0
        self.pseudo_var = 0.0

        self.min = None
        self.max = None

    def __call__(self, x, output: bool = True):

        r_mean_error = (x - self.mean)

        self.N += 1
        self.mean += r_mean_error / self.N
        self.pseudo_var += r_mean_error * (x - self.mean)

        if (self.min is None) or (x < self.min):
            self.min = x

        if (self.max is None) or (x > self.max):
            self.max = x

        if output:
            return self._return_stats()

    @property
    def var(self):
        if self.N < 2:
            return np.nan
            #raise ValueError(f"Needs more than {self.N} data-points.")
        return self.pseudo_var / (self.N - 1)

    @property
    def se(self):
        if self.N < 2:
            return np.nan

        return np.sqrt(self.var / self.N)

    def _return_stats(self):
        return self.mean, self.var, self.se, self.min, self.max

@dataclass
class RunningStatsBatch():
    # Inspired by:
    # https://www.statstodo.com/CombineMeansSDs.php

    N: int = 0

    sum_sum: float = 0.0
    sum_squared_sum: float = 0.0

    def clear(self):

        self.N = 0
        self.sum_sum = 0.0
        self.sum_squared_sum = 0.0

    def __call__(self, x, mask = None,  output: bool = True):

        if mask is not None:
            self.N += np.nansum(mask)
            x = mask * x
        else:
            self.N += np.prod(x.shape)

        self.sum_sum += np.nansum(x)
        self.sum_squared_sum += np.nansum(np.power(x, 2))

        if output:
            return self._return_stats()

    @property
    def mean(self):
        if self.N < 1:
            return np.nan
            #raise ValueError(f"Needs more than {self.N} data-points.")

        return self.sum_sum / self.N

    @property
    def var(self):
        if self.N < 2:
            return np.nan
            #raise ValueError(f"Needs more than {self.N} data-points.")

        return (self.sum_squared_sum - np.power(self.sum_sum,2) / self.N) / (self.N - 1)

    @property
    def se(self):
        if self.N < 2:
            return np.nan

        return np.sqrt(self.var / self.N)

    def _return_stats(self):
        return self.mean, self.var, self.se

@dataclass
class RunningF1():

    numerator: int = 0
    precision_denom: int = 0
    recall_denom: int = 0
    beta: int = 2
    _eps: float = 1e-8

    def clear(self):
        self.numerator = 0
        self.precision_denom = 0
        self.recall_denom = 0

    def __call__(self, preds, targets, mask = None):

        morph_union_size = np.sum(np.logical_and(preds, targets), axis=-1)
        preds_size = np.sum(preds, axis=-1)
        targets_size = np.sum(targets, axis=-1)

        if mask is None:
            self.numerator += np.sum(morph_union_size)
            self.precision_denom += np.sum(preds_size)
            self.recall_denom += np.sum(targets_size)
        else:
            self.numerator += np.nansum(mask * morph_union_size)
            self.precision_denom += np.nansum(mask * preds_size)
            self.recall_denom += np.nansum(mask * targets_size)

        return self._return_stats()

    @property
    def precision(self):
        return self.numerator / (self.precision_denom or 1)

    @property
    def recall(self):
        return self.numerator / (self.recall_denom or 1)

    @property
    def f1(self):
        precision = self.precision
        recall = self.recall

        return self.beta * (precision * recall) / (precision + recall + self._eps)

    def _return_stats(self):

        return self.precision, self.recall, self.f1


def clf_metrics(K: int, prefix: str, ignore_idx: int = -1):
    """Some useful multilabel, multiclass metrics to track.

    Args:
        K (int): number of classes
        prefix (str): prefix to append to logging name

    Returns:
        MetricCollection
    """
    clf_metrics = MetricCollection(
        {"acc": torchmetrics.Accuracy(num_classes=K, ignore_index=ignore_idx),
        "accweighted": torchmetrics.Accuracy(num_classes=K, average='weighted', ignore_index=ignore_idx),
        "f1micro": torchmetrics.F1(num_classes=K, average='micro', ignore_index=ignore_idx),
        "f1macro": torchmetrics.F1(num_classes=K, average='macro', ignore_index=ignore_idx)}
    )

    return clf_metrics.clone(prefix=prefix)

@torch.no_grad()
def binary_ml_clf_metrics(logits, targets, prefix: str, ignore_idx: int = -1) -> Dict[str, torch.Tensor]:
    """A function for getting useful classification metrics in the case of multi-dimensional, multi-label, but binary classification.

    Args:
        logits ([type]): [description]
        targets ([type]): [description]
        prefix (str): [description]
        ignore_idx (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """

    mask = torch.where(
        targets != ignore_idx,
        1.0,
        torch.nan)

    preds = torch.round(torch.sigmoid(logits))
    match = mask * (preds == targets).float()

    acc = torch.nanmean(match, dim=(0,1))

    fn = torch.nansum(mask * torch.logical_and(match == 0, targets == 0).float(), dim=(0,1))
    fp = torch.nansum(mask * torch.logical_and(match == 0, targets == 1).float(), dim=(0,1))
    tn = torch.nansum(mask * torch.logical_and(match == 1, targets == 0).float(), dim=(0,1))
    tp = torch.nansum(mask * torch.logical_and(match == 1, targets == 1).float(), dim=(0,1))

    prevalence = torch.sum(targets == 1, dim=(0,1)) / torch.sum(torch.logical_or(targets == 0, targets == 1), dim=(0,1))

    precision = tp / (tp + fp)
    precision[torch.isnan(precision)] = 0.0

    recall = tp / (tp + fn)
    recall[torch.isnan(recall)] = 0.0

    f1 = 2 * (precision * recall / (precision + recall))
    f1[torch.isnan(f1)] = 0.0

    return {
        f"{prefix}_accuracy_marco": torch.nanmean(acc),
        f"{prefix}_accuracy_micro": torch.sum(prevalence * acc) / torch.sum(prevalence),
        f"{prefix}_precision_macro": torch.mean(precision),
        f"{prefix}_precision_micro": torch.sum(prevalence * precision) / torch.sum(prevalence),
        f"{prefix}_recall_macro": torch.mean(recall),
        f"{prefix}_recall_micro": torch.sum(prevalence * recall) / torch.sum(prevalence),
        f"{prefix}_f1_macro": torch.mean(f1),
        f"{prefix}_f1_micro": torch.sum(prevalence * f1) / torch.sum(prevalence)
    }