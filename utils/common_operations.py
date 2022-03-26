import torch

def label_smooth(epsilon: float, labels: torch.LongTensor, K: int = 2):
    """Label smoothing applied to a one-hot tensor.

    Args:
        epsilon (float): degree of label smoothing
        labels (torch.LongTensor): binary tensor
        K (int, optional): Number of classes. Defaults to 2.

    """

    labels_ = labels.float() * (1 - epsilon)
    labels_ += epsilon / K

    return labels_