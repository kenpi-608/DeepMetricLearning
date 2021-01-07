from typing import Tuple

import torch
from torch import nn, Tensor


def convert_label_to_similarity(feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    normed_feature = nn.functional.normalize(feature)
    # コサイン類似度計算
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    # 同じラベルのものを見つけるために
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    # triu(diagonal=1)で行列の上三角部分を取ってこれて、かつ対角成分(自分自身)が消える
    positive_matrix = label_matrix.triu(diagonal=1)
    # logical_not → True False反転
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    # 1行に直してspとsnに  (view = reshape)
    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss
