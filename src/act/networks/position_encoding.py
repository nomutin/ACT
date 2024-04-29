# Copyright (c) 2017 Facebook, Inc. and its affiliates. All Rights Reserved
"""
位置埋め込み.

References
----------
- https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
- https://github.com/tonyzhaozh/act/blob/main/detr/models/position_encoding.py

"""

import math

import numpy as np
import torch
from torch import Tensor, nn


class PositionEmbeddingSine(nn.Module):
    """
    DETRで用いられている位置埋め込み.

    Note
    ----
    * 通常の位置エンコーディングと異なり, 入力は画像.
    * 詳しくは [DETR](https://arxiv.org/pdf/2005.12872)
    * TODO(野村): `cumsum` はdeterministicモードで使えない!

    Parameters
    ----------
    num_pos_feats: int
        x軸またはy軸に沿った各位置の特徴次元.
        最終的に返される各位置の次元はこの2倍.
    temperature: int
        位置埋め込みの温度パラメータ.
    scale: float
        位置埋め込みの係数.
    eps: float
        計算の安定のために追加する値.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
    ) -> None:
        """変数の定義."""
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = scale
        self.eps = eps

    def forward(self, tensor: Tensor) -> Tensor:
        """
        位置埋め込みの計算.

        Note
        ----
        * DETRでは入力はマスクだが, ACTでは画像そのものが入力.

        Parameters
        ----------
        tensor : Tensor
            入力テンソル. shape: [batch, height, width]

        Returns
        -------
        Tensor
            位置埋め込みされたテンソル.
            shape: [batch, num_pos_feats * 2, height, width]
        """
        x = tensor
        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(
            end=self.num_pos_feats, dtype=torch.float32, device=x.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> Tensor:
    """
    位置エンコーディングの係数を生成するらしい.

    Parameters
    ----------
    n_position : int
        ?
    d_hid : int
        ?
    """

    def get_position_angle_vec(position: int) -> list[float]:
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array([
        get_position_angle_vec(pos_i) for pos_i in range(n_position)
    ])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
