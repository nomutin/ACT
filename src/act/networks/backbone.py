# Copyright (c) 2017 Facebook, Inc. and its affiliates. All Rights Reserved
# ruff: noqa: PLC2701
"""観測のエンコーダに用いる(事前学習済み)モデル."""

import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


class Resnet18Encoder(nn.Module):
    """学習済みResnet18を用いた画像エンコーダ."""

    def __init__(self) -> None:
        super().__init__()
        backbone = torchvision.models.resnet18(
            replace_stride_with_dilation=[False, False, False],
            pretrained=True,
            norm_layer=FrozenBatchNorm2d,
        )
        self.body = IntermediateLayerGetter(
            model=backbone, return_layers={"layer4": "0"}
        )
        self.num_channels = 512

    def forward(self, tensor: Tensor) -> Tensor:
        """
        順伝播.

        Parameters
        ----------
        tensor: torch.Tensor
            入力画像. shapeは[N, C, H, W]だが、H, Wは任意.

        Returns
        -------
        torch.Tensor
            エンコードされた特徴量. shape = [N, 512, H/32, W/32].
        """
        return self.body(tensor)["0"]  # type: ignore[no-any-return]
