# Copyright (c) 2017 Facebook, Inc. and its affiliates. All Rights Reserved
"""
ACT に用いられるコンポーネントの定義.

References
----------
- https://github.com/facebookresearch/detr/blob/main/models
- https://github.com/tonyzhaozh/act/blob/main/detr/models

Attributes
----------
backbone
    観測のエンコーダに用いる(事前学習済み)モデル.
    論文では ResNet-18 を用いているので、ここでも同様に定義.
position_encoding
    観測への位置埋め込み.
transformer
    Transformer モデルの定義.
    ACT のために `torch.nn.Transformer` 及び `detr/models/transformer` から
    細かい変更が加えられているっぽい.
"""
