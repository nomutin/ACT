# Copyright (c) 2017 Facebook, Inc. and its affiliates. All Rights Reserved
# ruff: noqa: PLR0913, FBT003
"""
Action Chunking with Transformers (ACT) の定義.

References
----------
* https://github.com/tonyzhaozh/act

"""

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.autograd import Variable

from act.networks.backbone import Resnet18Encoder
from act.networks.position_encoding import (
    PositionEmbeddingSine,
    get_sinusoid_encoding_table,
)
from act.networks.transformer import (
    Transformer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class ACT(nn.Module):
    """
    Action Chunking with Transformers (ACT).

    Note
    ----
    * 本家ACTの./policy.py・detr/main.py・detr/detr_vae.pyをまとめたもの.
    * Temporal Ensamble や損失の計算はこのクラスには含んでいない.

    Methods
    -------
    _create_latent_parameters
        CVAEの潜在状態(正規分布)のパラメータを生成する. 外部から参照はしない.
    _predict_action
        行動を予測する. 外部から参照はしない.
    training_step
        学習時の処理.
        マスターの行動データを含めたデータから潜在状態を生成, 行動を生成する.
    inference_step
        推論時の処理.
        CVAEの潜在状態を0行列として行動を生成する.

    Parameters
    ----------
    action_dim : int, optional
            エージェントの行動次元数.
    enc_layers : int, optional
        Transfomerエンコーダのレイヤ数.
        CVAE encoderとTransfomer Encoderで共通.
    dec_layers : int, optional
        Transformerデコーダのレイヤ数.
    nheads : int, optional
        Transformerのヘッド数.
    dropout : float, optional
        Transformer内で用いるドロップアウト率
    chunk_size : int, optional
        予測するタイムステップ数. 論文では100を推奨.
        DETRが検出できる最大オブジェクト数でもあるらしい(?).
    dim_feedforward : int, optional
        Transformerブロック内のFNNの中間次元数.
    hidden_dim : int, optional
        Transformerの埋め込みの次元数.
    latent_dim : int, optional
        潜在状態 z の次元数.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        enc_layers: int = 4,
        dec_layers: int = 7,
        nheads: int = 8,
        dropout: float = 0.1,
        chunk_size: int = 100,
        dim_feedforward: int = 2048,
        hidden_dim: int = 256,
        latent_dim: int = 32,
    ) -> None:
        """
        各モデルの定義・初期化.

        各デフォルトパラメータは論文で使用されているもの.
        """
        super().__init__()
        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            activation="relu",
        )
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=enc_layers,
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        self.backbone = Resnet18Encoder()
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2)
        self.input_proj = nn.Conv2d(
            in_channels=self.backbone.num_channels,
            out_channels=hidden_dim,
            kernel_size=1,
        )
        self.input_proj_robot_state = nn.Linear(action_dim, hidden_dim)
        self.latent_dim = latent_dim
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(action_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + chunk_size, hidden_dim),
        )
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

        self.chunk_size = chunk_size

    def _create_latent_parameters(
        self,
        *,
        slave_proprio: Tensor,
        master_actions: Tensor,
        is_pad: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        CVAEの潜在状態(正規分布)のパラメータを生成する.

        学習時(`training_step`)にのみ使用され, 推論時は使用されない.

        Parameters
        ----------
        slave_proprio : Tensor
            Slaveの関節角度. shape: [batch, action_dim]
        master_actions : Tensor
            エキスパートの行動データ. shape: [batch, episode_len, action_dim]
        is_pad : Tensor
            パディング情報. shape: [batch, episode_len]

        Returns
        -------
        tuple[Tensor, Tensor]
            潜在変数の平均と分散(の対数). shape: [batch, latent_dim]

        """
        batch_size, _ = slave_proprio.shape
        actions_embed = self.encoder_action_proj(master_actions)
        query_action_embed = rearrange(
            self.encoder_joint_proj(slave_proprio),
            "batch hidden -> batch () hidden",
        )
        cls_embed = repeat(
            self.cls_embed.weight,
            "1 hidden -> batch 1 hidden",
            batch=batch_size,
        )
        encoder_input = rearrange(
            torch.cat([cls_embed, query_action_embed, actions_embed], dim=1),
            "batch seq hidden -> seq batch hidden",
        )
        cls_joint_is_pad = torch.full((batch_size, 2), False)
        cls_joint_is_pad = cls_joint_is_pad.to(slave_proprio.device)
        is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)
        pos_embed = rearrange(
            self.pos_table.clone().detach(),
            "1 seq hidden -> seq 1 hidden",
        )
        # query model
        encoder_output = self.encoder.forward(
            src=encoder_input,
            pos=pos_embed,
            src_key_padding_mask=is_pad,
        )[0]
        latent_info = self.latent_proj(encoder_output)
        mu, logvar = torch.chunk(latent_info, 2, dim=1)
        return mu, logvar

    def _predict_action(
        self,
        *,
        latent: Tensor,
        observation: Tensor,
        slave_proprio: Tensor,
    ) -> Tensor:
        src = self.backbone(observation)
        pos = self.pos_embed(src)
        hs = self.transformer.forward(
            src=self.input_proj.forward(src),
            query_embed=self.query_embed.weight,
            pos_embed=pos,
            latent_input=self.latent_out_proj(latent),
            proprio_input=self.input_proj_robot_state(slave_proprio),
            additional_pos_embed=self.additional_pos_embed.weight,
        )[0]
        return self.action_head.forward(hs)

    def training_step(
        self,
        *,
        slave_proprio: Tensor,
        observation: Tensor,
        master_actions: Tensor,
        is_pad: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        学習時の処理.

        マスターの行動データを含めたデータから潜在状態を生成, 行動を生成する.

        Parameters
        ----------
        slave_proprio : Tensor
            Slaveの関節角度. shape: [batch, action_dim]
        observation : Tensor
            観測. shape: [batch, 3, w, h]
        master_actions : Tensor
            エキスパートの行動データ. shape: [batch, episode_len, action_dim]
        is_pad : Tensor
            パディング情報. shape: [batch, episode_len]

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            予測された行動, 潜在変数の平均, 潜在変数の分散(の対数).
            shape: [batch, episode_len, action_dim], [batch, latent_dim]

        """
        mu, logvar = self._create_latent_parameters(
            slave_proprio=slave_proprio,
            master_actions=master_actions,
            is_pad=is_pad,
        )
        latent = reparametrize(mu, logvar)
        action_prediction = self._predict_action(
            latent=latent,
            observation=observation,
            slave_proprio=slave_proprio,
        )
        return action_prediction, mu, logvar

    def inference_step(
        self,
        *,
        slave_proprio: Tensor,
        observation: Tensor,
    ) -> Tensor:
        """
        推論時の処理.

        CVAEの潜在状態を0行列として行動を生成する.

        Parameters
        ----------
        slave_proprio : Tensor
            Slaveの関節角度. shape: [batch, action_dim]
        observation : Tensor
            観測. shape: [batch, 3, w, h]

        Returns
        -------
        Tensor
            予測された行動. shape: [batch, episode_len, action_dim]

        """
        batch_size = slave_proprio.shape[0]
        latent = torch.zeros(
            [batch_size, self.latent_dim],
            dtype=torch.float32,
            device=slave_proprio.device,
        )
        return self._predict_action(
            latent=latent,
            observation=observation,
            slave_proprio=slave_proprio,
        )


def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
    """平均と分散の対数から潜在変数をサンプリングする."""
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps
