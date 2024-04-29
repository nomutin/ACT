# Copyright (c) 2017 Facebook, Inc. and its affiliates. All Rights Reserved
# ruff: noqa: D102, PLR0913, PLC2701, FBT001, FBT002, PGH003
"""
DETRで用いられているTransformer.

基本的に `torch.nn.Transformer`と同じだが, 以下の変更点がある:
    * 位置エンコーディングはMHattentionに渡される
    * エンコーダの最後に追加されていたLNは削除
    * デコーダは全てのレイヤの結果をスタックして返す

References
----------
- https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py

"""

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_activation_fn, _get_clones


class Transformer(nn.Module):
    """
    少し修正・簡略化した`torch.nn.Transformer`.

    Parameters
    ----------
    d_model: int
        encoder/decoderの入力次元数.
    nhead: int
        multi-head attentionのヘッド数.
    num_encoder_layers: int
        `TransformerEncoder`のレイヤ数.
    num_decoder_layers: int
        `TransfomerDecoder`のレイヤ数.
    dim_feedforward: int
        feedforward層の次元数.
    dropout: float
        ドロップアウト率.
    activation: str
        活性化関数の名前 (`"relu"` or `"gelu"`).
    """

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
    ) -> None:
        """各モデルの定義."""
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self) -> None:
        """パラメータの初期化(xavier)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        *,
        src: Tensor,
        query_embed: Tensor,
        pos_embed: Tensor,
        additional_pos_embed: Tensor,
        proprio_input: Tensor,
        latent_input: Tensor,
    ) -> Tensor:
        """
        順伝搬.

        Parameters
        ----------
        src: Tensor
            エンコーダへの入力テンソル(観測).
        query_embed: Tensor
            クエリ埋め込み(の重み?).
        pos_embed: Tensor
            (観測の)位置埋め込み.
        additional_pos_embed: Tensor
            追加の位置埋め込み.
            ACTでは, actionとlatentの位置埋め込み.
        proprio_input: Tensor
            query_action の線形埋め込み.
        latent_input: Tensor
            行動系列から得られた潜在表現.

        Returns
        -------
        Tensor
            デコーダの出力.
            shape: [batch, num_queries, d_model]
        """
        src = rearrange(src, "b c h w -> (h w) b c")
        pos_embed = repeat(
            rearrange(pos_embed, "b c h w -> (h w) b c"),
            "hw n c -> hw (repeat n) c",
            repeat=src.shape[1],
        )
        query_embed = repeat(query_embed, "q c -> q b c", b=src.shape[1])
        additional_pos_embed = repeat(
            additional_pos_embed,
            "p c -> p b c",
            b=src.shape[1],
        )
        pos_embed = torch.cat([additional_pos_embed, pos_embed], dim=0)
        addition_input = torch.stack([latent_input, proprio_input], dim=0)
        src = torch.cat([addition_input, src], dim=0)
        memory = self.encoder.forward(src=src, pos=pos_embed)
        hs = self.decoder.forward(
            tgt=torch.zeros_like(query_embed),
            memory=memory,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs.transpose(1, 2)


class TransformerEncoderLayer(nn.Module):
    """
    MHAとFFNから成るエンコーダのレイヤ.

    Note
    ----
    * 以下のハイパーパラメータは基本的にDecoderLayerと共有される.

    Parameters
    ----------
    d_model: int
        入力次元数.
    nhead: int
        ヘッド数.
    dim_feedforward: int
        feedforward層の次元数.
    dropout: float
        ドロップアウト率.
    activation: str
        活性化関数の名前 (`"relu"` or `"gelu"`).
    """

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
    ) -> None:
        """モデル要素の初期化."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        """
        順伝播.

        Parameters
        ----------
        src: Tensor
            エンコーダへの入力テンソル.
        src_key_padding_mask: Tensor
            バッチごとの`src`のマスク. Optional.
            ACTでは`is_pad`.
        pos: Tensor
            位置埋め込み.
        """
        q = k = with_pos_embed(src, pos)
        src2 = self.self_attn.forward(
            query=q,
            key=k,
            value=src,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2.forward(src + self.dropout2(src2))


class TransformerEncoder(nn.Module):
    """
    `TransformerEncoderLayer`を複数重ねたエンコーダ.

    Parameters
    ----------
    encoder_layer: TransformerEncoderLayer
        エンコーダのレイヤ.
    num_layers: int
        レイヤ数.

    """

    def __init__(
        self,
        *,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)  # type: ignore
        self.num_layers = num_layers

    def forward(
        self,
        *,
        src: Tensor,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        """
        順伝播.

        Note
        ----
        * 間にLNとか挟まなくていいのかな?

        Parameters
        ----------
        src: Tensor
            エンコーダへの入力テンソル.
        src_key_padding_mask: Tensor
            バッチごとの`src`のマスク. Optional.
            ACTでは`is_pad`.
        pos: Tensor
            位置埋め込み.
        """
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        return output


class TransformerDecoderLayer(nn.Module):
    """
    MHAとFFNから成るデコーダのレイヤ.

    Parameters
    ----------
    d_model: int
        入力次元数.
    nhead: int
        ヘッド数.
    dim_feedforward: int
        feedforward層の次元数.
    dropout: float
        ドロップアウト率.
    activation: str
        活性化関数の名前 (`"relu"` or `"gelu"`).

    """

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """モデルの定義."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        *,
        tgt: Tensor,
        memory: Tensor,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        """
        順伝播.

        Parameters
        ----------
        tgt: Tensor
            デコーダへの入力テンソル. ACTでは0行列.
        memory: Tensor
            エンコーダの出力.
        pos: Tensor
            位置埋め込み.
        query_pos: Tensor
            クエリ埋め込み.
        """
        q = k = with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn.forward(q, k, value=tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn.forward(
            query=with_pos_embed(tgt, query_pos),
            key=with_pos_embed(memory, pos),
            value=memory,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return self.norm3.forward(tgt + self.dropout3(tgt2))


class TransformerDecoder(nn.Module):
    """
    `TransformerDecoderLayer`を複数重ねたデコーダ.

    Parameters
    ----------
    decoder_layer: TransformerDecoderLayer
        デコーダのレイヤ.
    num_layers: int
        レイヤ数.
    norm: nn.Module
        レイヤごとの正規化.

    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: nn.Module,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)  # type: ignore
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        *,
        tgt: Tensor,
        memory: Tensor,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=memory,
                pos=pos,
                query_pos=query_pos,
            )
            intermediate.append(self.norm(output))

        output = self.norm(output)
        intermediate.pop()
        intermediate.append(output)

        return torch.stack(intermediate)


def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
    """Positional encodingをtensorに追加する."""
    return tensor if pos is None else tensor + pos
