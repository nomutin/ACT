# ruff: noqa: PLR0913
"""
LightningModule の定義.

References
----------
- https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import wandb
from custom_types import DataGroup, LossDict
from lightning import LightningModule
from torch import Tensor
from torch.nn import functional as tf

from act import ACT


class ACTModule(LightningModule):
    """
    ACT の LigitningModuleラッパー.

    Parameters
    ----------
    kl_weight : float
        KL Divergence 項の重み.
        推論時は0行列を CVAE の潜在状態とするため, 重めに設定する. 論文では10.
    wandb_reference : str, optional
        Weight & Biases の Artifact の参照(full name).
        推論時にモデルの重みを読み込むために使用する.
    その他のハイパーパラメータ:
        src/act/core.pyを参照.
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
        kl_weight: float = 10.0,
        wandb_reference: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = ACT(
            action_dim=action_dim,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            nheads=nheads,
            dropout=dropout,
            chunk_size=chunk_size,
            dim_feedforward=dim_feedforward,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        self.kl_weight = kl_weight

        if wandb_reference is not None:
            self = self.load_from_wandb(wandb_reference)  # noqa: PLW0642

    def shared_step(self, batch: DataGroup) -> LossDict:
        """ACTの`training_step`を実行し, 各損失を計算する."""
        slave_proprio, observation, master_actions, is_pad = batch
        master_actions = master_actions[:, : self.model.chunk_size]
        is_pad = is_pad[:, : self.model.chunk_size]
        action_predictions, mean, logvar = self.model.training_step(
            slave_proprio=slave_proprio,
            observation=observation,
            master_actions=master_actions,
            is_pad=is_pad,
        )

        action_loss = calc_action_loss(
            prediction=action_predictions,
            target=master_actions,
            is_pad=is_pad,
        )
        kl_divergence = calc_kl_divergence(mean, logvar)
        return {
            "loss": action_loss + self.kl_weight * kl_divergence,
            "action_loss": action_loss,
            "kl_divergence": kl_divergence,
        }

    def training_step(self, batch: DataGroup, **_: int) -> LossDict:
        """
        学習ステップ.

        `shared_step`の結果をself.loggerに記録後, パラメータの更新を行う.
        """
        loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def validation_step(self, batch: DataGroup, **_: int) -> LossDict:
        """
        検証ステップ.

        `shared_step`の結果のkeyに"val_"を付与してself.loggerに記録する.
        """
        loss_dict = self.shared_step(batch)
        loss_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def test_step(self, batch: DataGroup, **_: int) -> dict[str, Tensor]:
        """
        テストステップ(open-loop).

        Note
        ----
        * 元実装に合わせて バッチサイズ=1 でのみ動作する.
        * `EpisodeDataset(sample_full_episode=True).test_dataloader()`の
          出力に合わせる.

        Parameters
        ----------
        batch : DataGroup
            slave_actions : Tensor
                shape: [1, episode_len, action_dim]
            observations : Tensor
                shape: [1, episode_len, observation_dim]
            master_actions : Tensor
                推論には使用しない.
            is_pad : Tensor
                使用しない.

        Returns
        -------
        dict[str, Tensor]
            target : Tensor
                エキスパートの行動データ. shape: [episode_len, action_dim]
            prediction : Tensor
                予測された行動. shape: [episode_len, action_dim]

        """
        slave_actions, observations, master_actions, _pad = batch
        _batch_size, episode_len, action_dim = master_actions.shape
        size = [episode_len, self.model.chunk_size + episode_len, action_dim]
        predictions = torch.zeros(size, device=self.device, dtype=torch.half)
        action_ensambled = []
        for t in range(master_actions.shape[1]):
            prediction = self.model.inference_step(
                slave_proprio=slave_actions[:, t],
                observation=observations[:, t],
            )
            predictions[[t], t:t + self.model.chunk_size] = prediction
            actions_for_curr_step = predictions[:, t]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step)))
            exp_weights /= exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            raw_action = actions_for_curr_step * exp_weights
            action_ensambled.append(raw_action.sum(dim=0, keepdim=True))
        return {
            "target": master_actions.squeeze(0),
            "prediction": torch.stack(action_ensambled, dim=1).squeeze(0),
        }

    @classmethod
    def load_from_wandb(cls, reference: str) -> "ACTModule":
        """
        Weight & Biases の Artifact からモデルの重みを読み込む.

        Parameters
        ----------
        reference : str
            Artifact の参照(full name).
            [name]/[project]/[artifact]:[version] の形式.

        """
        run = wandb.Api().artifact(reference)  # type: ignore[no-untyped-call]
        with TemporaryDirectory() as tmpdir:
            ckpt = Path(run.download(root=tmpdir))
            model = cls.load_from_checkpoint(
                checkpoint_path=ckpt / "model.ckpt",
                map_location=torch.device("cpu"),
            )
        if not isinstance(model, cls):
            msg = f"Model is not an instance of {cls}"
            raise TypeError(msg)
        return model


def calc_action_loss(
    prediction: Tensor,
    target: Tensor,
    is_pad: Tensor,
) -> Tensor:
    """
    行動の誤差項の計算.

    Note
    ----
    * L1 Lossのが精密な行動生成につながるらしい.
    * パディング部分の誤差は無視する.

    Parameters
    ----------
    prediction : Tensor
        モデルが予測した行動. shape: [batch, episode_len, action_dim]
    target : Tensor
        エキスパートの行動データ. shape: [batch, episode_len, action_dim]
    is_pad : Tensor
        パディング情報. shape: [batch, episode_len]
    """
    all_l1 = tf.l1_loss(prediction, target, reduction="none")
    return (all_l1 * ~is_pad.unsqueeze(-1)).mean()


def calc_kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    KLダイバージェンスの計算.

    潜在状態の次元方向に和をとり, バッチ方向に平均をとる.

    Parameters
    ----------
    mu : Tensor
        平均. shape: [batch, latent_dim]
    logvar : Tensor
        分散の対数. shape: [batch, latent_dim]
    """
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(-1).mean()
