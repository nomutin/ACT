"""モデル出力の可視化用関数・コールバック."""

from custom_types import DataGroup
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt
from torch import Tensor
from wandb import Image


def visualize_prediction(target: Tensor, prediction: Tensor) -> Image:
    """
    予測値と正解値を`wandb.Image`にプロットする.

    TODO(野村): wandbで画像が表示されない.

    Parameters
    ----------
    target : Tensor
        正解値. shape: [episode_len, dims]
    prediction : Tensor
        予測値. shape: [episode_len, dims]
    """
    dims = target.shape[-1]
    fig, axes = plt.subplots(dims, 1, figsize=(8, dims), tight_layout=True)
    for dim in range(target.shape[-1]):
        axes[dim].plot(target[:, dim].detach().cpu(), label="target")
        axes[dim].plot(prediction[:, dim].detach().cpu(), label="prediction")
    return Image(fig)


class LopPredictionCallback(Callback):
    """モデル出力の可視化用コールバック."""

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: DataGroup,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """`test_step`の出力をwandbに記録する."""
        if not isinstance(pl_module.logger, WandbLogger):
            return
        if not isinstance(outputs, dict):
            return
        prediction_figure = visualize_prediction(
            prediction=outputs["prediction"],
            target=outputs["target"],
        )
        pl_module.logger.experiment.log(
            {f"prediction#{batch_idx}": prediction_figure}
        )
