"""
モデルの学習・検証を行う.

References
----------
- https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html

"""

from lightning.pytorch.cli import LightningCLI


def main() -> None:
    """Lightning CLIを実行する."""
    LightningCLI(save_config_callback=None)


if __name__ == "__main__":
    main()
