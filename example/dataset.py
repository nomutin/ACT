"""
pytorch Dataset と lightning DataModule の定義.

References
----------
- https://lightning.ai/docs/pytorch/stable/data/datamodule.html
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from custom_types import DataGroup
from lightning import LightningDataModule
from torch import Tensor, from_numpy
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose


class EpisodeDataset(Dataset[DataGroup]):
    """
    ACT で用いられているデータセット.

    tonyzhaozh/ACT の データ収集スクリプトで得られる
    hdf5 ファイルをそのまま使用する前提で作っている.
    詳しくはREADME.mdを参照.

    Parameters
    ----------
    data_path_list : list[Path]
        データの`Path`オブジェクトのリスト.
        全てのデータは観測・行動がまとまった hdf5 ファイル.
    action_transforms : Transform
        行動データに適用する変換. 本家で使用しているのは正規化だけ.
    observation_transforms : Transform
        観測データに適用する変換. 本家で使用しているのは正規化だけ.
    sample_full_episode : bool, optional
        データセットからランダムなタイムステップからデータを取得するか.
        Falseの場合は0から取得し、paddingなどは行われない.

    """

    def __init__(
        self,
        data_path_list: list[Path],
        action_transforms: Compose,
        observation_transforms: Compose,
        sample_full_episode: bool = False,
    ) -> None:
        self.data_path_list = data_path_list
        self.action_transforms = action_transforms
        self.observation_transforms = observation_transforms
        self.sample_full_episode = sample_full_episode
        self.randgen = np.random.default_rng()

    def __len__(self) -> int:
        """データセットのサイズ."""
        return len(self.data_path_list)

    def __getitem__(self, idx: int) -> DataGroup:
        """
        1イタレーション分のデータ取得.

        Note
        ----
        * return するデータは t=0 からではなく, ランダムなタイムステップから.
        * そのため, 時系列のデータ (`master_actions`) は0でパディングされる.
        * パディングを含めた系列長は元データの長さで統一(`episode_len`).
        * ACTが学習するのは `chunk_size` (<`episode_len`)分のデータのみだが,
          モデル側で処理する.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            slave_proprio: Tensor
                現在時刻のSlaveの関節角度. shape: [action_dim]
                行動生成のqueryとして利用される. 時系列では無いので注意.
                無いなら master_actions[0] でも良さそう.
            observation: Tensor
                現在時刻の観測. shape: [3, w, h], w,hは大きい方が良い
                行動生成のqueryとして利用される. 時系列では無いので注意.
            master_actions: Tensor
                エキスパートの行動データ. shape: [episode_len, action_dim]
                時系列データ. 後半は0でパディングされている.
                latentの生成のみに使用されるため, inferenceの際には必要ない.
            is_pad: Tensor(bool)
                パディング情報. shape: [episode_len]
                Transformerのマスク(よくわからない)と損失の計算に使用される.

        Returns(sample_full_episode=True)
        ---------------------------------
        tuple[Tensor, Tensor, Tensor, Tensor]
            slave_actions: Tensor
                shape: [episode_len, action_dim]
            observations: Tensor
                shape: [episode_len, 3, w, h]
            master_actions: Tensor
                shape: [episode_len, action_dim]
            is_pad: Tensor(bool)

        """
        if self.sample_full_episode:
            with h5py.File(self.data_path_list[idx], "r") as root:
                master_actions_h5 = get_h5(root, "/action")
                slave_actions_h5 = get_h5(root, "/observations/qpos")
                observations_h5 = get_h5(root, "/observations/images/top")
                episode_len = master_actions_h5.shape[0]

                master_actions = from_numpy(master_actions_h5[...]).float()
                slave_actions = from_numpy(slave_actions_h5[...]).float()
                observations = from_numpy(observations_h5[...]).float()
                episode_len = master_actions_h5.shape[0]
                is_pad = torch.zeros(episode_len, dtype=torch.bool)

                return (
                    self.action_transforms(slave_actions),
                    self.observation_transforms(observations),
                    self.action_transforms(master_actions),
                    is_pad,
                )

        with h5py.File(self.data_path_list[idx], "r") as root:
            master_actions_h5 = get_h5(root, "/action")
            original_action_shape = master_actions_h5.shape

            episode_len = original_action_shape[0]
            start_ts = self.randgen.choice(episode_len)
            slave_proprio_h5 = get_h5(root, "/observations/qpos")[start_ts]
            master_actions_h5 = get_h5(root, "/action")
            observation_h5 = get_h5(root, "/observations/images/top")[start_ts]

            action_len = episode_len - start_ts
            padded_action = np.zeros(original_action_shape, dtype=np.float32)
            padded_action[:action_len] = master_actions_h5[start_ts:]

        master_actions = from_numpy(padded_action).float()
        slave_proprio = from_numpy(slave_proprio_h5).float()
        observation = from_numpy(observation_h5).float()
        is_pad = torch.zeros(episode_len, dtype=torch.bool)
        is_pad[action_len:] = True

        return (
            self.action_transforms(slave_proprio),
            self.observation_transforms(observation),
            self.action_transforms(master_actions),
            is_pad,
        )


class EpisodeDataModule(LightningDataModule):
    """
    `lightning`用のDataModule.

    Parameters
    ----------
    data_dir : str
        `hdf5` を置いたディレクトリのプロジェクトルートからの相対パス.
    num_workers : int
        データローダーの並列数.
    batch_size : int
        学習・検証のバッチサイズ.
    action_transforms : Compose
        行動データに適用する変換. 本家で使用しているのは正規化だけ.
    observation_transforms : Compose
        観測データに適用する変換. 本家で使用しているのは正規化だけ.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        action_transforms: Compose | None = None,
        observation_transforms: Compose | None = None,
    ) -> None:
        super().__init__()
        self.data_dir_path = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.action_transforms = action_transforms or Compose([])
        self.observation_transforms = observation_transforms or Compose([])

    def setup(self, stage: str) -> None:
        """
        学習・検証データセットの定義.

        `self.data_dir_path` 以下の hdf5 ファイルを8:2で分割する.
        """
        data_path_list = list(self.data_dir_path.glob("*.hdf5"))
        train_path_list, val_path_list = split_train_validation(
            path_list=data_path_list,
        )

        self.train_dataset = EpisodeDataset(
            data_path_list=train_path_list,
            action_transforms=self.action_transforms,
            observation_transforms=self.observation_transforms,
        )
        self.val_dataset = EpisodeDataset(
            data_path_list=val_path_list,
            action_transforms=self.action_transforms,
            observation_transforms=self.observation_transforms,
        )

        if stage == "test":
            self.test_dataset = EpisodeDataset(
                data_path_list=train_path_list + val_path_list,
                action_transforms=self.action_transforms,
                observation_transforms=self.observation_transforms,
                sample_full_episode=True,
            )

    def train_dataloader(self) -> DataLoader[DataGroup]:
        """学習用の`DataLoader`."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=1,
        )

    def val_dataloader(self) -> DataLoader[DataGroup]:
        """検証用の`DataLoader`."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=1,
        )

    def test_dataloader(self) -> DataLoader[DataGroup]:
        """テスト用の`DataLoader`."""
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=1,
        )


def get_h5(data: h5py.File, index: str) -> h5py.Dataset:
    """
    `h5.file` から指定したデータを取得する.

    Note
    ----
    * 別にこの関数は無くてもいいが, 型を厳密化するために使用.
    """
    if isinstance(content := data[index], h5py.Dataset):
        return content
    msg = f"Invalid data type: {type(content)}"
    raise TypeError(msg)


def split_train_validation(
    path_list: list[Path],
    train_ratio: float = 0.8,
) -> tuple[list[Path], list[Path]]:
    """Pathのリストを`train_ratio`で分割する."""
    split_point = int(len(path_list) * train_ratio)
    return path_list[:split_point], path_list[split_point:]


class NormalizeAction:
    """
    行動データの正規化.

    torchvision等と同様, `data = (data - mean) / std` の計算を行う.

    Parameters
    ----------
    mean : list[float]
        平均値. `(max + min) / 2` で求める.
    std : list[float]
        標準偏差. `(max - min) / 2` で求める.

    """

    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean = Tensor(mean)
        self.std = Tensor(std)

    def __call__(self, data: Tensor) -> Tensor:
        """
        正規化.

        Parameters
        ----------
        data : Tensor
            正規化するデータ.
            最後の次元数が `len(mean)` と一致してればバッチサイズは何でもいい
        """
        return data.sub(self.mean).div(self.std)
