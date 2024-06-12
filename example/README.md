# Example

[lightning](https://lightning.ai/docs/pytorch/stable/) を使った ACT の学習・テストコード. [rye](https://rye-up.com/) を使うと便利です.

## 使い方

1. [tonyzhaozh/ACT](https://github.com/tonyzhaozh/act.git) の [Simulated experiments](https://github.com/tonyzhaozh/act/tree/main?tab=readme-ov-file#simulated-experiments) を実行し, シミュレーションデータを収集

    > We use `sim_transfer_cube_scripted` task in the examples below. Another option is `sim_insertion_scripted`. To generated 50 episodes of scripted data, run:
    >
    > ```bash
    > python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data --num_episodes 50
    >```

2. 収集したデータ (`.hdf5`) を [examples/data ディレクトリ](data)に配置

3. 依存関係のインストール

    ｀dev-dependencies` に含まれるパッケージもインストールする必要がある

    ```bash
    rye sync --no-lock
    ```

4. 学習

    [lightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) を使って学習を行う

    ```bash
    cd example
    python run.py fit --config config/sim_insertion_scripted.yaml
    ```

5. テスト

    wandb にログが記録される

    ```bash
    python run.py test --config config/test_sim_insertion_scripted.yaml
    ```
