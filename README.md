# ACT

![python](https://img.shields.io/badge/python-3.10+-blue)
![cuda](https://img.shields.io/badge/cuda-12.0+-blue)
[![Rye](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/rye/main/artwork/badge.json)](https://rye-up.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[Action Chunking with Transformer (ACT)](https://tonyzhaozh.github.io/aloha/) の再実装

**[サンプルコードはこちら >>](./example)**

## インストール

```bash
pip install git+https://github.com/keio-crl/ACT.git
```

## API

```python
import torch
from act import ACT

batch_size, action_size = 8, 14
chunk_size = 50

act = ACT(action_size=action_size, chunk_size=chunk_size, ...)

action_prediction = act.inference_step(
    slave_proprio=torch.rand(batch_size, action_size),
    observation=torch.rand(batch_size, 3, 224, 224),
)  # [8, 50, 14]

```

## 参考文献

### 論文

- **[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://tonyzhaozh.github.io/aloha/)**
- [DE⫶TR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- [Visuo-Tactile Pretraing for Cable Plugging](https://sites.google.com/andrew.cmu.edu/visuo-tactile-cable-plugging/home)

### コード

- **[thonyzaozh/act](https://github.com/tonyzhaozh/act)**
- [Shaka-Labs/act](https://github.com/Shaka-Labs/ACT)
- [facebookresearch/detr](https://github.com/facebookresearch/detr)
- [Abraham190137/TactileACT](https://github.com/Abraham190137/TactileACT)
