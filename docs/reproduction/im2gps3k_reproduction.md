# Im2GPS3K 复现与对比

本文档固定 GeoCLIP 预训练权重在 Im2GPS3K 上的评测流程，并给出论文结果与当前仓库复现结果的对比。

## 1. 固化脚本

评测脚本：

- [scripts/eval_im2gps3k_pretrained.py](scripts/eval_im2gps3k_pretrained.py)

脚本默认输入和输出：

- 测试标注: [data/im2gps3k/test_subset.csv](data/im2gps3k/test_subset.csv)
- 测试图像目录: [data/im2gps3k/images/im2gps3ktest](data/im2gps3k/images/im2gps3ktest)
- 评测 JSON: [data/im2gps3k/geoclip_pretrained_eval.json](data/im2gps3k/geoclip_pretrained_eval.json)
- 对比表 Markdown: [docs/im2gps3k_reproduction_table.md](docs/im2gps3k_reproduction_table.md)

## 2. 复现命令

```bash
cd /mnt/d/ML/project/geo-clip
conda activate geoclip
python -m pip install -e .
python scripts/eval_im2gps3k_pretrained.py
```

## 3. 论文对比表

论文来源：GeoCLIP arXiv:2309.16020v2，Table 1(a) 的 Im2GPS3k (Ours) 行。

当前复现值来源：

- [data/im2gps3k/geoclip_pretrained_eval.json](data/im2gps3k/geoclip_pretrained_eval.json)

| 阈值 | 论文 Acc | 当前复现 Acc | 差值 (复现-论文) |
|---:|---:|---:|---:|
| 1km | 0.1411 | 0.1318 | -0.0093 |
| 25km | 0.3447 | 0.3227 | -0.0220 |
| 200km | 0.5065 | 0.4721 | -0.0344 |
| 750km | 0.6967 | 0.6473 | -0.0494 |
| 2500km | 0.8382 | 0.8045 | -0.0337 |

## 4. 说明

- 首次运行脚本时，Hugging Face 会下载 CLIP 主干权重；后续复现会走本地缓存。
- 论文与当前复现存在轻微差距是常见现象，常见原因包括推理硬件、库版本、预处理差异、测试集可用图像交集差异等。
- 如需重新生成对比表，请直接重跑 [scripts/eval_im2gps3k_pretrained.py](scripts/eval_im2gps3k_pretrained.py)。
