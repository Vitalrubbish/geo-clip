# 数据集来源说明

## 1. 本次新增与扩展的数据

| 本地文件/目录 | 类型 | 样本规模 | 来源 | 生成方式 |
|---|---|---:|---|---|
| data/train_images | 图像文件夹 | ~15,000 张 | Hugging Face 数据集 `blalexa/google-streetview-panoramas-geotagged` | 由脚本 `scripts/download_week1_data.py` 下载 parquet 分片，动态调整为 224*224 并以 JPEG (Quality 85) 格式存储 |
| data/all_subset.csv | 图像-GPS 标注 | ~15,000 条 | 同上（字段 `pano_id/lat/lon/image`） | 脚本汇总为 `IMG_FILE,LAT,LON` |
| data/train_subset.csv | 训练集标注 | ~13,500 条 | 由 `all_subset.csv` 划分得到 | 按 90% 训练划分，固定随机种子 `42` |
| data/val_subset.csv | 验证集标注 | ~13,500 条 | 由 `all_subset.csv` 划分得到 | 按 10% 验证划分，固定随机种子 `42` |
| data/im2gps3k/images/im2gps3ktest | 测试图像文件夹 | 3,000 张 | Im2GPS3k 测试集（MediaFire 发布页） | 由脚本 `scripts/download_im2gps3k_test_data.py` 自动解析下载链接并解压 |
| data/im2gps3k/im2gps3k_places365.csv | 测试元数据（原始） | 2,997 条 | Hugging Face 仓库 `Jia-py/G3-checkpoint`（文件 `im2gps3k_places365.csv`） | 脚本自动下载原始元数据 |
| data/im2gps3k/test_subset.csv | 测试集标注（GeoCLIP格式） | 2,997 条 | 由 `im2gps3k_places365.csv` 与本地图像交集生成 | 脚本输出 `IMG_FILE,LAT,LON` 三列 |

## 2. 仓库中已有数据（本次未改动来源）

| 本地文件/目录 | 说明 | 来源状态 |
|---|---|---|
| data/baseline_features.pt | baseline 特征文件 | 仓库已有文件 |
| data/images | 预置图片目录 | 仓库已有目录 |

## 3. 复现命令（训练集与测试集）

```bash
cd /mnt/d/ML/project/geo-clip
source ~/.bashrc >/dev/null 2>&1 || true
conda activate geoclip
python -m pip install -e .
# 开始全量下载（约135K张图片，实时压缩为224*224 JPEG）
python scripts/download_week1_data.py \
  --batch-size 1000 \
  --train-ratio 0.9 \
  --csv-path data/all_subset.csv \
  --train-csv-path data/train_subset.csv \
  --val-csv-path data/val_subset.csv

# 下载 Im2GPS3k 测试集并生成 GeoCLIP 格式测试 CSV
python scripts/download_im2gps3k_test_data.py \
  --root-dir data/im2gps3k \
  --output-csv data/im2gps3k/test_subset.csv
```

## 4. 存储说明

- **实时处理**：为了节省磁盘空间（原始数据约 400GB+），脚本在下载过程中将图片即时压缩为 `224*224*3` 的 JPEG 格式，最终图片目录总量约为几十 GB。
- **数据划分**：全量数据划分为 9:1 的训练集与验证集，不再单独预留测试集（遵循课程 Week1 任务要求）。

## 5. 许可与使用注意事项

- 新增数据来自 Hugging Face 数据集页面：`https://huggingface.co/datasets/blalexa/google-streetview-panoramas-geotagged`
- Im2GPS3k 图像下载来源（Revisiting Im2GPS README 提供）：`http://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip`
- Im2GPS3k 元数据来源：`https://huggingface.co/Jia-py/G3-checkpoint`（`im2gps3k_places365.csv`）
- 使用前请以该数据集页面中的 License/Terms 为准，确保与课程项目或论文复现用途兼容。
