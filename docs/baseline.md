# Baseline 实验记录

## 1. 数据集三分结果

基于 data/all_subset.csv（2500 条）按随机种子 42 做 8:1:1 划分：

- 训练集: data/train_subset.csv（2000）
- 验证集: data/val_subset.csv（250）
- 测试集: data/test_subset.csv（250）

完整性校验结果：

- 训练/验证/测试三者互斥（两两交集均为 0）
- 标注文件中的图片均存在于 data/train_images（missing = 0）

## 2. 运行环境与执行命令

按照 AGENT.md 要求，在 conda 环境 geoclip 中执行，并先构建模块：

```bash
cd /mnt/d/ML/project/geo-clip
source ~/.bashrc >/dev/null 2>&1 || true
conda activate geoclip
python -m pip install -e .

python scripts/split_dataset_three_way.py \
  --input-csv data/all_subset.csv \
  --train-csv data/train_subset.csv \
  --val-csv data/val_subset.csv \
  --test-csv data/test_subset.csv \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --seed 42

python scripts/run_baseline_train_eval.py \
  --train-csv data/train_subset.csv \
  --val-csv data/val_subset.csv \
  --test-csv data/test_subset.csv \
  --image-dir data/train_images \
  --epochs 1 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --output-json data/baseline_results.json \
  --checkpoint data/baseline_model.pt
```

## 3. Baseline 配置

- 模型: GeoCLIP(from_pretrained=True)
- 设备: CUDA
- Epoch: 1
- Batch size: 32
- Learning rate: 1e-4
- 训练样本: 2000
- 验证样本: 250
- 测试样本: 250

## 4. Baseline 结果

验证集（Epoch 1）：

- acc@2500km: 0.844
- acc@750km: 0.628
- acc@200km: 0.248
- acc@25km: 0.060
- acc@1km: 0.008

测试集（最佳验证模型，best val acc@200km=0.248）：

- acc@2500km: 0.796
- acc@750km: 0.548
- acc@200km: 0.312
- acc@25km: 0.040
- acc@1km: 0.000

## 5. 结果文件

- Baseline 指标: data/baseline_results.json
- Baseline 模型参数: data/baseline_model.pt
- 训练/验证/测试划分脚本: scripts/split_dataset_three_way.py
- 训练+评估脚本: scripts/run_baseline_train_eval.py
