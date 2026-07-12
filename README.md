# Astronomical Object Classification

基于天体测光数据的表格分类实验。任务是根据不同波段下的测光特征和红移信息，预测天体的光谱类型 `spectral_type`。

## 数据

输入文件路径：

```text
data/star_classification.csv
```

数据文件至少包含以下字段：

```text
u, g, r, i, z, redshift, spectral_type
```

其中 `u`、`g`、`r`、`i`、`z` 为不同波段下的测光信息，`redshift` 为红移特征，`spectral_type` 为目标类别。

## 方法

实验流程包括：

1. 构造颜色指数特征：`u_g`、`g_r`、`r_i`、`i_z`；
2. 使用分层抽样划分训练集和测试集；
3. 在训练集上拟合分位数分箱器，并应用到测试集；
4. 训练加权直方图朴素贝叶斯分类器。

默认配置使用均匀类别先验，并提高 `redshift` 特征在似然计算中的权重。

## 评价指标

实验输出以下结果：

- Accuracy
- Balanced Accuracy
- Classification Report
- Ablation Table

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── data/
├── results/
└── src/
    ├── features.py
    ├── quantile_binner.py
    ├── train.py
    └── weighted_hist_nb.py
```

## 运行

```bash
pip install -r requirements.txt
python -m src.train --data data/sample_star_classification.csv --target spectral_type --output-dir results
```

运行结果保存在 `results/` 目录下。
