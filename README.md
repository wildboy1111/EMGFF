# EMGFF
Selected code and recording-level five-fold splits for EMGFF
本仓库公开主模型及其必要模块的 PyTorch 实现。

- `model.py`：包含 Mel/MATS 分支编码、多尺度融合、DEMON 通道门控、Transformer、分类头和中心损失，以及主模型自检代码。
- `README.md`：代码公开范围与基本说明。

模型输入为按 Mel、DEMON、MATS 顺序排列的三路特征，形状为 `(B, 3, F, T)`；输出为特征向量、主分类分数和 DEMON 辅助分类分数。Mel 与 MATS 需提前完成物理频带对齐。

依赖 `torch` 和 `einops`，运行 `python model.py` 可进行前向自检。创建优化器或加载权重前，需先用实际特征尺寸执行一次前向计算，完成分类头初始化。

本次公开不包含数据集、预处理与训练评估脚本、训练权重及消融模型。
