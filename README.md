# 基于 ViT-Tiny 的 STL10 实现

## 1. 项目简介

本项目是一个基于 ViT-Tiny 的 STL10 图像分类学习项目，主要用于学习和实践深度学习模型的完整流程，包括数据加载、模型构建、训练、验证、测试以及结果保存。

项目使用 Vision Transformer 作为主干网络，并实现了 attention map 可视化功能，可以观察模型在不同 Transformer 层中关注图像区域的变化，帮助理解 ViT 模型的特征学习过程。

## 2. 项目结构

```text
STL_10/
├── datasets/
│   ├── data.py              # 数据集加载与数据增强
│   └── __init__.py
├── engine/
│   ├── train.py             # 训练与验证流程
│   └── __init__.py
├── model/
│   ├── vit_tiny.py          # ViT-Tiny 模型定义
│   └── __init__.py
├── outputs/
│   ├── best_model.pth       # 保存的最优模型权重
│   └── __init__.py
├── main.py                  # 项目训练入口
└── visualize_attention.py   # attention map 可视化脚本