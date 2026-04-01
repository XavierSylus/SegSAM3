
# SegSAM3

[返回首页](./README.md) | [English](./README_en.md)

> 基于 **SAM3 + Adapter + Federated Aggregation** 的异构多模态医学图像分割实验框架

---

## 项目简介

SegSAM3 是一个面向 **医学图像分割** 的研究型联邦学习代码仓库，核心目标是：

- 支持 **异构客户端** 场景下的联邦训练
- 以 **SAM3** 作为视觉主干
- 通过 **Adapter / 轻量参数微调** 降低训练与聚合成本
- 统一支持三类客户端：
  - `text_only`
  - `image_only`
  - `multimodal`
- 提供完整训练链路：
  - 联邦训练
  - 验证评估
  - 检查点恢复
  - 曲线绘制
  - 分割掩码导出

当前默认配置 `configs/exp_group_a.yaml` 对应一个 **GroupA image-only** 受控实验；但底层代码已经具备扩展到异构多模态联邦学习的实现基础。

---

## 主要特性

### SAM3 医学分割适配
- 支持真实 SAM3 权重加载
- 支持 mock 模式快速打通训练链路
- 支持 RoPE 频率重置与尺寸适配

### 轻量参数微调
- 在 Transformer Block 中注入 Adapter
- 冻结主干，仅训练少量关键参数
- 支持不同参数组使用不同学习率

### 异构客户端支持
- `text_only`
- `image_only`
- `multimodal`

### 完整联邦训练闭环
- 串行客户端训练
- 服务器端参数聚合
- 全局图像 / 文本表征维护
- Early Stopping、Checkpoint、日志、可视化

### 研究友好
- 支持 Dice / IoU / HD95
- 导出训练历史 JSON
- 自动绘制训练曲线
- 支持保存预测掩码与真值掩码

---

## 仓库结构

```text
SegSAM3/
├─ main.py
├─ configs/
│  └─ exp_group_a.yaml
├─ src/
│  ├─ config_manager.py
│  ├─ federated_trainer.py
│  ├─ integrated_model.py
│  ├─ client.py
│  ├─ server.py
│  └─ ...
├─ data/
│  ├─ heterogeneous_dataset_loader.py
│  └─ ...
├─ scripts/
├─ tests/
├─ requirements.txt
└─ requirements-lock.txt
````

---

## 核心模块

### main.py

训练总入口，负责：

* 解析命令行参数
* 加载 YAML 配置
* 应用 CLI 覆盖项
* 创建 FederatedTrainer
* 启动联邦训练

### src/config_manager.py

配置管理器，将 YAML 中的嵌套字段展平为 FederatedConfig，统一管理训练、验证、日志、checkpoint 与设备等参数。

### src/federated_trainer.py

联邦训练主控模块，负责：

* 初始化全局模型
* 初始化服务器聚合器
* 设置客户端与状态缓存
* 构建验证集
* 执行每轮联邦训练
* 验证、Early Stopping、保存 checkpoint、绘图

### src/integrated_model.py

核心模型 SAM3MedicalIntegrated，主要包含：

* SAM3 主干加载
* Adapter 注入
* 冻结主干参数
* 文本 / 图像投影头
* 文本 Prompt 注入
* 输出通道适配
* 医学分割头初始化
* 对比学习辅助接口

### src/client.py

客户端训练器：

* TextOnlyTrainer
* ImageOnlyTrainer
* MultimodalTrainer

### src/server.py

服务器聚合器 CreamAggregator，负责：

* 客户端权重聚合
* 模态解耦路由
* 全局图像 / 文本表征维护
* 梯度冲突探针
* heterogeneous client 三元组上传处理

### data/heterogeneous_dataset_loader.py

异构客户端数据加载器，支持：

* text_only
* image_only
* multimodal

实现：

* BraTS 风格病例读取
* 2D slice 构建
* `.nii / .nii.gz` 图像与掩码读取
* 文本特征 `.npy` 读取
* 训练 / 验证 / 公共数据加载

---

## 支持的客户端类型

### text_only

* 私有数据格式：(text_feature,)
* 公共数据格式：(text_feature,)

### image_only

* 私有数据格式：(image, mask)
* 公共数据格式：(image,)

### multimodal

* 私有数据格式：(image, mask, text_feature)
* 公共数据格式：(image, text_feature)

---

## 数据组织方式

```text
data_root/
  train/
    client_x/
      private/
        BraTSxxxx/
          *_flair.nii.gz / *_t1.nii.gz
          *_seg.nii.gz
          *_text.npy
      public/
        ...
  val/
    client_x/
      private/
        ...
```

说明：

* private：客户端私有数据
* public：公共数据
* *_text.npy：文本特征
* *_seg.nii.gz：分割标签

---

## 掩码编码说明

* channel0 = background
* channel1 = WT
* channel2 = ET

默认按 **WT 通道** 计算 Dice / IoU / HD95。

---

## 默认实验配置（GroupA）

配置文件：

```text
configs/exp_group_a.yaml
```

关键参数：

* client: client_2
* modality: image_only
* batch_size: 1
* accumulation_steps: 4
* effective_batch_size: 4
* learning_rate: 5e-5
* rounds: 60
* local_epochs: 1
* aggregation_method: fedavg
* lambda_cream: 0.0
* num_classes: 3
* device: cuda

---

## 环境安装

```bash
pip install -r requirements.txt
```

或：

```bash
pip install -r requirements-lock.txt
```

---

## 主要依赖

* torch
* torchvision
* numpy
* timm
* tensorboard
* monai
* nibabel
* matplotlib
* scikit-image
* scikit-learn
* pandas
* pyyaml
* fvcore
* fairscale
* hydra-core

---

## 运行方式

### 1. Smoke Test

```bash
python main.py --config configs/exp_group_a.yaml --rounds 1 --use_mock
```

### 2. 标准运行

```bash
python main.py --config configs/exp_group_a.yaml
```

### 3. CLI 覆盖

```bash
python main.py \
  --config configs/exp_group_a.yaml \
  --data_root data/federated_split \
  --rounds 10 \
  --batch_size 1 \
  --lr 5e-5 \
  --device cuda
```

---

## 输出内容

### 1. Checkpoints

```text
<log_dir>/checkpoints/
```

包含：

* checkpoint_round_*.pth
* latest_checkpoint.pth
* best_model.pth
* final_model.pth

### 2. 训练历史

```text
training_history.json
```

记录：

* loss
* seg / cream loss
* metrics
* lr history
* GPU 显存
* 时间
* 梯度冲突角

### 3. 曲线

* loss_curve.png
* metrics_dice_iou.png
* metrics_hd95.png

### 4. 分割结果

* 预测 mask
* GT mask

---

## 验证指标

* Dice
* IoU
* HD95

特点：

* 全局像素统计
* 验证结束统一计算
* HD95 仅 GT 非空时统计
* 默认 WT 通道

---

## 恢复训练

```yaml
checkpoint:
  resume_from_checkpoint: path/to/checkpoint.pth
```

---

## 使用前注意

### 1. 需要 SAM3 权重

```text
data/checkpoints/sam3.pt
```

或使用 `--use_mock`

### 2. 依赖外部工程

* core_projects/sam3-main
* core_projects/SAM-Adapter-PyTorch-main
* core_projects/CreamFL-main/src
* core_projects/FedFMS-main

### 3. 数据不包含

需自行准备：

* 医学影像
* 文本特征
* 权重
* 日志

---

## 推荐使用流程

### Step 1

```bash
python main.py --config configs/exp_group_a.yaml --rounds 1 --use_mock
```

### Step 2

```bash
python main.py --config configs/exp_group_a.yaml
```

### Step 3

扩展：

* clients
* 数据划分
* lambda_cream
* aggregation_method

---

## 项目特点总结

* SAM3 医学分割适配
* Adapter 轻量微调
* 异构客户端训练
* 模态解耦聚合
* 全局表征维护
* 医学分割诊断工具链

---

## 致谢

参考方向：

* SAM / SAM3
* Adapter / PEFT
* CreamFL
* 医学分割联邦学习

---

## 引用说明

请在论文或项目中注明本仓库来源，并结合具体实验引用相关基础工作。

```
```
