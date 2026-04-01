
---

````markdown
# SegSAM3

> 基于 **SAM3 + Adapter + 联邦聚合** 的异构多模态医学图像分割实验框架  
> 当前仓库默认提供 **GroupA（image-only）** 实验配置，同时底层代码已支持 `text_only / image_only / multimodal` 三类客户端。

---

## 项目简介

SegSAM3 是一个面向 **医学图像分割** 的研究型联邦学习代码仓库，核心目标是：

- 支持 **异构客户端** 场景下的联邦训练
- 以 **SAM3** 作为视觉主干
- 通过 **Adapter/轻量参数微调** 降低训练与聚合成本
- 提供 **图像 / 文本 / 多模态** 三类客户端统一训练接口
- 支持 **训练、验证、检查点恢复、曲线绘制、掩码导出**

当前默认配置 `configs/exp_group_a.yaml` 主要用于一个受控的 **GroupA image-only 实验**；但训练器、聚合器、模型和数据加载器已经具备扩展到异构多模态联邦学习的实现基础。

---

## 主要特性

- **SAM3 医学分割适配**
  - 支持真实 SAM3 权重加载
  - 支持 mock 模式快速打通训练链路
  - 支持 RoPE 频率重置与尺寸适配

- **轻量参数微调**
  - 在 Transformer Block 中注入 Adapter
  - 冻结主干，仅训练少量关键参数
  - 支持不同参数组使用不同学习率

- **异构客户端支持**
  - `text_only`
  - `image_only`
  - `multimodal`

- **联邦训练完整链路**
  - 串行客户端训练
  - 服务器端参数聚合
  - 全局表征维护
  - 验证、Early Stopping、Checkpoint、日志、绘图

- **研究友好**
  - 保留训练诊断信息
  - 支持 Dice / IoU / HD95
  - 导出训练历史 JSON
  - 可保存预测掩码与曲线图

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

## 核心模块说明

### 1. `main.py`

训练总入口，负责：

* 解析命令行参数
* 加载 YAML 配置
* 应用 CLI 覆盖项
* 创建 `FederatedTrainer`
* 启动训练流程

---

### 2. `src/config_manager.py`

配置管理器，负责将 YAML 中的嵌套字段展平为 `FederatedConfig`，统一管理：

* 数据路径
* 训练轮数
* batch size / 学习率
* AMP
* checkpoint
* validation
* logging
* aggregation 等参数

---

### 3. `src/federated_trainer.py`

联邦训练主控模块，负责：

* 初始化全局模型
* 初始化服务器聚合器
* 设置客户端与状态缓存
* 构建验证集
* 执行每轮联邦训练
* 验证、Early Stopping、保存 checkpoint
* 输出训练曲线和训练历史

---

### 4. `src/integrated_model.py`

核心模型 `SAM3MedicalIntegrated`，主要包含：

* SAM3 主干加载
* Adapter 注入
* 冻结主干参数
* 文本 / 图像投影头
* 文本 Prompt 注入
* 输出通道适配
* 医学分割头初始化
* 对比学习辅助接口

---

### 5. `src/client.py`

客户端训练器，采用抽象基类 + 三个子类设计：

* `TextOnlyTrainer`
* `ImageOnlyTrainer`
* `MultimodalTrainer`

职责包括：

* 私有/公共 batch 解包
* 分割损失与对比损失计算
* AMP 与梯度累加
* 梯度裁剪
* 本地表征聚合
* 返回客户端更新结果

---

### 6. `src/server.py`

服务器端聚合器 `CreamAggregator`，负责：

* 客户端权重聚合
* 模态解耦路由
* 全局图像 / 文本表征维护
* 梯度冲突探针
* heterogeneous client 三元组上传的统一处理

---

### 7. `data/heterogeneous_dataset_loader.py`

异构客户端数据加载器，支持：

* `text_only`
* `image_only`
* `multimodal`

并实现：

* BraTS 风格病例读取
* 2D slice 构建
* `.nii/.nii.gz` 图像与掩码读取
* 文本特征 `.npy` 读取
* 训练 / 验证 / 公共数据加载

---

## 支持的客户端类型

### `text_only`

仅使用文本特征进行训练，不参与分割任务。

* 私有数据格式：`(text_feature,)`
* 公共数据格式：`(text_feature,)`

---

### `image_only`

仅使用图像与分割标签进行训练，是当前默认实验主路径。

* 私有数据格式：`(image, mask)`
* 公共数据格式：`(image,)`

---

### `multimodal`

同时使用图像、掩码和文本特征。

* 私有数据格式：`(image, mask, text_feature)`
* 公共数据格式：`(image, text_feature)`

---

## 数据组织方式

当前代码面向 **BraTS 风格目录**，实际加载逻辑更接近如下结构：

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

* `private/`：客户端私有训练/验证数据
* `public/`：公共数据，用于表征提取或对比学习
* `*_text.npy`：文本特征文件
* `*_seg.nii.gz`：分割标签

---

## 掩码编码说明

当前 `heterogeneous_dataset_loader.py` 中，BraTS 掩码会被转换为三通道：

* `channel0 = background`
* `channel1 = WT`
* `channel2 = ET`

当前验证流程默认按 **WT 通道** 计算 Dice / IoU / HD95。

---

## 默认实验配置（GroupA）

当前默认实验配置文件：

```bash
configs/exp_group_a.yaml
```

其主要含义如下：

* 使用 `client_2`
* 模态为 `image_only`
* `batch_size = 1`
* `accumulation_steps = 4`
* `effective_batch_size = 4`
* `learning_rate = 5e-5`
* `rounds = 60`
* `local_epochs = 1`
* `aggregation_method = fedavg`
* `lambda_cream = 0.0`
* `num_classes = 3`
* `device = cuda`
* 每轮验证一次
* 保存 mask

也就是说，当前仓库默认是一个 **受控的 image-only 联邦分割实验**，而不是默认直接跑完整的三模态异构联邦设置。

---

## 环境安装

建议使用独立虚拟环境。

### 安装依赖

```bash
pip install -r requirements.txt
```

如需更严格复现：

```bash
pip install -r requirements-lock.txt
```

---

## 主要依赖

当前仓库核心依赖包括：

* `torch`
* `torchvision`
* `numpy`
* `timm`
* `tensorboard`
* `monai`
* `nibabel`
* `matplotlib`
* `scikit-image`
* `scikit-learn`
* `pandas`
* `pyyaml`
* `fvcore`
* `fairscale`
* `hydra-core`

---

## 运行方式

### 1. 快速 smoke test

当你还没有准备好真实 SAM3 权重时，可先使用 mock 模式验证训练链路：

```bash
python main.py --config configs/exp_group_a.yaml --rounds 1 --use_mock
```

---

### 2. 标准运行

```bash
python main.py --config configs/exp_group_a.yaml
```

---

### 3. 常见命令行覆盖项

```bash
python main.py \
  --config configs/exp_group_a.yaml \
  --data_root data/federated_split \
  --rounds 10 \
  --batch_size 1 \
  --lr 5e-5 \
  --device cuda
```

支持的主要参数包括：

* `--config`
* `--data_root`
* `--rounds`
* `--batch_size`
* `--lr`
* `--lambda_cream`
* `--use_mock`
* `--device`

---

## 输出内容

训练结束后，通常会在日志目录下生成以下内容：

### 1. Checkpoints

位于：

```text
<log_dir>/checkpoints/
```

包含：

* `checkpoint_round_*.pth`
* `latest_checkpoint.pth`
* `best_model.pth`
* `final_model.pth`

---

### 2. 训练历史

导出为：

```text
training_history.json
```

记录内容包括：

* 每轮训练损失
* seg loss / cream loss
* 验证指标
* 学习率历史
* GPU 峰值显存
* 每轮耗时
* 梯度冲突角

---

### 3. 曲线图

自动绘制并保存：

* `loss_curve.png`
* `metrics_dice_iou.png`
* `metrics_hd95.png`

---

### 4. 分割结果

若开启 `save_masks`，会导出：

* 预测 mask
* 真实 mask

用于可视化检查。

---

## 验证指标

当前验证流程支持：

* **Dice**
* **IoU**
* **HD95**

特点：

* 使用全验证集全局像素累加
* Dice / IoU 在验证结束时统一计算
* HD95 仅在 GT 非空时统计
* 多通道输出时按 WT 通道评估

---

## 恢复训练

如需恢复训练，可在配置文件中设置：

```yaml
checkpoint:
  resume_from_checkpoint: path/to/checkpoint.pth
```

训练器会尝试恢复：

* server state
* training history
* client states

然后继续后续轮次训练。

---

## 使用前请注意

### 1. 真实训练需要本地 SAM3 权重

例如：

```text
data/checkpoints/sam3.pt
```

如果没有，请先用 `--use_mock` 打通链路。

---

### 2. 当前仓库依赖外部项目目录

`src/integrated_model.py` 会动态加入以下路径：

* `core_projects/sam3-main`
* `core_projects/SAM-Adapter-PyTorch-main`
* `core_projects/CreamFL-main/src`
* `core_projects/FedFMS-main`

这意味着当前仓库 **不是完全自包含** 的，需要你本地具备这些依赖工程。

---

### 3. 数据不包含在仓库中

本仓库不提供：

* 原始医学影像
* 文本特征文件
* 训练日志
* 模型权重产物

需要用户自行准备。

---

## 建议的使用顺序

### 第一步：先跑通训练链路

```bash
python main.py --config configs/exp_group_a.yaml --rounds 1 --use_mock
```

### 第二步：接入真实权重和真实数据

```bash
python main.py --config configs/exp_group_a.yaml
```

### 第三步：再扩展到异构客户端实验

修改：

* `config.clients`
* 数据划分
* `lambda_cream`
* `aggregation_method`

逐步扩展到 text/image/multimodal 联邦实验。

---

## 项目特点总结

SegSAM3 当前最有价值的地方不在于“已经打包成一个完整产品”，而在于它已经具备了以下研究骨架：

* SAM3 医学分割适配
* Adapter 轻量微调
* 异构客户端训练器
* 模态解耦聚合
* 全局表征维护
* 医学分割验证与诊断工具链

对于做课程设计、毕业设计或联邦医学分割研究来说，它是一个较好的实验基础版本。

---

## 致谢

本仓库的实现思路参考并融合了以下方向的研究与工程实践：

* SAM3 / SAM 系列视觉分割模型
* Adapter / PEFT 轻量微调方法
* CreamFL 风格跨客户端表征聚合思想
* 医学图像分割中的异构联邦学习需求

---

## 引用说明

如果你在论文、课程项目或毕业设计中使用了本仓库，请结合你的具体实验设置，引用相关基础工作，并注明本仓库实现来源。

```

---

