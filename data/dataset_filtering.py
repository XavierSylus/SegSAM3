"""
数据过滤模块 - 解决空标签切片导致的模型坍塌问题
专门用于医学影像分割任务中过滤空标签（全背景）的切片

问题背景:
- BraTS 等 3D 医学影像数据集中存在大量空标签切片（边缘切片）
- 空标签切片会导致 Dice Loss 塌陷到 0，模型只学习预测背景
- 在联邦学习中，每个客户端每轮只跑 1 个 Batch 时，空标签问题会被放大

解决方案:
1. 提供 MONAI CacheDataset 的过滤函数
2. 支持按前景像素比例过滤
3. 支持优先采样含病灶切片
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import json

# MONAI imports
try:
    from monai.data import CacheDataset, Dataset
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, ScaleIntensityRanged, CropForegroundd,
        RandCropByPosNegLabeld, ToTensord
    )
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False
    print("警告: 未安装 MONAI。安装方法: pip install monai")


def filter_empty_labels(data_dicts: List[Dict],
                        min_foreground_ratio: float = 0.001,
                        verbose: bool = True) -> List[Dict]:
    """
    过滤掉不含前景像素（或前景像素极少）的数据项

    Args:
        data_dicts: MONAI 数据字典列表，每个字典包含 {'image': path, 'label': path}
        min_foreground_ratio: 最小前景像素比例（默认 0.1%）
        verbose: 是否打印过滤日志

    Returns:
        过滤后的数据字典列表
    """
    if not HAS_MONAI:
        raise ImportError("需要安装 MONAI: pip install monai")

    from monai.transforms import LoadImage

    loader = LoadImage(image_only=True)
    filtered_data = []

    total = len(data_dicts)
    filtered_count = 0

    for data_dict in data_dicts:
        label_path = data_dict.get('label', data_dict.get('mask'))
        if label_path is None:
            # 如果没有标签，跳过
            continue

        # 加载标签
        label = loader(label_path)
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        # 计算前景像素比例
        total_pixels = label.size
        foreground_pixels = np.sum(label > 0)
        foreground_ratio = foreground_pixels / total_pixels

        # 过滤判断
        if foreground_ratio >= min_foreground_ratio:
            filtered_data.append(data_dict)
        else:
            filtered_count += 1

    if verbose:
        print(f"[数据过滤] 总样本: {total}, 过滤掉: {filtered_count}, 保留: {len(filtered_data)}")
        if len(filtered_data) > 0:
            print(f"[数据过滤] 保留率: {len(filtered_data)/total*100:.1f}%")

    return filtered_data


def create_foreground_balanced_dataset(
    data_dicts: List[Dict],
    foreground_ratio: float = 0.8,
    min_foreground_pixels: int = 100,
    verbose: bool = True
) -> List[Dict]:
    """
    创建前景平衡的数据集：优先保留含前景的切片，少量保留背景切片

    适用场景：
    - 训练集：需要保留一定比例的背景切片，避免模型过拟合前景
    - 验证集：可以只保留含前景的切片，专注于病灶分割性能

    Args:
        data_dicts: MONAI 数据字典列表
        foreground_ratio: 含前景切片的目标比例（默认 80%）
        min_foreground_pixels: 判断为"含前景"的最小像素数（默认 100）
        verbose: 是否打印日志

    Returns:
        平衡后的数据字典列表
    """
    if not HAS_MONAI:
        raise ImportError("需要安装 MONAI: pip install monai")

    from monai.transforms import LoadImage

    loader = LoadImage(image_only=True)
    foreground_data = []
    background_data = []

    # 分类：前景 vs 背景
    for data_dict in data_dicts:
        label_path = data_dict.get('label', data_dict.get('mask'))
        if label_path is None:
            continue

        label = loader(label_path)
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        foreground_pixels = np.sum(label > 0)

        if foreground_pixels >= min_foreground_pixels:
            foreground_data.append(data_dict)
        else:
            background_data.append(data_dict)

    # 计算目标数量
    total_foreground = len(foreground_data)
    target_background = int(total_foreground * (1 - foreground_ratio) / foreground_ratio)
    target_background = min(target_background, len(background_data))

    # 采样背景数据
    if target_background > 0 and len(background_data) > 0:
        sampled_background = np.random.choice(
            len(background_data),
            size=target_background,
            replace=False
        )
        background_subset = [background_data[i] for i in sampled_background]
    else:
        background_subset = []

    # 合并
    balanced_data = foreground_data + background_subset
    np.random.shuffle(balanced_data)

    if verbose:
        print(f"[前景平衡] 原始: 前景={len(foreground_data)}, 背景={len(background_data)}")
        print(f"[前景平衡] 平衡后: 前景={len(foreground_data)}, 背景={len(background_subset)}")
        print(f"[前景平衡] 总计: {len(balanced_data)} 样本，前景比例: {len(foreground_data)/len(balanced_data)*100:.1f}%")

    return balanced_data


def create_filtered_monai_dataset(
    data_dicts: List[Dict],
    transforms: Compose,
    cache_rate: float = 1.0,
    num_workers: int = 4,
    filter_mode: str = 'strict',  # 'strict', 'balanced', 'none'
    min_foreground_ratio: float = 0.001,
    foreground_target_ratio: float = 0.8,
    verbose: bool = True
) -> CacheDataset:
    """
    创建带过滤功能的 MONAI CacheDataset

    Args:
        data_dicts: MONAI 数据字典列表
        transforms: MONAI 数据增强管道
        cache_rate: 缓存比例（1.0 = 全部缓存到内存）
        num_workers: 数据加载线程数
        filter_mode: 过滤模式
            - 'strict': 严格过滤，只保留含前景的切片
            - 'balanced': 平衡过滤，保留 80% 前景 + 20% 背景
            - 'none': 不过滤
        min_foreground_ratio: 最小前景像素比例（用于 strict 模式）
        foreground_target_ratio: 前景目标比例（用于 balanced 模式）
        verbose: 是否打印日志

    Returns:
        过滤后的 CacheDataset
    """
    if not HAS_MONAI:
        raise ImportError("需要安装 MONAI: pip install monai")

    # 应用过滤
    if filter_mode == 'strict':
        filtered_data = filter_empty_labels(
            data_dicts,
            min_foreground_ratio=min_foreground_ratio,
            verbose=verbose
        )
    elif filter_mode == 'balanced':
        filtered_data = create_foreground_balanced_dataset(
            data_dicts,
            foreground_ratio=foreground_target_ratio,
            verbose=verbose
        )
    else:
        filtered_data = data_dicts
        if verbose:
            print(f"[数据加载] 未启用过滤，总样本: {len(filtered_data)}")

    # 创建 CacheDataset
    dataset = CacheDataset(
        data=filtered_data,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers
    )

    return dataset


# ============================================================
# 使用示例：针对 BraTS 数据集
# ============================================================

def create_brats_data_dicts(
    data_root: str,
    fold: int = 0,
    split: str = 'train'
) -> List[Dict]:
    """
    创建 BraTS 数据字典（示例）

    Args:
        data_root: BraTS 数据根目录
        fold: 交叉验证折数
        split: 'train' 或 'val'

    Returns:
        数据字典列表
    """
    data_root = Path(data_root)
    data_dicts = []

    # 假设数据结构：data_root/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz
    # 实际使用时需要根据你的数据结构调整
    case_dirs = sorted(data_root.glob("BraTS*"))

    for case_dir in case_dirs:
        case_name = case_dir.name

        # 检查文件是否存在
        flair = case_dir / f"{case_name}_flair.nii.gz"
        t1 = case_dir / f"{case_name}_t1.nii.gz"
        t1ce = case_dir / f"{case_name}_t1ce.nii.gz"
        t2 = case_dir / f"{case_name}_t2.nii.gz"
        seg = case_dir / f"{case_name}_seg.nii.gz"

        if all([flair.exists(), t1.exists(), t1ce.exists(), t2.exists(), seg.exists()]):
            data_dicts.append({
                'image': str(flair),  # 或者使用多模态：[str(flair), str(t1), str(t1ce), str(t2)]
                'label': str(seg)
            })

    return data_dicts


def example_usage():
    """完整使用示例"""
    if not HAS_MONAI:
        print("请先安装 MONAI: pip install monai")
        return

    # 1. 创建数据字典
    data_dicts = create_brats_data_dicts(
        data_root="data/BraTS2020/MICCAI_BraTS2020_TrainingData",
        fold=0,
        split='train'
    )

    # 2. 定义数据增强
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=3,  # 正样本（前景）采样权重
            neg=1,  # 负样本（背景）采样权重
            num_samples=4
        ),
        ToTensord(keys=["image", "label"])
    ])

    # 3. 创建过滤后的数据集
    # 训练集：使用 balanced 模式（保留 80% 前景 + 20% 背景）
    train_dataset = create_filtered_monai_dataset(
        data_dicts=data_dicts,
        transforms=train_transforms,
        cache_rate=0.5,  # 缓存 50% 的数据到内存
        num_workers=4,
        filter_mode='balanced',
        foreground_target_ratio=0.8,
        verbose=True
    )

    # 验证集：使用 strict 模式（只保留含前景的切片）
    val_dataset = create_filtered_monai_dataset(
        data_dicts=data_dicts,
        transforms=train_transforms,
        cache_rate=1.0,  # 验证集较小，全部缓存
        num_workers=4,
        filter_mode='strict',
        min_foreground_ratio=0.001,
        verbose=True
    )

    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 4. 创建 DataLoader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"训练 Batch 数: {len(train_loader)}")


if __name__ == "__main__":
    example_usage()
