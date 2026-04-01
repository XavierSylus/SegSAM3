"""
多模态医学图像数据集加载器 - 支持加载文本特征
专门用于加载包含预提取文本特征的医学图像分割数据
支持 TextBraTS.json 格式的元数据文件
"""
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List, Dict
from pathlib import Path

# 尝试导入 nibabel（用于加载 .nii.gz 文件）
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("警告: 未安装 nibabel，无法加载 .nii.gz 格式的医学图像")
    print("安装方法: pip install nibabel")


class MultimodalMedicalDataset(Dataset):
    """
    多模态医学图像分割数据集（支持文本特征）

    【数据格式】
    元数据文件（JSON）结构:
    {
        "data": [
            {
                "fold": 0,
                "image": ["path/to/t1.nii.gz", "path/to/t2.nii.gz", ...],
                "label": "path/to/seg.nii.gz",
                "text_feature": "path/to/text_feature.npy"
            },
            ...
        ]
    }

    【使用示例】
    dataset = MultimodalMedicalDataset(
        json_path="data/source_images/TextBraTS.json",
        data_root="data/source_images",
        fold=0,  # 只加载 fold 0 的数据
        image_size=1024
    )

    image, mask, text_features = dataset[0]
    """

    def __init__(
        self,
        json_path: str,
        data_root: Optional[str] = None,
        fold: Optional[int] = None,
        image_size: int = 1024,
        transform: Optional[transforms.Compose] = None,
        normalize_text: bool = True,
        max_samples: Optional[int] = None,
        embed_dim: int = 768
    ):
        """
        Args:
            json_path: JSON 元数据文件路径（如 "data/source_images/TextBraTS.json"）
            data_root: 数据根目录（如果 JSON 中的路径是相对路径，需要提供根目录）
                       如果为 None，则使用 JSON 文件所在目录作为根目录
            fold: 只加载指定 fold 的数据（None 表示加载所有 fold）
            image_size: 目标图像尺寸
            transform: 图像变换（如果为 None，使用默认变换）
            normalize_text: 是否归一化文本特征（L2 normalization）
            max_samples: 最大样本数（用于调试/测试）
            embed_dim: 文本嵌入维度（用于生成空文本特征，默认768）
        """
        self.json_path = Path(json_path)
        self.image_size = image_size
        self.normalize_text = normalize_text
        self.embed_dim = embed_dim

        # 设置数据根目录
        if data_root is None:
            # 默认使用 JSON 文件所在目录
            self.data_root = self.json_path.parent
        else:
            self.data_root = Path(data_root)

        # 检查 JSON 文件是否存在
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"❌ 找不到 JSON 元数据文件: {self.json_path}\n"
                f"请检查路径是否正确。"
            )

        # 加载 JSON 元数据
        print(f"[MultimodalDataset] 正在加载元数据: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # 解析数据列表
        if isinstance(metadata, dict) and 'data' in metadata:
            data_list = metadata['data']
        elif isinstance(metadata, list):
            data_list = metadata
        else:
            raise ValueError(
                f"❌ JSON 文件格式不正确！\n"
                f"期望格式: {{'data': [...]}} 或 [...]\n"
                f"实际格式: {list(metadata.keys()) if isinstance(metadata, dict) else type(metadata)}"
            )

        # 过滤指定的 fold
        if fold is not None:
            data_list = [item for item in data_list if item.get('fold') == fold]
            print(f"[MultimodalDataset] 只加载 fold {fold} 的数据")

        # 限制样本数（用于调试）
        if max_samples is not None and max_samples > 0:
            data_list = data_list[:max_samples]
            print(f"[MultimodalDataset] 已限制加载 {len(data_list)} 个样本 (max_samples={max_samples})")

        self.data_list = data_list

        if len(self.data_list) == 0:
            raise ValueError(
                f"❌ 没有找到任何数据！\n"
                f"JSON 文件: {self.json_path}\n"
                f"Fold: {fold}\n"
                f"请检查 JSON 文件内容和 fold 参数。"
            )

        print(f"[MultimodalDataset] 成功加载 {len(self.data_list)} 个样本")

        # 验证文件是否存在（可选，用于早期错误检测）
        self._validate_files(check_first_n=5)  # 只检查前 5 个样本

        # 设置默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def _validate_files(self, check_first_n: int = 5):
        """验证文件是否存在（早期错误检测）"""
        print(f"[MultimodalDataset] 正在验证前 {check_first_n} 个样本的文件...")

        for i, item in enumerate(self.data_list[:check_first_n]):
            # 检查图像文件
            image_paths = item.get('image', [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            for img_path_str in image_paths:
                img_path = self._resolve_path(img_path_str)
                if not img_path.exists():
                    print(f"  ⚠️  样本 {i}: 图像文件不存在: {img_path}")

            # 检查标签文件
            label_path_str = item.get('label')
            if label_path_str:
                label_path = self._resolve_path(label_path_str)
                if not label_path.exists():
                    print(f"  ⚠️  样本 {i}: 标签文件不存在: {label_path}")

            # 检查文本特征文件
            text_feature_path_str = item.get('text_feature')
            if text_feature_path_str:
                text_feature_path = self._resolve_path(text_feature_path_str)
                if not text_feature_path.exists():
                    print(f"  ⚠️  样本 {i}: 文本特征文件不存在: {text_feature_path}")

        print(f"[MultimodalDataset] 文件验证完成 ✓")

    def _resolve_path(self, path_str: str) -> Path:
        """
        解析路径（处理相对路径和绝对路径）

        Args:
            path_str: 路径字符串（可能是相对路径或绝对路径）

        Returns:
            解析后的绝对路径
        """
        path = Path(path_str)

        # 如果是绝对路径，直接返回
        if path.is_absolute():
            return path

        # 如果是相对路径，相对于 data_root 解析
        return self.data_root / path

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取数据项

        Returns:
            (image, mask, text_features)
            - image: (3, H, W) - 图像张量
            - mask: (1, H, W) - 分割掩码张量
            - text_features: (D,) - 文本特征向量
        """
        item = self.data_list[idx]

        # ============================================================
        # 1. 加载图像
        # ============================================================
        image_paths = item.get('image', [])
        if isinstance(image_paths, str):
            # 检查是否为 "empty" 标记
            if image_paths == 'empty':
                # 返回全零图像张量
                image = torch.zeros(3, self.image_size, self.image_size)
            else:
                image_paths = [image_paths]
                image = self._load_image(image_paths)
        elif len(image_paths) == 0:
            raise ValueError(f"样本 {idx} 没有图像路径")
        else:
            # 加载第一个图像（或多模态图像）
            image = self._load_image(image_paths)

        # ============================================================
        # 2. 加载标签（掩码）
        # ============================================================
        label_path_str = item.get('label')
        if label_path_str:
            label_path = self._resolve_path(label_path_str)
            mask = self._load_mask(label_path)
        else:
            # 如果没有标签，创建全零掩码
            mask = torch.zeros(1, self.image_size, self.image_size)

        # ============================================================
        # 3. 加载文本特征
        # ============================================================
        text_feature_path_str = item.get('text_feature')
        if text_feature_path_str:
            # 检查是否为 "empty" 标记
            if text_feature_path_str == 'empty':
                # 返回全零文本特征向量
                text_features = torch.zeros(self.embed_dim)
            else:
                text_feature_path = self._resolve_path(text_feature_path_str)
                text_features = self._load_text_features(text_feature_path)
        else:
            # 如果没有文本特征，返回空张量
            print(f"⚠️  样本 {idx} 没有文本特征，返回零向量")
            text_features = torch.zeros(self.embed_dim)

        return image, mask, text_features

    def _load_image(self, image_paths: List[str]) -> torch.Tensor:
        """
        加载图像（支持多模态医学图像）

        Args:
            image_paths: 图像文件路径列表（如 BraTS 的 4 个模态）

        Returns:
            image: (3, H, W) 图像张量
        """
        # 加载第一个图像文件
        img_path = self._resolve_path(image_paths[0])

        if not img_path.exists():
            raise FileNotFoundError(
                f"❌ 图像文件不存在: {img_path}\n"
                f"数据根目录: {self.data_root}\n"
                f"原始路径: {image_paths[0]}"
            )

        # 检查文件格式
        if str(img_path).endswith('.nii.gz') or str(img_path).endswith('.nii'):
            # 加载 NIfTI 格式（医学图像）
            if not HAS_NIBABEL:
                raise ImportError("需要安装 nibabel 来加载 .nii.gz 文件: pip install nibabel")

            nii_img = nib.load(img_path)
            image_data = nii_img.get_fdata()

            # 处理 3D 数据（选择中间切片）
            if image_data.ndim == 3:
                mid_slice = image_data.shape[2] // 2
                image_data = image_data[:, :, mid_slice]  # (H, W)
            elif image_data.ndim == 4:
                mid_slice = image_data.shape[2] // 2
                image_data = image_data[:, :, mid_slice, 0]  # (H, W)

            # 归一化到 [0, 1]
            image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)

            # 转换为 tensor (1, H, W)
            image = torch.from_numpy(image_data).float().unsqueeze(0)

            # 转换为 3 通道
            image = image.repeat(3, 1, 1)  # (3, H, W)

            # 调整尺寸
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        else:
            # 加载标准图像格式（jpg, png 等）
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)

        return image

    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        """
        加载掩码

        Args:
            mask_path: 掩码文件路径

        Returns:
            mask: (1, H, W) 掩码张量
        """
        if not mask_path.exists():
            raise FileNotFoundError(
                f"❌ 掩码文件不存在: {mask_path}\n"
                f"数据根目录: {self.data_root}"
            )

        # 检查文件格式
        if str(mask_path).endswith('.nii.gz') or str(mask_path).endswith('.nii'):
            # 加载 NIfTI 格式掩码
            if not HAS_NIBABEL:
                raise ImportError("需要安装 nibabel 来加载 .nii.gz 文件: pip install nibabel")

            nii_mask = nib.load(mask_path)
            mask_data = nii_mask.get_fdata()

            # 处理 3D 掩码（选择中间切片）
            if mask_data.ndim == 3:
                mid_slice = mask_data.shape[2] // 2
                mask_data = mask_data[:, :, mid_slice]

            # 转换为 tensor
            mask = torch.from_numpy(mask_data).float()

            # 调整尺寸
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='nearest'
            ).squeeze(0)

            # BraTS 掩码：0=背景, 1=坏死, 2=水肿, 4=增强肿瘤
            # 转换为二值掩码（所有非零值 -> 1）
            mask = (mask > 0).float()

        else:
            # 加载标准图像格式掩码
            mask = Image.open(mask_path).convert('L')  # 灰度图

            # 调整尺寸
            mask_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])
            mask = mask_transform(mask)

            # 二值化
            mask = (mask > 0.5).float()

        return mask

    def _load_text_features(self, text_feature_path: Path) -> torch.Tensor:
        """
        加载文本特征（从 .npy 文件）

        Args:
            text_feature_path: 文本特征文件路径（.npy）

        Returns:
            text_features: (D,) 文本特征向量
        """
        if not text_feature_path.exists():
            raise FileNotFoundError(
                f"❌ 文本特征文件不存在: {text_feature_path}\n"
                f"数据根目录: {self.data_root}\n"
                f"请检查 JSON 文件中的 'text_feature' 路径是否正确。"
            )

        try:
            # 加载 .npy 文件
            text_features = np.load(text_feature_path)

            # 转换为 torch tensor
            text_features = torch.from_numpy(text_features).float()

            # 处理维度
            if text_features.dim() == 0:
                # 标量 -> (1,)
                text_features = text_features.unsqueeze(0)
            elif text_features.dim() > 1:
                # 多维 -> 展平为 1D
                text_features = text_features.flatten()

            # L2 归一化（可选）
            if self.normalize_text:
                text_features = F.normalize(text_features, p=2, dim=0)

            return text_features

        except Exception as e:
            raise RuntimeError(
                f"❌ 加载文本特征失败: {text_feature_path}\n"
                f"错误: {e}\n"
                f"请检查 .npy 文件是否完整且格式正确。"
            )


def create_multimodal_dataloaders(
    json_path: str,
    data_root: Optional[str] = None,
    batch_size: int = 4,
    image_size: int = 1024,
    num_workers: int = 0,
    shuffle_train: bool = True,
    folds: Optional[Dict[str, List[int]]] = None
) -> Dict[str, DataLoader]:
    """
    创建多模态数据加载器（支持交叉验证）

    Args:
        json_path: JSON 元数据文件路径
        data_root: 数据根目录
        batch_size: 批次大小
        image_size: 图像尺寸
        num_workers: DataLoader 工作进程数
        shuffle_train: 是否打乱训练集
        folds: 折划分配置（用于交叉验证）
               例如: {'train': [0, 1, 2], 'val': [3], 'test': [4]}
               如果为 None，则加载所有数据到训练集

    Returns:
        {'train': train_loader, 'val': val_loader, 'test': test_loader}
    """
    loaders = {}

    if folds is None:
        # 默认：加载所有数据到训练集
        print("[MultimodalDataLoader] 加载所有数据到训练集（未指定 fold）")

        train_dataset = MultimodalMedicalDataset(
            json_path=json_path,
            data_root=data_root,
            fold=None,  # 加载所有 fold
            image_size=image_size
        )

        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        print(f"[MultimodalDataLoader] 训练集: {len(train_dataset)} 个样本")

    else:
        # 按 fold 划分数据
        for split_name, fold_list in folds.items():
            print(f"[MultimodalDataLoader] 创建 {split_name} 集（folds: {fold_list}）")

            # 为每个 fold 创建数据集
            datasets = []
            for fold in fold_list:
                dataset = MultimodalMedicalDataset(
                    json_path=json_path,
                    data_root=data_root,
                    fold=fold,
                    image_size=image_size
                )
                datasets.append(dataset)

            # 合并多个 fold 的数据集
            if len(datasets) == 1:
                merged_dataset = datasets[0]
            else:
                from torch.utils.data import ConcatDataset
                merged_dataset = ConcatDataset(datasets)

            # 创建 DataLoader
            loaders[split_name] = DataLoader(
                merged_dataset,
                batch_size=batch_size,
                shuffle=(shuffle_train if split_name == 'train' else False),
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False
            )

            print(f"[MultimodalDataLoader] {split_name} 集: {len(merged_dataset)} 个样本")

    return loaders


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("多模态医学图像数据集加载器 - 测试")
    print("=" * 80)

    # 配置参数
    json_path = "data/source_images/TextBraTS.json"
    data_root = "data/source_images"

    # 方案 1：加载所有数据
    print("\n【方案 1】加载所有数据到训练集")
    print("-" * 80)
    try:
        loaders = create_multimodal_dataloaders(
            json_path=json_path,
            data_root=data_root,
            batch_size=2,
            image_size=1024,
            shuffle_train=True
        )

        train_loader = loaders['train']
        print(f"✓ 成功创建训练数据加载器，共 {len(train_loader)} 个批次")

        # 测试加载一个批次
        print("\n测试加载第一个批次...")
        images, masks, text_features = next(iter(train_loader))
        print(f"  - 图像形状: {images.shape}")
        print(f"  - 掩码形状: {masks.shape}")
        print(f"  - 文本特征形状: {text_features.shape}")

    except Exception as e:
        print(f"✗ 失败: {e}")

    # 方案 2：按 fold 划分（5 折交叉验证）
    print("\n\n【方案 2】按 fold 划分（5 折交叉验证）")
    print("-" * 80)
    try:
        folds = {
            'train': [0, 1, 2, 3],  # fold 0-3 用于训练
            'val': [4]               # fold 4 用于验证
        }

        loaders = create_multimodal_dataloaders(
            json_path=json_path,
            data_root=data_root,
            batch_size=2,
            image_size=1024,
            shuffle_train=True,
            folds=folds
        )

        train_loader = loaders['train']
        val_loader = loaders['val']

        print(f"✓ 训练集: {len(train_loader.dataset)} 个样本")
        print(f"✓ 验证集: {len(val_loader.dataset)} 个样本")

        # 测试加载
        print("\n测试加载训练集批次...")
        images, masks, text_features = next(iter(train_loader))
        print(f"  - 图像形状: {images.shape}")
        print(f"  - 掩码形状: {masks.shape}")
        print(f"  - 文本特征形状: {text_features.shape}")

    except Exception as e:
        print(f"✗ 失败: {e}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
