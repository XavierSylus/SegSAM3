"""
数据集加载器 - 用于从文件系统加载真实的医学图像分割数据
支持训练集、验证集和测试集
支持多种数据格式和目录结构
"""
import os
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path



try:
    from src.model import SAM3_Medical, DEVICE, BATCH_SIZE, LR
    from src.cream_losses import CreamContrastiveLoss
except ImportError:
    # 如果找不到源文件，使用 Mock 类代替，确保 MCP 服务能启动
    print("Warning: src.model not found, using Mock classes.")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 6
    LR = 2e-4

    class SAM3_Medical(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 256
            self.dummy_layer = nn.Linear(1, 1) # 为了有参数
        def forward(self, x): 
            # 返回模拟的 logits (B, 1, H, W)
            return torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
        def extract_features(self, x): 
            # 返回模拟的特征 (B, N, D)
            return torch.randn(x.shape[0], 256, self.embed_dim).to(x.device)
        def get_trainable_params(self): return self.parameters()

    class CreamContrastiveLoss(nn.Module):
        def __init__(self, tau=0.07): super().__init__()
        def forward(self, a, b, c): return torch.tensor(0.5), torch.tensor(0.5)

# 尝试导入 nibabel（用于加载 .nii.gz 文件）
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("警告: 未安装 nibabel，无法加载 .nii.gz 格式的医学图像")
    print("安装方法: pip install nibabel")


@functools.lru_cache(maxsize=1)
def _load_nii_cached(path: str) -> np.ndarray:
    """缓存最近一次读取的 NIfTI volume。

    maxsize=1 确保同患者连续切片顺序遍历时磁盘 I/O 只发生一次。
    多进程 DataLoader 中每个 worker 持有独立缓存副本，不共享内存，无 OOM 风险。
    前提：DataLoader 验证时必须 shuffle=False，否则同患者切片被打散缓存命中率归零。
    """
    return nib.load(path).get_fdata()


class MedicalImageDataset(Dataset):
    """
    医学图像分割数据集加载器
    
    支持的数据目录结构:
    data/
    ├── train/ (或 val/, test/)
    │   ├── client_A/
    │   │   ├── private/
    │   │   │   ├── images/
    │   │   │   │   ├── img_001.jpg
    │   │   │   │   └── img_002.jpg
    │   │   │   └── masks/
    │   │   │       ├── mask_001.png
    │   │   │       └── mask_002.png
    │   │   └── public/
    │   │       └── images/
    │   │           ├── pub_001.jpg
    │   │           └── pub_002.jpg
    │   ├── client_B/
    │   │   └── ...
    │   └── client_C/
    │       └── ...
    """
    
    def __init__(
        self,
        data_dir: str,
        mode: str = "private",  # "private" or "public"
        image_size: int = 1024,
        transform: Optional[transforms.Compose] = None,
        has_mask: bool = True,
        max_samples: Optional[int] = None,
        is_val: bool = False
    ):
        """
        Args:
            data_dir: 数据目录路径（如 "data/train/client_A" 或 "data/val/client_A"）
            mode: 数据模式 ("private" 或 "public")
            image_size: 目标图像尺寸
            transform: 图像变换
            has_mask: 是否有掩码（public 模式通常为 False）
            max_samples: 最大样本数（用于测试/调试）
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.image_size = image_size
        self.has_mask = has_mask
        self.is_val = is_val
        # is_val=True 时由 _build_val_samples() 填充，训练时保持 None
        self.samples: Optional[List[tuple]] = None
        
        # 设置图像和掩码目录（支持单复数形式：image/images）
        if mode == "private":
            # 优先级：private/images > private/image > private
            if (self.data_dir / "private" / "images").exists():
                self.image_dir = self.data_dir / "private" / "images"
                self.mask_dir = self.data_dir / "private" / "masks" if has_mask else None
            elif (self.data_dir / "private" / "image").exists():
                self.image_dir = self.data_dir / "private" / "image"
                self.mask_dir = self.data_dir / "private" / "image" if has_mask else None
            elif (self.data_dir / "private").exists():
                self.image_dir = self.data_dir / "private"
                self.mask_dir = self.data_dir / "private" if has_mask else None
            else:
                self.image_dir = self.data_dir / "private" / "images"
                self.mask_dir = self.data_dir / "private" / "masks" if has_mask else None
        else:  # public
            # 优先级：public/images > public/image > public
            if (self.data_dir / "public" / "images").exists():
                self.image_dir = self.data_dir / "public" / "images"
                self.mask_dir = None
            elif (self.data_dir / "public" / "image").exists():
                self.image_dir = self.data_dir / "public" / "image"
                self.mask_dir = None
            elif (self.data_dir / "public").exists():
                self.image_dir = self.data_dir / "public"
                self.mask_dir = None
            else:
                self.image_dir = self.data_dir / "public" / "images"
                self.mask_dir = None
        
        # 检查目录是否存在
        if not self.image_dir.exists():
            raise ValueError(f"图像目录不存在: {self.image_dir}")
        
        # 获取所有图像文件（支持多种格式）
        image_files = []
        # 标准图像格式
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(self.image_dir.glob(ext)))
        # 医学图像格式（.nii.gz）
        if HAS_NIBABEL:
            image_files.extend(list(self.image_dir.glob("*.nii.gz")))
            image_files.extend(list(self.image_dir.glob("*.nii")))
        # 如果数据是按病例组织的（每个病例一个文件夹）
        # 检查是否有子目录包含图像文件
        if len(image_files) == 0:
            # 尝试在子目录中查找
            for subdir in self.image_dir.iterdir():
                if subdir.is_dir():
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                        image_files.extend(list(subdir.glob(ext)))
                    if HAS_NIBABEL:
                        image_files.extend(list(subdir.glob("*.nii.gz")))
                        image_files.extend(list(subdir.glob("*.nii")))
        
        # 过滤掉掩码文件 (seg.nii, seg.nii.gz)
        image_files = [f for f in image_files if '_seg.nii' not in f.name and '_seg.nii.gz' not in f.name]

        # 对于 BraTS 数据，优先选择 flair 模态（因为有预生成的文本特征）
        # 按病例分组，每个病例只保留一个模态
        if any('flair' in str(f) for f in image_files):
            # 检测到 BraTS 数据格式
            case_to_files = {}
            for f in image_files:
                # 提取病例名称（例如：BraTS20_Training_001）
                case_name = f.stem.rsplit('_', 1)[0] if '_' in f.stem else f.stem
                if case_name not in case_to_files:
                    case_to_files[case_name] = []
                case_to_files[case_name].append(f)

            # 为每个病例选择最佳模态（优先级：flair > t1ce > t1 > t2）
            filtered_files = []
            for case_name, files in case_to_files.items():
                # 优先选择 flair
                flair_files = [f for f in files if 'flair' in f.name.lower()]
                if flair_files:
                    filtered_files.append(flair_files[0])
                    continue
                # 其次选择 t1ce
                t1ce_files = [f for f in files if 't1ce' in f.name.lower()]
                if t1ce_files:
                    filtered_files.append(t1ce_files[0])
                    continue
                # 再次选择 t1
                t1_files = [f for f in files if 't1' in f.name.lower() and 't1ce' not in f.name.lower()]
                if t1_files:
                    filtered_files.append(t1_files[0])
                    continue
                # 最后选择其他文件
                filtered_files.append(files[0])

            image_files = filtered_files

        self.image_files = sorted(image_files)
        
        # 如果指定了最大样本数，进行截断
        if max_samples is not None and max_samples > 0:
            self.image_files = self.image_files[:max_samples]
            print(f"  [INFO] 已限制加载 {len(self.image_files)} 个样本 (max_samples={max_samples})")
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {self.image_dir} 中未找到图像文件")
        
        # 如果有掩码，检查掩码文件
        if self.has_mask and self.mask_dir is not None:
            if not self.mask_dir.exists():
                raise ValueError(f"掩码目录不存在: {self.mask_dir}")
            
            # 验证每个图像都有对应的掩码
            self.mask_files = []
            for img_file in self.image_files:
                mask_file = None
                
                # 尝试多种命名方式
                possible_mask_names = [
                    img_file.name,  # 相同文件名
                    f"mask_{img_file.stem}.png",
                    f"{img_file.stem}_mask.png",
                    f"{img_file.stem}.png",
                    # 对于 .nii.gz 文件，尝试对应的掩码文件
                    img_file.name.replace('.nii.gz', '_seg.nii.gz'),
                    img_file.name.replace('.nii.gz', '.nii.gz').replace('_flair', '_seg'),
                    img_file.name.replace('.nii.gz', '.nii.gz').replace('_t1', '_seg'),
                    img_file.name.replace('.nii.gz', '.nii.gz').replace('_t1ce', '_seg'),
                    img_file.name.replace('.nii.gz', '.nii.gz').replace('_t2', '_seg'),
                ]
                
                # 如果图像文件在子目录中，掩码可能在同一个子目录
                if img_file.parent != self.image_dir:
                    # 图像在子目录中，检查子目录和主目录
                    search_dirs = [img_file.parent, self.mask_dir]
                else:
                    search_dirs = [self.mask_dir]
                
                for search_dir in search_dirs:
                    for mask_name in possible_mask_names:
                        candidate = search_dir / mask_name
                        if candidate.exists():
                            mask_file = candidate
                            break
                    if mask_file:
                        break
                
                if not mask_file:
                    # 如果图像在子目录中，检查掩码目录是否有对应的子目录
                    if img_file.parent != self.image_dir:
                        subdir_name = img_file.parent.name
                        mask_subdir = self.mask_dir / subdir_name
                        if mask_subdir.exists():
                            for mask_name in possible_mask_names:
                                candidate = mask_subdir / mask_name
                                if candidate.exists():
                                    mask_file = candidate
                                    break
                
                if not mask_file:
                    raise ValueError(f"未找到对应的掩码文件: {img_file.name} (搜索目录: {self.mask_dir})")
                
                self.mask_files.append(mask_file)
        
        # 设置默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # 验证模式：扫描所有 NIfTI header，构建全切片平铺列表
        if self.is_val and HAS_NIBABEL:
            self._build_val_samples()
    
    def _build_val_samples(self) -> None:
        """扫描所有 NIfTI header，构建 (file_idx, z) 全切片平铺列表。

        只读 header（不读体素），内存开销可忽略。
        所有切片包含纯背景切片，不施加任何前景过滤。
        """
        self.samples = []
        for file_idx, img_path in enumerate(self.image_files):
            path_str = str(img_path)
            if path_str.endswith('.nii.gz') or path_str.endswith('.nii'):
                header = nib.load(path_str).header
                D = header.get_data_shape()[2]
                for z in range(D):
                    self.samples.append((file_idx, z))
            else:
                # 非 NIfTI 文件视为单切片
                self.samples.append((file_idx, 0))
        print(f"  [ValDataset] 全切片平铺完成：{len(self.image_files)} volumes -> {len(self.samples)} slices")

    def __len__(self) -> int:
        if self.is_val and self.samples is not None:
            return len(self.samples)
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        获取数据项

        Returns:
            如果是 private 模式且有文本特征: (image, mask, text_feature)
            如果是 private 模式但无文本特征: (image, mask)
            如果是 public 模式且有文本特征: (image, text_feature)
            如果是 public 模式但无文本特征: (image,)
        """
        # 根据模式确定文件路径：验证时 idx 是平铺切片索引，需先解析 file_idx
        if self.is_val and self.samples is not None:
            file_idx, _ = self.samples[idx]
            img_path = self.image_files[file_idx]
        else:
            img_path = self.image_files[idx]

        # 尝试加载预生成的文本特征（如果存在）
        text_feature = None
        # 检查是否存在对应的 _text.npy 文件
        # 例如: BraTS20_Training_001_flair.nii -> BraTS20_Training_001_flair_text.npy
        text_feature_path = None
        if str(img_path).endswith('.nii.gz'):
            text_feature_path = Path(str(img_path).replace('.nii.gz', '_text.npy'))
        elif str(img_path).endswith('.nii'):
            text_feature_path = Path(str(img_path).replace('.nii', '_text.npy'))
        else:
            # 对于其他格式（jpg, png等），也尝试查找
            text_feature_path = Path(str(img_path.with_suffix('')) + '_text.npy')

        if text_feature_path and text_feature_path.exists():
            try:
                text_feature = np.load(str(text_feature_path))
                text_feature = torch.from_numpy(text_feature).float()
            except Exception as e:
                print(f"警告: 无法加载文本特征 {text_feature_path}: {e}")
                text_feature = None
        
        _SLICE_VALID_THRESHOLD: int = 10

        # 验证模式：按 (file_idx, z) 索引取固定切片，不过滤背景
        if self.is_val and self.samples is not None:
            file_idx, selected_z = self.samples[idx]
            img_path = self.image_files[file_idx]
            mask_path_val = self.mask_files[file_idx] if (self.has_mask and hasattr(self, 'mask_files')) else None

            if str(img_path).endswith('.nii.gz') or str(img_path).endswith('.nii'):
                image_data = _load_nii_cached(str(img_path))
                if image_data.ndim == 4:
                    slice_2d = image_data[:, :, selected_z, :]
                else:
                    slice_2d = image_data[:, :, selected_z]

                slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)

                if slice_2d.ndim == 3:
                    image_tensor = torch.from_numpy(np.transpose(slice_2d, (2, 0, 1))).float()
                else:
                    image_tensor = torch.from_numpy(slice_2d[np.newaxis]).float()

                if image_tensor.shape[0] == 1:
                    image_tensor = image_tensor.repeat(3, 1, 1)
                elif image_tensor.shape[0] == 4:
                    image_tensor = image_tensor[:3]
                elif image_tensor.shape[0] != 3:
                    image_tensor = image_tensor[:3] if image_tensor.shape[0] > 3 else torch.cat(
                        [image_tensor] * (3 // image_tensor.shape[0] + 1), dim=0
                    )[:3]

                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0)

                if mask_path_val is not None and (str(mask_path_val).endswith('.nii.gz') or str(mask_path_val).endswith('.nii')):
                    mask_data = _load_nii_cached(str(mask_path_val))
                    mask_slice = mask_data[:, :, selected_z]
                    mask_tensor = torch.from_numpy(mask_slice).float()
                    mask_tensor = F.interpolate(
                        mask_tensor.unsqueeze(0).unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode='nearest'
                    ).squeeze(0).squeeze(0).unsqueeze(0)
                    mask_tensor = (mask_tensor > 0).float()
                    return image_tensor, mask_tensor
                return (image_tensor,)
            else:
                # 非 NIfTI fallback：走标准训练路径
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image)
                return (image_tensor,)

        # 检查文件格式
        if str(img_path).endswith('.nii.gz') or str(img_path).endswith('.nii'):
            # 加载 NIfTI 格式（医学图像）
            if not HAS_NIBABEL:
                raise ImportError("需要安装 nibabel 来加载 .nii.gz 文件: pip install nibabel")

            nii_img = nib.load(img_path)
            image_data = nii_img.get_fdata()  # numpy (H, W, D) or (H, W, D, C)

            # ── 确定 Z 轴深度 D ─────────────────────────────────────────
            # BraTS 单模态: (H, W, D); 多模态: (H, W, D, C)
            D = image_data.shape[2]

            # ── 前景感知切片采样 ─────────────────────────────────────────
            # 掩码同步加载以计算每切片前景像素数（避免图像/掩码切片不一致）
            selected_z = D // 2  # fallback: center slice
            if self.has_mask and self.mask_dir is not None:
                mask_path_for_fg = self.mask_files[idx]
                if str(mask_path_for_fg).endswith('.nii.gz') or str(mask_path_for_fg).endswith('.nii'):
                    _mask_3d = nib.load(mask_path_for_fg).get_fdata()  # (H, W, D)
                    if _mask_3d.ndim == 3:
                        # 向量化计算所有切片前景像素数: (D,)
                        per_slice_fg = torch.from_numpy(
                            (_mask_3d > 0).sum(axis=(0, 1)).astype(np.int64)
                        )
                        # 精确提取所有有效切片索引，O(D) 一次完成，无循环重试
                        valid_indices = torch.where(per_slice_fg > _SLICE_VALID_THRESHOLD)[0]
                        if len(valid_indices) > 0:
                            rand_pos = torch.randint(0, len(valid_indices), (1,)).item()
                            selected_z = valid_indices[rand_pos].item()
                        else:
                            # 整个 3D volume 无病灶：取前景最多的切片（即便为 0）
                            selected_z = torch.argmax(per_slice_fg).item()

            # ── 按 selected_z 提取 2D 切片 ──────────────────────────────
            if image_data.ndim == 4:  # (H, W, D, C) 多模态
                image_data = image_data[:, :, selected_z, :]  # (H, W, C)
            else:  # (H, W, D) 单模态
                image_data = image_data[:, :, selected_z]     # (H, W)

            # 归一化到 [0, 1]
            image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)

            # 转换为 tensor
            if image_data.ndim == 3:
                image_data = np.transpose(image_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            elif image_data.ndim == 2:
                image_data = image_data[np.newaxis, :, :]          # (H, W) -> (1, H, W)

            image = torch.from_numpy(image_data).float()

            # 统一到 3 通道
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            elif image.shape[0] == 4:
                image = image[:3, :, :]
            elif image.shape[0] != 3:
                if image.shape[0] > 3:
                    image = image[:3, :, :]
                else:
                    image = torch.cat([image] * (3 // image.shape[0] + 1), dim=0)[:3, :, :]

            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        else:
            # 加载标准图像格式（jpg, png等）
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            selected_z = None  # 非 3D 数据无需切片索引

        # 如果是 private 模式且有掩码，加载掩码
        if self.has_mask and self.mask_dir is not None:
            mask_path = self.mask_files[idx]

            # 检查掩码文件格式
            if str(mask_path).endswith('.nii.gz') or str(mask_path).endswith('.nii'):
                # 加载 NIfTI 格式掩码
                if not HAS_NIBABEL:
                    raise ImportError("需要安装 nibabel 来加载 .nii.gz 文件: pip install nibabel")

                nii_mask = nib.load(mask_path)
                mask_data = nii_mask.get_fdata()  # (H, W, D)

                # 使用与图像完全相同的 selected_z 提取切片
                if mask_data.ndim == 3 and selected_z is not None:
                    mask_data = mask_data[:, :, selected_z]  # (H, W)
                elif mask_data.ndim == 3:
                    mask_data = mask_data[:, :, mask_data.shape[2] // 2]

                mask = torch.from_numpy(mask_data).float()

                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode='nearest'
                ).squeeze(0).squeeze(0)

                mask = mask.unsqueeze(0)       # (1, H, W)
                mask = (mask > 0).float()      # BraTS: 所有非零值 -> 1
            else:
                # 加载标准图像格式掩码
                mask = Image.open(mask_path).convert('L')  # 灰度图
                
                # 调整掩码尺寸
                mask_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ])
                mask = mask_transform(mask)
                
                # 二值化掩码（如果需要）
                mask = (mask > 0.5).float()

            # 返回图像、掩码和文本特征（如果有）
            if text_feature is not None:
                return image, mask, text_feature
            else:
                return image, mask
        else:
            # 返回图像和文本特征（如果有）
            if text_feature is not None:
                return image, text_feature
            else:
                return (image,)


class TextOnlyDataset(Dataset):
    """
    ★ Fix (2026-03-14): 纯文本数据集（text_only 客户端专属）

    专门为 text_only 客户端设计，只加载 _text.npy 特征文件，
    完全跳过图像文件的搜索和加载，避免 text_only 目录下找不到
    jpg/png/nii.gz 导致的 ValueError → 致命错误崩溃。

    返回格式：(text_feature,)  —— 与 TextOnlyTrainer.unpack_private_batch 一致
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "private",
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: 客户端数据目录（如 "data/federated_split/val/client_1"）
            mode:     "private" 或 "public"
            max_samples: 最大样本数
        """
        self.data_dir = Path(data_dir)
        self.mode = mode

        # text_only 数据目录：private/ 或 public/
        self.case_dir = self.data_dir / mode
        if not self.case_dir.exists():
            raise ValueError(
                f"[TextOnlyDataset] 数据目录不存在: {self.case_dir}\n"
                f"请确认 {mode} 目录已创建，且包含 *_text.npy 文件（或子目录）。"
            )

        # 递归扫描所有 _text.npy 文件
        text_files = sorted(self.case_dir.glob("**/*_text.npy"))

        if len(text_files) == 0:
            raise ValueError(
                f"[TextOnlyDataset] 在 {self.case_dir} 中未找到任何 *_text.npy 文件。\n"
                f"text_only 客户端必须在 {mode}/ 目录下包含预计算的文本特征文件。"
            )

        if max_samples is not None and max_samples > 0:
            text_files = text_files[:max_samples]

        self.text_files = text_files
        print(f"  [TextOnlyDataset] {mode.capitalize()} - 加载了 {len(self.text_files)} 个文本特征")

    def __len__(self) -> int:
        return len(self.text_files)

    def __getitem__(self, idx: int):
        """返回 (text_feature,) 元组，与 TextOnlyTrainer.unpack_private_batch 兼容"""
        text_path = self.text_files[idx]
        try:
            text_feature = np.load(str(text_path))
            text_feature = torch.from_numpy(text_feature).float()
        except Exception as e:
            raise RuntimeError(
                f"[TextOnlyDataset] 加载文本特征失败: {text_path}\n原因: {e}"
            ) from e
        return (text_feature,)


def create_data_loaders(
    data_root: str,
    split: str = "train",  # "train", "val", or "test"
    client_configs: List[dict] = None,
    batch_size: int = 6,
    image_size: int = 1024,
    num_workers: int = 0,
    shuffle: bool = True,
    max_samples: Optional[int] = None
) -> List[Tuple[DataLoader, Optional[DataLoader]]]:
    """
    为多个客户端创建数据加载器
    
    Args:
        data_root: 数据根目录（如 "data"）
        split: 数据集划分（"train", "val", 或 "test"）
        client_configs: 客户端配置列表，每个配置包含:
            {
                'client_id': 'client_A',
                'has_private': True,  # 是否有私有数据
                'has_public': True   # 是否有公开数据（验证/测试集通常为False）
            }
        batch_size: 批次大小
        image_size: 图像尺寸
        num_workers: DataLoader 工作进程数
        shuffle: 是否打乱数据（验证/测试集通常为False）
        max_samples: 每个客户端最大样本数（None 表示不限制）
    
    Returns:
        List of (private_loader, public_loader) tuples
        public_loader 可能为 None（如果 has_public=False）
    """
    if client_configs is None:
        # 默认配置：3个客户端
        client_configs = [
            {'client_id': 'client_A', 'has_private': True, 'has_public': True},
            {'client_id': 'client_B', 'has_private': True, 'has_public': True},
            {'client_id': 'client_C', 'has_private': True, 'has_public': True},
        ]
    
    loaders = []
    data_root_path = Path(data_root)
    
    for config in client_configs:
        client_id = config['client_id']
        client_data_dir = data_root_path / split / client_id
        
        # 创建私有数据加载器
        private_loader = None
        if config.get('has_private', True):
            try:
                # ★ Fix (2026-03-14): text_only 客户端使用专属 TextOnlyDataset，
                # 跳过图像文件搜索，避免 text_only 目录下无图像时崩溃。
                _is_text_only = (
                    config.get('modality') == 'text_only'
                    or config.get('is_text_only', False)
                )

                if _is_text_only:
                    private_dataset = TextOnlyDataset(
                        data_dir=str(client_data_dir),
                        mode="private",
                        max_samples=max_samples
                    )
                    # text_only 批次：[(text_feat,), (text_feat,), ...]
                    # collate_fn 将其合并为 (stacked_text_feat,)
                    def _text_only_collate(batch):
                        feats = torch.stack([b[0] for b in batch], dim=0)
                        return (feats,)

                    private_loader = DataLoader(
                        private_dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        collate_fn=_text_only_collate,
                        pin_memory=True if torch.cuda.is_available() else False
                    )
                else:
                    # val/test 时激活全切片平铺模式（is_val=True），shuffle 强制 False
                    # 以确保 _load_nii_cached(maxsize=1) 顺序遍历时缓存命中率最大化
                    _is_val_split = (split != "train")
                    private_dataset = MedicalImageDataset(
                        data_dir=str(client_data_dir),
                        mode="private",
                        image_size=image_size,
                        has_mask=True,
                        max_samples=max_samples,
                        is_val=_is_val_split
                    )
                    private_loader = DataLoader(
                        private_dataset,
                        batch_size=batch_size,
                        shuffle=False if _is_val_split else shuffle,
                        num_workers=num_workers,
                        pin_memory=True if torch.cuda.is_available() else False
                    )

                print(f"  [OK] {client_id} ({split}) - Private data: {len(private_dataset)} samples")

            except Exception as e:
                # ★ Fail Fast 策略：拒绝构造假数据，强制用户检查数据集完整性
                _is_text_only = (
                    config.get('modality') == 'text_only'
                    or config.get('is_text_only', False)
                )
                _hint = (
                    f"  2. 确认目录中包含 *_text.npy 文件（text_only 模式）\n"
                    if _is_text_only else
                    f"  2. 确认目录中包含有效的图像文件 (jpg/png/nii.gz)\n"
                )
                error_msg = (
                    f"\n{'=' * 80}\n"
                    f"❌ 致命错误: 客户端 {client_id} ({split}) 私有数据加载失败！\n"
                    f"{'=' * 80}\n"
                    f"原因: {e}\n"
                    f"数据目录: {client_data_dir}\n"
                    f"\n"
                    f"💡 解决方案:\n"
                    f"  1. 检查数据集目录是否存在: {client_data_dir / 'private'}\n"
                    f"{_hint}"
                    f"  3. 如果该客户端不应参与训练，请在配置中移除或禁用它\n"
                    f"  4. 如果数据集未准备好，请运行数据准备脚本\n"
                    f"{'=' * 80}\n"
                )
                print(error_msg)
                raise FileNotFoundError(error_msg) from e
        
        # 创建公开数据加载器（验证/测试集通常不需要）
        public_loader = None
        if config.get('has_public', True) and split == "train":
            try:
                public_dataset = MedicalImageDataset(
                    data_dir=str(client_data_dir),
                    mode="public",
                    image_size=image_size,
                    has_mask=False,
                    max_samples=max_samples
                )
                public_loader = DataLoader(
                    public_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                print(f"  [OK] {client_id} ({split}) - Public data: {len(public_dataset)} samples")
            except Exception as e:
                # ★ 兼容策略：公共数据加载失败时，降级为 image_only 模式
                print(f"  [警告] {client_id} ({split}) - 公共数据加载失败，降级为 image_only 模式")
                print(f"    原因: {e}")
                print(f"    数据目录: {client_data_dir / 'public'}")
                print(f"    💡 提示: 如不需要 public 数据，请在配置中设置 has_public=False")
                public_loader = None  # 降级为 image_only
        
        loaders.append((private_loader, public_loader))
    
    return loaders


# 示例：如何使用
if __name__ == "__main__":
    # 示例配置
    data_root = "data/federated_split"  # 数据根目录
    
    client_configs = [
        {
            'client_id': 'client_A',
            'has_private': True,
            'has_public': True  # 训练集需要公共数据
        },
        {
            'client_id': 'client_B',
            'has_private': False,  # 文本客户端，没有图像私有数据
            'has_public': True
        },
        {
            'client_id': 'client_C',
            'has_private': True,
            'has_public': True
        }
    ]
    
    # 创建训练集数据加载器
    print("=" * 60)
    print("创建训练集数据加载器")
    print("=" * 60)
    try:
        train_loaders = create_data_loaders(
            data_root=data_root,
            split="train",
            client_configs=client_configs,
            batch_size=6,
            image_size=1024,
            shuffle=True
        )
        print(f"\n成功创建 {len(train_loaders)} 个客户端的训练数据加载器")
    except Exception as e:
        print(f"错误: {e}")
        print("\n提示: 请确保数据目录结构正确，或使用虚拟数据进行测试")
    
    # 创建验证集数据加载器
    print("\n" + "=" * 60)
    print("创建验证集数据加载器")
    print("=" * 60)
    val_client_configs = [
        {'client_id': 'client_A', 'has_private': True, 'has_public': False},
        {'client_id': 'client_B', 'has_private': True, 'has_public': False},
        {'client_id': 'client_C', 'has_private': True, 'has_public': False},
    ]
    try:
        val_loaders = create_data_loaders(
            data_root=data_root,
            split="val",
            client_configs=val_client_configs,
            batch_size=6,
            image_size=1024,
            shuffle=False  # 验证集不打乱
        )
        print(f"\n成功创建 {len(val_loaders)} 个客户端的验证数据加载器")
    except Exception as e:
        print(f"错误: {e}")
        print("\n提示: 验证集目录可能不存在，这是正常的")
    
    # 创建测试集数据加载器
    print("\n" + "=" * 60)
    print("创建测试集数据加载器")
    print("=" * 60)
    test_client_configs = [
        {'client_id': 'client_A', 'has_private': True, 'has_public': False},
        {'client_id': 'client_B', 'has_private': True, 'has_public': False},
        {'client_id': 'client_C', 'has_private': True, 'has_public': False},
    ]
    try:
        test_loaders = create_data_loaders(
            data_root=data_root,
            split="test",
            client_configs=test_client_configs,
            batch_size=6,
            image_size=1024,
            shuffle=False  # 测试集不打乱
        )
        print(f"\n成功创建 {len(test_loaders)} 个客户端的测试数据加载器")
    except Exception as e:
        print(f"错误: {e}")
        print("\n提示: 测试集目录可能不存在，这是正常的")


def _create_mock_dataloaders(batch_size=4):
    """辅助函数：创建模拟数据，为了让工具开箱即用"""
    # 模拟数据：20个样本，3通道，256x256
    dummy_private_imgs = torch.randn(20, 3, 256, 256)
    dummy_private_masks = torch.randn(20, 1, 256, 256)
    dummy_public_imgs = torch.randn(20, 3, 256, 256)
    
    priv_ds = TensorDataset(dummy_private_imgs, dummy_private_masks)
    pub_ds = TensorDataset(dummy_public_imgs)
    
    return (DataLoader(priv_ds, batch_size=batch_size), 
            DataLoader(pub_ds, batch_size=batch_size))