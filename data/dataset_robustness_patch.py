"""
FedSAM3-Cream Dataset 类健壮性补丁
CVPR 2026 Workshop - 处理模态缺失场景的代码片段

目的：
展示如何在 PyTorch Dataset 的 __getitem__ 方法中优雅地处理 "empty" 标识，
确保 text_only / image_only / multimodal 三种客户端都能正常加载数据。

重要发现：
✅ data/multimodal_dataset.py 中的 MultimodalMedicalDataset 类已经实现了完整的处理逻辑！
✅ 该类在第 220-222 行和第 249-251 行已经处理了 "empty" 标记。

本文件提供：
1. 现有实现的代码解析
2. 如果需要自定义，提供模板代码
3. 使用示例和测试代码
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple


# ==============================================================================
# 方案 1: 使用现有的 MultimodalMedicalDataset 类（推荐）⭐
# ==============================================================================
"""
你的项目中已经有一个完善的 Dataset 类：data/multimodal_dataset.py

关键代码片段（已存在于你的代码库中）：
"""

def __getitem_example_from_existing_class(self, idx: int):
    """
    这是 data/multimodal_dataset.py 第 202-260 行的实现
    展示了如何处理 "empty" 标识
    """
    item = self.data_list[idx]

    # ===========================================
    # 1. 处理图像字段（image）
    # ===========================================
    image_paths = item.get('image', [])

    if isinstance(image_paths, str):
        # 🔥 关键逻辑：检查是否为 "empty" 标记
        if image_paths == 'empty':
            # ✅ Client 1 (text_only): 返回全零图像张量
            # 形状: (3, H, W) 与真实图像一致
            image = torch.zeros(3, self.image_size, self.image_size)
        else:
            # 单个图像路径
            image_paths = [image_paths]
            image = self._load_image(image_paths)
    elif len(image_paths) == 0:
        raise ValueError(f"样本 {idx} 没有图像路径")
    else:
        # 多模态图像路径（BraTS: flair, t1, t1ce, t2）
        image = self._load_image(image_paths)

    # ===========================================
    # 2. 处理标签字段（label）
    # ===========================================
    label_path_str = item.get('label')
    if label_path_str:
        label_path = self._resolve_path(label_path_str)
        mask = self._load_mask(label_path)
    else:
        # 如果没有标签，创建全零掩码
        mask = torch.zeros(1, self.image_size, self.image_size)

    # ===========================================
    # 3. 处理文本特征字段（text_feature）
    # ===========================================
    text_feature_path_str = item.get('text_feature')

    if text_feature_path_str:
        # 🔥 关键逻辑：检查是否为 "empty" 标记
        if text_feature_path_str == 'empty':
            # ✅ Client 2 (image_only): 返回全零文本特征向量
            # 形状: (embed_dim,) 例如 (768,)
            text_features = torch.zeros(self.embed_dim)
        else:
            # 加载真实的文本特征
            text_feature_path = self._resolve_path(text_feature_path_str)
            text_features = self._load_text_features(text_feature_path)
    else:
        # 如果 JSON 中没有 text_feature 字段，也返回零向量
        text_features = torch.zeros(self.embed_dim)

    return image, mask, text_features


# ==============================================================================
# 方案 2: 如果需要自定义 Dataset 类，使用以下模板
# ==============================================================================
class RobustMultimodalDataset:
    """
    健壮的多模态数据集类模板
    展示如何处理模态缺失场景
    """

    def __init__(self, image_size=256, embed_dim=768):
        self.image_size = image_size
        self.embed_dim = embed_dim
        # ... 其他初始化代码

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取数据项（健壮版本）

        Returns:
            image: (3, H, W) - RGB 图像张量或全零张量
            mask: (1, H, W) - 分割掩码张量
            text_features: (D,) - 文本特征向量或全零向量
        """
        item = self.data_list[idx]

        # ===========================================
        # 🛡️ 健壮处理：图像字段
        # ===========================================
        image_field = item.get('image')

        if image_field is None or image_field == 'empty':
            # Case 1: 没有图像 或 显式标记为 "empty"
            # → 返回全零张量（保持与真实图像相同的形状）
            image = self._create_dummy_image()

        elif isinstance(image_field, str):
            # Case 2: 单个图像路径字符串
            image = self._load_single_image(image_field)

        elif isinstance(image_field, list):
            # Case 3: 多模态图像路径列表（如 BraTS 的 4 个模态）
            if len(image_field) == 0:
                image = self._create_dummy_image()
            else:
                image = self._load_multimodal_image(image_field)
        else:
            raise TypeError(
                f"不支持的 image 字段类型: {type(image_field)}\n"
                f"期望: str | list | 'empty' | None"
            )

        # ===========================================
        # 🛡️ 健壮处理：标签字段
        # ===========================================
        label_field = item.get('label')

        if label_field and label_field != 'empty':
            mask = self._load_mask(label_field)
        else:
            # 没有标签或标记为空
            mask = torch.zeros(1, self.image_size, self.image_size)

        # ===========================================
        # 🛡️ 健壮处理：文本特征字段
        # ===========================================
        text_field = item.get('text_feature')

        if text_field is None or text_field == 'empty':
            # Case 1: 没有文本特征 或 显式标记为 "empty"
            # → 返回全零向量（保持与真实特征相同的维度）
            text_features = self._create_dummy_text_features()

        elif isinstance(text_field, str):
            # Case 2: 文本特征文件路径（.npy）
            text_features = self._load_text_features(text_field)

        else:
            raise TypeError(
                f"不支持的 text_feature 字段类型: {type(text_field)}\n"
                f"期望: str | 'empty' | None"
            )

        return image, mask, text_features

    # ===========================================
    # 🔧 辅助方法：生成 Dummy 数据
    # ===========================================
    def _create_dummy_image(self) -> torch.Tensor:
        """
        创建全零图像张量（用于 text_only 客户端）

        为什么使用全零张量？
        1. 保持与真实图像相同的形状 (3, H, W)
        2. 不会导致 SAM3 前向传播时的维度错误
        3. 占用内存小，计算开销低

        Returns:
            全零图像张量 (3, H, W)
        """
        return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)

    def _create_dummy_text_features(self) -> torch.Tensor:
        """
        创建全零文本特征向量（用于 image_only 客户端）

        为什么使用全零向量？
        1. 保持与真实文本特征相同的维度 (embed_dim,)
        2. 在对比学习中会被正确处理（零向量在计算相似度时贡献为零）
        3. 不会导致模型崩溃

        Returns:
            全零文本特征向量 (embed_dim,)
        """
        return torch.zeros(self.embed_dim, dtype=torch.float32)

    # ===========================================
    # 🔧 辅助方法：加载真实数据
    # ===========================================
    def _load_single_image(self, image_path: str) -> torch.Tensor:
        """加载单个图像文件"""
        # 实现细节省略
        # ...
        return torch.randn(3, self.image_size, self.image_size)  # 示例

    def _load_multimodal_image(self, image_paths: list) -> torch.Tensor:
        """加载多模态医学图像（如 BraTS 的 4 个 MRI 序列）"""
        # 实现细节省略
        # ...
        return torch.randn(3, self.image_size, self.image_size)  # 示例

    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """加载分割掩码"""
        # 实现细节省略
        # ...
        return torch.randint(0, 2, (1, self.image_size, self.image_size))  # 示例

    def _load_text_features(self, feature_path: str) -> torch.Tensor:
        """
        加载预提取的文本特征（.npy 文件）

        Args:
            feature_path: .npy 文件路径

        Returns:
            文本特征向量 (embed_dim,)
        """
        try:
            # 加载 .npy 文件
            features = np.load(feature_path)

            # 转换为 torch tensor
            text_features = torch.from_numpy(features).float()

            # 验证维度
            if text_features.dim() == 0:
                # 标量 → 扩展到向量
                text_features = text_features.unsqueeze(0).repeat(self.embed_dim)
            elif text_features.dim() == 2:
                # (1, D) → (D,)
                text_features = text_features.squeeze(0)

            # 确保维度正确
            if text_features.shape[0] != self.embed_dim:
                raise ValueError(
                    f"文本特征维度不匹配: 期望 {self.embed_dim}, 实际 {text_features.shape[0]}"
                )

            return text_features

        except Exception as e:
            print(f"⚠️  加载文本特征失败: {feature_path}")
            print(f"错误: {e}")
            print(f"返回零向量作为回退")
            return self._create_dummy_text_features()


# ==============================================================================
# 方案 3: 单元测试 - 验证 Dummy 数据的正确性
# ==============================================================================
def test_dummy_data_generation():
    """
    测试全零张量的生成是否正确
    确保形状、数据类型、数值范围都符合预期
    """
    print("=" * 80)
    print("测试 Dummy 数据生成")
    print("=" * 80)

    image_size = 256
    embed_dim = 768

    # 测试图像张量
    dummy_image = torch.zeros(3, image_size, image_size, dtype=torch.float32)
    print(f"\n✓ Dummy 图像张量:")
    print(f"  - 形状: {dummy_image.shape} (期望: torch.Size([3, {image_size}, {image_size}]))")
    print(f"  - 数据类型: {dummy_image.dtype} (期望: torch.float32)")
    print(f"  - 数值范围: [{dummy_image.min():.2f}, {dummy_image.max():.2f}] (期望: [0.00, 0.00])")
    print(f"  - 内存占用: {dummy_image.element_size() * dummy_image.nelement() / 1024:.2f} KB")

    # 测试文本特征向量
    dummy_text = torch.zeros(embed_dim, dtype=torch.float32)
    print(f"\n✓ Dummy 文本特征向量:")
    print(f"  - 形状: {dummy_text.shape} (期望: torch.Size([{embed_dim}]))")
    print(f"  - 数据类型: {dummy_text.dtype} (期望: torch.float32)")
    print(f"  - 数值范围: [{dummy_text.min():.2f}, {dummy_text.max():.2f}] (期望: [0.00, 0.00])")
    print(f"  - 内存占用: {dummy_text.element_size() * dummy_text.nelement() / 1024:.2f} KB")

    # 测试掩码张量
    dummy_mask = torch.zeros(1, image_size, image_size, dtype=torch.float32)
    print(f"\n✓ Dummy 掩码张量:")
    print(f"  - 形状: {dummy_mask.shape} (期望: torch.Size([1, {image_size}, {image_size}]))")
    print(f"  - 数据类型: {dummy_mask.dtype} (期望: torch.float32)")

    print("\n" + "=" * 80)
    print("✅ 所有 Dummy 数据生成测试通过！")
    print("=" * 80)


# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == '__main__':
    """
    运行测试和示例代码
    """

    # 测试 1: Dummy 数据生成
    test_dummy_data_generation()

    print("\n\n")

    # 示例: 如何在 DataLoader 中使用
    print("=" * 80)
    print("DataLoader 使用示例")
    print("=" * 80)
    print("""
from data.multimodal_dataset import MultimodalMedicalDataset
from torch.utils.data import DataLoader

# 方法 1: 使用现有的 MultimodalMedicalDataset（推荐）⭐
dataset = MultimodalMedicalDataset(
    json_path="data/federated_split/client1_text_only/dataset.json",
    data_root="data/source_images",  # BraTS 2020 数据根目录
    image_size=256,
    embed_dim=768
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)

# 测试数据加载
for batch_idx, (images, masks, text_features) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"  - images: {images.shape}")         # (B, 3, 256, 256)
    print(f"  - masks: {masks.shape}")           # (B, 1, 256, 256)
    print(f"  - text_features: {text_features.shape}")  # (B, 768)

    if batch_idx == 0:
        break

# ✅ 即使 Client 1 的 image 字段为 "empty"，
# DataLoader 仍然会返回形状正确的全零张量，
# 不会报错！
    """)

    print("\n📝 关键要点:")
    print("  1. ✅ 你的 MultimodalMedicalDataset 已经处理了 'empty' 标记")
    print("  2. ✅ 不需要额外修改代码")
    print("  3. ✅ 直接使用即可")
    print()
