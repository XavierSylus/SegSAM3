"""
文本专用数据加载器
用于 text_only 客户端（如医学报告文本）

支持的数据格式:
1. 独立 .txt 文件: 每个文本样本一个文件
2. JSON 文件: 包含所有文本样本的列表
3. 预计算的 .npy 特征文件: BERT/PubMedBERT 编码后的特征

作者: FedSAM3-Cream Team
日期: 2026-02-28
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np
from typing import Tuple, Optional, List


class TextOnlyDataset(Dataset):
    """
    纯文本数据集

    数据目录结构示例:

    方式1: 独立文本文件
    data/federated_split/train/client_1/
    ├── private/
    │   └── texts/
    │       ├── report_001.txt
    │       ├── report_002.txt
    │       └── ...
    └── public/
        └── texts/
            ├── pub_report_001.txt
            └── ...

    方式2: JSON 格式
    data/federated_split/train/client_1/
    ├── private/
    │   └── texts.json  # [{"id": 1, "text": "Patient shows..."}]
    └── public/
        └── texts.json

    方式3: 预计算特征 (推荐用于加速训练)
    data/federated_split/train/client_1/
    ├── private/
    │   ├── texts/
    │   │   └── ...
    │   └── features/  # 预计算的BERT特征
    │       ├── report_001.npy  # shape: (768,)
    │       └── ...
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "private",  # "private" or "public"
        text_feature_dim: int = 768,  # BERT/PubMedBERT 特征维度
        max_samples: Optional[int] = None,
        use_precomputed_features: bool = True  # 是否使用预计算的特征
    ):
        """
        Args:
            data_dir: 数据目录路径（如 "data/federated_split/train/client_1"）
            mode: 数据模式 ("private" 或 "public")
            text_feature_dim: 文本特征维度
            max_samples: 最大样本数（用于测试/调试）
            use_precomputed_features: 是否使用预计算的 .npy 特征文件
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.text_feature_dim = text_feature_dim
        self.use_precomputed_features = use_precomputed_features

        # 设置文本目录
        if mode == "private":
            self.text_dir = self.data_dir / "private" / "texts"
            self.feature_dir = self.data_dir / "private" / "features"  # 预计算特征
            self.json_file = self.data_dir / "private" / "texts.json"
        else:  # public
            self.text_dir = self.data_dir / "public" / "texts"
            self.feature_dir = self.data_dir / "public" / "features"
            self.json_file = self.data_dir / "public" / "texts.json"

        # 加载文本数据
        self.texts = []
        self.text_features = []

        # 方式1: 从 JSON 文件加载
        if self.json_file.exists():
            self._load_from_json()

        # 方式2: 从 .txt 文件加载
        elif self.text_dir.exists():
            self._load_from_txt_files()

        else:
            raise ValueError(
                f"文本数据不存在！\n"
                f"检查路径:\n"
                f"  - JSON 文件: {self.json_file}\n"
                f"  - 文本目录: {self.text_dir}\n"
                f"至少需要其中一个存在。"
            )

        # 加载预计算的文本特征（如果启用）
        if use_precomputed_features and self.feature_dir.exists():
            self._load_precomputed_features()

        # 限制样本数
        if max_samples is not None and max_samples > 0:
            self.texts = self.texts[:max_samples]
            if self.text_features:
                self.text_features = self.text_features[:max_samples]
            print(f"  [TextDataset] 限制样本数: {len(self.texts)}")

        if len(self.texts) == 0:
            raise ValueError(f"未加载到任何文本数据！检查目录: {self.text_dir}")

    def _load_from_json(self):
        """从 JSON 文件加载文本"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 支持两种 JSON 格式:
        # 1. [{"text": "..."}, ...]
        # 2. [{"id": 1, "text": "..."}, ...]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'text' in item:
                    self.texts.append(item['text'])
                elif isinstance(item, str):
                    self.texts.append(item)

        print(f"  [TextDataset] 从 JSON 加载了 {len(self.texts)} 条文本: {self.json_file}")

    def _load_from_txt_files(self):
        """从独立的 .txt 文件加载文本"""
        text_files = sorted(list(self.text_dir.glob("*.txt")))

        for txt_file in text_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:  # 跳过空文件
                    self.texts.append(text)

        print(f"  [TextDataset] 从 {len(text_files)} 个 .txt 文件加载了 {len(self.texts)} 条文本")

    def _load_precomputed_features(self):
        """加载预计算的文本特征 (.npy 文件)"""
        feature_files = sorted(list(self.feature_dir.glob("*.npy")))

        for feat_file in feature_files:
            try:
                feat = np.load(feat_file)

                # 验证特征维度
                if feat.shape[-1] != self.text_feature_dim:
                    print(f"  [警告] 特征维度不匹配: {feat_file.name} "
                          f"({feat.shape[-1]} vs 期望 {self.text_feature_dim})")
                    continue

                # 如果是多维特征，取平均
                if feat.ndim > 1:
                    feat = feat.mean(axis=0)

                self.text_features.append(torch.from_numpy(feat).float())

            except Exception as e:
                print(f"  [警告] 加载特征失败 {feat_file.name}: {e}")
                continue

        if len(self.text_features) == len(self.texts):
            print(f"  [TextDataset] 加载了 {len(self.text_features)} 个预计算特征 ✓")
        elif len(self.text_features) > 0:
            print(f"  [警告] 预计算特征数量 ({len(self.text_features)}) != 文本数量 ({len(self.texts)})")
            print(f"  [警告] 将使用零向量填充缺失的特征")
            # 不清空，后续在 __getitem__ 中处理
        else:
            print(f"  [TextDataset] 未找到预计算特征，将使用零向量（需要动态编码）")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            text: 原始文本字符串
            text_feature: 预计算的文本特征 (D,) 或零向量
        """
        text = self.texts[idx]

        # 返回预计算特征（如果有）
        if self.text_features and idx < len(self.text_features):
            text_feature = self.text_features[idx]
        else:
            # 占位符：返回零向量
            # 注意：实际使用时需要在训练循环中动态编码
            text_feature = torch.zeros(self.text_feature_dim)

        return text, text_feature


def text_collate_fn(batch):
    """
    文本数据的 collate 函数

    将一个 batch 的样本整理成批次张量

    输入: [(text1, feat1), (text2, feat2), ...]
    输出: (texts, features)
        - texts: List[str] - 原始文本列表
        - features: torch.Tensor (B, D) - 特征张量
    """
    texts, features = zip(*batch)

    # 堆叠特征张量
    features_stacked = torch.stack(features, dim=0)  # (B, D)

    return list(texts), features_stacked


def create_text_data_loaders(
    data_root: str,
    split: str = "train",
    client_id: str = "client_1",
    batch_size: int = 8,
    text_feature_dim: int = 768,
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: Optional[int] = None
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    为 text_only 客户端创建数据加载器

    Args:
        data_root: 数据根目录 (如 "data/federated_split")
        split: 数据集划分 ("train", "val", "test")
        client_id: 客户端ID (如 "client_1")
        batch_size: 批次大小
        text_feature_dim: 文本特征维度
        shuffle: 是否打乱数据
        num_workers: DataLoader 工作进程数（文本数据通常设为0）
        max_samples: 最大样本数（用于测试）

    Returns:
        (private_loader, public_loader)
        - private_loader: 私有数据加载器
        - public_loader: 公共数据加载器（可能为 None）
    """
    data_path = Path(data_root) / split / client_id

    private_loader = None
    public_loader = None

    # 加载私有数据
    try:
        private_dataset = TextOnlyDataset(
            data_dir=str(data_path),
            mode="private",
            text_feature_dim=text_feature_dim,
            max_samples=max_samples
        )

        private_loader = DataLoader(
            private_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=text_collate_fn,
            pin_memory=torch.cuda.is_available()
        )

        print(f"  [OK] {client_id} ({split}) - Private text data: {len(private_dataset)} samples")

    except Exception as e:
        print(f"  [错误] {client_id} ({split}) - Private text data 加载失败!")
        print(f"    原因: {e}")
        raise  # 私有数据必须存在，加载失败应该中断

    # 加载公共数据（可选）
    try:
        public_dataset = TextOnlyDataset(
            data_dir=str(data_path),
            mode="public",
            text_feature_dim=text_feature_dim,
            max_samples=max_samples
        )

        public_loader = DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=text_collate_fn,
            pin_memory=torch.cuda.is_available()
        )

        print(f"  [OK] {client_id} ({split}) - Public text data: {len(public_dataset)} samples")

    except Exception as e:
        print(f"  [警告] {client_id} ({split}) - Public text data 加载失败，跳过")
        print(f"    原因: {e}")
        print(f"    💡 提示: 公共数据用于对比学习，如果不需要可以忽略此警告")
        public_loader = None

    return private_loader, public_loader


# ============================================================================
# 测试和示例代码
# ============================================================================

def create_test_data():
    """创建测试数据（用于开发和测试）"""
    print("=" * 70)
    print("创建测试文本数据")
    print("=" * 70)

    test_dir = Path("data/federated_split/train/client_1")

    # 创建私有文本数据（方式1: 独立 .txt 文件）
    private_text_dir = test_dir / "private" / "texts"
    private_text_dir.mkdir(parents=True, exist_ok=True)

    sample_reports = [
        "Patient ID 001: Brain MRI shows enhancing mass in left frontal lobe measuring 3.2 cm. Surrounding vasogenic edema present.",
        "Patient ID 002: Post-contrast T1 reveals ring-enhancing lesion in right temporal region. Differential includes glioblastoma.",
        "Patient ID 003: FLAIR sequence demonstrates hyperintense signal abnormality in periventricular white matter. No mass effect.",
        "Patient ID 004: Multiple enhancing foci in bilateral hemispheres consistent with metastatic disease.",
        "Patient ID 005: Large necrotic mass in posterior fossa with significant mass effect and midline shift.",
    ]

    for i, report in enumerate(sample_reports):
        with open(private_text_dir / f"report_{i+1:03d}.txt", 'w', encoding='utf-8') as f:
            f.write(report)

    print(f"✓ 创建了 {len(sample_reports)} 个私有文本文件")

    # 创建公共文本数据（方式2: JSON 格式）
    public_json = test_dir / "public" / "texts.json"
    public_json.parent.mkdir(parents=True, exist_ok=True)

    public_data = [
        {"id": 1, "text": "Normal brain MRI without evidence of acute pathology."},
        {"id": 2, "text": "Small non-enhancing lesion in white matter, likely chronic microvascular change."},
    ]

    with open(public_json, 'w', encoding='utf-8') as f:
        json.dump(public_data, f, indent=2, ensure_ascii=False)

    print(f"✓ 创建了公共文本 JSON 文件: {len(public_data)} 条记录")

    # 创建模拟的预计算特征
    private_feature_dir = test_dir / "private" / "features"
    private_feature_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(sample_reports)):
        # 生成随机特征（实际使用时应该是 BERT 编码）
        fake_feature = np.random.randn(768).astype(np.float32)
        np.save(private_feature_dir / f"report_{i+1:03d}.npy", fake_feature)

    print(f"✓ 创建了 {len(sample_reports)} 个预计算特征文件")
    print(f"\n测试数据路径: {test_dir}")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("文本数据加载器 - 测试模式")
    print("=" * 70)
    print()

    # 创建测试数据
    create_test_data()
    print()

    # 测试数据加载
    print("=" * 70)
    print("测试数据加载")
    print("=" * 70)

    try:
        private_loader, public_loader = create_text_data_loaders(
            data_root="data/federated_split",
            split="train",
            client_id="client_1",
            batch_size=2,
            text_feature_dim=768,
            max_samples=5
        )

        print("\n" + "=" * 70)
        print("测试批次迭代")
        print("=" * 70)

        # 测试私有数据迭代
        if private_loader:
            print("\n[Private Data]")
            for batch_idx, (texts, features) in enumerate(private_loader):
                print(f"  Batch {batch_idx + 1}:")
                print(f"    Texts ({len(texts)} samples):")
                for i, text in enumerate(texts):
                    print(f"      [{i+1}] {text[:80]}...")
                print(f"    Features shape: {features.shape}")
                print(f"    Features stats: mean={features.mean().item():.4f}, std={features.std().item():.4f}")

                if batch_idx >= 1:  # 只显示前2个batch
                    break

        # 测试公共数据迭代
        if public_loader:
            print("\n[Public Data]")
            for batch_idx, (texts, features) in enumerate(public_loader):
                print(f"  Batch {batch_idx + 1}:")
                print(f"    Texts: {texts}")
                print(f"    Features shape: {features.shape}")
                break

        print("\n" + "=" * 70)
        print("✓ 所有测试完成！")
        print("=" * 70)
        print("\n使用说明:")
        print("1. 将此文件复制到你的项目: data/text_dataset_loader.py")
        print("2. 准备你的文本数据（.txt 文件或 .json 文件）")
        print("3. （可选）预计算 BERT 特征以加速训练")
        print("4. 在训练脚本中导入并使用:")
        print("   from data.text_dataset_loader import create_text_data_loaders")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
