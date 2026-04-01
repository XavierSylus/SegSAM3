"""
数据集拆分工具 - 将原始数据集拆分为私有数据和公共数据
用于准备联邦学习的数据集
"""
import os
import shutil
import random
from pathlib import Path
from typing import Tuple, Optional
import argparse


def split_dataset(
    source_images_dir: str,
    source_masks_dir: str,
    output_base_dir: str,
    client_id: str,
    private_ratio: float = 0.7,
    public_ratio: float = 0.3,
    seed: int = 42,
    copy_mode: bool = True
) -> Tuple[int, int]:
    """
    将数据集拆分为私有数据和公共数据
    
    Args:
        source_images_dir: 源图像目录路径
        source_masks_dir: 源掩码目录路径（私有数据需要）
        output_base_dir: 输出基础目录（如 "data"）
        client_id: 客户端ID（如 "client_A"）
        private_ratio: 私有数据比例（默认0.7，即70%）
        public_ratio: 公共数据比例（默认0.3，即30%）
        seed: 随机种子
        copy_mode: True=复制文件，False=移动文件
    
    Returns:
        Tuple of (私有数据数量, 公共数据数量)
    """
    # 设置随机种子
    random.seed(seed)
    
    # 转换为Path对象
    source_images = Path(source_images_dir)
    source_masks = Path(source_masks_dir)
    output_base = Path(output_base_dir)
    
    # 检查源目录是否存在
    if not source_images.exists():
        raise ValueError(f"源图像目录不存在: {source_images_dir}")
    if not source_masks.exists():
        raise ValueError(f"源掩码目录不存在: {source_masks_dir}")
    
    # 创建输出目录结构
    private_images_dir = output_base / client_id / "private" / "images"
    private_masks_dir = output_base / client_id / "private" / "masks"
    public_images_dir = output_base / client_id / "public" / "images"
    
    private_images_dir.mkdir(parents=True, exist_ok=True)
    private_masks_dir.mkdir(parents=True, exist_ok=True)
    public_images_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(source_images.glob(f"*{ext}")))
    
    if len(image_files) == 0:
        raise ValueError(f"在 {source_images_dir} 中未找到图像文件")
    
    # 随机打乱
    random.shuffle(image_files)
    
    # 计算拆分点
    total_files = len(image_files)
    private_count = int(total_files * private_ratio)
    public_count = total_files - private_count
    
    print(f"\n数据集拆分信息:")
    print(f"  总文件数: {total_files}")
    print(f"  私有数据: {private_count} ({private_ratio*100:.1f}%)")
    print(f"  公共数据: {public_count} ({public_ratio*100:.1f}%)")
    
    # 拆分文件
    private_images = image_files[:private_count]
    public_images = image_files[private_count:]
    
    # 复制/移动私有数据（图像+掩码）
    print(f"\n处理私有数据...")
    private_processed = 0
    for img_file in private_images:
        # 复制图像
        dest_img = private_images_dir / img_file.name
        if copy_mode:
            shutil.copy2(img_file, dest_img)
        else:
            shutil.move(str(img_file), str(dest_img))
        
        # 查找对应的掩码文件
        mask_file = None
        # 尝试多种命名方式
        possible_mask_names = [
            img_file.name,  # 相同文件名
            f"mask_{img_file.stem}.png",
            f"{img_file.stem}_mask.png",
            f"{img_file.stem}.png"
        ]
        
        for mask_name in possible_mask_names:
            mask_path = source_masks / mask_name
            if mask_path.exists():
                mask_file = mask_path
                break
        
        if mask_file:
            dest_mask = private_masks_dir / mask_file.name
            if copy_mode:
                shutil.copy2(mask_file, dest_mask)
            else:
                shutil.move(str(mask_file), str(dest_mask))
            private_processed += 1
        else:
            print(f"  警告: 未找到 {img_file.name} 对应的掩码文件")
    
    print(f"  已处理: {private_processed}/{private_count} 个私有图像-掩码对")
    
    # 复制/移动公共数据（只有图像，无掩码）
    print(f"\n处理公共数据...")
    public_processed = 0
    for img_file in public_images:
        dest_img = public_images_dir / img_file.name
        if copy_mode:
            shutil.copy2(img_file, dest_img)
        else:
            shutil.move(str(img_file), str(dest_img))
        public_processed += 1
    
    print(f"  已处理: {public_processed} 个公共图像")
    
    return private_processed, public_processed


def split_dataset_simple(
    source_dir: str,
    output_base_dir: str,
    client_id: str,
    private_ratio: float = 0.7,
    public_ratio: float = 0.3,
    seed: int = 42,
    copy_mode: bool = True
):
    """
    简化版拆分：假设源目录中 images/ 和 masks/ 在同一目录下
    
    Args:
        source_dir: 源目录，包含 images/ 和 masks/ 子目录
        output_base_dir: 输出基础目录
        client_id: 客户端ID
        private_ratio: 私有数据比例
        public_ratio: 公共数据比例
        seed: 随机种子
        copy_mode: True=复制，False=移动
    """
    source_path = Path(source_dir)
    images_dir = source_path / "images"
    masks_dir = source_path / "masks"
    
    return split_dataset(
        source_images_dir=str(images_dir),
        source_masks_dir=str(masks_dir),
        output_base_dir=output_base_dir,
        client_id=client_id,
        private_ratio=private_ratio,
        public_ratio=public_ratio,
        seed=seed,
        copy_mode=copy_mode
    )


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description="将数据集拆分为私有数据和公共数据"
    )
    
    parser.add_argument(
        '--source_images',
        type=str,
        required=True,
        help='源图像目录路径'
    )
    parser.add_argument(
        '--source_masks',
        type=str,
        required=True,
        help='源掩码目录路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/federated_split',
        help='输出基础目录（默认: data/federated_split）'
    )
    parser.add_argument(
        '--client_id',
        type=str,
        required=True,
        help='客户端ID（如: client_A）'
    )
    parser.add_argument(
        '--private_ratio',
        type=float,
        default=0.7,
        help='私有数据比例（默认: 0.7，即70%%）'
    )
    parser.add_argument(
        '--public_ratio',
        type=float,
        default=None,
        help='公共数据比例（默认: 1 - private_ratio）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认: 42）'
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help='移动文件而不是复制（默认: 复制）'
    )
    
    args = parser.parse_args()
    
    # 计算公共数据比例
    if args.public_ratio is None:
        args.public_ratio = 1.0 - args.private_ratio
    
    # 验证比例
    if abs(args.private_ratio + args.public_ratio - 1.0) > 1e-6:
        print(f"警告: 私有比例({args.private_ratio}) + 公共比例({args.public_ratio}) != 1.0")
        print("将自动调整公共比例")
        args.public_ratio = 1.0 - args.private_ratio
    
    print("=" * 60)
    print("数据集拆分工具")
    print("=" * 60)
    print(f"源图像目录: {args.source_images}")
    print(f"源掩码目录: {args.source_masks}")
    print(f"输出目录: {args.output}/{args.client_id}")
    print(f"私有数据比例: {args.private_ratio*100:.1f}%")
    print(f"公共数据比例: {args.public_ratio*100:.1f}%")
    print(f"模式: {'移动' if args.move else '复制'}")
    print("=" * 60)
    
    try:
        private_count, public_count = split_dataset(
            source_images_dir=args.source_images,
            source_masks_dir=args.source_masks,
            output_base_dir=args.output,
            client_id=args.client_id,
            private_ratio=args.private_ratio,
            public_ratio=args.public_ratio,
            seed=args.seed,
            copy_mode=not args.move
        )
        
        print("\n" + "=" * 60)
        print("拆分完成！")
        print("=" * 60)
        print(f"私有数据: {private_count} 个图像-掩码对")
        print(f"公共数据: {public_count} 个图像")
        print(f"\n输出目录结构:")
        print(f"  {args.output}/{args.client_id}/private/images/")
        print(f"  {args.output}/{args.client_id}/private/masks/")
        print(f"  {args.output}/{args.client_id}/public/images/")
        
    except Exception as e:
        print(f"\n错误: {e}")
        return 1
    
    return 0


# 示例使用
if __name__ == "__main__":
    import sys
    
    # 如果直接运行且没有参数，显示使用示例
    if len(sys.argv) == 1:
        print("=" * 60)
        print("数据集拆分工具 - 使用示例")
        print("=" * 60)
        print("\n命令行用法:")
        print("  python split_dataset.py \\")
        print("    --source_images /path/to/images \\")
        print("    --source_masks /path/to/masks \\")
        print("    --output data \\")
        print("    --client_id client_A \\")
        print("    --private_ratio 0.7")
        print("\nPython代码用法:")
        print("""
from split_dataset import split_dataset

# 拆分数据集
private_count, public_count = split_dataset(
    source_images_dir="raw_data/images",
    source_masks_dir="raw_data/masks",
    output_base_dir="data",
    client_id="client_A",
    private_ratio=0.7,  # 70% 私有数据
    public_ratio=0.3,   # 30% 公共数据
    seed=42,
    copy_mode=True      # True=复制，False=移动
)
        """)
        print("\n简化用法（如果images和masks在同一目录下）:")
        print("""
from split_dataset import split_dataset_simple

split_dataset_simple(
    source_dir="raw_data",  # 包含 images/ 和 masks/ 子目录
    output_base_dir="data",
    client_id="client_A",
    private_ratio=0.7
)
        """)
        sys.exit(0)
    
    # 运行命令行接口
    sys.exit(main())
