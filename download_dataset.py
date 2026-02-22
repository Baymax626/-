import os
import random
import shutil
from PIL import Image

def prepare_catdog_data(source_dir='./data/PetImages', target_dir='./data', train_ratio=0.8):
    """
    整理猫狗数据集，划分训练集和验证集，并跳过损坏的图片。
    """
    categories = ['Cat', 'Dog']
    target_names = ['cats', 'dogs']  # 目标子文件夹名（对应类别名）

    # 创建目标目录
    for split in ['train', 'val']:
        for name in target_names:
            os.makedirs(os.path.join(target_dir, split, name), exist_ok=True)

    for cat_idx, category in enumerate(categories):
        src_folder = os.path.join(source_dir, category)
        if not os.path.exists(src_folder):
            print(f"警告：源文件夹 {src_folder} 不存在，跳过。")
            continue

        # 获取所有图片文件（假设是 .jpg）
        images = [f for f in os.listdir(src_folder) if f.lower().endswith('.jpg')]
        print(f"在 {category} 中找到 {len(images)} 张图片，正在检查损坏文件...")

        valid_images = []
        # 检查图片是否可打开
        for img_name in images:
            img_path = os.path.join(src_folder, img_name)
            try:
                with Image.open(img_path) as img:
                    img.verify()  # 验证完整性
                valid_images.append(img_name)
            except Exception as e:
                print(f"跳过损坏图片: {img_path}")

        print(f"有效图片数: {len(valid_images)}")
        random.shuffle(valid_images)

        # 划分
        split_idx = int(len(valid_images) * train_ratio)
        train_images = valid_images[:split_idx]
        val_images = valid_images[split_idx:]

        # 复制训练集
        for img_name in train_images:
            src = os.path.join(src_folder, img_name)
            dst = os.path.join(target_dir, 'train', target_names[cat_idx], img_name)
            shutil.copy2(src, dst)

        # 复制验证集
        for img_name in val_images:
            src = os.path.join(src_folder, img_name)
            dst = os.path.join(target_dir, 'val', target_names[cat_idx], img_name)
            shutil.copy2(src, dst)

        print(f"{category} 处理完成：训练集 {len(train_images)} 张，验证集 {len(val_images)} 张")

    print("数据集整理完毕！")

if __name__ == '__main__':
    prepare_catdog_data()