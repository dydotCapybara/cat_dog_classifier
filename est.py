import os
from PIL import Image


def check_images(directory):
    for subdir, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                img = Image.open(filepath)
                img.verify()  # 检查图片是否损坏
            except (IOError, SyntaxError) as e:
                print(f"图片损坏: {filepath}")


# 检查训练集和验证集
check_images('F:/dataset/PetImages/Train')
check_images('F:/dataset/PetImages/Validation')
