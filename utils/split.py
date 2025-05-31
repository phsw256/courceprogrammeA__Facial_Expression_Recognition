import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from shutil import copyfile

csv_path = '../data/ProjectA_FER2013_20250422/fer2013.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"未找到 {csv_path} 文件，请确保文件存在。")

# 读取 CSV 文件
data = pd.read_csv(csv_path)
if 'Usage' not in data.columns or 'emotion' not in data.columns or 'pixels' not in data.columns:
    raise ValueError("CSV 文件缺少必要的列：'Usage', 'emotion', 'pixels'。")

# 根据 'Usage' 列划分数据
train_data = data[data['Usage'] == 'Training']
test_data = data[data['Usage'] == 'PublicTest']

# 创建类别子文件夹并保存图像
def save_images(dat, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for _, row in dat.iterrows():
        label = str(row['emotion'])
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        # 将像素值转换为图像
        pixels = list(map(int, row['pixels'].split()))
        img = np.array(pixels, dtype=np.uint8).reshape(48, 48)

        # 保存图像
        img_path = os.path.join(label_dir, f"{row.name}.png")
        plt.imsave(img_path, img, cmap='gray')

save_images(train_data, '../data/ProjectA_FER2013_20250422/train')
save_images(test_data, '../data/ProjectA_FER2013_20250422/test')