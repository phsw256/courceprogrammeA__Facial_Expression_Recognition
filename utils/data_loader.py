import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms


# 设置随机种子，确保实验可重复性
def set_seed(seed=42):
    """
    设置随机种子，确保实验可重复性

    参数:
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 基本数据变换 - 只进行标准化
basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 使用数据增强的变换
augmented_transform = transforms.Compose([
    transforms.RandomCrop(48, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def load_FER2013(use_augmentation=False, valid_size=0.1, batch_size=128, num_workers=2):
    """
    kl
    加载FER2013数据集，并分割出验证集

    参数:
        use_augmentation: 是否对训练集使用数据增强
        valid_size: 验证集比例
        batch_size: 批次大小
        num_workers: 数据加载器使用的工作进程数

    返回:
        train_loader, valid_loader, test_loader: 数据加载器
        classes: 类别名称
    """
    transform = augmented_transform if use_augmentation else basic_transform

    #train 和 test 文件夹
    train_dir = './data/ProjectA_FER2013_20250422/train'
    test_dir = './data/ProjectA_FER2013_20250422/test'

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=basic_transform)

    # 计算验证集大小
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(valid_size * num_train)
    train_idx, valid_idx = indices[split:], indices[:split]

    # 创建数据采样器
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"训练集大小: {len(train_idx)}")
    print(f"验证集大小: {len(valid_idx)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 获取类别名称
    classes = ('angry', 'disgust', 'fear', 'happy', 'sad',
               'surprise', 'neutral')

    return train_loader, valid_loader, test_loader, classes


def visualize_samples(dataloader, classes, num_samples=5):
    """
    可视化数据样本

    参数:
        dataloader: 数据加载器
        classes: 类别名称
        num_samples: 每个类别要显示的样本数
    """
    # 获取batch数据
    images, labels = next(iter(dataloader))

    # 创建样本计数器
    class_counts = {i: 0 for i in range(len(classes))}
    indices = []

    for i, label in enumerate(labels):
        label = label.item()
        if class_counts[label] < num_samples:
            indices.append(i)
            class_counts[label] += 1

        # 如果所有类别都有足够的样本，则停止
        if all(count >= num_samples for count in class_counts.values()):
            break

    # 获取选定的图像和标签
    selected_images = images[indices]
    selected_labels = labels[indices]

    # 创建图像网格
    fig, axes = plt.subplots(10, num_samples, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.5)

    # 对于每个类别
    for class_idx in range(len(classes)):
        # 找到该类别的所有样本
        class_indices = [i for i, label in enumerate(selected_labels) if label == class_idx]

        for i in range(min(num_samples, len(class_indices))):
            img_idx = class_indices[i]
            img = selected_images[img_idx].numpy().transpose((1, 2, 0))
            # 反标准化
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            ax = axes[class_idx, i]
            ax.imshow(img)
            ax.set_title(classes[class_idx])
            ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 设置随机种子
    set_seed()

    # 检查是否有可用的GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    train_loader, valid_loader, test_loader, classes = load_FER2013(use_augmentation=False)

    # 可视化一些样本
    visualize_samples(train_loader, classes, num_samples=5)
