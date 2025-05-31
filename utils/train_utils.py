import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import streamlit as st
import datetime

import torchvision


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None,
                num_epochs=10, device=None, save_dir='./checkpoints', recorder=None):
    """
    训练模型并记录性能指标

    参数:
        model: 要训练的模型
        train_loader, valid_loader: 训练和验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        num_epochs: 训练轮数
        device: 使用的设备
        save_dir: 模型保存目录

    返回:
        history: 包含训练历史的字典
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    model = model.to(device)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_times': []
    }

    best_val_acc = 0.0

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    bar=st.empty()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        bar.progress(epoch/num_epochs,f"Epoch {epoch + 1}/{num_epochs}")

        # 计算训练指标
        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 统计
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算验证指标
        val_loss = val_loss / len(valid_loader.sampler)
        val_acc = val_correct / val_total

        # 更新学习率
        if scheduler:
            scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 记录每个epoch的时间
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        history['epoch_times'].append(epoch_time)

        filename=f"{save_dir}/{model.__class__.__name__}_best({recorder}).pth"

        # 如果是最佳模型，保存权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), filename)
            print(f"模型已保存到 {filename}")
            st.success(f"模型已保存到 {filename}")

        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"本轮用时: {epoch_time:.2f}s")
        print("-" * 50)
        st.success(f"Epoch {epoch + 1}/{num_epochs}"+f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}\n"+f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}\n"+f"本轮用时: {epoch_time:.2f}s")

    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"总训练时间: {total_time:.2f}s")

    return model, history

def evaluate_model(model, test_loader, criterion, device=None, classes=None, timestamp=None):
    """
    评估模型在测试集上的性能

    参数:
        model: 要评估的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 使用的设备
        classes: 类别名称列表

    返回:
        test_loss: 测试损失
        test_acc: 测试准确率
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 统计
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # 收集真实标签和预测标签
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 计算测试指标
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total

    print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

    # 如果提供了类别名称，计算混淆矩阵
    if classes:
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            import seaborn as sns

            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')  # 英文标签
            plt.ylabel('True')  # 英文标签
            plt.title('Confusion Matrix')  # 英文标题
            os.makedirs('./reportfig', exist_ok=True)  # 确保文件夹存在
            plt.savefig(f"./reportfig/{timestamp}_confusion_matrix.png")
            st.pyplot(plt.gcf())

            # 打印分类报告
            print("分类报告:")
            st.write("分类报告:")
            print(classification_report(y_true, y_pred, target_names=classes))
            st.dataframe(pd.DataFrame(classification_report(y_true, y_pred, target_names=classes, output_dict=True)).transpose())

        except ImportError:
            st.error("警告: 未安装sklearn或seaborn，无法生成混淆矩阵和分类报告")

    return test_loss, test_acc

def plot_training_history(history, title="Training History"):
    """
    绘制训练历史曲线

    参数:
        history: 包含训练历史的字典
        title: 图表标题
    """
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')  # 英文标签
    plt.plot(history['val_loss'], label='Validation Loss')  # 英文标签
    plt.xlabel('Epochs')  # 英文标签
    plt.ylabel('Loss')  # 英文标签
    plt.title('Loss Curves')  # 英文标题
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')  # 英文标签
    plt.plot(history['val_acc'], label='Validation Accuracy')  # 英文标签
    plt.xlabel('Epochs')  # 英文标签
    plt.ylabel('Accuracy')  # 英文标签
    plt.title('Accuracy Curves')  # 英文标题
    plt.legend()

    plt.suptitle(title)  # 英文总标题
    plt.tight_layout()
    os.makedirs('./reportfig', exist_ok=True)  # 确保文件夹存在
    plt.savefig(f"./reportfig/{title.replace(' ', '_')}.png")
    st.pyplot(plt.gcf())

def visualize_model_predictions(model, test_loader, classes, device=None, num_images=25, title="visualize predictions"):
    """
    可视化模型预测

    参数:
        model: 要评估的模型
        test_loader: 测试数据加载器
        classes: 类别名称列表
        device: 使用的设备
        num_images: 要显示的图像数量
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    # 获取batch数据
    images, labels = next(iter(test_loader))

    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)

    # 将预测和标签转换为CPU上的numpy数组
    preds = preds.cpu().numpy()
    labels = labels.numpy()

    # 计算display_grid的尺寸
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    for i, ax in enumerate(axes.flat):
        if i < min(num_images, len(preds)):
            img = images[i].numpy().transpose((1, 2, 0))
            # 反标准化
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            color = "green" if preds[i] == labels[i] else "red"
            ax.set_title(f"Predicted: {classes[preds[i]]}\nTrue: {classes[labels[i]]}", color=color)  # 英文标签
        ax.axis('off')

    plt.tight_layout()
    os.makedirs('./reportfig', exist_ok=True)  # 确保文件夹存在
    plt.savefig(f"./reportfig/{title.replace(' ', '_')}.png")
    st.pyplot(plt.gcf())

def visualize_conv_filters(model, layer_name='conv1', title="visualize filters"):
    """
    可视化卷积核

    参数:
        model: 模型
        layer_name: 要可视化的卷积层名称
    """
    model.eval()

    # 获取指定层的权重
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            weights = module.weight.data.clone().cpu()
            break
    else:
        print(f"未找到名为 {layer_name} 的卷积层")
        return

    # 规范化权重以便可视化
    weights = weights - weights.min()
    weights = weights / weights.max()

    # 绘制卷积核
    num_filters = min(16, weights.size(0))
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(f'Conv Layer {layer_name} Filters')  # 英文标题

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # 如果是3通道的卷积核，直接显示RGB
            if weights.size(1) == 3:
                ax.imshow(weights[i].permute(1, 2, 0))
            else:
                # 如果不是3通道，只显示第一个通道
                ax.imshow(weights[i, 0], cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    os.makedirs('./reportfig', exist_ok=True)  # 确保文件夹存在
    plt.savefig(f"./reportfig/{title.replace(' ', '_')}.png")
    st.pyplot(plt.gcf())

def model_complexity(model, input_size=(3, 48, 48), batch_size=128, device=None):
    """
    计算模型参数量和推理时间

    参数:
        model: 要评估的模型
        input_size: 输入尺寸
        batch_size: 批量大小
        device: 使用的设备

    返回:
        num_params: 参数量
        inference_time: 每批次推理时间
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 创建随机输入
    dummy_input = torch.randn(batch_size, *input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end_time = time.time()

    inference_time = (end_time - start_time) / 100

    print(f"参数量: {num_params:,}")
    print(f"每批次({batch_size}个样本)推理时间: {inference_time*1000:.2f}ms")
    st.success("已分析模型复杂度"+'\n'+f"参数量: {num_params:,}"+'\n'+f"每批次({batch_size}个样本)推理时间: {inference_time*1000:.2f}ms")

    return num_params, inference_time


def predict(image, model, device=None):
    emotion_mapping = {0: 'Angry 😠', 1: 'Disgust 😣', 2: 'Fear 😱', 3: 'Happy 😀', 4: 'Sad 😢', 5: 'Surprise 😲',
                       6: 'Neutral 😐'}
    '''
    kl：总之就是突发奇想将其放进头文件了
    我用到并非vit模型，是吗？
    '''
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    model = model.to(device)
    model.eval()

    image = image.to(device)
    image = image.unsqueeze(0)

    logits = model(image)
    st.write(logits)

    y_prob = torch.nn.functional.softmax(logits[0], dim=-1)

    df_result = pd.DataFrame(
        y_prob.T.cpu().detach().numpy(),
        columns=['Probability'],
        index=emotion_mapping.values()
    )

    return df_result