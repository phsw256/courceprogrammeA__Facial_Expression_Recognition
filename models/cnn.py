import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """简单的CNN模型，包含两个卷积层和两个全连接层"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)##我以为这个10随便填的呢
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 输出大小: 16x16x16
        x = self.pool(self.relu(self.conv2(x)))  # 输出大小: 8x8x32
        x = x.view(-1, 32 * 12 * 12)  # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MediumCNN(nn.Module):
    """中等复杂度的CNN模型，包含批标准化和Dropout"""
    def __init__(self, use_bn=True, out_channel1=32):
        super(MediumCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel1) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_channel1, out_channel1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel1) if use_bn else nn.Identity()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(out_channel1, 2*out_channel1, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(2*out_channel1) if use_bn else nn.Identity()
        self.conv4 = nn.Conv2d(2*out_channel1, 2*out_channel1, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(2*out_channel1) if use_bn else nn.Identity()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2*out_channel1 * 12 * 12, 512)
        self.bn5 = nn.BatchNorm1d(512) if use_bn else nn.Identity()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


class VGGStyleNet(nn.Module):
    """VGG风格的CNN网络，使用连续的3x3卷积和池化"""
    def __init__(self, out_channel1=64):
        super(VGGStyleNet, self).__init__()

        # VGG风格：连续的3x3卷积 + 池化
        self.features = nn.Sequential(
            nn.Conv2d(3, out_channel1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 第二块：2倍通道数
            nn.Conv2d(out_channel1, 2 * out_channel1, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * out_channel1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * out_channel1, 2 * out_channel1, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * out_channel1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 第三块：4倍通道数
            nn.Conv2d(2 * out_channel1, 4 * out_channel1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * out_channel1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * out_channel1, 4 * out_channel1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * out_channel1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * out_channel1 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 7)  # 最终输出7个通道
        )

        # 权重初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class ResidualBlock(nn.Module):
    """卷积神经网络的残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class SimpleResNet(nn.Module):
    """简化版ResNet模型"""
    def __init__(self, num_blocks=[2, 2, 2], num_classes=7, in_channels=16):
        super(SimpleResNet, self).__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(self.in_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.in_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.in_channels*2, num_blocks[2], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
