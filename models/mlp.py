import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    """单隐层MLP模型"""
    def __init__(self, input_dim=6912, hidden_dim=512, output_dim=7):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepMLP(nn.Module):
    """深层MLP模型，具有多个隐藏层、批标准化和dropout"""
    def __init__(self, input_dim=6912, dropout_rate=0.5, use_bn=True, use_dropout=True):
        super(DeepMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        # 第一层
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.bn1 = nn.BatchNorm1d(input_dim // 2) if use_bn else nn.Identity()

        # 第二层
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)
        self.bn2 = nn.BatchNorm1d(input_dim // 4) if use_bn else nn.Identity()

        # 第三层
        self.fc3 = nn.Linear(input_dim // 4, input_dim // 8)
        self.bn3 = nn.BatchNorm1d(input_dim // 8) if use_bn else nn.Identity()

        # 输出层
        self.fc4 = nn.Linear(input_dim // 8, 7)

        # 激活和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()

    def forward(self, x):
        x = self.flatten(x)

        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 第三层
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 输出层
        x = self.fc4(x)

        return x


class ResidualBlock(nn.Module):
    """MLP的残差块"""
    def __init__(self, input_dim, output_dim, activation, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()

        self.linear1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

        # 如果输入维度不等于输出维度，添加一个线性变换
        self.shortcut = nn.Identity()
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )

    def forward(self, x):
        residual = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.activation(out)

        return out


class ResidualMLP(nn.Module):
    """带有残差连接的MLP模型"""
    def __init__(self, input_dim=6912, hidden_dims=[1024, 1024, 1024, 512, 512, 512], output_dim=7,
                 dropout_rate=0.5, activation='relu'):
        super(ResidualMLP, self).__init__()
        self.flatten = nn.Flatten()

        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        # 输入层
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout_rate))

        # 隐藏层，带残差连接
        for i in range(1, len(hidden_dims)):
            layers.append(ResidualBlock(hidden_dims[i-1], hidden_dims[i], self.activation, dropout_rate))

        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x
