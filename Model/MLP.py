import torch.nn as nn
'''
2-layer MLP with ReLU activation.
'''
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate):
        # input_dim：输入特征的维度。
        # hidden_dim：隐藏层的维度。
        # num_classes：输出的类别数量。
        # dropout_rate：dropout 正则化的丢弃概率。
        super().__init__()

        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        # linear_1：一个全连接层，将输入特征映射到隐藏层。
        self.relu = nn.ReLU()
        # relu：ReLU 激活函数，用于增加模型的非线性拟合能力。
        self.linear_2 = nn.Linear(hidden_dim, num_classes)
        # linear_2：另一个全连接层，将隐藏层的输出映射到输出类别空间。
        self.dropout = nn.Dropout(dropout_rate)
        # dropout：dropout 正则化层，用于防止过拟合。
    

    def forward(self, x):
        # x：输入特征，形状为 (batch_size, input_dim)。
        return self.dropout(self.linear_2(self.relu(self.linear_1(x))))
        # 输入特征 x 经过第一个全连接层 linear_1 进行线性变换。
        # 然后通过 ReLU 激活函数 relu 进行非线性映射。
        # 随后，经过第二个全连接层 linear_2 进行线性变换，将其映射到输出类别空间。
        # 最后，应用 dropout 正则化层 dropout 对全连接层的输出进行随机丢弃，以减少过拟合风险。
        # 返回模型的预测结果，形状为 (batch_size, num_classes)，表示每个样本属于每个类别的概率分布。