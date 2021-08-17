import torch
import torch.nn as nn

# 【判别器】
class Discriminator(nn.Module):

    def __init__(self, input_size=(1682, 1682)):
        super().__init__()

        self.model = nn.Sequential(
            # 判断一个128维的向量是不是后生成的
            nn.Linear(input_size[0] + input_size[1], 600),  # 1682是vector大小，1682是CGAN的扩展维度,condition=self
            nn.LayerNorm(600),
            nn.LeakyReLU(0.02),

            nn.Linear(600, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(0.02),

            nn.Linear(300, 1),
            nn.Sigmoid()
        )

        self.loss_function = nn.BCELoss()

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.0001)

        self.counter = 0
        self.progress = []

    def forward(self, vector_tensor, label_tensor):
        inputs = torch.cat((vector_tensor, label_tensor))
        return self.model(inputs)


# 【生成器】
class Generator(nn.Module):

    def __init__(self, input_size=(128, 1682), output_size=1682):
        super().__init__()

        self.model = nn.Sequential(
            # 拿到一个128+1682维的加了白噪声的随机初始化向量，生成一个1682维的向量
            nn.Linear(input_size[0] + input_size[1], 300),  # 128是随机种子的维度，1682是CGAN的扩展维度
            nn.LayerNorm(300),  # channel 方向做归一化
            nn.LeakyReLU(0.02),

            nn.Linear(300, 600),
            nn.LayerNorm(600),
            nn.LeakyReLU(0.02),

            nn.Linear(600, output_size),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.0001)

        self.counter = 0
        self.progress = []

    def forward(self, seed_tensor, label_tensor):
        inputs = torch.cat((seed_tensor, label_tensor))
        return self.model(inputs)
