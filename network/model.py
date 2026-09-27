import torch.nn as nn

class ConvBlock(nn.Module):
    '''卷积块搭建'''
    def __init__(self,num_filters):
        super().__init__()
        self.conv = nn.Conv2d(num_filters,num_filters,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out
        
class ResBlock(nn.Module):
    '''残差块搭建'''
    
class GomokuNet(nn.Module):
    def __init__(self,num_filters=64,num_blocks=4):
        super().__init__()
        
        # 主干神经网络
        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=num_filters,
                kernel_size=3,
                padding=1
                ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            *[ConvBlock(num_filters) for _ in range(num_blocks)]
            )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2,
                kernel_size=1,
                padding=0
            ),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(450,225)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2,
                kernel_size=1,
                padding=0
            ),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(450,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Tanh()
        )
        
    def forward(self,x):
        x = self.backbone(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy,value