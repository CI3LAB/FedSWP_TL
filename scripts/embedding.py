# -*- coding: utf-8 -*-
# File              : embedding.py
# Author            : Joy
# Date              : 2024/06/09
# Last Modified Date: 2024/08/09
# Last Modified By  : Joy
# Reference         : pip install -U openmim && mim install "mmpretrain>=1.0.0rc8"
# Description       : embedding architecture and network architecture
# ******************************************************
import torch.nn as nn
from mmpretrain import get_model

class MobileViT_Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 加载预训练MobileViT backbone (移除原分类头)
        self.backbone = get_model('mobilevit-xsmall_3rdparty_in1k', pretrained=True ).backbone

        # 得到表征数据
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局池化
            nn.Flatten()
        )

        # 自定义分类头 (输入维度需匹配MobileViT输出)
        self.head = nn.Sequential(
            nn.Linear(384, 64),  # MobileViT-small最后层通道数为640；MobileViT-xsmall最后层通道数为384
            nn.PReLU(),
            nn.Linear(64, num_classes)  # 二分类输出
        )

    def forward(self, x):
        embeddings = self.get_embedding(x)
        return self.head(embeddings)  # 输出 [B, 2]

    def get_embedding(self, x):
        features = self.backbone(x)  # 输入尺寸: [B, 3, 224, 224]
        # print(f"All features shapes: {[f.shape for f in features]}")  # 调试输出
        features = features[-1]
        # print(f"All features shapes: {[f.shape for f in features]}")  # 调试输出
        embeddings = self.embedding(features)
        # print(f"All features shapes: {[f.shape for f in embeddings]}")  # 调试输出
        return embeddings  # 输出 [B, 384]


class TripletNet(nn.Module):
        def __init__(self, embedding_net):
            super(TripletNet, self).__init__()
            self.embedding_net = embedding_net

        def forward(self, x1, x2, x3):
            output1 = self.embedding_net(x1)
            output2 = self.embedding_net(x2)
            output3 = self.embedding_net(x3)
            return output1, output2, output3

        def get_embedding(self, x):
            return self.embedding_net(x)

##################### archived ######################
class MobileViT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 加载预训练MobileViT backbone (移除原分类头)
        self.backbone = get_model('mobilevit-xsmall_3rdparty_in1k', pretrained=True, ).backbone

        # 自定义分类头 (输入维度需匹配MobileViT输出)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局池化
            nn.Flatten(),
            nn.Linear(384, 64),  # MobileViT-small最后层通道数为640；MobileViT-xsmall最后层通道数为384
            nn.PReLU(),
            nn.Linear(64, num_classes)  # 二分类输出
        )

    def forward(self, x):
        all_features = self.backbone(x)
        features = all_features[-1]
        return self.head(features)  # 输出 [B, 2]
