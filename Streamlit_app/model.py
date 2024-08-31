
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import v2

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class NetVLAD(nn.Module):
    """NetVLAD layer implementation https://github.com/lyakaap/NetVLAD-pytorch"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        #print(x.shape)
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        #TO DO I can change the shape directly here
        x_flatten = x.view(N, C, -1)
        #print(x_flatten.shape)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        
################################################# THIS IS THE MAIN NETWORKs ####################        
class SlowFusionNetVLAD(nn.Module):
    def __init__(self, num_classes=12, vocab_size=256,dropout=0.4):
        super(SlowFusionNetVLAD, self).__init__()

        # EfficientNetV2 as a 2D Encoder
        efficient_net = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        self.encoder_2d = nn.Sequential(
            efficient_net.features,
            nn.Conv2d(1280, 192, kernel_size=1)  # Reduce the dimension to 192
        )

        # 3D Encoder
        self.encoder_3d = nn.Sequential(
            ResidualBlock(192, 256),#I think thy problem is here(shattered gradiend problem)
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )

        # NetVLAD Pooling Layer
        #TO DO cercare un modo per prendere in automatico dim==(T*C)
        self.pool_layer = NetVLAD(num_clusters=vocab_size, dim=1280,alpha=1.0)
        #self.pool_layer_after = NetVLAD(cluster_size=vocab_size//2, feature_size=256, add_batch_norm=True)

        # Classifier
        #TO DO cercare un modo per prendere in automatico in_features
        self.fc = nn.Linear(in_features=327680, out_features=num_classes)
        self.drop = nn.Dropout(p=dropout)
        #self.sigmoid = nn.Sigmoid()#If I use the sigmoid I assign to each label a p independently(2 event at the same time)


    def forward(self, x):
        B, T, C, H, W = x.shape  # x is expected to have shape (B, T, C, H, W)
        #print(x.shape)

        # Shared weights among the 5 stack
        x = x.view(B * T, C, H, W)
        x = self.encoder_2d(x)
        #print(x.shape)

        # Reshape to (B, T, C, H, W)
        _, C, H, W = x.shape
        x = x.view(B, T, C, H, W)
        #print(x.shape)

        # Permute to (B, C, T, H, W) for 3D convolution
        x = x.permute(0, 2, 1, 3, 4)
        #print(x.shape)

        # Pass through the 3D encoder
        x = self.encoder_3d(x)
        #print(x.shape)

        # Reshape for NetVLAD  # x [BS, T, D]
        #Concat temporal feature
        B, T, C, H, W = x.shape
        x=x.view(B,T*C,H,W)
        #print(x.shape)

        x=self.pool_layer(x)
        #print(x.shape)

        # Classifier
        x=self.drop(x)
        x = self.fc(x)

        return x