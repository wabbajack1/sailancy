import torch
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import wandb
import os
from torch import nn

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    return torch.matmul( (gauss / gauss.sum()).unsqueeze(-1), (gauss / gauss.sum()).unsqueeze(-1).t()) # (window_size, 1) x (1, window_size) = (window_size, window_size), i.e. cross-correlation kernel


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out += identity # skip connection
        out = self.relu(out)
        
        return out

class SEBlock(nn.Module):
    """Sequeeze and Excitation Block.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, channels, reduction=16, droput_rate=0.1):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = x.mean((2, 3), keepdim=True)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se


class FCNHead(nn.Module):
    def __init__(self, num_channels=1, dropout_rate=0.1):
        super(FCNHead, self).__init__()
        self.layer1 = ResidualBlock(2048, 512, dropout_rate=dropout_rate)
        self.se1 = SEBlock(512)
        self.layer2 = ResidualBlock(512, 256, dropout_rate=dropout_rate)
        self.se2 = SEBlock(256)
        self.layer3 = ResidualBlock(256, 128, dropout_rate=dropout_rate)
        self.se3 = SEBlock(128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.final_conv = nn.Conv2d(128, num_channels, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.dropout(x)
        x = self.final_conv(x)
        return x

class Eye_Fixation(torch.nn.Module):
    def __init__(self, args, window_size:int = 25, sigma:float = 11.2, path="cv2_project_data"):
        super(Eye_Fixation, self).__init__()

        # Freeze the backbone
        model = fcn_resnet50(pretrained=True) # load the pre-trained model
        self.dropout = torch.nn.Dropout(args.dropout_rate)
        
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        # Modify the classifier with a full convolutional layer (new head)
        model.classifier = FCNHead(num_channels=1, dropout_rate=args.dropout_rate)
        
        # set the backbone and classifier
        self.backbone = model.backbone
        self.decoder = model.classifier # pre-defined classifier is already full convolutional layer
        print(self.decoder)

        # for post-processing
        self.window_size = window_size
        self.sigma = sigma
        self.weight_kernel = torch.nn.Parameter(gaussian(window_size, sigma), requires_grad=False) # (window_size, window_size)

        # Center bias for post-processing
        center_bias = torch.tensor(np.load(os.path.join(path, "center_bias_density.npy")))
        log_center_bias = torch.log(center_bias)
        self.center_bias = torch.nn.Parameter(log_center_bias)  # 224 depends on the input size

        try:
            wandb.watch(self, log="all", log_freq=100)
        except:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_features = self.backbone(x)["out"]
        x_decoded = self.decoder(x_features) # (B, 1, H/8, W/8)
        x = torch.nn.functional.interpolate(x_decoded, size=x.shape[-2:], mode='bilinear', align_corners=False) # (B, 1, H, W)

        # post-processing raw decoder outputs
        smoothed_output = torch.nn.functional.conv2d(x, self.weight_kernel.view(1, 1, self.window_size, self.window_size), padding=self.window_size // 2)
        smoothed_output += self.center_bias

        return smoothed_output



if __name__ == '__main__':
    from argparse import Namespace
    args = Namespace(dropout_rate=0.2)

    model = Eye_Fixation(args)
    x = torch.randn(3, 3, 224, 224) # (B, C, H, W)
    out = model(x) # (B, 1, H, W)
    print(out.shape)