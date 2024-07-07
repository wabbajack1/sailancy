import torch
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import wandb
import os
from torchvision.models import resnet50
import torch.nn as nn

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    return torch.matmul( (gauss / gauss.sum()).unsqueeze(-1), (gauss / gauss.sum()).unsqueeze(-1).t()) # (window_size, 1) x (1, window_size) = (window_size, window_size), i.e. cross-correlation kernel


class Eye_Fixation(torch.nn.Module):
    def __init__(self, args, window_size:int = 25, sigma:float = 11.2, path="cv2_project_data"):
        super(Eye_Fixation, self).__init__()

        # Freeze the backbone
        resnet = resnet50(pretrained=True)

        # Remove the fully connected layer (classification head)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2)
        )
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        # for post-processing
        self.window_size = window_size
        self.sigma = sigma
        self.weight_kernel = torch.nn.Parameter(gaussian(window_size, sigma), requires_grad=False) # (window_size, window_size)

        # Center bias for post-processing
        center_bias = torch.tensor(np.load(os.path.join(path, "center_bias_density.npy")))
        log_center_bias = torch.log(center_bias)
        self.center_bias = torch.nn.Parameter(log_center_bias)  # 224 depends on the input size
        self.dropout = torch.nn.Dropout(args.dropout_rate)

        try:
            wandb.watch(self, log="all", log_freq=100)
        except:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_features = self.backbone(x) # (B, 2048, H/32, W/32)
        x_features = self.dropout(x_features)
        x_decoded = self.decoder(x_features) # (B, 1, H/8, W/8)
        x_features = self.dropout(x_features)
        x = torch.nn.functional.interpolate(x_decoded, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # post-processing raw decoder outputs
        smoothed_output = torch.nn.functional.conv2d(x, self.weight_kernel.view(1, 1, self.window_size, self.window_size), padding="same")
        smoothed_output += self.center_bias

        return smoothed_output






if __name__ == '__main__':
    from argparse import Namespace
    args = Namespace(dropout_rate=0.5)
    model = Eye_Fixation(args=args)
    x = torch.randn(3, 3, 224, 224) # (B, C, H, W)
    out = model(x) # (B, 1, H, W)
    print(out.shape)