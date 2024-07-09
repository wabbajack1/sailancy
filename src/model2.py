import torch
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import wandb
import os
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
        self.model = fcn_resnet50(pretrained=True) # load the pre-trained model
        self.dropout = torch.nn.Dropout(args.dropout_rate)
        
        # Modify the final classifier to have the desired number of classes
        self.model.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        
        # set the backbone and classifier
        self.backbone = self.model.backbone
        self.decoder = self.model.classifier # pre-defined classifier is already full convolutional layer
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
        x_features = self.dropout(x_features)
        x_decoded = self.decoder(x_features) # (B, 1, H/8, W/8)
        x_features = self.dropout(x_features)
        x = torch.nn.functional.interpolate(x_decoded, size=x.shape[-2:], mode='bilinear', align_corners=False)

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