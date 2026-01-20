import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableFilters(nn.Module):
    def __init__(self):
        super().__init__()

    def adjust_gamma(self, img, gamma_param):
        # Param [0,1] -> Gamma [0.1, 5.0]
        gamma = gamma_param * 4.9 + 0.1 
        return torch.pow(img + 1e-6, gamma) # +1e-6 cegah NaN

    def adjust_exposure(self, img, exp_param):
        # Param [0,1] -> Exposure [-2, 2] stops
        exp = (exp_param - 0.5) * 4 
        return img * torch.exp(exp * 0.693)

    def adjust_contrast(self, img, cont_param):
        # Param [0,1] -> Contrast [0.5, 1.5]
        alpha = cont_param + 0.5
        mean_lum = torch.mean(img, dim=[2, 3], keepdim=True)
        return torch.clamp((img - mean_lum) * alpha + mean_lum, 0, 1)

    def forward(self, img, params):
        # params: [Batch, 3] -> (Gamma, Exposure, Contrast)
        x = self.adjust_gamma(img, params[:, 0].view(-1, 1, 1, 1))
        x = self.adjust_exposure(x, params[:, 1].view(-1, 1, 1, 1))
        x = self.adjust_contrast(x, params[:, 2].view(-1, 1, 1, 1))
        return x

class DIP_Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Input 3 channel -> Output 3 parameter
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.InstanceNorm2d(16), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 3), 
            nn.Sigmoid() # Wajib Sigmoid agar output 0-1
        )

    def forward(self, x):
        # Downsample dulu ke 256x256 biar cepat
        x_small = F.interpolate(x, size=(256, 256), mode='bilinear')
        return self.net(x_small)