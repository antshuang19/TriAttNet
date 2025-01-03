import torch
import torch.nn as nn
import torchvision.models as models

class VDSRLoss(nn.Module):
    def __init__(self):
        super(VDSRLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        return self.mse_loss(output, target)


class PerceptualLoss(nn.Module):
    def __init__(self, layers=[2, 7, 12]):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.vgg = vgg[:max(layers)+1] 
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss() 

    def forward(self, output, target):
        output_features = self.get_vgg_features(output)
        target_features = self.get_vgg_features(target)
        loss = sum(self.mse_loss(o_f, t_f) for o_f, t_f in zip(output_features, target_features))
        return loss

    def get_vgg_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in [2, 7, 12]: 
                features.append(x)
        return features


class TotalVariationLoss(nn.Module):
    def __init__(self, weight=1e-6):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        tv_loss = torch.sum(torch.abs(torch.diff(x, dim=3))) + torch.sum(torch.abs(torch.diff(x, dim=2)))
        return self.weight * tv_loss


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = VDSRLoss()  
        self.perceptual_loss = PerceptualLoss()  
        self.tv_loss = TotalVariationLoss() 

        self.alpha = alpha  
        self.beta = beta  
        self.gamma = gamma  

    def forward(self, output, target):
        device = output.device
        self.mse_loss = self.mse_loss.to(device)
        self.perceptual_loss = self.perceptual_loss.to(device)
        self.tv_loss = self.tv_loss.to(device)
        mse = self.mse_loss(output, target)
        perceptual = self.perceptual_loss(output, target)
        tv = self.tv_loss(output)
        total_loss = self.alpha * mse + self.beta * perceptual + self.gamma * tv
        return total_loss
