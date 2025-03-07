import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import models
from torchvision.models import VGG16_Weights

# Custom color loss - Blue loss
def blue_loss(images):
    # How far are the blue channel values to 0.9:
    error = torch.abs(images[:,2] - 0.9).mean() # [:,2] -> all images in batch, only the blue channel
    return error

# Custom color loss - Yellow-Purple gradient loss
def custom_color_loss(images):
    # Encourage yellow (high R&G, low B) in top half and purple (high R&B, low G) in bottom half
    height = images.shape[2]
    top_half = images[:, :, :height//2, :]
    bottom_half = images[:, :, height//2:, :]
    
    # Yellow loss for top half (high R&G, low B)
    top_yellow_loss = (
        torch.abs(top_half[:,0] - 0.9).mean() +  # Red high
        torch.abs(top_half[:,1] - 0.9).mean() +  # Green high
        torch.abs(top_half[:,2] - 0.1).mean()    # Blue low
    )
    
    # Purple loss for bottom half (high R&B, low G)
    bottom_purple_loss = (
        torch.abs(bottom_half[:,0] - 0.9).mean() +  # Red high
        torch.abs(bottom_half[:,1] - 0.1).mean() +  # Green low
        torch.abs(bottom_half[:,2] - 0.9).mean()    # Blue high
    )
    
    return (top_yellow_loss + bottom_purple_loss) / 2

# Gradient Loss 
def gradient_loss(img):
    """Computes edge preservation loss using image gradients."""
    def get_gradient(img):
        # Ensure the convolution kernel matches input dimensions
        kernel = torch.tensor([[[-1, 1], [-1, 1]]], dtype=img.dtype, device=img.device)
        kernel = kernel.unsqueeze(0).repeat(img.shape[1], 1, 1, 1)
        
        gx = F.conv2d(img, kernel, padding=1, groups=img.shape[1])
        gy = F.conv2d(img, kernel.transpose(2, 3), padding=1, groups=img.shape[1])
        return gx, gy
    
    img_gx, img_gy = get_gradient(img)
    return F.mse_loss(img_gx, torch.zeros_like(img_gx)) + F.mse_loss(img_gy, torch.zeros_like(img_gy))

# EPSLoss =============================================================================
# Edge-Preserving Structural Loss
class EPSLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=1.0):
        """Edge-Preserving Structural Loss"""
        super(EPSLoss, self).__init__()
        self.lambda1 = lambda1  # Weight for Smooth L1 Loss
        self.lambda2 = lambda2  # Weight for Edge Loss
        self.lambda3 = lambda3  # Weight for SSIM Loss
        self.smooth_l1 = nn.SmoothL1Loss()
        self.ssim = StructuralSimilarityIndexMeasure()
    
    def forward(self, img):
        """Computes the loss directly on the input image."""
        # Handle each channel separately for gradient loss
        loss_edge = sum(gradient_loss(img[:, i:i+1, :, :]) for i in range(img.shape[1]))
        
        # Other losses can handle multi-channel input
        loss_smooth = self.smooth_l1(img, torch.zeros_like(img))
        loss_ssim = 1 - self.ssim(img, torch.zeros_like(img))
        
        total_loss = self.lambda1 * loss_smooth + self.lambda2 * loss_edge + self.lambda3 * loss_ssim
        return total_loss

# TAPLoss =============================================================================
# Texture-Aware Perceptual Loss
def gram_matrix(feature_map):
    """Computes the Gram matrix for texture loss."""
    (b, c, h, w) = feature_map.shape
    features = feature_map.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)

class TAPLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=1.0, weights=VGG16_Weights.IMAGENET1K_V1):
        """Texture-Aware Perceptual Loss"""
        super(TAPLoss, self).__init__()
        self.lambda1 = lambda1  # Perceptual Loss Weight
        self.lambda2 = lambda2  # Texture Loss Weight
        self.lambda3 = lambda3  # Edge Loss Weight
        
        # Load pre-trained VGG with updated weights parameter
        vgg = models.vgg16(weights=weights).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:9]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False  # Freeze VGG weights
    
    def forward(self, img):
        """Computes the loss based on perceptual, texture, and edge preservation."""
        zero_img = torch.zeros_like(img)
        img_features = self.vgg_layers(img)
        zero_features = self.vgg_layers(zero_img)
        
        # Perceptual Loss
        loss_perceptual = F.mse_loss(img_features, zero_features)
        
        # Texture Loss (Gram Matrix)
        loss_texture = F.mse_loss(gram_matrix(img_features), gram_matrix(zero_features))
        
        # Edge Loss using Sobel filters instead of gradient
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img.device).float()
        
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1)
        
        edge_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
        edge_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
        
        loss_edge = torch.mean(torch.abs(edge_x) + torch.abs(edge_y))

        # Weighted Sum
        total_loss = (
            self.lambda1 * loss_perceptual +
            self.lambda2 * loss_texture +
            self.lambda3 * loss_edge
        )
        return total_loss

# SALoss =============================================================================    
# Laplacian Pyramid Loss
def laplacian_pyramid(img, levels=3):
    """Computes a Laplacian Pyramid loss by progressively downsampling and computing differences."""
    pyramid = []
    current = img
    for _ in range(levels):
        down = F.avg_pool2d(current, kernel_size=2, stride=2)
        up = F.interpolate(down, size=current.shape[2:], mode='bilinear', align_corners=False)
        pyramid.append(current - up)
        current = down
    return pyramid

# Structure-Aware Loss
class SALoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=1.0):
        """Structure-Aware Loss using Laplacian Pyramid and Edge Detection."""
        super(SALoss, self).__init__()
        self.lambda1 = lambda1  # Laplacian Pyramid Loss Weight
        self.lambda2 = lambda2  # Edge Loss Weight
        self.lambda3 = lambda3  # Pixel-Level Smooth L1 Loss
        self.smooth_l1 = nn.SmoothL1Loss()
    
    def forward(self, img):
        """Computes the loss based on structural, edge, and pixel similarity."""
        zero_img = torch.zeros_like(img)
        
        # Laplacian Pyramid Loss
        pyramid_loss = sum(F.mse_loss(lap, torch.zeros_like(lap)) for lap in laplacian_pyramid(img))
        
        # Edge Loss using Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img.device).float()
        
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1)
        
        edge_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
        edge_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
        
        edge_loss = torch.mean(torch.abs(edge_x) + torch.abs(edge_y))
        
        # Pixel-Level Loss
        pixel_loss = self.smooth_l1(img, zero_img)
        
        # Weighted Sum
        total_loss = (
            self.lambda1 * pyramid_loss +
            self.lambda2 * edge_loss +
            self.lambda3 * pixel_loss
        )
        return total_loss    