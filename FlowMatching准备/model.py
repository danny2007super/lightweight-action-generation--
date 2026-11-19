"""Compact image-conditioned conditional flow matching policy.

Straight path: x_t=(1-t)*noise+t*action; target velocity=action-noise.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


class FlowPolicy(nn.Module):
    def __init__(self, *, state_dim=14, action_dim=14, horizon=30, hidden=256,
                 layers=4, heads=8, pretrained=True):
        super().__init__()
        self.action_dim, self.horizon = action_dim, horizon
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.vision = nn.Sequential(*list(backbone.children())[:-1])
        self.image_proj = nn.Linear(512, hidden)
        self.state_proj = nn.Linear(state_dim, hidden)
        self.action_proj = nn.Linear(action_dim, hidden)
        self.time_proj = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.position = nn.Parameter(torch.zeros(1, horizon, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.denoiser = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(hidden, action_dim)

    def velocity(self, images, state, noisy, time):
        # Same ResNet18 is shared by both cameras; input is [B,2,3,H,W].
        batch, cameras = images.shape[:2]
        features = self.vision(images.reshape(batch * cameras, *images.shape[2:])).flatten(1)
        visual_tokens = self.image_proj(features.reshape(batch, cameras, -1))
        state_token = self.state_proj(state).unsqueeze(1)
        time_token = self.time_proj(time.reshape(batch, 1)).unsqueeze(1)
        action_tokens = self.action_proj(noisy) + self.position
        tokens = torch.cat((visual_tokens, state_token, time_token, action_tokens), dim=1)
        return self.out(self.denoiser(tokens)[:, 4:])

    def flow_loss(self, images, state, actions, padding):
        noise = torch.randn_like(actions)
        time = torch.rand(actions.shape[0], device=actions.device)
        mixture = (1 - time[:, None, None]) * noise + time[:, None, None] * actions
        target = actions - noise
        prediction = self.velocity(images, state, mixture, time)
        valid = (~padding).unsqueeze(-1)
        return ((prediction.float() - target.float()).square() * valid).sum() / (valid.sum() * actions.shape[-1])

    @torch.no_grad()
    def sample(self, images, state, steps=10):
        if steps < 1:
            raise ValueError("steps must be positive")
        x = torch.randn(state.shape[0], self.horizon, self.action_dim,
                        device=state.device, dtype=state.dtype)
        for i in range(steps):
            t = torch.full((state.shape[0],), i / steps, device=state.device)
            x = x + self.velocity(images, state, x, t).float() / steps
        return x


def prepare_images(images, size=224):
    batch, cameras = images.shape[:2]
    height, width = (size, size) if isinstance(size, int) else tuple(size)
    images = images.reshape(batch * cameras, *images.shape[2:])
    if images.shape[-2:] != (height, width):
        images = F.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)
    images = images.reshape(batch, cameras, 3, height, width)
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
    return (images - mean) / std
