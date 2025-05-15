import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class CLIPFeatureExtractor(nn.Module):
    def __init__(self, device, model_name="openai/clip-vit-base-patch32"):
        super(CLIPFeatureExtractor, self).__init__()
        # Load CLIP model
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # CLIP ViT-B/32 input resolution
        self.input_resolution = 224

        # Get feature dimension
        self.output_dim = self.model.config.projection_dim

    def forward(self, x):
        # Resize input to CLIP expected dimensions
        x = F.interpolate(x,
                          size=(self.input_resolution, self.input_resolution),
                          mode='bilinear',
                          align_corners=False)

        # Convert tensor range from [0,1] to [-1,1]
        x = x * 2 - 1

        # Extract features
        with torch.no_grad():
            vision_outputs = self.model.vision_model(x)
            image_features = self.model.visual_projection(vision_outputs[1])

        return image_features

    @property
    def feature_dim(self):
        """Return feature dimension"""
        return self.output_dim