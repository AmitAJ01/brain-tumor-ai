"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) Explainer
Generates visual explanations for CNN decisions on Brain MRI images.
Workflow: MRI -> EfficientNet -> Predicted Class -> Gradients at Target Layer ->
          Weighted Heatmap -> Jet Colormap -> MRI Alpha Blend.
"""

from typing import Tuple, Optional
import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAMExplainer:
    """
    Grad-CAM Explainer for Convolutional Neural Networks.
    Hooks into intermediate feature maps to compute gradient-weighted class activations.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.features[-1]
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple where element 0 contains gradients w.r.t layer output
            self.gradients = grad_out[0].detach()

        self._hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
        original_image_uint8: Optional[np.ndarray] = None,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Generate Grad-CAM heatmap and overlay image.

        Args:
            input_tensor: 4D Tensor (1, C, H, W)
            target_class_idx: Integer index of target class (if None, argmax predicted class)
            original_image_uint8: (H, W) or (H, W, 3) original MRI slice
            alpha: Blending weight between original image and colormap

        Returns:
            Tuple of:
              - raw_heatmap: 2D float array [0.0, 1.0]
              - overlay_bgr: 3D uint8 array (H, W, 3) BGR image
              - overlay_base64: Base64 data URL string for web rendering
        """
        self.model.eval()
        self.model.zero_grad()

        input_var = input_tensor.clone().requires_grad_(True)
        device = next(self.model.parameters()).device
        input_var = input_var.to(device)

        # Forward pass
        logits = self.model(input_var)
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        # Backward pass on the specific class score
        score = logits[0, target_class_idx]
        score.backward()

        # Compute channel weights via global average pooling of gradients: alpha_k = (1/Z) * sum(grad)
        gradients = self.gradients[0]     # (C, H_feat, W_feat)
        activations = self.activations[0] # (C, H_feat, W_feat)

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True) # (C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0) # (H_feat, W_feat)
        cam = F.relu(cam) # Pass through ReLU (only features that contribute positively)

        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        # Determine target image dimensions
        if original_image_uint8 is not None:
            H, W = original_image_uint8.shape[:2]
        else:
            H, W = input_tensor.shape[2], input_tensor.shape[3]

        # Upsample heatmap to full image resolution
        heatmap_resized = cv2.resize(cam_norm, (W, H), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

        # Colorize with Jet colormap
        heatmap_uint8 = (heatmap_resized * 255.0).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Base image formatting
        if original_image_uint8 is not None:
            if original_image_uint8.ndim == 2:
                base_bgr = cv2.cvtColor(original_image_uint8, cv2.COLOR_GRAY2BGR)
            elif original_image_uint8.shape[2] == 1:
                base_bgr = cv2.cvtColor(original_image_uint8[:, :, 0], cv2.COLOR_GRAY2BGR)
            else:
                base_bgr = cv2.cvtColor(original_image_uint8, cv2.COLOR_RGB2BGR)
        else:
            # Reconstruct from input tensor
            disp = (input_tensor[0].mean(dim=0).cpu().numpy() * 255.0).astype(np.uint8)
            base_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        base_bgr = cv2.resize(base_bgr, (W, H))

        # Alpha blend: original * (1 - alpha) + heatmap * alpha
        overlay_bgr = cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        # Encode to base64
        pil_img = Image.fromarray(overlay_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_str}"

        return heatmap_resized, overlay_rgb, data_url
