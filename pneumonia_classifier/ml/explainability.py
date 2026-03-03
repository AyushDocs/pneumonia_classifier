import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.fwd_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        # Vectorized weighting: (1, C, H, W) * (C, 1, 1) -> (1, C, H, W)
        activations = self.activations * pooled_gradients.view(1, -1, 1, 1)

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap) + 1e-10
        return heatmap.detach().cpu().numpy()

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

def apply_heatmap(image_path_or_pil, heatmap, alpha=0.4):
    if isinstance(image_path_or_pil, str):
        img = cv2.imread(image_path_or_pil)
    else:
        img = cv2.cvtColor(np.array(image_path_or_pil), cv2.COLOR_RGB2BGR)

    height, width, _ = img.shape
    heatmap_resized = cv2.resize(heatmap, (width, height))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
    return superimposed_img

def get_medical_heatmap(model, input_tensor, original_image, target_layer_name=None):
    """
    Generates a Grad-CAM heatmap for the given CNN model and input.
    """
    if target_layer_name is None:
        target_layer_name = "convolution_block9"

    # Find the target layer
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break

    if target_layer is None:
        # Fallback: find the last 2D convolutional layer
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

    if target_layer is None:
        return None

    # Generate Heatmap
    cam = GradCAM(model, target_layer)
    try:
        heatmap = cam.generate_heatmap(input_tensor)
        combined_img = apply_heatmap(original_image, heatmap)
    finally:
        cam.remove_hooks()

    return combined_img



