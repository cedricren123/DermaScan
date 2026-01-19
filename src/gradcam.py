import torch
import torch.nn.functional as F

def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model for Grad-CAM.")
    return last_conv

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer or find_last_conv_layer(model)
        self.activations = None
        self.gradients = None

        # Hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        input_tensor: shape [1,3,H,W]
        returns: CAM tensor shape [H,W] normalized 0..1 (CPU)
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)  # [1, num_classes]
        score = logits[0, class_idx]

        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        # activations: [1, C, h, w], gradients: [1, C, h, w]
        grads = self.gradients
        acts = self.activations

        # Global average pooling over spatial dims
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=False)  # [1,h,w]
        cam = F.relu(cam)[0]  # [h,w]

        cam = self._normalize_cam(cam).cpu()
        return cam
