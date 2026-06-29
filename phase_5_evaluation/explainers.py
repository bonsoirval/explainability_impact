import torch
import numpy as np
from captum.attr import Saliency, LayerGradCam, KernelShap
from lime import lime_image
from skimage.segmentation import mark_boundaries

class ExplainabilitySuite:
    def __init__(self, model, target_layer=None):
        self.device = next(model.parameters()).device
        self.model = model
        self.model.eval()
        
        self.saliency = Saliency(self.model)
        
        self.target_layer = target_layer if target_layer is not None else self._get_last_conv_layer(self.model)
        
        if self.target_layer is not None:
            self.gradcam = LayerGradCam(self.model, self.target_layer)
        else:
            self.gradcam = None
            
        self.lime_explainer = lime_image.LimeImageExplainer()
        self.shap_explainer = KernelShap(self.model)

    def _get_last_conv_layer(self, model):
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        return None

    def explain_saliency(self, input_tensor, target_class):
        input_tensor = input_tensor.clone().detach().requires_grad_(True).to(self.device)
        attribution = self.saliency.attribute(input_tensor, target=target_class)
        saliency_map = torch.max(torch.abs(attribution), dim=1)[0].squeeze().cpu().numpy()
        return saliency_map

    def explain_gradcam(self, input_tensor, target_class):
        if self.gradcam is None:
            raise ValueError("Grad-CAM has no conv layer targets.")
        input_tensor = input_tensor.to(self.device)
        attribution = self.gradcam.attribute(input_tensor, target=target_class)
        upsampled = LayerGradCam.interpolate(attribution, input_tensor.shape[2:])
        gradcam_map = upsampled.squeeze().cpu().detach().numpy()
        if gradcam_map.max() > gradcam_map.min():
            gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min())
        return gradcam_map

    def explain_lime(self, input_tensor, target_class, num_samples=100):
        img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        
        def batch_predict(images):
            tensors = []
            for img in images:
                tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                tensors.append(tensor)
            batch_tensor = torch.cat(tensors, dim=0)
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

        explanation = self.lime_explainer.explain_instance(
            img_np.astype(np.float64), 
            batch_predict, 
            top_labels=5, 
            hide_color=0, 
            num_samples=num_samples
        )
        temp, mask = explanation.get_image_and_mask(
            target_class, 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )
        return mark_boundaries(temp, mask), mask

    def explain_shap(self, input_tensor, target_class, n_samples=50):
        input_tensor = input_tensor.to(self.device)
        baseline = torch.zeros_like(input_tensor).to(self.device)
        attribution = self.shap_explainer.attribute(
            input_tensor, 
            baselines=baseline, 
            target=target_class, 
            n_samples=n_samples
        )
        shap_map = torch.sum(attribution.squeeze(), dim=0).cpu().detach().numpy()
        return shap_map
