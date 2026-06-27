import os
import torch
import torch.nn as nn
from torchvision import models

def export_to_onnx(model_name="mobilenet_v3", num_classes=3, weights_path=None, export_path="phase_6_deployment/model.onnx"):
    """
    Exports a trained lightweight model checkpoint to ONNX format for deployment.
    """
    model_name = model_name.lower()
    if model_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == "shufflenet_v2":
        model = models.shufflenet_v2_x1_0(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture {model_name}")

    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print("Warning: No weights found. Exporting randomly initialized/untrained model structure.")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    print(f"Exporting model to ONNX format at {export_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("ONNX export complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX.")
    parser.add_argument("--model", type=str, default="mobilenet_v3")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--out", type=str, default="phase_6_deployment/model.onnx")
    args = parser.parse_args()
    
    export_to_onnx(model_name=args.model, weights_path=args.weights, export_path=args.out)
