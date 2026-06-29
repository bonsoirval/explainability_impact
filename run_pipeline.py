import os
import argparse
import torch
import pandas as pd
import numpy as np

# Import CRISP-DM phases modules
from phase_2_data_understanding.download_dataset import download_real
from phase_3_data_preparation.preprocess import get_dataloaders
from phase_4_modeling.train import get_lightweight_model, train_model
from phase_5_evaluation.explainers import ExplainabilitySuite
from phase_5_evaluation.benchmark import XAIBenchmarker
from phase_5_evaluation.utils import (
    plot_explanations,
    plot_benchmark_distributions,
    plot_training_curves,
    plot_confusion_matrix,
    plot_multi_class_explanations,
    plot_latency_vs_ram,
    plot_run_over_run,
    plot_cv_stability,
    plot_confidence_bar,
)
from phase_6_deployment.export_onnx import export_to_onnx

def main():
    parser = argparse.ArgumentParser(description="CRISP-DM XAI Computational Efficiency Benchmarking Pipeline")
    parser.add_argument("--model", type=str, default="mobilenet_v3", choices=["mobilenet_v3", "shufflenet_v2"])
    parser.add_argument("--dataset_path", type=str, default="data/PlantVillage")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--results_dir", type=str, default="phase_5_evaluation/results")
    parser.add_argument("--methods", type=str, nargs="+", default=["Saliency", "Grad-CAM", "LIME", "SHAP"],
                        choices=["Saliency", "Grad-CAM", "LIME", "SHAP"],
                        help="List of XAI methods to run/benchmark. E.g. --methods Saliency Grad-CAM")
    parser.add_argument("--crop", type=str, default="Tomato",
                        help="Crop prefix to filter class folders (default: Tomato).")
    parser.add_argument("--kaggle_slug", type=str, default="abdallahalidev/plantvillage-dataset",
                        help="Kaggle dataset slug (owner/dataset-name).")
    parser.add_argument("--image_type", type=str, default="color",
                        choices=["color", "segmented", "grayscale"],
                        help="Image variant to use from the PlantVillage zip (default: color).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[PHASE 1] Starting CRISP-DM Pipeline on Device: {device}")

    # PHASE 2: Data Understanding / Ingestion
    print("\n[PHASE 2] Data Understanding (Downloading real PlantVillage Tomato dataset)...")
    download_real(
        dest=args.dataset_path,
        crop=args.crop,
        kaggle_slug=args.kaggle_slug,
        image_type=args.image_type,
    )

    # PHASE 3: Data Preparation
    print("\n[PHASE 3] Data Preparation (Splitting & preprocessing)...")
    train_loader, val_loader, class_names = get_dataloaders(root_dir=args.dataset_path, batch_size=4)
    print(f"Loaded {len(class_names)} classes: {class_names}")

    # PHASE 4: Modeling
    print(f"\n[PHASE 4] Modeling (Fine-tuning {args.model})...")
    model = get_lightweight_model(args.model, num_classes=len(class_names), pretrained=True)
    model, checkpoint_path, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        save_dir="phase_4_modeling/models",
        model_name=args.model
    )

    # Plot training curves (loss + accuracy per epoch)
    curves_path = os.path.join(args.results_dir, f"{args.model}_training_curves.png")
    plot_training_curves(history, model_name=args.model, save_path=curves_path)
    print(f"Saved training curves to {curves_path}")

    # Plot confusion matrix on validation set
    cm_path = os.path.join(args.results_dir, f"{args.model}_confusion_matrix.png")
    plot_confusion_matrix(model, val_loader, class_names, device, cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # PHASE 5: Evaluation
    print("\n[PHASE 5] Evaluation (Benchmarking XAI efficiency)...")
    model.eval()
    suite = ExplainabilitySuite(model)
    benchmarker = XAIBenchmarker(suite)

    # Get a sample for explanation
    img_tensor, label = val_loader.dataset[0]
    input_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()

    print(f"Visualizing explanation maps for predicted class: {class_names[pred_class]}")

    # Softmax confidence bar for the explained sample
    conf_path = os.path.join(args.results_dir, f"{args.model}_confidence_bar.png")
    plot_confidence_bar(model, input_tensor, class_names, device, conf_path)
    print(f"Saved confidence bar to {conf_path}")

    saliency_map = suite.explain_saliency(input_tensor, pred_class) if "Saliency" in args.methods else np.zeros((224, 224))
    gradcam_map = suite.explain_gradcam(input_tensor, pred_class) if "Grad-CAM" in args.methods else np.zeros((224, 224))
    
    if "LIME" in args.methods:
        lime_img, _ = suite.explain_lime(input_tensor, pred_class, num_samples=50)
    else:
        # Fallback to a normalized raw image representation
        lime_img = img_tensor.clone().squeeze()
        if lime_img.is_cuda:
            lime_img = lime_img.cpu()
        lime_img = lime_img.permute(1, 2, 0).numpy()
        lime_img = np.clip(lime_img, 0, 1)

    shap_map = suite.explain_shap(input_tensor, pred_class, n_samples=30) if "SHAP" in args.methods else np.zeros((224, 224))

    plot_path = os.path.join(args.results_dir, f"{args.model}_explanations.png")
    plot_explanations(
        original_tensor=img_tensor,
        saliency_map=saliency_map,
        gradcam_map=gradcam_map,
        lime_img=lime_img,
        shap_map=shap_map,
        save_path=plot_path
    )
    print(f"Saved visual report to {plot_path}")

    # Multi-class explanation grid (Original | Saliency | Grad-CAM per disease class)
    multi_path = os.path.join(args.results_dir, f"{args.model}_multi_class_explanations.png")
    print("Generating multi-class explanation grid (Saliency + Grad-CAM per class)...")
    plot_multi_class_explanations(
        suite=suite,
        val_dataset=val_loader.dataset,
        class_names=class_names,
        device=device,
        save_path=multi_path,
    )
    print(f"Saved multi-class explanation grid to {multi_path}")

    # Run Benchmark stats
    results_df, raw_runs_df = benchmarker.benchmark_sample(
        input_tensor, pred_class, runs=args.runs, methods_to_run=args.methods
    )
    
    # Save reports
    report_path = os.path.join(args.results_dir, f"{args.model}_benchmark_report.csv")
    results_df.to_csv(report_path, index=False)
    
    raw_report_path = os.path.join(args.results_dir, f"{args.model}_raw_runs.csv")
    raw_runs_df.to_csv(raw_report_path, index=False)
    print(f"Saved aggregated benchmark report to {report_path}")
    print(f"Saved raw run-by-run details to {raw_report_path}")
    
    # Plot benchmark distributions
    dist_plot_path = os.path.join(args.results_dir, f"{args.model}_benchmark_distributions.png")
    plot_benchmark_distributions(raw_runs_df, dist_plot_path)
    print(f"Saved computational benchmarking distributions plot to {dist_plot_path}")

    # Latency vs Peak RAM scatter
    lv_path = os.path.join(args.results_dir, f"{args.model}_latency_vs_ram.png")
    plot_latency_vs_ram(raw_runs_df, model_name=args.model, save_path=lv_path)
    print(f"Saved latency vs RAM scatter to {lv_path}")

    # Run-over-run latency line chart
    ror_path = os.path.join(args.results_dir, f"{args.model}_run_over_run.png")
    plot_run_over_run(raw_runs_df, model_name=args.model, save_path=ror_path)
    print(f"Saved run-over-run chart to {ror_path}")

    # Coefficient of Variation (stability) bar chart
    cv_path = os.path.join(args.results_dir, f"{args.model}_cv_stability.png")
    plot_cv_stability(raw_runs_df, model_name=args.model, save_path=cv_path)
    print(f"Saved CV stability chart to {cv_path}")
    
    print("\n" + "="*60)
    print(f"BENCHMARK RESULTS FOR {args.model.upper()}")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)

    # PHASE 6: Deployment
    print("\n[PHASE 6] Deployment (Exporting to ONNX)...")
    export_path = os.path.join("phase_6_deployment", f"{args.model}.onnx")
    export_to_onnx(model_name=args.model, num_classes=len(class_names), weights_path=checkpoint_path, export_path=export_path)
    
    print("\nCRISP-DM Pipeline Run Completed Successfully!")

if __name__ == "__main__":
    main()
