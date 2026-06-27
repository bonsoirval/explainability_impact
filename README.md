# XAI Computational Efficiency Benchmarking

This repository implements a systematic benchmarking suite to evaluate the computational efficiency and resource overhead of Explainable AI (XAI) models when applied to lightweight neural networks for plant disease detection.

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.

## Project Structure
```
explainability_impact/
│
├── phase_1_business_understanding/
│   └── README.md              # Research goals and target metrics
│
├── phase_2_data_understanding/
│   └── download_dataset.py    # Ingestion and raw data exploration setup
│
├── phase_3_data_preparation/
│   └── preprocess.py          # Transformations, splits, and loaders
│
├── phase_4_modeling/
│   ├── train.py               # Lightweight model definition & train loops
│   └── models/                # Trained checkpoints (.pth)
│
├── phase_5_evaluation/
│   ├── explainers.py          # Saliency, Grad-CAM, LIME, SHAP wrappers
│   ├── benchmark.py           # Latency and memory profiling suite
│   ├── utils.py               # Matplotlib explanation plot helpers
│   └── results/               # Saved explanation PNGs and reports
│
├── phase_6_deployment/
│   ├── export_onnx.py         # ONNX compilation script
│   └── model.onnx             # Compiled ONNX weight files
│
├── requirements.txt           # Project dependencies
├── run_pipeline.py            # Global CRISP-DM pipeline runner
└── README.md                  # Project documentation
```

## Methodology (CRISP-DM)
1. **Business/Research Understanding**: Establishing the latency vs. trust trade-off for edge devices.
2. **Data Understanding**: Exploring class imbalance, image quality, and resolution in the PlantVillage dataset.
3. **Data Preparation**: Resizing to $224 \times 224$, normalizing with ImageNet stats, and split generation.
4. **Modeling**: Transfer learning using **MobileNetV3** and **ShuffleNetV2**.
5. **Evaluation**: Benchmarking latency, RAM/VRAM usage, and energy footprint of **Vanilla Saliency**, **Grad-CAM**, **LIME**, and **SHAP**.
6. **Deployment**: Exporting optimized models and outputting a decision matrix for deployment.

## Installation & Execution
1. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full CRISP-DM pipeline (dataset download, training, benchmarking, and export):
   ```bash
   python run_pipeline.py --model mobilenet_v3 --epochs 1 --runs 3
   ```
