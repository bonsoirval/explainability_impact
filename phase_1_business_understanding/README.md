# Phase 1: Business / Research Understanding

## Research Objective
The goal of this thesis is to benchmark and analyze the computational and resource overhead of four Explainable AI (XAI) models:
1. **Vanilla Saliency Maps** (Gradient-based)
2. **Grad-CAM** (Gradient-weighted Class Activation Mapping)
3. **LIME** (Local Interpretable Model-agnostic Explanations)
4. **SHAP** (Shapley Additive exPlanations)

These will be evaluated when explaining predictions made by lightweight convolutional neural networks (**MobileNetV3** and **ShuffleNetV2**) trained on plant disease image datasets.

## Success Criteria & Target Metrics
- **Model Performance:** Lightweight models should achieve at least **90% classification accuracy** (F1-score) on the validation set.
- **Explainability Quality:** Visual explanations must highlight features relevant to the pathology (e.g. leaf spots, rust spots, discolored veins).
- **Computational Benchmark:** Capture statistical means and standard deviations across runs for:
  - **Latency (seconds):** Time required to generate an explanation map.
  - **Memory Footprint (MB):** Peak RAM/VRAM usage.
  - **Energy Footprint (mWh / CO2 emissions):** Measured using the CodeCarbon library.
