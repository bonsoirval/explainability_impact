Viewed download_dataset.py:14-40

To progress this thesis task under the **CRISP-DM** methodology, here are the immediate next steps:

### 1. Verify the Benchmark Output (Phase 5: Evaluation)
Once your background execution of `./reproduce.sh` completes, verify that the pipeline successfully created the output results:
* Check `phase_5_evaluation/results/` for:
  - `mobilenet_v3_benchmark_report.csv` (contains the latency, RAM, and VRAM measurements).
  - `mobilenet_v3_explanations.png` (the multi-panel plot comparing Saliency, Grad-CAM, LIME, and SHAP).
* Check `phase_6_deployment/` for:
  - `mobilenet_v3.onnx` (the exported model graph for edge compilation).

### 2. Transition to the Real PlantVillage Dataset (Phase 2 & 3)
Currently, the pipeline runs on **mock/synthetic leaf images** to ensure the code works end-to-end. For your actual research:
* Download the official **PlantVillage Dataset** (specifically target one crop crop like *Tomato* or *Potato* leaf diseases to keep the volume manageable).
* Unpack it into `data/PlantVillage/`. The directory structure will automatically be parsed by `phase_3_data_preparation/preprocess.py`.

### 3. Run Benchmarks on both Architectures (Phase 4 & 5)
Run the pipeline for both of your lightweight architectures to generate the comparative datasets:
```bash
# Run for MobileNetV3
python run_pipeline.py --model mobilenet_v3 --dataset_path data/PlantVillage --epochs 5 --runs 5

# Run for ShuffleNetV2
python run_pipeline.py --model shufflenet_v2 --dataset_path data/PlantVillage --epochs 5 --runs 5
```

### 4. Construct the Thesis Trade-Off Matrix (Phase 6)
Combine the results of both models and write your analysis comparing:
* **Computational Cost vs. Visual Faithfulness:** Does the fast Vanilla Saliency Map or Grad-CAM localize the leaf disease spots as accurately as the computationally heavy SHAP or LIME?
* **Resource Bounds:** Determine if LIME or SHAP can physically fit on an edge node with low RAM (e.g., $<512\text{ MB}$), or if you must resort to lightweight options like Vanilla Saliency and Grad-CAM.cd 


### **How to Run run_pipeline.py**
python run_pipeline.py --model mobilenet_v3 --epochs 5 --runs 10