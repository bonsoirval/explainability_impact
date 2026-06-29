import time
import tracemalloc
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable
from phase_5_evaluation.explainers import ExplainabilitySuite

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

class XAIBenchmarker:
    def __init__(self, suite: ExplainabilitySuite):
        self.suite = suite
        self.device = suite.device

    def profile_latency_and_memory(self, explainer_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            
        tracemalloc.start()
        tracemalloc.reset_peak()
        
        start_time = time.perf_counter()
        result = explainer_func(*args, **kwargs)
        end_time = time.perf_counter()
        
        _, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        latency = end_time - start_time
        peak_ram_mb = peak_ram / (1024 * 1024)
        
        peak_vram_mb = 0.0
        if torch.cuda.is_available() and self.device.type == 'cuda':
            peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            
        return {
            "latency_sec": latency,
            "peak_ram_mb": peak_ram_mb,
            "peak_vram_mb": peak_vram_mb,
            "result": result
        }

    def benchmark_sample(self, input_tensor: torch.Tensor, target_class: int, runs: int = 3, methods_to_run: list = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        all_methods = {
            "Saliency": lambda: self.suite.explain_saliency(input_tensor, target_class),
            "Grad-CAM": lambda: self.suite.explain_gradcam(input_tensor, target_class),
            "LIME": lambda: self.suite.explain_lime(input_tensor, target_class, num_samples=50),
            "SHAP": lambda: self.suite.explain_shap(input_tensor, target_class, n_samples=30)
        }
        
        # Filter methods if specified
        if methods_to_run is not None:
            methods = {k: v for k, v in all_methods.items() if k in methods_to_run}
        else:
            methods = all_methods
            
        results = []
        raw_runs = []
        
        for name, func in methods.items():
            print(f"Benchmarking {name} over {runs} runs...")
            latencies = []
            rams = []
            vrams = []
            
            # Warmup
            try:
                self.profile_latency_and_memory(func)
            except Exception as e:
                print(f"Warmup failed for {name}: {e}")
                continue
                
            for run in range(runs):
                stats = self.profile_latency_and_memory(func)
                latency = stats["latency_sec"]
                ram = stats["peak_ram_mb"]
                vram = stats["peak_vram_mb"]
                
                latencies.append(latency)
                rams.append(ram)
                vrams.append(vram)
                
                raw_runs.append({
                    "Method": name,
                    "Run": run + 1,
                    "Latency (s)": latency,
                    "Peak RAM (MB)": ram,
                    "Peak VRAM (MB)": vram
                })
                
            results.append({
                "Method": name,
                "Latency Mean (s)": np.mean(latencies),
                "Latency Std (s)": np.std(latencies),
                "Peak RAM Mean (MB)": np.mean(rams),
                "Peak VRAM Mean (MB)": np.mean(vrams)
            })
            
        return pd.DataFrame(results), pd.DataFrame(raw_runs)
