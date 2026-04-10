import time
import torch
import json
import os
import statistics

class InferenceProfiler:
    def __init__(self, use_cuda=True):
        self.times = []            # 每个 batch 推理时间
        self.image_count = 0       # 总图像数
        self.batch_sizes = []      # 用于计算平均 batch size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.max_memory = 0        # 最大显存（MB）
        self.memory_samples = []   # 每个 batch 的当前显存占用（MB）

    def start_timer(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.start_time = time.time()

    def stop_timer(self, batch_size=1):
        if self.use_cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)
        self.image_count += batch_size
        self.batch_sizes.append(batch_size)

        if self.use_cuda:
            # 当前显存（非峰值）
            current_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            self.memory_samples.append(current_mem)

            # 峰值显存
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.max_memory = max(self.max_memory, peak_mem)
            torch.cuda.reset_peak_memory_stats()

    def summary(self, save_name='inference_profile'):
        save_path = f"inference_profile/{save_name}.json"
        total_time = sum(self.times)
        avg_time = total_time / self.image_count if self.image_count else 0
        max_time = max(self.times) if self.times else 0
        min_time = min(self.times) if self.times else 0
        std_time = statistics.stdev(self.times) if len(self.times) >= 2 else 0.0
        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 1
        fps = self.image_count / total_time if total_time > 0 else 0
        avg_memory = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else None

        result = {
            'total_images': self.image_count,
            'batch_size_avg': round(avg_batch_size, 2),
            'total_inference_time_sec': round(total_time, 6),
            'avg_time_per_image_sec': round(avg_time, 6),
            'min_time_per_batch_sec': round(min_time, 6),
            'max_time_per_batch_sec': round(max_time, 6),
            'std_time_per_batch_sec': round(std_time, 6),
            'fps': round(fps, 2),
            'max_gpu_memory_MB': round(self.max_memory, 2) if self.use_cuda else None,
            'avg_gpu_memory_MB': round(avg_memory, 2) if avg_memory is not None else None
        }

        print("\n=== Inference Profile Summary ===")
        for k, v in result.items():
            print(f"{k}: {v}")

        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nSaved profiling results to {save_path}\n")

        return result
