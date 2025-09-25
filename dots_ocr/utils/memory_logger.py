import os
import psutil
import tracemalloc
import gc
from datetime import datetime

class MemoryLogger:
    def __init__(self, log_file: str = "parse_pdf_stream_enhanced.log", to_stdout: bool = False):
        self.log_file = log_file
        self.to_stdout = to_stdout
        tracemalloc.start()

        if os.path.dirname(self.log_file):
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log(self, stage: str):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        rss_mb = mem_info.rss / 1024 / 1024
        vms_mb = mem_info.vms / 1024 / 1024

        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024

        obj_count = len(gc.get_objects())

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = (
            f"[{ts}] Stage: {stage} | "
            f"RSS={rss_mb:.2f} MB | VMS={vms_mb:.2f} MB | "
            f"PyAlloc={current_mb:.2f} MB (peak {peak_mb:.2f} MB) | "
            f"Objects={obj_count}"
        )

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        if self.to_stdout:
            print(log_line)

    def reset_tracemalloc(self):
        tracemalloc.reset_peak()
