import time
import threading
import psutil
import os


class Monitor:
    """Tracks inference server metrics in real time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_tokens = 0
        self._ttft_sum = 0.0
        self._tps_sum = 0.0
        self._start_time = time.time()

    def record_request(self, tokens_generated, ttft_ms, tokens_per_sec):
        with self._lock:
            self._total_requests += 1
            self._total_tokens += tokens_generated
            self._ttft_sum += ttft_ms
            self._tps_sum += tokens_per_sec

    def get_metrics(self):
        with self._lock:
            uptime = time.time() - self._start_time
            process = psutil.Process(os.getpid())
            mem = process.memory_info()
            peak = getattr(mem, "peak_wset", None) or mem.rss

            avg_ttft = self._ttft_sum / self._total_requests if self._total_requests > 0 else 0
            avg_tps = self._tps_sum / self._total_requests if self._total_requests > 0 else 0

            return {
                "uptime_seconds": round(uptime, 1),
                "total_requests": self._total_requests,
                "total_tokens_generated": self._total_tokens,
                "avg_ttft_ms": round(avg_ttft, 1),
                "avg_tokens_per_sec": round(avg_tps, 1),
                "memory_mb": round(peak / (1024 * 1024), 1),
            }