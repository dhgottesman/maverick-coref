import os
import json
import time
import torch
import gc
from multiprocessing import get_context, Process, Queue
from collections import deque
from maverick import Maverick
import queue
from tqdm import tqdm

MAX_MEMORY_GB = 75
SECTION_MEMORY_GB = 3

REFINED_OUTPUT_DIR = "/home/morg/dataset/refined"
OUTPUT_DIR = "/home/morg/dataset/maverick"
FAILED_OUTPUT_FILE = None

MAX_WORKERS = None
MAX_PENDING = None
SCALE_UP_THRESHOLD = None

WORKER_IDLE_TIMEOUT = 60 * 200  # seconds

def init_model():
    model = Maverick(hf_name_or_path="sapienzanlp/maverick-mes-ontonotes", flash=True)
    print("[Worker]: After init model", flush=True)
    return model

import json

def stream_ndjson(file_path, target_index):
    with open(file_path, 'r', encoding='utf-8') as f:
        low, high = 0, f.seek(0, 2)  # Move to end to get file size
        
        # Binary search for line with target_index
        while low < high:
            mid = (low + high) // 2
            f.seek(mid)
            f.readline()  # Skip partial line
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                index = json.loads(line).get('line_index')
            except (json.JSONDecodeError, KeyError):
                high = mid
                continue

            if index is None:
                high = mid
            elif index < target_index:
                low = pos + 1
            else:
                high = mid

        # Stream from the first complete line after or at `low`
        f.seek(low)
        if low != 0:
            f.readline()  # Skip partial line

        for line in f:
            try:
                record = json.loads(line)
                i = record.get('line_index')
                if i is not None and i >= target_index:
                    yield (i, record)
            except json.JSONDecodeError:
                continue

def get_start_index(input_path, output_path, skip_indices):
    if not os.path.exists(output_path):
        with open(input_path, 'r') as f:
            return json.loads(f.readline()).get("line_index", 0)
    with open(output_path, 'rb') as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    i = json.loads(last_line).get("line_index", -1) + 1
    if len(skip_indices) > 0 and i == max(skip_indices):
        i += 1
    return i

def split_text(text, section_starts, max_chars=50000):
    section_starts = sorted(set(section_starts))
    if not section_starts or section_starts[0] != 0:
        section_starts = [0] + section_starts
    section_starts.append(len(text))
    sections = [text[section_starts[i]:section_starts[i+1]] for i in range(len(section_starts) - 1)]
    chunks, chunk_starts = [], []
    current_chunk, current_start = "", 0
    for start, section in zip(section_starts[:-1], sections):
        if len(current_chunk) + len(section) <= max_chars:
            current_chunk += section
        else:
            chunks.append(current_chunk)
            chunk_starts.append(current_start)
            current_chunk = section
            current_start = start
    if current_chunk:
        chunks.append(current_chunk)
        chunk_starts.append(current_start)
    return chunk_starts, chunks

def combine_dicts(results, chunk_starts):
    combined = {}
    for r, start in zip(results, chunk_starts):
        if isinstance(r, dict):
            for k, v in r.items():
                if k == "clusters_char_offsets":
                    adjusted = [[(t[0] + start, t[1] + start) for t in cluster] for cluster in v]
                    combined.setdefault(k, []).extend(adjusted)
                elif k == "clusters_char_text":
                    combined.setdefault(k, []).extend(v)
    return combined

def process_line(model, line):
    idx = line.get("line_index")
    text = line["text"]
    section_starts = line.get("section_starts", [])
    chunk_starts, chunks = split_text(text, section_starts)

    preds = []
    for chunk in chunks:
        try:
            pred = model.predict(chunk)
            preds.append(pred)
            del pred
            torch.cuda.empty_cache()
        except RuntimeError as e:
            torch.cuda.empty_cache()
            gc.collect()
            raise e

    combined = combine_dicts(preds, chunk_starts)

    # Sanity check
    for token_offsets, token_texts in zip(combined["clusters_char_offsets"], combined["clusters_char_text"]):
        for off, t in zip(token_offsets, token_texts):
            assert(t == text[off[0]:off[1]])

    line["coref"] = combined
    line["entity_linking"] = line.pop("spans", [])

    del text, section_starts, chunk_starts, chunks, preds, combined
    torch.cuda.empty_cache()
    gc.collect()

    return idx, line

def worker_loop(task_queue: Queue, result_queue: Queue):
    print("[Worker] Initializing model", flush=True)
    model = init_model()
    pid = os.getpid()

    try:
        while True:
            wait_start = time.time()
            task = task_queue.get()
            wait_time = time.time() - wait_start

            if task is None:
                print(f"[Worker PID {pid}] Exiting", flush=True)
                break

            if wait_time > 1:
                print(f"[Worker PID {pid}] Blocked for {wait_time:.2f} seconds waiting for task", flush=True)

            idx, line = task
            try:
                result = process_line(model, line)
                result_queue.put(("success", idx, result, pid))
            except Exception as e:
                print(f"[Worker PID {pid}] Failure (maybe CUDA OOM) encountered. Cleaning up and exiting {e}.", flush=True)
                result_queue.put(("fail", idx, None, pid))
                break
    finally:
        print(f"[Worker PID {pid}] Final cleanup: clearing CUDA & garbage", flush=True)
        torch.cuda.empty_cache()
        gc.collect()

class WorkerPoolManager:
    def __init__(self, max_workers=MAX_WORKERS):
        ctx = get_context("spawn")
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.workers = []
        self.success_log = deque(maxlen=SCALE_UP_THRESHOLD * MAX_WORKERS)
        self.max_workers = max_workers

    def scale_up(self):
        if len(self.workers) < self.max_workers:
            print("[Manager] Scaling up", flush=True)
            p = Process(target=worker_loop, args=(self.task_queue, self.result_queue))
            p.start()
            self.workers.append(p)

    def monitor_and_scale(self):
        N = SCALE_UP_THRESHOLD * len(self.workers)
        success_count = sum(1 for s in list(self.success_log)[-N:] if s == "success")
        if success_count >= N and len(self.workers) < self.max_workers:
            self.scale_up()
            self.success_log.clear()

    def shutdown(self):
        print("[Manager] Sending shutdown signals to workers...", flush=True)
        for _ in self.workers:
            self.task_queue.put(None)

        for p in self.workers:
            print(f"[Manager] Joining PID {p.pid}...", flush=True)
            p.join(timeout=10)

            if p.is_alive():
                print(f"[Manager] Worker PID {p.pid} stuck. Forcing termination...", flush=True)
                p.terminate()
                p.join(timeout=5)
            else:
                print(f"[Manager] Worker PID {p.pid} exited cleanly.", flush=True)

        self.workers.clear()

    def _process_next_result(self, pending_tasks, result_buffer, f, next_index):
        try:
            print(next_index, flush=True)
            status, res_idx, result, pid = self.result_queue.get(timeout=WORKER_IDLE_TIMEOUT)
            line, _ = pending_tasks[res_idx]
            
            if status == "success":
                self.success_log.append("success")
                result_buffer[res_idx] = (line, result)
                
                while next_index in result_buffer:
                    _, output = result_buffer.pop(next_index)
                    f.write(json.dumps(output[1]) + "\n")
                    f.flush()
                    del pending_tasks[next_index]
                    next_index += 1

                self.monitor_and_scale()

            elif status == "fail":
                # Check how many workers are alive
                live_workers = sum(1 for p in self.workers if p.is_alive())

                if live_workers == 1:
                    print(f"[Manager] Logging failed line {res_idx} (only 1 live worker)", flush=True)

                print(f"[Manager] Error at line {res_idx} from PID {pid}. Exiting.", flush=True)
                raise SystemExit(f"[Fatal Error] {res_idx} — shutting down.")

        except queue.Empty:
            print(f"[Manager] Waiting on workers (idle)...", flush=True)

        return next_index

    def process_loop(self, input_path, output_path):
        print("[Manager] Starting process loop", flush=True)

        skip_indices = set()
        if os.path.exists(FAILED_OUTPUT_FILE):
            with open(FAILED_OUTPUT_FILE, 'r') as f:
                skip_indices = set()
                for line in f:
                    skip_indices.add(int(line.strip()))
        print(f"SKIP INDICES: {skip_indices}", flush=True)

        start_index = get_start_index(input_path, output_path, skip_indices)
        print("[Manager] Starting from index", start_index, flush=True)
        
        result_buffer = {}
        pending_tasks = {}
        next_index = start_index
        input_iter = stream_ndjson(input_path, start_index)

        with open(output_path, "a") as f:
            for idx, line in input_iter:
                while len(pending_tasks) >= MAX_PENDING:
                    next_index = self._process_next_result(pending_tasks, result_buffer, f, next_index)
                if idx == start_index:
                    print(f"PUT ON TASK QUEUE {idx}")
                    print(line)
                self.task_queue.put((idx, line))
                pending_tasks[idx] = (line, 0)

            while pending_tasks:
                next_index = self._process_next_result(pending_tasks, result_buffer, f, next_index)

if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--max_memory_gb", type=int, default=MAX_MEMORY_GB)
    args = parser.parse_args()

    MAX_WORKERS = args.max_memory_gb // SECTION_MEMORY_GB
    MAX_PENDING = MAX_WORKERS * 8
    SCALE_UP_THRESHOLD = MAX_WORKERS * 2

    print(f"Max memory: {args.max_memory_gb} GB", flush=True)
    print(f"Max workers: {MAX_WORKERS}", flush=True)
    print(f"Max pending: {MAX_PENDING}", flush=True)
    print(f"Scale up threshold: {SCALE_UP_THRESHOLD}", flush=True)

    INPUT_FILE = os.path.join(REFINED_OUTPUT_DIR, f"wikipedia_links_sections_{args.gpu_id}.json")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"maverick_{args.gpu_id}.json")
    FAILED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"maverick_fail_{args.gpu_id}.txt")

    while True:
        pool = WorkerPoolManager(max_workers=MAX_WORKERS)
        pool.scale_up()
        try:
            pool.process_loop(INPUT_FILE, OUTPUT_FILE)
            break  # Done processing all input; exit loop
        except SystemExit as e:
            print(f"[Main] Executor crashed with fatal error: {e}. hutting down and restarting pool...", flush=True)
        except Exception as e:
            print(f"[Main] Unexpected exception: {e}. Shutting down and restarting pool...", flush=True)
        finally:
            pool.shutdown()
            time.sleep(5) # Avoid rapid restarts
