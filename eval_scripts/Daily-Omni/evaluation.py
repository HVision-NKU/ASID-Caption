#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM-based Multiple-Choice QA Evaluator for Video Captions
"""

import os
import json
import time
import traceback
import multiprocessing
import random
import string
import numpy as np
import argparse
import hashlib
from tqdm import tqdm

from openai import AzureOpenAI
from openai import APIError, RateLimitError, AuthenticationError, APIConnectionError

# ========== Volcano Engine Configuration ==========
VOLCANO_API_KEY = os.environ.get("ARK_API_KEY", "")
AZURE_ENDPOINT = ""
API_VERSION = "2024-03-01-preview"
MODEL = "gemini-2.5-pro"

# ========== Thinking Configuration ==========
THINKING_ENABLE = True
THINKING_BUDGET_TOKENS = 4096
THINKING_INCLUDE_THOUGHTS = True

# ========== Result Saving Configuration ==========
RESULT_FILENAME = "real_time_results.jsonl"  # Output filename (dynamic path)
ACC_LOG_FILENAME = "real_time_acc.log"       # Accuracy log filename (dynamic path)

# ========== System Instruction ==========
SYSTEM_INSTRUCTION = '''
You are a precise QA assistant. Your task is to answer multiple-choice questions based ONLY on the video caption provided. 
Do not use any outside knowledge or assumptions—your answer must strictly reflect information from the caption. 
Always output only the capital letter corresponding to your choice (e.g., A, B, C, D). 
If the caption does not provide enough information to answer the question, output "N/A" instead.
'''

# ========== Generation Configuration ==========
GEN_CONFIG = {
    "temperature": 0,
    "top_p": 0.001,
    "max_tokens": 4096,
    "seed": 42,
    "stream": False
}

# ========== Output Validation Configuration ==========
VALID_OUTPUTS = {"A", "B", "C", "D", "N/A"}
INVALID_RETRY_MAX = 5       # Max retries for invalid outputs (non A/B/C/D/N/A)
MAX_TOTAL_RETRIES = 10      # Total retry limit (prevent infinite loop)
BASE_DELAY = 3.0            # Base delay for exponential backoff
MAX_DELAY = 30.0            # Max delay cap

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)

SEED = 42
set_seed(SEED)

def generate_tt_logid():
    """Generate random 32-character LOGID for Volcano Engine"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

def init_volcano_client():
    """Initialize Volcano Engine AzureOpenAI client with timeout"""
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=VOLCANO_API_KEY,
        api_version=API_VERSION,
        default_headers={"X-TT-LOGID": generate_tt_logid()},
        timeout=60.0,      # Critical: prevent hanging requests
        max_retries=0      # Avoid overlapping retries (SDK + custom)
    )
    return client

def init_worker_process():
    """Pool initializer: create client once per worker process"""
    global client
    client = init_volcano_client()

def generate_task_id(vid, question):
    """Generate unique task ID from video ID + question (MD5 hash)"""
    task_str = f"{vid}_{question}".encode("utf-8")
    return hashlib.md5(task_str).hexdigest()

def load_processed_tasks(result_file_path):
    """
    Load processed task IDs and statistics from result file
    Returns: (processed_task_ids, total_count, correct_count)
    """
    processed_task_ids = set()
    total = 0
    correct = 0

    if os.path.exists(result_file_path):
        try:
            with open(result_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        result = json.loads(line)
                        task_id = result.get("task_id") or generate_task_id(result.get("video_id", ""), result.get("question", ""))
                        processed_task_ids.add(task_id)
                        total += 1
                        if result.get("is_correct", False):
                            correct += 1
                    except Exception as e:
                        print(f"[WARN] Failed to parse history line: {e}")
            print(f"[Resume] Loaded {len(processed_task_ids)} processed tasks from {result_file_path}")
            print(f"[History] Total: {total} | Correct: {correct} | Accuracy: {correct/total if total>0 else 0:.4f}")
        except Exception as e:
            print(f"[WARN] Failed to read result file, starting fresh: {e}")
    else:
        print(f"[Resume] No result file found ({result_file_path}), starting fresh")

    return processed_task_ids, total, correct

def clear_results(result_file_path, acc_log_file_path):
    """Clear existing result and accuracy log files"""
    for file_path in [result_file_path, acc_log_file_path]:
        if os.path.exists(file_path):
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("")
                print(f"[Clear] Cleared {file_path}")
            except Exception as e:
                print(f"[WARN] Failed to clear {file_path}: {e}")

def extract_thinking_from_response(response):
    """Extract thinking/reasoning content from API response"""
    thinking_content = {}
    try:
        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            thinking_content["reasoning_content"] = response.choices[0].message.reasoning_content
        if hasattr(response.choices[0].message, 'multimodal_contents') and response.choices[0].message.multimodal_contents:
            for item in response.choices[0].message.multimodal_contents:
                if isinstance(item, dict) and item.get('thought') is True:
                    thinking_content["multimodal_thought"] = item.get('text', '')
                elif hasattr(item, 'thought') and item.thought is True:
                    thinking_content["multimodal_thought"] = item.text if hasattr(item, 'text') else ''
        if hasattr(response, 'usage') and hasattr(response.usage, 'reasoning_tokens'):
            thinking_content["reasoning_tokens"] = response.usage.reasoning_tokens
    except Exception:
        pass
    return thinking_content if thinking_content else None

def generate(prompt):
    """
    Generate answer from LLM with retry logic
    Handles: invalid outputs, API errors (no N/A retries)
    Returns: (final_answer, thinking_content)
    """
    final_answer = None
    final_thinking = None
    total_retry_count = 0       # Total retry counter
    invalid_retry_count = 0     # Invalid output retry counter

    while total_retry_count < MAX_TOTAL_RETRIES:
        try:
            request_logid = generate_tt_logid()
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION.strip()},
                {"role": "user", "content": prompt.strip()}
            ]

            # API call
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=GEN_CONFIG["max_tokens"],
                temperature=GEN_CONFIG["temperature"],
                top_p=GEN_CONFIG["top_p"],
                seed=GEN_CONFIG["seed"],
                messages=messages,
                extra_headers={"X-TT-LOGID": request_logid},
                extra_body={
                    "thinking": {
                        "include_thoughts": THINKING_INCLUDE_THOUGHTS,
                        "budget_tokens": THINKING_BUDGET_TOKENS
                    }
                } if THINKING_ENABLE else {}
            )
            
            raw_answer = (response.choices[0].message.content or "").strip().upper()
            final_thinking = extract_thinking_from_response(response)

            # Valid output (A/B/C/D/N/A) - return immediately
            if raw_answer in VALID_OUTPUTS:
                final_answer = raw_answer
                break
            
            # Invalid output (non A/B/C/D/N/A) - retry specified times
            else:
                invalid_retry_count += 1
                total_retry_count += 1
                
                if invalid_retry_count < INVALID_RETRY_MAX:
                    delay = BASE_DELAY * (2 ** (invalid_retry_count - 1))
                    delay = min(delay, MAX_DELAY)
                    print(f"[WARN] Invalid output: {raw_answer}, retry {invalid_retry_count}/{INVALID_RETRY_MAX} (delay {delay:.1f}s)")
                    time.sleep(delay)
                    continue
                else:
                    final_answer = "N/A"
                    print(f"[INFO] Invalid output retries exhausted ({INVALID_RETRY_MAX} attempts), returning N/A")
                    break

        # API error handling
        except (RateLimitError, AuthenticationError, APIConnectionError, APIError) as e:
            total_retry_count += 1
            if total_retry_count >= MAX_TOTAL_RETRIES:
                print(f"\n[ERROR] API retry limit reached: {type(e).__name__}")
                traceback.print_exc()
                break
            
            delay = 10.0 if isinstance(e, RateLimitError) else BASE_DELAY * (2 ** (total_retry_count - 1))
            delay = min(delay, MAX_DELAY)
            print(f"\n[ERROR] API call failed ({type(e).__name__}), retry {total_retry_count}/{MAX_TOTAL_RETRIES} (delay {delay:.1f}s)")
            traceback.print_exc()
            time.sleep(delay)
            
        # Generic error handling
        except Exception as e:
            total_retry_count += 1
            if total_retry_count >= MAX_TOTAL_RETRIES:
                print(f"\n[ERROR] Max retries reached for unknown error")
                traceback.print_exc()
                break
            
            delay = BASE_DELAY * (2 ** (total_retry_count - 1))
            delay = min(delay, MAX_DELAY)
            print(f"\n[ERROR] Unknown error, retry {total_retry_count}/{MAX_TOTAL_RETRIES} (delay {delay:.1f}s)")
            traceback.print_exc()
            time.sleep(delay)

    # Fallback: return N/A if no valid answer
    if not final_answer:
        final_answer = "N/A"
        print(f"[INFO] Total retries exhausted ({MAX_TOTAL_RETRIES} attempts), returning N/A")

    return final_answer, final_thinking

def save_single_result(result, result_file_path):
    """Append single result to JSONL file (flush immediately)"""
    try:
        with open(result_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
    except Exception as e:
        print(f"\n[ERROR] Failed to save result: {e}")

def worker(task_with_path):
    """
    Worker process: handle single QA task
    Args: task_with_path - (task_tuple, result_file_path)
    Returns: processed result dict
    """
    task, result_file_path = task_with_path
    task_id, vid, video_duration, question, choices, answer, caption = task

    choices_text = "\n".join([f"{c}" for c in choices])
    prompt_filled = f'''
Here is the video caption:
"{caption}"

Question: {question}
Choices:
    {choices_text}'''

    try:
        resp, thinking = generate(prompt_filled)
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_id": vid,
            "video_duration": video_duration,
            "question": question,
            "choices": choices,
            "ground_truth": answer,
            "model_answer": resp,
            "is_correct": (resp == answer),
            "caption": caption,
            "thinking": thinking,
            "task_id": task_id
        }
        save_single_result(result, result_file_path)
        return result

    except Exception as e:
        traceback.print_exc()
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_id": vid,
            "video_duration": video_duration,
            "question": question,
            "choices": choices,
            "ground_truth": answer,
            "model_answer": "N/A",
            "is_correct": False,
            "caption": caption,
            "thinking": None,
            "error": str(e),
            "task_id": task_id
        }
        save_single_result(result, result_file_path)
        return result

def run_multiprocess_tasks(tasks, result_file_path, acc_log_file_path, num_processes=None, resume=True):
    """
    Run QA tasks with multiprocessing
    Args:
        tasks: List of task tuples
        result_file_path: Path to save results
        acc_log_file_path: Path to save accuracy logs
        num_processes: Number of worker processes (default: CPU count)
        resume: Whether to resume from existing results
    Returns: List of processed results
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    if not resume:
        clear_results(result_file_path, acc_log_file_path)

    processed_task_ids, history_total, history_correct = load_processed_tasks(result_file_path)

    pending_tasks = [t for t in tasks if t[0] not in processed_task_ids]
    pending_tasks_with_path = [(task, result_file_path) for task in pending_tasks]
    
    total_task_count = len(tasks)
    pending_task_count = len(pending_tasks)

    print(f"[Task Filter] Total tasks: {total_task_count} | Processed: {history_total} | Pending: {pending_task_count}")

    if pending_task_count == 0:
        final_acc = history_correct / history_total if history_total > 0 else 0.0
        final_acc_str = f"\n[FINAL ACCURACY] Total: {history_total} | Correct: {history_correct} | Accuracy: {final_acc:.4f}"
        print(final_acc_str)
        with open(acc_log_file_path, "a", encoding="utf-8") as f:
            f.write(final_acc_str + "\n")
        return []

    # Real-time statistics tracking
    total = history_total
    correct = history_correct

    pbar = tqdm(
        total=total_task_count,
        initial=history_total,
        desc="QA Task Progress",
        unit="task",
        ncols=120,
        dynamic_ncols=True,
        smoothing=0.1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    results = []

    # Use maxtasksperchild to prevent resource leaks in long-running processes
    with multiprocessing.Pool(
        processes=num_processes,
        initializer=init_worker_process,
        maxtasksperchild=200
    ) as pool:

        # Use imap_unordered for streaming results (avoids blocking)
        for result in pool.imap_unordered(worker, pending_tasks_with_path, chunksize=1):
            results.append(result)
            total += 1
            if result.get("is_correct"):
                correct += 1

            pbar.update(1)

            # Log accuracy every 10 tasks (reduce I/O frequency)
            if total % 10 == 0:
                acc = correct / total if total > 0 else 0.0
                acc_str = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total: {total} | Correct: {correct} | Accuracy: {acc:.4f}"
                print(f"\n[REAL-TIME ACCURACY] {acc_str}", flush=True)
                with open(acc_log_file_path, "a", encoding="utf-8") as f:
                    f.write(acc_str + "\n")
                    f.flush()

    pbar.close()

    final_acc = correct / total if total > 0 else 0.0
    final_acc_str = f"\n[FINAL ACCURACY] Total: {total} | Correct: {correct} | Accuracy: {final_acc:.4f}"
    print(final_acc_str)
    with open(acc_log_file_path, "a", encoding="utf-8") as f:
        f.write(final_acc_str + "\n")
        f.flush()

    return results

def eval_dailyomni_caption_qas(file_path, resume=True, num_processes=32):
    """
    Main evaluation function: load data, generate tasks, run multiprocess evaluation
    Args:
        file_path: Path to input QA JSONL file
        resume: Whether to resume from existing results
        num_processes: Number of worker processes
    Returns: List of evaluation results
    """
    # Generate output paths (same directory as input file)
    input_dir = os.path.dirname(os.path.abspath(file_path))
    result_file_path = os.path.join(input_dir, RESULT_FILENAME)
    acc_log_file_path = os.path.join(input_dir, ACC_LOG_FILENAME)
    
    print(f"[File Paths] Results will be saved to: {result_file_path}")
    print(f"[File Paths] Accuracy log will be saved to: {acc_log_file_path}")

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"\n[WARN] Failed to parse line: {e}, skipping")
                continue

    # Generate task list with unique IDs
    tasks = []
    for video_info in data:
        vid = video_info.get("video_id", "")
        video_duration = video_info.get("video_duration", "")
        caption = video_info.get("caption", "")
        for q in video_info["questions"]:
            task_id = generate_task_id(vid, q["Question"])
            task_item = (
                task_id,
                vid,
                video_duration,
                q["Question"],
                q["Choice"],
                q["Answer"],
                caption
            )
            tasks.append(task_item)

    # Run multiprocess evaluation
    results = run_multiprocess_tasks(
        tasks, 
        result_file_path=result_file_path,
        acc_log_file_path=acc_log_file_path,
        num_processes=num_processes, 
        resume=resume
    )
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Caption QA Evaluator")
    parser.add_argument("merged_file", type=str, help="Path to QA JSONL file")
    parser.add_argument("--resume", action="store_true", default=True, help="Enable resume (default)")
    parser.add_argument("--restart", action="store_false", dest="resume", help="Run from scratch (clear results)")
    parser.add_argument("--num_processes", type=int, default=32, help="Number of processes (default: 32)")
    args = parser.parse_args()

    eval_dailyomni_caption_qas(args.merged_file, resume=args.resume, num_processes=args.num_processes)