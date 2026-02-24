#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caption-only Temporal Grounding Evaluation (Gemini via AzureOpenAI)

Input:
- Charades annotations: dict[vid] = {duration, timestamps[list], sentences[list]}
- Caption file: jsonl lines with {"video_id": ..., "caption": ...} (your model-generated caption, includes "At Xs" anchors)

"""

import os
import re
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

# ========== Volcano/Azure config ==========
VOLCANO_API_KEY = os.environ.get("ARK_API_KEY", "")
AZURE_ENDPOINT = "https://search.bytedance.net/gpt/openapi/online/v2/crawl"
API_VERSION = "2024-03-01-preview"
MODEL = "gemini-2.5-pro"

# ========== Thinking config (keep) ==========
THINKING_ENABLE = True
THINKING_BUDGET_TOKENS = 4096
THINKING_INCLUDE_THOUGHTS = True

# ========== Output files ==========
RESULT_FILENAME = "real_time_results.jsonl"
ACC_LOG_FILENAME = "real_time_acc.log"

# ========== Generation config ==========
GEN_CONFIG = {
    "temperature": 0,
    "top_p": 0.001,
    "max_tokens": 1024,   # grounding output is short; can be smaller
    "seed": 42,
    "stream": False
}

# ========== Retry / backoff ==========
MAX_TOTAL_RETRIES = 10
BASE_DELAY = 3.0
MAX_DELAY = 30.0

# ========== Metrics ==========
THRESH = np.array([0.3, 0.5, 0.7], dtype=np.float32)

# -------------------- Prompt --------------------
SYSTEM_INSTRUCTION = r"""
You are a temporal grounding assistant.

You will be given:
(1) a long video caption with multiple timestamp anchors like "At 0s, ... At 5s, ...",
(2) an event description (a sentence).

Goal:
Infer the most likely continuous time interval (start and end in seconds) when the event happens, using ONLY the caption.

Rules (very important):
1) Always try to output a time interval, even if the evidence is partial. Use best-effort inference.
2) Prefer intervals aligned to existing anchors. If the event is mentioned near "At Ts", choose a range that covers that anchor and the most plausible neighboring anchors.
3) If the event is implied by related actions/objects (synonyms/paraphrases), still infer the interval by matching the closest described segment.
4) Output "N/A" ONLY if the caption provides absolutely no usable clue to localize the event (no matching action/object/context anywhere).

Output format:
- "start to end" (seconds, one decimal preferred), e.g., "1.2 to 10.8"
- or "N/A"
Output ONLY the final answer. No extra words.
""".strip()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

SEED = 42
set_seed(SEED)

def generate_tt_logid():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

def init_volcano_client():
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=VOLCANO_API_KEY,
        api_version=API_VERSION,
        default_headers={"X-TT-LOGID": generate_tt_logid()},
        timeout=60.0,
        max_retries=0
    )
    return client

def init_worker_process():
    global client
    client = init_volcano_client()

def generate_task_id(vid, sentence_idx):
    task_str = f"{vid}_{sentence_idx}".encode("utf-8")
    return hashlib.md5(task_str).hexdigest()

def extract_thinking_from_response(response):
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

def parse_timestamp_output(output_string):
    """
    Parse:
    - "12.54 to 17.83"
    - "12.54 and 17.83"
    - possibly inside <answer>...</answer> (robust)
    """
    if not output_string:
        return None, None

    s = output_string.strip()
    if s.upper() == "N/A":
        return None, None

    matches = re.findall(r"(\d+\.?\d*)\s*(to|and)\s*(\d+\.?\d*)", s, flags=re.IGNORECASE)
    if not matches:
        answer_match = re.search(r"<answer>(.*?)</answer>", s, flags=re.DOTALL | re.IGNORECASE)
        if answer_match:
            inner = answer_match.group(1).strip()
            inner_matches = re.findall(r"(\d+\.?\d*)\s*(to|and)\s*(\d+\.?\d*)", inner, flags=re.IGNORECASE)
            if inner_matches:
                last = inner_matches[-1]
                try:
                    return float(last[0]), float(last[2])
                except ValueError:
                    return None, None
        return None, None

    last = matches[-1]
    try:
        return float(last[0]), float(last[2])
    except ValueError:
        return None, None

def calc_iou_1d(pred_s, pred_e, gt_s, gt_e):
    if pred_s is None or pred_e is None:
        return 0.0
    if pred_e < pred_s:
        pred_s, pred_e = pred_e, pred_s
    inter = min(pred_e, gt_e) - max(pred_s, gt_s)
    union = max(pred_e, gt_e) - min(pred_s, gt_s)
    if union <= 0:
        return 0.0
    return max(inter, 0.0) / union

def call_llm(prompt):
    final_answer = None
    final_thinking = None
    total_retry = 0

    while total_retry < MAX_TOTAL_RETRIES:
        try:
            request_logid = generate_tt_logid()
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt.strip()}
            ]
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
            raw = (response.choices[0].message.content or "").strip()
            final_thinking = extract_thinking_from_response(response)

            # accept either N/A or a parseable range
            if raw.upper() == "N/A":
                final_answer = "N/A"
                break

            ps, pe = parse_timestamp_output(raw)
            if ps is not None and pe is not None:
                final_answer = raw
                break

            # invalid -> retry
            total_retry += 1
            delay = min(BASE_DELAY * (2 ** (total_retry - 1)), MAX_DELAY)
            print(f"[WARN] Invalid grounding output: {raw!r}. retry {total_retry}/{MAX_TOTAL_RETRIES} sleep {delay:.1f}s")
            time.sleep(delay)

        except (RateLimitError, AuthenticationError, APIConnectionError, APIError) as e:
            total_retry += 1
            if total_retry >= MAX_TOTAL_RETRIES:
                print(f"[ERROR] API failed to max retries: {type(e).__name__}")
                traceback.print_exc()
                break
            delay = 10.0 if isinstance(e, RateLimitError) else min(BASE_DELAY * (2 ** (total_retry - 1)), MAX_DELAY)
            print(f"[ERROR] API error {type(e).__name__}, retry {total_retry}/{MAX_TOTAL_RETRIES}, sleep {delay:.1f}s")
            traceback.print_exc()
            time.sleep(delay)

        except Exception as e:
            total_retry += 1
            if total_retry >= MAX_TOTAL_RETRIES:
                print("[ERROR] Unknown error to max retries")
                traceback.print_exc()
                break
            delay = min(BASE_DELAY * (2 ** (total_retry - 1)), MAX_DELAY)
            print(f"[ERROR] Unknown error, retry {total_retry}/{MAX_TOTAL_RETRIES}, sleep {delay:.1f}s")
            traceback.print_exc()
            time.sleep(delay)

    if not final_answer:
        final_answer = "N/A"
    return final_answer, final_thinking

def save_single_result(result, result_file_path):
    try:
        with open(result_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
    except Exception as e:
        print(f"[ERROR] Save result failed: {e}")

def load_processed_tasks(result_file_path):
    processed = set()
    total = 0
    sum_iou = 0.0
    recall_cnt = np.array([0, 0, 0], dtype=np.int64)

    if not os.path.exists(result_file_path):
        print(f"[Resume] No result file: {result_file_path}")
        return processed, total, sum_iou, recall_cnt

    bad = 0
    with open(result_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tid = obj.get("task_id")
                if tid:
                    processed.add(tid)
                iou = obj.get("iou", None)
                if iou is not None:
                    iou = float(iou)
                    total += 1
                    sum_iou += iou
                    recall_cnt += (THRESH <= iou).astype(np.int64)
            except Exception:
                bad += 1

    miou = sum_iou / total if total > 0 else 0.0
    recall = (recall_cnt / total).tolist() if total > 0 else [0.0, 0.0, 0.0]
    print(f"[Resume] processed_tasks={len(processed)} scored={total} bad_lines={bad}")
    print(f"[Resume] mIoU={miou:.4f} R@0.3/0.5/0.7={recall}")
    return processed, total, sum_iou, recall_cnt

def worker(task_with_path):
    task, result_file_path = task_with_path
    task_id, vid, duration, sentence, gt_s, gt_e, caption = task

    prompt = f"""
Here is the video caption:
\"\"\"{caption}\"\"\"

Event: {sentence}

Return the time interval (start to end) in seconds.
""".strip()

    t0 = time.time()
    try:
        raw_ans, thinking = call_llm(prompt)
        ps, pe = parse_timestamp_output(raw_ans)
        iou = calc_iou_1d(ps, pe, gt_s, gt_e)

        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_id": task_id,
            "video_id": vid,
            "video_duration": duration,
            "sentence": sentence,
            "gt": [gt_s, gt_e],
            "pred_raw": raw_ans,
            "pred": [ps, pe],
            "iou": float(iou),
            "hit@0.3": bool(iou >= 0.3),
            "hit@0.5": bool(iou >= 0.5),
            "hit@0.7": bool(iou >= 0.7),
            "latency_sec": round(time.time() - t0, 4),
            "caption": caption,
            "thinking": thinking
        }
        save_single_result(result, result_file_path)
        return result

    except Exception as e:
        traceback.print_exc()
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_id": task_id,
            "video_id": vid,
            "video_duration": duration,
            "sentence": sentence,
            "gt": [gt_s, gt_e],
            "pred_raw": "N/A",
            "pred": [None, None],
            "iou": 0.0,
            "hit@0.3": False,
            "hit@0.5": False,
            "hit@0.7": False,
            "latency_sec": round(time.time() - t0, 4),
            "caption": caption,
            "thinking": None,
            "error": f"{type(e).__name__}: {str(e)}"
        }
        save_single_result(result, result_file_path)
        return result

def run_multiprocess_tasks(tasks, result_file_path, acc_log_file_path, num_processes=32, resume=True):
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)

    processed, hist_scored, hist_sum_iou, hist_recall_cnt = load_processed_tasks(result_file_path)

    pending = [t for t in tasks if t[0] not in processed]
    pending_with_path = [(t, result_file_path) for t in pending]

    print(f"[Tasks] total={len(tasks)} | already_done={len(processed)} | pending={len(pending)}")

    # metric state (scored only)
    scored = hist_scored
    sum_iou = hist_sum_iou
    recall_cnt = hist_recall_cnt.copy()

    pbar = tqdm(
        total=len(tasks),
        initial=len(processed),
        desc="Caption-only Grounding",
        unit="item",
        ncols=120,
        dynamic_ncols=True,
        smoothing=0.1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    if len(pending) == 0:
        miou = sum_iou / scored if scored > 0 else 0.0
        recall = (recall_cnt / scored).tolist() if scored > 0 else [0.0, 0.0, 0.0]
        final_line = f"[FINAL] scored={scored} mIoU={miou:.4f} R@0.3={recall[0]:.4f} R@0.5={recall[1]:.4f} R@0.7={recall[2]:.4f}"
        print(final_line)
        with open(acc_log_file_path, "a", encoding="utf-8") as f:
            f.write(final_line + "\n")
        return []

    results = []
    with multiprocessing.Pool(
        processes=num_processes,
        initializer=init_worker_process,
        maxtasksperchild=200
    ) as pool:
        for result in pool.imap_unordered(worker, pending_with_path, chunksize=1):
            results.append(result)
            pbar.update(1)

            # update metrics
            iou = float(result.get("iou", 0.0))
            scored += 1
            sum_iou += iou
            recall_cnt += (THRESH <= iou).astype(np.int64)

            if scored % 10 == 0:
                miou = sum_iou / scored if scored > 0 else 0.0
                recall = (recall_cnt / scored).tolist() if scored > 0 else [0.0, 0.0, 0.0]
                line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] scored={scored} mIoU={miou:.4f} R@0.3={recall[0]:.4f} R@0.5={recall[1]:.4f} R@0.7={recall[2]:.4f}"
                print("\n[REAL-TIME]", line, flush=True)
                with open(acc_log_file_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()

    pbar.close()

    miou = sum_iou / scored if scored > 0 else 0.0
    recall = (recall_cnt / scored).tolist() if scored > 0 else [0.0, 0.0, 0.0]
    final_line = f"[FINAL] scored={scored} mIoU={miou:.4f} R@0.3={recall[0]:.4f} R@0.5={recall[1]:.4f} R@0.7={recall[2]:.4f}"
    print(final_line)
    with open(acc_log_file_path, "a", encoding="utf-8") as f:
        f.write(final_line + "\n")
        f.flush()

    return results

def load_charades_ann(ann_file):
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize timestamps to [start,end] single pair per sentence_idx (your sample shows list of [s,e])
    return data

def load_caption_map(caption_file):
    cap = {}
    with open(caption_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            vid = obj.get("video_id")
            caption = obj.get("caption", "")
            if vid:
                cap[vid] = caption
    return cap

def build_tasks(ann_data, caption_map):
    tasks = []
    missing_caps = 0
    missing_ts = 0

    for vid, ann in ann_data.items():
        duration = ann.get("duration", ann.get("video_duration", None))
        sentences = ann.get("sentences", [])
        timestamps = ann.get("timestamps", [])

        caption = caption_map.get(vid, None)
        if caption is None:
            missing_caps += 1
            continue

        for i, sent in enumerate(sentences):
            if i >= len(timestamps):
                missing_ts += 1
                continue
            gt = timestamps[i]
            # Charades format: [start,end]
            gt_s, gt_e = float(gt[0]), float(gt[1])

            task_id = generate_task_id(vid, i)
            tasks.append((task_id, vid, duration, sent, gt_s, gt_e, caption))

    print(f"[Build] tasks={len(tasks)} missing_caption_vids={missing_caps} missing_timestamps={missing_ts}")
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Caption-only Grounding Eval (Gemini)")
    parser.add_argument("--ann_file", type=str, required=True, help="Charades annotation json (dict)")
    parser.add_argument("--caption_file", type=str, required=True, help="Caption jsonl with video_id+caption")
    parser.add_argument("--num_processes", type=int, default=32)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--restart", action="store_false", dest="resume")
    args = parser.parse_args()

    input_dir = os.path.dirname(os.path.abspath(args.caption_file))
    result_file_path = os.path.join(input_dir, RESULT_FILENAME)
    acc_log_file_path = os.path.join(input_dir, ACC_LOG_FILENAME)

    print(f"[Paths] ann_file: {args.ann_file}")
    print(f"[Paths] caption_file: {args.caption_file}")
    print(f"[Paths] result_file: {result_file_path}")
    print(f"[Paths] acc_log_file: {acc_log_file_path}")

    ann_data = load_charades_ann(args.ann_file)
    caption_map = load_caption_map(args.caption_file)
    tasks = build_tasks(ann_data, caption_map)

    run_multiprocess_tasks(
        tasks,
        result_file_path=result_file_path,
        acc_log_file_path=acc_log_file_path,
        num_processes=args.num_processes,
        resume=args.resume
    )

if __name__ == "__main__":
    main()
