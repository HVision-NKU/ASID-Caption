#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
One-shot Instruction-Following / Attribute-Purity Eval for Captions (Gemini)

Judge: ONE call
- Model outputs JSON only:
  - levels: none/minor/explicit for all 8 attrs
  - missing_targets / extra_explicit
  - evidence (required for extra_explicit)
  - absent_attributes + absent_reason

We recompute FOLLOW ourselves for consistency:
FOLLOW = YES iff:
  - every target attr is explicit
  - no non-target attr is explicit

Inputs:
- caption_file jsonl: {"video_id": "...", "caption": "..."}
- prompt_file  jsonl: {"video_rel": "...", "video": "...", "k": 1|2|3|4, "attributes": [...], "prompt": "..."}

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
from pathlib import Path
from tqdm import tqdm

from openai import AzureOpenAI
from openai import APIError, RateLimitError, AuthenticationError, APIConnectionError

# ========== Volcano/Azure config ==========
VOLCANO_API_KEY = os.environ.get("ARK_API_KEY", "")
AZURE_ENDPOINT = "https://search.bytedance.net/gpt/openapi/online/v2/crawl"
API_VERSION = "2024-03-01-preview"
MODEL = "gemini-2.5-pro"

# ========== Output files ==========
RESULT_FILENAME = "real_time_results.jsonl"
ACC_LOG_FILENAME = "real_time_acc.log"

# ========== Retry / backoff ==========
MAX_TOTAL_RETRIES = 8
BASE_DELAY = 2.0
MAX_DELAY = 20.0

# ========== Generation config ==========
# Enough for JSON + evidence + absent reasons
GEN = dict(temperature=0, top_p=0.001, max_tokens=16384, seed=42)

# Disable thinking for stability
THINKING_ENABLE = False

# =========================
# Attribute space (8)
# =========================
ATTRS_8 = [
    "scene",
    "characters",
    "objects",
    "actions",
    "narrative elements",
    "speech",
    "camera",
    "emotions",
]

PROMPT_PARSE_KEYS = {
    "scene": ["scene"],
    "characters": ["characters", "character"],
    "objects": ["objects", "object"],
    "actions": ["actions", "action"],
    "narrative elements": ["narrative elements", "narrative"],
    "speech": ["speech", "audio", "dialogue"],
    "camera": ["camera"],
    "emotions": ["emotions", "emotion"],
}

# -------------------- One-shot judge instruction (ULTRA-CONSERVATIVE extra) --------------------
SYSTEM = r"""
You are an evaluator for *attribute purity* of a caption relative to a target attribute set.
You must NOT judge factual correctness. Ignore whether the caption is true.

Closed-set attributes (exact keys):
scene, characters, objects, actions, narrative elements, speech, camera, emotions

MOST IMPORTANT:
Detect whether the caption contains ANY *non-target* attribute as an EXPLICIT description.
You must be VERY CONSERVATIVE.
Only mark a non-target attribute as explicit when it is unmistakably and clearly developed.
If unsure, label it as minor or none (do NOT penalize).

Levels:
- none: not mentioned.
- minor: brief hint/mention without clear development (allowed for non-target attributes).
- explicit: clearly developed with concrete details. Must satisfy either:
  (a) at least ONE sentence mainly about this attribute AND contains >=2 concrete details for this attribute, OR
  (b) this attribute is developed across >=2 sentences (clearly a focus).

Concrete details (examples):
scene: location type + environment elements / lighting / weather / background activities
characters: number + appearance/clothing + posture/interactions
objects: multiple objects + properties/relations (type/quantity/position/material)
actions: action sequence + body/hand/facial movements + speed/temporal order
speech: spoken content OR language/tone/volume + who speaks (dialogue/narration)
camera: shot type/angle + camera motion/transition/zoom
emotions: emotion type + intensity/changes + expressed by whom/how
narrative elements: cause-effect, intent/goal, event transitions, story progression

Evaluation:
- Target attributes: require EXPLICIT.
- Non-target attributes: label EXPLICIT ONLY if very obvious (otherwise minor/none).

Output MUST be valid JSON ONLY (no markdown, no extra text).
Return these keys:
{
  "levels": { "<attr>": "none|minor|explicit", ... (all 8 attrs) },

  "missing_targets": [ ... ],      // target attrs whose level != explicit
  "extra_explicit": [ ... ],       // non-target attrs whose level == explicit (ULTRA-CONSERVATIVE!)

  "evidence": { "<attr>": ["verbatim quote", ...] },   // REQUIRED for each attr in extra_explicit; optional for targets
  "absent_attributes": [ ... ],    // attrs whose level == none
  "absent_reason": { "<attr>": "short why-not-present", ... },  // brief reason, e.g., "no dialogue content", "no camera description"

  "notes": "optional short note"
}

Constraints:
- Be strict for target explicitness, but ultra-conservative for extra_explicit.
- For extra_explicit, provide at least 1 short verbatim quote as evidence whenever possible.
- For absent_reason, keep it short and generic (do not hallucinate specifics).
""".strip()

# -------------------- Utils --------------------
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

def generate_tt_logid():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

def init_volcano_client():
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=VOLCANO_API_KEY,
        api_version=API_VERSION,
        default_headers={"X-TT-LOGID": generate_tt_logid()},
        timeout=60.0,
        max_retries=0,
    )

def init_worker_process():
    global client
    client = init_volcano_client()

def normalize_attrs(attrs):
    out = []
    if not isinstance(attrs, list):
        return out
    for a in attrs:
        if isinstance(a, str):
            a2 = a.strip().lower()
            if a2 in ATTRS_8:
                out.append(a2)
    out = sorted(set(out), key=ATTRS_8.index)
    return out

def parse_attrs_from_prompt_text(prompt_text: str):
    t = (prompt_text or "").lower()
    found = []
    for canon, keys in PROMPT_PARSE_KEYS.items():
        for k in keys:
            if k in t:
                found.append(canon)
                break
    return normalize_attrs(found)

def safe_get_prompt_text(item):
    p = item.get("prompt", "")
    if isinstance(p, dict):
        return str(p.get("value", "") or p.get("text", "") or "")
    return str(p) if p is not None else ""

def prompt_key(item):
    if isinstance(item.get("video_rel"), str) and item["video_rel"].strip():
        return item["video_rel"].strip()
    v = item.get("video", "")
    if isinstance(v, str) and v.strip():
        return Path(v).name
    return ""

def build_user_prompt(target_attrs, caption):
    target_attrs = target_attrs or []
    non_targets = [a for a in ATTRS_8 if a not in target_attrs]
    return f"""
TASK:
1) Label all 8 attributes with levels (none/minor/explicit).
2) Target coverage: all target attrs must be explicit.
3) Purity: only flag non-target attrs as extra_explicit if VERY OBVIOUS; otherwise minor/none.
4) Also list absent_attributes (level==none) and absent_reason (short why-not-present).

Target attributes (must be EXPLICIT): {target_attrs}
Non-target attributes (ONLY flag as EXPLICIT if very obvious): {non_targets}

Caption:
\"\"\"{caption}\"\"\"
""".strip()

def _extract_first_json_block(text: str):
    if not text:
        return None
    raw = text.strip()
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    return m.group(0)

def parse_judge_output(text: str):
    """
    Truncation-tolerant JSON parser:
    - Extract first {...} block
    - Require levels dict
    - Normalize levels, evidence, absent fields
    """
    if not text:
        return None
    raw = text.strip()
    js = _extract_first_json_block(raw)
    if not js:
        return None

    try:
        obj = json.loads(js)
    except Exception:
        return None

    levels = obj.get("levels", {})
    if not isinstance(levels, dict):
        return None

    # normalize levels
    norm_levels = {}
    for a in ATTRS_8:
        v = str(levels.get(a, "none")).strip().lower()
        if v not in ("none", "minor", "explicit"):
            v = "none"
        norm_levels[a] = v

    # evidence normalize
    evidence = obj.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    norm_evidence = {}
    for k, v in evidence.items():
        kk = str(k).strip().lower()
        if kk not in ATTRS_8:
            continue
        if isinstance(v, list):
            norm_evidence[kk] = [str(z) for z in v if str(z).strip()]
        elif isinstance(v, str) and v.strip():
            norm_evidence[kk] = [v.strip()]

    # model-provided missing/extra (we recompute later; keep for debug)
    missing_targets = obj.get("missing_targets", [])
    extra_explicit = obj.get("extra_explicit", [])
    if not isinstance(missing_targets, list):
        missing_targets = []
    if not isinstance(extra_explicit, list):
        extra_explicit = []
    missing_targets = [x for x in [str(y).strip().lower() for y in missing_targets] if x in ATTRS_8]
    extra_explicit = [x for x in [str(y).strip().lower() for y in extra_explicit] if x in ATTRS_8]

    # absent fields
    absent_attributes = obj.get("absent_attributes", [])
    if not isinstance(absent_attributes, list):
        absent_attributes = []
    absent_attributes = [x for x in [str(y).strip().lower() for y in absent_attributes] if x in ATTRS_8]

    absent_reason = obj.get("absent_reason", {})
    if not isinstance(absent_reason, dict):
        absent_reason = {}
    norm_absent_reason = {}
    for k, v in absent_reason.items():
        kk = str(k).strip().lower()
        if kk in ATTRS_8 and isinstance(v, str) and v.strip():
            norm_absent_reason[kk] = v.strip()

    return {
        "levels": norm_levels,
        "missing_model": missing_targets,
        "extra_model": extra_explicit,
        "evidence": norm_evidence,
        "absent_attributes": absent_attributes,
        "absent_reason": norm_absent_reason,
        "raw": raw,
    }

def call_gemini(system_text: str, user_text: str, gen_cfg: dict):
    total_retry = 0
    while total_retry < MAX_TOTAL_RETRIES:
        try:
            request_logid = generate_tt_logid()
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=gen_cfg["max_tokens"],
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                seed=gen_cfg["seed"],
                messages=messages,
                extra_headers={"X-TT-LOGID": request_logid},
                extra_body={},  # thinking disabled
            )
            return (resp.choices[0].message.content or "").strip()

        except (RateLimitError, AuthenticationError, APIConnectionError, APIError) as e:
            total_retry += 1
            if total_retry >= MAX_TOTAL_RETRIES:
                print(f"[ERROR] API failed max retries: {type(e).__name__}")
                traceback.print_exc()
                return ""
            delay = 10.0 if isinstance(e, RateLimitError) else min(BASE_DELAY * (2 ** (total_retry - 1)), MAX_DELAY)
            print(f"[ERROR] API {type(e).__name__}, retry {total_retry}/{MAX_TOTAL_RETRIES}, sleep {delay:.1f}s")
            time.sleep(delay)

        except Exception:
            total_retry += 1
            if total_retry >= MAX_TOTAL_RETRIES:
                print("[ERROR] Unknown error max retries")
                traceback.print_exc()
                return ""
            delay = min(BASE_DELAY * (2 ** (total_retry - 1)), MAX_DELAY)
            print(f"[ERROR] Unknown error, retry {total_retry}/{MAX_TOTAL_RETRIES}, sleep {delay:.1f}s")
            time.sleep(delay)

    return ""

def judge_once(target_attrs, caption):
    """
    One-shot judge. Retries ONLY when JSON cannot be parsed.
    We compute follow/missing/extra/absent ourselves from levels for consistency.
    """
    user_prompt = build_user_prompt(target_attrs, caption)

    tries = 0
    last_raw = ""
    while tries < MAX_TOTAL_RETRIES:
        raw = call_gemini(SYSTEM, user_prompt, GEN)
        last_raw = raw
        parsed = parse_judge_output(raw)
        if parsed is not None:
            levels = parsed["levels"]
            target_attrs = target_attrs or []

            # recompute
            missing = [a for a in target_attrs if levels.get(a, "none") != "explicit"]
            extra = [a for a in ATTRS_8 if a not in target_attrs and levels.get(a, "none") == "explicit"]
            follow = (len(missing) == 0 and len(extra) == 0)

            # derive absent from levels (authoritative)
            derived_absent = [a for a in ATTRS_8 if levels.get(a, "none") == "none"]

            # keep absent_reason only for actually-absent attrs
            absent_reason = parsed.get("absent_reason", {}) or {}
            absent_reason = {k: v for k, v in absent_reason.items() if k in derived_absent}

            # evidence sanity: for extra attrs, if no evidence, keep empty list (still flagged)
            evidence = parsed.get("evidence", {}) or {}
            for a in extra:
                evidence.setdefault(a, [])

            return {
                "follow": follow,
                "missing": missing,
                "extra": extra,
                "levels": levels,
                "evidence": evidence,
                "absent_attributes": derived_absent,
                "absent_reason": absent_reason,
                "raw": parsed.get("raw", last_raw),
                # keep model-provided lists for debug
                "missing_model": parsed.get("missing_model", []),
                "extra_model": parsed.get("extra_model", []),
            }

        tries += 1
        delay = min(BASE_DELAY * (2 ** (tries - 1)), MAX_DELAY)
        print(f"[WARN] Invalid judge JSON. retry {tries}/{MAX_TOTAL_RETRIES} sleep {delay:.1f}s | raw={raw!r}")
        time.sleep(delay)

    return {
        "follow": False,
        "missing": [],
        "extra": [],
        "levels": {a: "none" for a in ATTRS_8},
        "evidence": {},
        "absent_attributes": ATTRS_8[:],
        "absent_reason": {},
        "raw": last_raw,
        "missing_model": [],
        "extra_model": [],
    }

def save_single_result(result, result_file_path):
    with open(result_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()

def load_processed_tasks(result_file_path):
    """
    Resume:
    - processed task ids
    - overall follow stats
    - per-k stats
    - optional: by combo stats
    """
    processed = set()
    scored = 0
    follow_cnt = 0
    by_k = {}
    by_combo = {}

    if not os.path.exists(result_file_path):
        print(f"[Resume] No result file: {result_file_path}")
        return processed, scored, follow_cnt, by_k, by_combo

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

                follow = obj.get("follow", None)
                k = obj.get("k", None)
                combo = obj.get("combo", "UNKNOWN")

                if follow is None:
                    continue

                follow = bool(follow)
                scored += 1
                follow_cnt += int(follow)

                by_k.setdefault(k, [0, 0])
                by_k[k][0] += 1
                by_k[k][1] += int(follow)

                by_combo.setdefault(combo, [0, 0])
                by_combo[combo][0] += 1
                by_combo[combo][1] += int(follow)
            except Exception:
                bad += 1

    rate = follow_cnt / scored if scored > 0 else 0.0
    print(f"[Resume] processed_tasks={len(processed)} scored={scored} bad_lines={bad}")
    print(f"[Resume] Overall FollowRate={rate:.4f}")
    for kk in sorted([x for x in by_k.keys() if isinstance(x, int)]):
        n, fcnt = by_k[kk]
        print(f"[Resume] k={kk} n={n} FollowRate={fcnt/n:.4f}")
    return processed, scored, follow_cnt, by_k, by_combo

def generate_task_id(vid, idx, k, targets, prompt_text):
    s = f"{vid}_{idx}_{k}_{'|'.join(targets)}_{prompt_text}".encode("utf-8")
    return hashlib.md5(s).hexdigest()

def worker(task_with_path):
    task, result_file_path = task_with_path
    task_id, vid, k, target_attrs, prompt_text, caption = task

    t0 = time.time()
    try:
        judged = judge_once(target_attrs, caption)
        follow = bool(judged.get("follow", False))
        combo = "|".join(target_attrs) if target_attrs else "UNKNOWN"

        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_id": task_id,
            "video_id": vid,
            "k": k,
            "targets": target_attrs,
            "combo": combo,
            "prompt": prompt_text,
            "caption": caption,

            "follow": follow,
            "missing_targets": judged.get("missing", []),
            "extra_attributes": judged.get("extra", []),

            "levels": judged.get("levels", {}),
            "evidence": judged.get("evidence", {}),

            "absent_attributes": judged.get("absent_attributes", []),
            "absent_reason": judged.get("absent_reason", {}),

            # debug (optional)
            "missing_model": judged.get("missing_model", []),
            "extra_model": judged.get("extra_model", []),

            "judge_raw": judged.get("raw", ""),
            "latency_sec": round(time.time() - t0, 4),
        }
        save_single_result(result, result_file_path)
        return result

    except Exception as e:
        traceback.print_exc()
        combo = "|".join(target_attrs) if target_attrs else "UNKNOWN"
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_id": task_id,
            "video_id": vid,
            "k": k,
            "targets": target_attrs,
            "combo": combo,
            "prompt": prompt_text,
            "caption": caption,

            "follow": False,
            "missing_targets": [],
            "extra_attributes": [],

            "levels": {a: "none" for a in ATTRS_8},
            "evidence": {},

            "absent_attributes": ATTRS_8[:],
            "absent_reason": {},

            "missing_model": [],
            "extra_model": [],

            "judge_raw": "",
            "latency_sec": round(time.time() - t0, 4),
            "error": f"{type(e).__name__}: {str(e)}",
        }
        save_single_result(result, result_file_path)
        return result

def log_scores(acc_log_file_path, scored, follow_cnt, by_k):
    overall = follow_cnt / scored if scored > 0 else 0.0
    parts = [f"scored={scored} OverallFollowRate={overall:.4f}"]
    for kk in [1, 2, 3, 4]:
        if kk in by_k and by_k[kk][0] > 0:
            n, fcnt = by_k[kk]
            parts.append(f"k{kk}={fcnt/n:.4f}(n={n})")
    line = "[SCORES] " + " ".join(parts)
    with open(acc_log_file_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
    return line

def run_multiprocess_tasks(tasks, result_file_path, acc_log_file_path, num_processes=32):
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)

    processed, hist_scored, hist_follow, by_k, by_combo = load_processed_tasks(result_file_path)

    pending = [t for t in tasks if t[0] not in processed]
    pending_with_path = [(t, result_file_path) for t in pending]

    print(f"[Tasks] total={len(tasks)} | already_done={len(processed)} | pending={len(pending)}")

    scored = hist_scored
    follow_cnt = hist_follow

    pbar = tqdm(
        total=len(tasks),
        initial=len(processed),
        desc="Attr-Purity (Gemini one-shot)",
        unit="item",
        ncols=120,
        dynamic_ncols=True,
        smoothing=0.1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    if len(pending) == 0:
        final_line = log_scores(acc_log_file_path, scored, follow_cnt, by_k)
        print(final_line)
        return []

    results = []
    with multiprocessing.Pool(
        processes=num_processes,
        initializer=init_worker_process,
        maxtasksperchild=300,
    ) as pool:
        for result in pool.imap_unordered(worker, pending_with_path, chunksize=1):
            results.append(result)
            pbar.update(1)

            scored += 1
            follow = bool(result.get("follow", False))
            follow_cnt += int(follow)

            k = result.get("k", None)
            by_k.setdefault(k, [0, 0])
            by_k[k][0] += 1
            by_k[k][1] += int(follow)

            if scored % 50 == 0:
                line = log_scores(acc_log_file_path, scored, follow_cnt, by_k)
                print("\n[REAL-TIME]", line, flush=True)

    pbar.close()

    final_line = log_scores(acc_log_file_path, scored, follow_cnt, by_k)
    print(final_line)

    print("\n[Final Breakdown by k]")
    for kk in [1, 2, 3, 4]:
        if kk in by_k and by_k[kk][0] > 0:
            n, fcnt = by_k[kk]
            print(f"  k={kk} n={n} FollowRate={fcnt/n:.4f}")

    return results

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

def load_prompts(prompt_file):
    data = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def build_tasks(prompt_data, caption_map):
    tasks = []
    missing_caps = 0
    missing_key = 0

    for i, p in enumerate(prompt_data):
        vid = prompt_key(p)
        if not vid:
            missing_key += 1
            continue

        caption = caption_map.get(vid, None)
        if caption is None:
            missing_caps += 1
            continue

        k = p.get("k", None)
        if isinstance(k, str) and k.strip().isdigit():
            k = int(k.strip())
        elif not isinstance(k, int):
            attrs0 = p.get("attributes", [])
            if isinstance(attrs0, list) and len(attrs0) > 0:
                k = len(attrs0)
            else:
                k = None

        targets = normalize_attrs(p.get("attributes", []))
        if not targets:
            targets = parse_attrs_from_prompt_text(safe_get_prompt_text(p))

        prompt_text = safe_get_prompt_text(p)
        task_id = generate_task_id(vid, i, k, targets, prompt_text)
        tasks.append((task_id, vid, k, targets, prompt_text, caption))

    print(f"[Build] tasks={len(tasks)} missing_caption={missing_caps} missing_prompt_key={missing_key}")
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Caption Attr-Purity Eval (Gemini) — one-shot JSON + scores by k")
    parser.add_argument("--caption_file", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=32)
    args = parser.parse_args()

    if not VOLCANO_API_KEY:
        raise RuntimeError("ARK_API_KEY is not set.")

    input_dir = os.path.dirname(os.path.abspath(args.caption_file))
    result_file_path = os.path.join(input_dir, RESULT_FILENAME)
    acc_log_file_path = os.path.join(input_dir, ACC_LOG_FILENAME)

    print(f"[Paths] caption_file: {args.caption_file}")
    print(f"[Paths] prompt_file: {args.prompt_file}")
    print(f"[Paths] result_file: {result_file_path}")
    print(f"[Paths] acc_log_file: {acc_log_file_path}")

    caption_map = load_caption_map(args.caption_file)
    prompt_data = load_prompts(args.prompt_file)
    tasks = build_tasks(prompt_data, caption_map)

    run_multiprocess_tasks(
        tasks,
        result_file_path=result_file_path,
        acc_log_file_path=acc_log_file_path,
        num_processes=args.num_processes,
    )

if __name__ == "__main__":
    main()
