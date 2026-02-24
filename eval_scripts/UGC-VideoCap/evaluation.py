import os
import re
import json
from datetime import datetime
from tqdm import tqdm
import multiprocessing
import sys
from pathlib import Path

try:
    with open("apikey.txt", "r") as f:
        api_key = f.read()
except:
    api_key = ''

def call_gpt4o(msg):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o", 
                messages=msg, 
                api_key=api_key, 
                request_timeout=5)
            break
        except:
            print("Timeout, retrying...")
            time.sleep(5)

    output_text = response['choices'][0]['message']['content']
    return output_text


INPUT_JSONL = sys.argv[1]
OUTPUT_JSON = sys.argv[2]
script_dir = Path(__file__).resolve().parent
GT_PATH = script_dir / "final_caption_qa.json"
NUM_PROCESSES = 10

def safe_parse_evaluation(response_str):
    try:
        match = re.search(r"\{.*\}", response_str, re.DOTALL)
        if match:
            return json.loads(match.group(0).replace("'", '"'))
    except Exception as e:
        print(f"Error parsing evaluation output: {e}")
    return {}

def evaluate_caption(sample_id, pred_caption, true_caption):

    system_msg = (
        "You are an assistant that compares a ground truth video description and a predicted video description. "
        "Evaluate the predicted description against the ground truth on the following three dimensions:\n"
        "1. **visual**: the accuracy and completeness of visual content including the scene setting, background, characters or objects, their actions or interactions, and any OCR text.\n"
        "2. **audio**: how well it captures voices, background music, sound effects, and their emotional tone.\n"
        "3. **details**: the completeness, thematic consistency, purpose, coherence, and integration of multimodal content.\n\n"

        "For each dimension, assign an integer score from 1 to 5, following these detailed grading criteria:\n\n"

        "**Score 1:** The description is mostly irrelevant or misleading. It misrepresents or omits most key information. "
        "At least 3 important elements are missing or incorrect. Severe hallucinations may be present.\n\n"

        "**Score 2:** The description captures a few elements (1-2 aspects) but is vague or inaccurate for the rest. "
        "It is poorly structured or confusing, with major omissions or incorrect details.\n\n"

        "**Score 3:** The description aligns with the video on most elements (3 or more), but lacks depth or specificity. "
        "Some key details are missing, or minor factual errors exist. It's generally correct but too generic or incomplete.\n\n"

        "**Score 4:** A mostly accurate and complete description. Captures nearly all key information (4+ aspects), "
        "with clear structure and appropriate level of detail. Minor omissions or simplifications are acceptable.\n\n"

        "**Score 5:** Exceptionally accurate and detailed. Covers all relevant aspects thoroughly, with well-integrated information. "
        "Captures subtle nuances (e.g., emotion, scene dynamics, audio-visual interplay) and reads like it was written by a domain expert.\n\n"


        "Respond only with a valid Python dictionary in this format:\n"
        "{'visual': int, 'audio': int, 'details': int}"
    )

    user_msg = (
        f"Sample ID: {sample_id}\n"
        f"Predicted Description: {pred_caption}\n"
        f"Ground Truth Description: {true_caption}\n"
    )

    for i in range(20):
        try:
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            content = call_gpt4o(messages)
            print(content)
            eval_dict = safe_parse_evaluation(content)

            v = int(eval_dict["visual"])
            a = int(eval_dict["audio"])
            d = int(eval_dict["details"])
            return {'visual': v, 'audio': a, 'details': d}
        except Exception as e:
            print(f"Error evaluating sample {sample_id}: {e}")

    v = a = d = 0
    return {'visual': v, 'audio': a, 'details': d}

def process_sample(args):
    video_id, pred_caption, true_caption = args
    scores = evaluate_caption(video_id, pred_caption, true_caption)
    avg_score = sum(scores.values()) / len(scores)
    
    result_data = {
        'visual_score': scores['visual'],
        'audio_score': scores['audio'],
        'details_score': scores['details'],
        'average_score': avg_score
    }
    return video_id, result_data

def run_evaluation():
    try:
        with open(GT_PATH, 'r', encoding='utf-8') as f_gt:
            gt_anno = json.load(f_gt)["samples"]
        gt = {anno["video_id"]: anno["answer"] for anno in gt_anno}
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return

    results = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                results = existing.get('evaluations', {})
        except Exception as e:
            print(f"Warning: Failed to read previous results: {e}")
            results = {}
    
    tasks = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line)
                video_id = sample.get('video_id')
                pred_caption = sample.get('caption', '').strip()
        
                if not video_id or not pred_caption:
                    continue

                if video_id in results:
                    continue
                if video_id == "PI-0abb79ae-18ae-40d9-a66f-3ad7744a6095": # no GT
                    continue
                
                true_caption = gt[video_id]
                tasks.append((video_id, pred_caption, true_caption))
            except json.JSONDecodeError:
                continue
    
    if not tasks:
        print("No new samples to evaluate.")
        return

    print(f"Found {len(tasks)} new samples to evaluate.")

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        with tqdm(total=len(tasks), desc="Evaluating samples") as pbar:
            for video_id, result_data in pool.imap_unordered(process_sample, tasks):
                if result_data:
                    results[video_id] = result_data
                    
                    all_scores = list(results.values())
                    count = len(all_scores)
                    sum_visual = sum(s['visual_score'] for s in all_scores)
                    sum_audio = sum(s['audio_score'] for s in all_scores)
                    sum_details = sum(s['details_score'] for s in all_scores)
                    sum_avg = sum(s['average_score'] for s in all_scores)

                    current_output = {
                        'evaluations': results,
                        'overall_visual_average': round(sum_visual / count / 5.0 * 100, 2),
                        'overall_audio_average': round(sum_audio / count / 5.0 * 100, 2),
                        'overall_details_average': round(sum_details / count / 5.0 * 100, 2),
                        'overall_average_percent': round((sum_avg / count) / 5.0 * 100, 2)
                    }

                    with open(OUTPUT_JSON, 'w', encoding='utf-8') as out_f:
                        json.dump(current_output, out_f, indent=2, ensure_ascii=False)
                
                pbar.update(1)

    print("\nEvaluation complete.")
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            final_output = json.load(f)
            print(f"Overall visual avg: {final_output.get('overall_visual_average', 0):.2f}, "
                  f"audio avg: {final_output.get('overall_audio_average', 0):.2f}, "
                  f"details avg: {final_output.get('overall_details_average', 0):.2f}, "
                  f"overall%: {final_output.get('overall_average_percent', 0):.2f}")

if __name__ == '__main__':
    run_evaluation()