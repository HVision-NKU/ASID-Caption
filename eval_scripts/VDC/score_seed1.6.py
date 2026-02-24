import os
import json
import random
import ast
import argparse
import time
import re
import threading  
from tqdm import tqdm
import concurrent.futures
from openai import OpenAI 


# ===================== RPM Rate Limiting Configuration =====================
MAX_RPM = 1800  # Maximum requests per minute
MIN_REQUEST_INTERVAL = 60.0 / MAX_RPM  # Minimum interval between requests (seconds)
_last_request_timestamp = 0.0  # Timestamp of last API request
_request_lock = threading.Lock()  # Thread lock for rate limiting safety

def enforce_1000_rpm_limit():
    """
    Enforce API rate limit (1000 RPM) - thread safe
    Ensures minimum interval between consecutive API requests
    """
    global _last_request_timestamp
    with _request_lock:
        current_time = time.time()
        time_elapsed_since_last_request = current_time - _last_request_timestamp
        if time_elapsed_since_last_request < MIN_REQUEST_INTERVAL:
            sleep_duration = MIN_REQUEST_INTERVAL - time_elapsed_since_last_request
            time.sleep(sleep_duration)
        _last_request_timestamp = time.time()
# ===========================================================================

def get_doubao_client():
    """Create and return Doubao OpenAI-compatible client"""
    return OpenAI(
        base_url="https://ark-cn-beijing.bytedance.net/api/v3",
        api_key=os.environ.get("ARK_API_KEY"),
    )


def gener_pred_response(pred_cap, q, max_retry=3):
    """Generate answer based on video description and question (with retry logic)"""
    # Define retryable exceptions
    retry_exceptions = [
        "qpm limit",
        "reach token limit",
        "Request timed out",
        "The service is temporarily unable to process your request",
        "upstream failed to respond",
        "502 Bad Gateway",
        "429 Too Many Requests",
        "Retrying request to"
    ]
    
    retry = 0
    client = get_doubao_client()
    while retry < max_retry:
        try:
            model_name = "ep-20250814094952-4t6vz"
            max_tokens = 1000
            
            system_message = (
                "You are an intelligent chatbot designed for providing accurate answers to questions related to the content based on a detailed description of a video or image.\n"
                "Here's how you can accomplish the task:\n"
                "------\n"
                "##INSTRUCTIONS: \n"
                "- Read the detailed description carefully.\n"
                "- Answer the question only based on the detailed description.\n"
                "- The answer should be a short sentence or phrase.\n"
            )
            
            user_message = (
                "Please provide accurate answers to questions related to the content based on a detailed description of a video or image:\n\n"
                f"detailed description: {pred_cap}, question: {q}\n"
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide short but accurate answer."
            )

            enforce_1000_rpm_limit()
            
            # Doubao API call (OpenAI SDK compatible)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens,
            )
            
            response = completion.choices[0].message.content
            print('response', response)
            # Clean response content
            response = response.strip()
            response = re.sub(r'\s+', ' ', response)
            return response
            
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Failed to generate answer (retry {retry+1}/{max_retry}): {error_msg}")
            
            # Check if retry is needed
            if any(exception in error_msg for exception in retry_exceptions):
                retry += 1
                # Exponential backoff
                sleep_time = 2 ** retry + random.uniform(0, 1)
                time.sleep(sleep_time)
            else:
                return ""
    
    # Max retries reached
    print(f"[ERROR] Max retries ({max_retry}) reached for answer generation, returning empty result")
    return ""


def gener_pred_score(qa, max_retry=3):
    """Evaluate answer correctness (with retry logic)"""
    # Define retryable exceptions
    retry_exceptions = [
        "qpm limit",
        "reach token limit",
        "Request timed out",
        "The service is temporarily unable to process your request",
        "upstream failed to respond",
        "502 Bad Gateway",
        "429 Too Many Requests",
        "Retrying request to"
    ]
    
    retry = 0
    client = get_doubao_client()
    while retry < max_retry:
        try:
            model_name = "ep-20250814094952-4t6vz"
            max_tokens = 256
            
            system_message = (
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n"
                "------\n"
                "##INSTRUCTIONS: \n"
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the prediction compared to the answer."
            )
            
            user_message = (
                "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {qa['question']}\n"
                f"Correct Answer: {qa['answer']}\n"
                f"Predicted Answer: {qa['response']}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'yes', 'score': 4}."
            )

            enforce_1000_rpm_limit()
            
            # Doubao API call (OpenAI SDK compatible)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens,
            )
            
            score_response = completion.choices[0].message.content.strip()
            
            # Clean response format
            if score_response.startswith("```json"):
                score_response = score_response.replace("```json", "").replace("```", "").strip()
            elif score_response.startswith("```python"):
                score_response = score_response.replace("```python", "").replace("```", "").strip()
            
            # Fix potential format issues
            if not score_response.startswith('{'):
                score_response = '{' + score_response
            if not score_response.endswith('}'):
                score_response = score_response + '}'
            
            score_response = score_response.replace("True", "true").replace("False", "false")
            score_response = score_response.replace("}, {", "}, {").replace("}{", "}, {")
            score_response = score_response.replace(",\n}", "\n}").replace(", }", "}").replace(",}", "}")
            
            return score_response
            
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Failed to score answer (retry {retry+1}/{max_retry}): {error_msg}")
            
            # Check if retry is needed
            if any(exception in error_msg for exception in retry_exceptions):
                retry += 1
                # Exponential backoff
                sleep_time = 2 ** retry + random.uniform(0, 1)
                time.sleep(sleep_time)
            else:
                return "{'pred': 'no', 'score': 0}"
    
    # Max retries reached
    print(f"[ERROR] Max retries ({max_retry}) reached for scoring, returning default result")
    return "{'pred': 'no', 'score': 0}"


def process_video(video_id, pred, answer, result_gtqa_list):
    """Process single video: generate answers and evaluate correctness"""
    tp_result_dict = {
        'id': video_id,
        'pred_caption': pred,
        'gt_caption': answer,
        'qa_tp_list': []
    }

    qa_list = []

    # Generate answers (concurrent processing)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_qa = {
            executor.submit(
                gener_pred_response,
                pred_cap=pred,
                q=qa_dict['question']
            ): qa_dict
            for qa_dict in result_gtqa_list
        }

        for future in concurrent.futures.as_completed(future_to_qa):
            qa_dict = future_to_qa[future]
            try:
                response = future.result()
                qa_list.append({
                    "question": qa_dict['question'],
                    "answer": qa_dict['answer'],
                    "response": response
                })
            except Exception as e:
                print(f"[ERROR] generate response failed for {video_id}: {e}")

    # Evaluate answers (concurrent processing)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_qa = {
            executor.submit(
                gener_pred_score,
                qa=qa
            ): qa
            for qa in qa_list
        }

        for future in concurrent.futures.as_completed(future_to_qa):
            qa = future_to_qa[future]
            try:
                score_response = future.result()
                response_dict = ast.literal_eval(score_response)
                qa.update(response_dict)
                tp_result_dict['qa_tp_list'].append(qa)
            except Exception as e:
                print(f"[ERROR] score evaluation failed for {video_id}: {e}")
                # Add default values on evaluation failure
                qa.update({'pred': 'no', 'score': 0})
                tp_result_dict['qa_tp_list'].append(qa)

    # Calculate total score and accuracy
    total_score, total_acc = 0, 0
    for qa in tp_result_dict['qa_tp_list']:
        total_score += float(qa.get('score', 0))
        if qa.get('pred') == 'yes':
            total_acc += 1

    tp_score = total_score / len(tp_result_dict['qa_tp_list']) if tp_result_dict['qa_tp_list'] else 0
    tp_acc = total_acc / len(tp_result_dict['qa_tp_list']) if tp_result_dict['qa_tp_list'] else 0

    return tp_score, tp_acc, tp_result_dict


def main():
    parser = argparse.ArgumentParser(description="Process VDC results and evaluate captions using Doubao API.")
    parser.add_argument('raw_file', type=str, help='Path to the raw input JSON file.')
    parser.add_argument('output_file', type=str, help='Path to the output JSONL file.')
    parser.add_argument('tp_qa_path', type=str, default='detailed.jsonl', 
                      help='Path to the TP QA JSONL file (default: eval_scripts/VDC/detailed.jsonl).')
    parser.add_argument('--num_workers', type=int, default=256, help='Number of parallel workers for processing videos')
    parser.add_argument('--max_retry', type=int, default=8, help='Maximum number of retries for API calls')
    args = parser.parse_args()

    # Validate API key exists
    if not os.environ.get("ARK_API_KEY"):
        raise ValueError("ARK_API_KEY environment variable not set")

    # Load TP QA data
    tp_gt_qa_dict = {}
    with open(args.tp_qa_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            tp_gt_qa_dict.update(data)

    # Load reference captions
    captions_dict = {}
    captions_path = 'eval_scripts/VDC/VDC_1k_captions.jsonl'
    with open(captions_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            key = data['video_id']
            caption = data['captions']['detailed_caption']
            captions_dict[key] = caption

    # Load prediction results
    preds_dict = {}
    with open(args.raw_file, "r") as f:
        for line in f:
            data = json.loads(line)
            preds_dict[data["video_id"]] = data["caption"]

    result_list, tp_scores, tp_accs = [], [], []

    # Filter valid videos to process
    valid_video_ids = [vid for vid in preds_dict if vid in tp_gt_qa_dict]
    video_count = len(valid_video_ids)
    print(f"Number of valid videos to process: {video_count}")

    # Process all videos with global progress bar
    global_pbar = tqdm(total=video_count, desc="Global Video Processing Progress", leave=True, ncols=80)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_vid = {
            executor.submit(
                process_video,
                video_id,
                preds_dict[video_id],
                captions_dict[video_id],
                tp_gt_qa_dict.get(video_id, [])
            ): video_id
            for video_id in valid_video_ids
        }

        for future in concurrent.futures.as_completed(future_to_vid):
            video_id = future_to_vid[future]
            try:
                tp_score, tp_acc, tp_result_dict = future.result()
                if tp_score is not None and tp_acc is not None:
                    tp_scores.append(tp_score)
                    tp_accs.append(tp_acc)
                    result_list.append({
                        'id': video_id,
                        'tp_score': tp_score,
                        'tp_acc': tp_acc,
                        'qa_tp_list': tp_result_dict['qa_tp_list']
                    })
            except Exception as e:
                print(f"[ERROR] processing video {video_id} failed: {e}")
            finally:
                global_pbar.update(1)

    global_pbar.close()

    # Calculate overall scores
    if not tp_scores:
        print("Warning: No valid video processing results, setting total score to 0")
        overall_tp_score = 0.0
        overall_tp_acc = 0.0
    else:
        overall_tp_score = sum(tp_scores) / len(tp_scores)
        overall_tp_acc = sum(tp_accs) / len(tp_accs)

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as file:
        for item in result_list:
            file.write(json.dumps(item) + '\n')
        file.write(json.dumps({'tp_score': overall_tp_score, 'tp_acc': overall_tp_acc}) + '\n')

    print(f"Results saved to {args.output_file}")
    print(f"Overall TP Score: {overall_tp_score:.4f}")
    print(f"Overall TP Accuracy: {overall_tp_acc:.4f}")


if __name__ == "__main__":
    main()