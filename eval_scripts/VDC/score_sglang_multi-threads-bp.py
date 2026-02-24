import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import random
import ast
import argparse
from tqdm import tqdm
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
import concurrent.futures


@function
def gener_pred_response(s, pred_cap, q):
    s += system(
        "You are an intelligent chatbot designed for providing accurate answers to questions related to the content based on a detailed description of a video or image."
        "Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Read the detailed description carefully.\n"
        "- Answer the question only based on the detailed description.\n"
        "- The answer should be a short sentence or phrase.\n"
    )
    s += user(
        "Please provide accurate answers to questions related to the content based on a detailed description of a video or image:\n\n"
        f"detailed description: {pred_cap}, question: {q}"
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide short but accurate answer."
    )
    s += assistant(gen("answer_1", max_tokens=256))


@function
def gener_pred_score(s, qa):
    s += system(
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
        "- Consider synonyms or paraphrases as valid matches.\n"
        "- Evaluate the correctness of the prediction compared to the answer."
    )
    s += user(
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {qa['question']}\n"
        f"Correct Answer: {qa['answer']}\n"
        f"Predicted Answer: {qa['response']}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        "For example, your response should look like this: {'pred': 'yes', 'score': 4}."
    )
    s += assistant(gen("answer_1", max_tokens=256))


def process_video(video_id, pred, answer, result_gtqa_list):
    tp_result_dict = {
        'id': video_id,
        'pred_caption': pred,
        'gt_caption': answer,
        'qa_tp_list': []
    }

    qa_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_qa = {
            executor.submit(
                gener_pred_response.run,
                pred_cap=pred,
                q=qa_dict['question']
            ): qa_dict
            for qa_dict in result_gtqa_list
        }

        for future in concurrent.futures.as_completed(future_to_qa):
            qa_dict = future_to_qa[future]
            try:
                state = future.result()
                qa_list.append({
                    "question": qa_dict['question'],
                    "answer": qa_dict['answer'],
                    "response": state["answer_1"]
                })
            except Exception as e:
                print(f"[ERROR] generate response failed for {video_id}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_qa = {
            executor.submit(
                gener_pred_score.run,
                qa=qa
            ): qa
            for qa in qa_list
        }

        for future in concurrent.futures.as_completed(future_to_qa):
            qa = future_to_qa[future]
            try:
                state = future.result()
                response_dict = ast.literal_eval(state["answer_1"])
                qa.update(response_dict)
                tp_result_dict['qa_tp_list'].append(qa)
            except Exception as e:
                print(f"[ERROR] score evaluation failed for {video_id}: {e}")

    total_score, total_acc = 0, 0
    for qa in qa_list:
        total_score += float(qa.get('score', 0))
        if qa.get('pred') == 'yes':
            total_acc += 1

    tp_score = total_score / len(qa_list) if qa_list else 0
    tp_acc = total_acc / len(qa_list) if qa_list else 0

    return tp_score, tp_acc, tp_result_dict


def main():
    parser = argparse.ArgumentParser(description="Process VDC results and evaluate captions.")
    parser.add_argument('--raw_file', type=str, help='Path to the raw input JSON file.')
    parser.add_argument('--output_file', type=str, help='Path to the output JSONL file.')
    parser.add_argument('--tp_qa_path', type=str, default='eval_scripts/VDC/detailed.jsonl', help='Path to the TP QA JSONL file (default: post_eval/background.jsonl).')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for processing videos.')
    args = parser.parse_args()
    set_default_backend(RuntimeEndpoint("http://127.0.0.1:12366"))

    tp_gt_qa_dict = {}
    with open(args.tp_qa_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            tp_gt_qa_dict.update(data)

    captions_dict = {}
    captions_path = 'eval_scripts/VDC/VDC_1k_captions.jsonl'
    with open(captions_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            key = data['video_id']
            caption = data['captions']['detailed_caption']
            captions_dict[key] = caption

    set_default_backend(RuntimeEndpoint("http://127.0.0.1:12366"))

    preds_dict = {}
    with open(args.raw_file, "r") as f:
        for line in f:
            data = json.loads(line)
            preds_dict[data["video_id"]] = data["caption"]

    result_list, tp_scores, tp_accs = [], [], []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_vid = {
            executor.submit(
                process_video,
                video_id,
                pred,
                captions_dict[video_id],
                tp_gt_qa_dict.get(video_id, [])
            ): video_id
            for video_id, pred in preds_dict.items()
            if video_id in tp_gt_qa_dict
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_vid), total=len(future_to_vid)):
            video_id = future_to_vid[future]
            try:
                tp_score, tp_acc, tp_result_dict = future.result()
                if tp_score and tp_acc:
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

    tp_score = sum(tp_scores) / len(tp_scores) if tp_scores else 0
    tp_acc = sum(tp_accs) / len(tp_accs) if tp_accs else 0

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as file:
        for item in result_list:
            file.write(json.dumps(item) + '\n')
        file.write(json.dumps({'tp_score': tp_score, 'tp_acc': tp_acc}) + '\n')

    print(f"Results saved to {args.output_file}")
    print(f"Overall TP Score: {tp_score}")
    print(f"Overall TP Accuracy: {tp_acc}")


if __name__ == "__main__":
    main()