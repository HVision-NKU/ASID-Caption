"""
ASID-Captioner: Audiovisual Video Captioning with Qwen2.5-Omni
Simplified Version: Remove global task pool/completed files, judge pending videos based on temporary/final files
Core Features: Multi-GPU support, breakpoint resume, fault tolerance, result merging
"""
import json
import os
import random
import re
import sys
import time
import logging
import gc
import hashlib
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Value, Lock
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# -------------------------- Core Constants (Essential Only) --------------------------
VIDEO_MAX_PIXELS = 401408  # Max pixels for video processing (512*28*28)
VIDEO_TOTAL_PIXELS = 20070400  # Total pixels limit for video
USE_AUDIO_IN_VIDEO = True  # Enable audio processing in video
os.environ['VIDEO_MAX_PIXELS'] = str(VIDEO_TOTAL_PIXELS)

# -------------------------- Caption Prompt Templates --------------------------
PROMPT_LIST = [
    "Provide a comprehensive description of all the content in the video, leaving out no details. Be sure to include as much of the audio information as possible, and ensure that your descriptions of the audio and video are closely aligned.",
    "Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.",
    "Please describe all the information in the video without sparing every detail in it. As you describe, you should also describe as much of the information in the audio as possible, and pay attention to the synchronization between the audio and video descriptions.",
    "Offer a detailed description of the video, making sure to include every detail. Also, incorporate as much information from the audio as you can, and ensure that your descriptions of the audio and video are in sync.",
    "Describe every aspect of the video in full detail, covering all the information it contains. Additionally, include as much of the audio content as you can, and make sure your descriptions of the audio and video are synchronized.",
    "Please provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so.",
    "Give a detailed account of everything in the video, capturing all the specifics. While doing so, also include as much information from the audio as possible, ensuring that the descriptions of audio and video are well-synchronized."
]

def get_args():
    """Parse command line arguments (core parameters only)"""
    import argparse
    parser = argparse.ArgumentParser(description='ASID-Captioner (Simplified - Multi-GPU/Breakpoint Resume)')
    parser.add_argument('--video_dir', type=str, default='demo_test', help='Directory containing video files')
    parser.add_argument('--model_path', type=str, default='Model/ASID-Captioner-3B', help='Path to Qwen2.5-Omni model checkpoint')
    parser.add_argument('--result_dir', type=str, default="./output/", help="Directory to save results (including temporary/final files)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU (reserved for future use)")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--resume", action="store_true", help="Resume from breakpoint (skip processed videos)")
    return parser.parse_args()

def get_unique_video_id(video_path):
    """Generate unique video ID (MD5 based on absolute path to avoid duplication)"""
    abs_path = os.path.abspath(video_path)
    md5_hash = hashlib.md5(abs_path.encode('utf-8')).hexdigest()[:8]
    filename = os.path.basename(video_path)
    name_no_ext = os.path.splitext(filename)[0]
    return f"{name_no_ext}_{md5_hash}"

def clean_invalid_chars(s):
    """Clean invalid characters (avoid JSON parsing errors)"""
    if not isinstance(s, str):
        s = str(s)
    return s.replace('\x00', '').replace('\ufffd', '').strip()

def setup_logger(rank, log_dir):
    """Setup independent logger for each GPU worker"""
    logger = logging.getLogger(f"worker_{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if not logger.handlers:
        log_file = os.path.join(log_dir, f"worker_{rank}.log")
        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def load_model_and_processor(model_path, device):
    """Load Qwen2.5-Omni model and processor (BF16 + Flash Attention 2)"""
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2"
    )
    model.disable_talker()  # Disable unused talker module
    
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor

def generate_caption(model, processor, video_path, prompt, device):
    """Generate video caption (core inference logic)"""
    # Build Qwen2.5-Omni conversation template
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "total_pixels": VIDEO_TOTAL_PIXELS,  
                    "max_pixels": VIDEO_MAX_PIXELS,
                },
                {"type": "text", "text": prompt}
            ],
        },
    ]
    
    # Process conversation and multimodal inputs
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    
    # Prepare model inputs (move to target GPU)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    inputs = inputs.to(device).to(model.dtype)
    
    # Generate caption (fixed parameters for consistency)
    with torch.no_grad():
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            do_sample=False,
            thinker_max_new_tokens=4096,
            repetition_penalty=1.1,
            use_cache=True
        )

    # Decode and extract results
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.split("\nassistant\n")[-1]

def get_all_completed_video_ids(result_dir, model_name):
    """Get processed video IDs from temporary/final files (replace global completed file)"""
    completed_ids = set()
    
    # 1. Final merged file
    final_file = os.path.join(result_dir, f"{model_name}_captions.jsonl")
    if os.path.exists(final_file):
        with open(final_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    if line.strip():
                        data = json.loads(line)
                        completed_ids.add(data['video_id'])
                except:
                    continue
        print(f"Loaded processed IDs from final file: {len(completed_ids)}")
    
    # 2. Temporary files (unmerged results)
    tmp_pattern = re.compile(r"tmp_captions_part\d+\.jsonl")
    for filename in os.listdir(result_dir):
        if tmp_pattern.match(filename):
            tmp_file = os.path.join(result_dir, filename)
            if os.path.exists(tmp_file):
                with open(tmp_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            if line.strip():
                                data = json.loads(line)
                                completed_ids.add(data['video_id'])
                        except:
                            continue
            print(f"Loaded processed IDs from temporary file {filename}, total: {len(completed_ids)}")
    
    return completed_ids

def get_all_undone_videos(video_dir, completed_ids):
    """Get list of all unprocessed videos"""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    undone_videos = []
    
    if os.path.exists(video_dir):
        for filename in os.listdir(video_dir):
            if filename.lower().endswith(video_extensions):
                video_path = os.path.abspath(os.path.join(video_dir, filename))
                video_id = get_unique_video_id(video_path)
                if video_id not in completed_ids:
                    undone_videos.append({"video_path": video_path, "video_id": video_id})
    
    print(f"Total number of unprocessed videos: {len(undone_videos)}")
    return undone_videos

def worker_proc(rank, gpu_id, model_path, tmp_out_path, video_list, counter, lock, log_dir):
    """Simplified Worker Process: Process assigned video list"""
    # Initialize logger and device
    logger = setup_logger(rank, log_dir)
    logger.info(f"Worker {rank} started (GPU {gpu_id}, PID: {os.getpid()})")
    device = f"cuda:{gpu_id}"

    # Load model (exit if failed)
    try:
        model, processor = load_model_and_processor(model_path, device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        return

    # Process assigned video list
    for task in video_list:
        video_id = task['video_id']
        video_path = task['video_path']
        
        try:
            # Randomly select prompt
            prompt = random.choice(PROMPT_LIST)
            # Generate caption
            caption = generate_caption(model, processor, video_path, prompt, device)
            
            # Prepare result
            result = {
                'video_id': clean_invalid_chars(video_id),
                'video_path': clean_invalid_chars(video_path),
                'prompt': clean_invalid_chars(prompt),
                'caption': clean_invalid_chars(caption),
                'timestamp': clean_invalid_chars(time.strftime("%Y-%m-%d %H:%M:%S"))
            }

            # Write to temporary file (append mode)
            with open(tmp_out_path, 'a', encoding='utf-8') as fout:
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                fout.flush()

            # Update counter
            with lock:
                counter.value += 1
            logger.info(f"Processed successfully: {video_id} (total completed: {counter.value})")

        except Exception as e:
            logger.error(f"Failed to process {video_id}: {str(e)}", exc_info=True)
            continue

    # Clean up GPU memory
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    logger.info(f"Worker {rank} completed all assigned tasks")

def merge_results(tmp_files, final_path):
    """Merge temporary files to final file (remove duplicates)"""
    merged_ids = set()
    
    with open(final_path, 'w', encoding='utf-8') as fout:  # Overwrite to avoid duplication
        for tmp in tmp_files:
            if os.path.exists(tmp):
                with open(tmp, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        line = clean_invalid_chars(line.strip())
                        if line:
                            try:
                                data = json.loads(line)
                                vid = data['video_id']
                                if vid not in merged_ids:
                                    merged_ids.add(vid)
                                    fout.write(line + '\n')
                            except:
                                continue
                os.remove(tmp)  # Delete temporary file after merging
    
    print(f"Merging completed, {len(merged_ids)} unique results saved to {final_path}")

def main():
    """Main Function: Simplified Version (Remove global task pool/completed files)"""
    args = get_args()
    
    # Create directories
    os.makedirs(args.result_dir, exist_ok=True)
    log_dir = os.path.join(args.result_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Path configuration
    model_name = os.path.basename(os.path.normpath(args.model_path))
    final_output_path = os.path.join(args.result_dir, f"{model_name}_captions.jsonl")
    tmp_files = [os.path.join(args.result_dir, f"tmp_captions_part{rank}.jsonl") for rank in range(args.num_gpus)]

    # Load processed video IDs (breakpoint resume)
    completed_video_ids = set()
    if args.resume:
        completed_video_ids = get_all_completed_video_ids(args.result_dir, model_name)
    print(f"Total number of processed videos: {len(completed_video_ids)}")

    # Get all unprocessed videos
    undone_videos = get_all_undone_videos(args.video_dir, completed_video_ids)
    if not undone_videos:
        print("All videos have been processed, exiting")
        return

    # Limit GPU count to available GPUs
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs for processing tasks")

    # Distribute video list evenly to each GPU
    video_chunks = []
    chunk_size = len(undone_videos) // num_gpus
    for i in range(num_gpus):
        start = i * chunk_size
        end = start + chunk_size if i < num_gpus - 1 else len(undone_videos)
        video_chunks.append(undone_videos[start:end])
    print(f"Task allocation completed: {[len(chunk) for chunk in video_chunks]}")

    # Initialize counter and lock
    counter = Value('i', 0)
    lock = Lock()

    # Start Worker processes
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=worker_proc,
            args=(rank, rank, args.model_path, tmp_files[rank], video_chunks[rank], counter, lock, log_dir),
            name=f"worker_{rank}"
        )
        p.start()
        processes.append(p)
        print(f"Started Worker {rank} (GPU {rank}, PID: {p.pid}), assigned {len(video_chunks[rank])} tasks")
    
    # Global progress bar
    with tqdm(total=len(undone_videos), desc="Global Processing Progress") as pbar:
        last_count = 0
        while True:
            with lock:
                current_done = counter.value
            if current_done > last_count:
                pbar.update(current_done - last_count)
                last_count = current_done
            # Exit when all tasks completed or all processes exited
            if current_done >= len(undone_videos) or all(not p.is_alive() for p in processes):
                break
            time.sleep(1)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Merge temporary files to final file
    print("Starting to merge temporary result files...")
    merge_results(tmp_files, final_output_path)
    print("All tasks completed!")

if __name__ == '__main__':
    # Multi-GPU compatible startup method
    mp.set_start_method('spawn', force=True)
    main()