import json
import os
import random
import re
import sys
import time
import logging
import gc
from tqdm import tqdm
from math import ceil
import multiprocessing as mp
from multiprocessing import Value, Lock
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from pathlib import Path

# --- Constants Definition ---
VIDEO_MAX_PIXELS = 401408  # 512*28*28
VIDEO_TOTAL_PIXELS = 20070400  # 512*28*28*50
USE_AUDIO_IN_VIDEO = True
os.environ['VIDEO_MAX_PIXELS'] = str(VIDEO_TOTAL_PIXELS)
script_dir = Path(__file__).resolve().parent

# --- Prompt Templates ---
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
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='ASID-Caption Video Captioning with Multi-GPU Support')
    parser.add_argument('--video_dir', type=str, default='Charades/video', help='Directory containing video files')
    parser.add_argument('--model_path', type=str, default='Model/ASID-Captioner-3B', help='Path to the ASID-Caption model checkpoint')
    parser.add_argument('--result_dir', type=str, default="./output/", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return parser.parse_args()


class VideoCaptionDataset(Dataset):
    """Dataset class for video captioning task"""
    def __init__(self, work_items):
        self.work_items = work_items

    def __len__(self):
        return len(self.work_items)

    def __getitem__(self, idx):
        item = self.work_items[idx]
        video_path = item['video_path']
        # Randomly select a prompt for each video
        prompt = random.choice(PROMPT_LIST)
        return {
            'video_path': video_path,
            'prompt': prompt,
            'video_id': item['video_id']
        }


def collate_fn(batch):
    """Simple collation function to maintain batch structure"""
    return batch


def setup_logger(rank, log_dir):
    """Set up logger for each worker process"""
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
    """Load model and processor on specified device"""
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


def generate_caption(model, processor, file_path, prompt, device):
    """Generate caption for a single video on specified device"""
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
                    "video": file_path,
                    "total_pixels": VIDEO_TOTAL_PIXELS,  
                    "max_pixels": VIDEO_MAX_PIXELS,
                },
                {"type": "text", "text": prompt}
            ],
        },
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    
    inputs = processor(
        text=text, 
        audio=audios, 
        images=images, 
        videos=videos, 
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    inputs = inputs.to(device).to(model.dtype)
    
    with torch.no_grad():
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            do_sample=False,
            thinker_max_new_tokens=4096,
            repetition_penalty=1.1,
            use_cache=True
        )

    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    model_generation = text.split("\nassistant\n")[-1]
    return model_generation


def worker_proc(rank, gpu_id, model_path, work_items, tmp_out_path, counter, lock, log_dir, args):
    """Worker process function for multi-GPU processing"""
    # Initialize logger
    logger = setup_logger(rank, log_dir)
    logger.info(f"Worker {rank} started on GPU {gpu_id}, PID: {os.getpid()}")

    # Initialize device
    device = f"cuda:{gpu_id}"
    logger.info(f"Using device: {device}")

    # Load model
    try:
        model, processor = load_model_and_processor(model_path, device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        return

    # Prepare dataset and dataloader
    dataset = VideoCaptionDataset(work_items)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1
    )

    # Load processed items for resume
    processed_items = set()
    if args.resume and os.path.exists(tmp_out_path):
        try:
            with open(tmp_out_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    processed_items.add(data['video_id'])
            logger.info(f"Resumed {len(processed_items)} processed items")
        except Exception as e:
            logger.warning(f"Failed to load resume data: {str(e)}")

    # Process data
    fout = open(tmp_out_path, 'a', encoding='utf-8')  # Append mode for resume
    try:
        for batch in tqdm(dataloader, desc=f"Worker {rank}", position=rank):
            for item in batch:
                video_id = item['video_id']
                if video_id in processed_items:
                    logger.info(f"Skipping already processed video: {video_id}")
                    continue

                try:
                    # Generate caption
                    caption = generate_caption(
                        model, 
                        processor, 
                        item['video_path'], 
                        item['prompt'], 
                        device
                    )
                    
                    # Save result
                    result = {
                        'video_id': video_id,
                        'video_path': item['video_path'],
                        'prompt': item['prompt'],
                        'caption': caption,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                    fout.flush()

                    # Update counter
                    with lock:
                        counter.value += 1

                except Exception as e:
                    logger.error(f"Error processing {video_id}: {str(e)}", exc_info=True)
                    continue

        logger.info(f"Worker {rank} finished. Processed {len(work_items) - len(processed_items)} videos")

    finally:
        fout.close()
        # Clean up memory
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


def get_video_files(video_dir):
    """Get all video files from directory (supports common video extensions)"""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    video_files = []
    for filename in os.listdir(video_dir):
        if filename.lower().endswith(video_extensions):
            video_path = os.path.abspath(os.path.join(video_dir, filename))
            video_id = os.path.splitext(filename)[0] 
            video_files.append({
                'video_path': video_path,
                'video_id': video_id
            })
    return video_files


def get_completed_items(fout_path):
    """Get IDs of completed videos for resume functionality"""
    completed = set()
    if os.path.exists(fout_path):
        with open(fout_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed.add(data['video_id'])
                except Exception as e:
                    print(f"Warning: Invalid line in {fout_path}: {line.strip()}")
    return completed


def split_work_items(work_items, num_gpus, completed_items):
    """Split work items across GPUs, excluding completed items"""
    filtered = [item for item in work_items if item['video_id'] not in completed_items]
    chunks = []
    for i in range(num_gpus):
        chunks.append(filtered[i::num_gpus])  # Even split across GPUs
    return chunks


def merge_results(tmp_files, final_path):
    """Merge all temporary result files into final output"""
    with open(final_path, 'a', encoding='utf-8') as fout:
        for tmp in tmp_files:
            if os.path.exists(tmp):
                with open(tmp, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        fout.write(line)
                os.remove(tmp)  # Clean up temporary files


def main():
    """Main function for multi-GPU video captioning"""
    args = get_args()
    
    # Create output directories
    os.makedirs(args.result_dir, exist_ok=True)
    log_dir = os.path.join(args.result_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Get all video files
    print(f"Scanning video directory: {args.video_dir}")
    work_items = get_video_files(args.video_dir)
    total_videos = len(work_items)
    print(f"Found {total_videos} video files")
    
    if total_videos == 0:
        print("No video files found. Exiting.")
        return
    
    # Prepare output paths
    model_name = os.path.basename(os.path.normpath(args.model_path))
    fout_path = os.path.join(args.result_dir, "model_caption.jsonl")
    
    # Get completed items for resume
    completed_items = get_completed_items(fout_path)
    print(f"Found {len(completed_items)} already processed videos")
    
    # Split work items across GPUs
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs")
    work_chunks = split_work_items(work_items, num_gpus, completed_items)
    total_remaining = sum(len(chunk) for chunk in work_chunks)
    print(f"Remaining videos to process: {total_remaining}")
    
    if total_remaining == 0:
        print("All videos already processed. Exiting.")
        return
    
    # Prepare temporary file paths
    tmp_files = [
        os.path.join(args.result_dir, f"tmp_captions_part{rank}.jsonl")
        for rank in range(num_gpus)
    ]
    
    # Initialize progress counter
    counter = Value('i', 0)
    lock = Lock()
    
    # Start worker processes
    processes = []
    for rank in range(num_gpus):
        if len(work_chunks[rank]) == 0:
            print(f"No work for GPU {rank}, skipping...")
            continue
            
        p = mp.Process(
            target=worker_proc,
            args=(
                rank,
                rank,  # GPU ID matches process rank
                args.model_path,
                work_chunks[rank],
                tmp_files[rank],
                counter,
                lock,
                log_dir,
                args
            ),
            name=f"worker_{rank}"
        )
        p.start()
        processes.append(p)
    
    # Display total progress
    with tqdm(total=total_remaining, desc="Total Progress") as pbar:
        last_count = 0
        while True:
            with lock:
                current = counter.value
            if current > last_count:
                pbar.update(current - last_count)
                last_count = current
            if current >= total_remaining:
                break
            # Check if all processes have finished
            if all(not p.is_alive() for p in processes):
                break
            time.sleep(1)  # Reduce CPU usage
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Merge results
    print("Merging results...")
    merge_results(tmp_files, fout_path)
    print(f"Final results saved to {fout_path}")


if __name__ == '__main__':
    # Use spawn start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()