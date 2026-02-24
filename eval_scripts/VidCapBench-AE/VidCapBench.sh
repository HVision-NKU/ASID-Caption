#!/bin/bash
MODEL_PATH="Model/ASID-Captioner-3B" 
OUTPUT_DIR="./output/"

mkdir -p "$OUTPUT_DIR"


python generate_caption.py \
    --video_dir VidCapBench/videos \
    --model_path "$MODEL_PATH" \
    --result_dir "$OUTPUT_DIR" \
    --num_gpus 8


python eval.py --caption_path "$OUTPUT_DIR/model_caption.jsonl"    