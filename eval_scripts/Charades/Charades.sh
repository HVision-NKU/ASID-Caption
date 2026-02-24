#!/bin/bash
MODEL_PATH="Model/ASID-Captioner-3B" 
OUTPUT_DIR="./output/"

mkdir -p "$OUTPUT_DIR"

# Step 1: caption generation
python generate_caption.py \
    --video_dir Charades/video \
    --model_path "$MODEL_PATH" \
    --result_dir "$OUTPUT_DIR" \
    --num_gpus 8



# Step 2: evaluation
python evaluation.py charades_test.json "$OUTPUT_DIR/model_caption.jsonl"





