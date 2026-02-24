#!/bin/bash
MODEL_PATH="Model/ASID-Captioner-3B" 
OUTPUT_DIR="./output/"

mkdir -p "$OUTPUT_DIR"

# Step 1: caption generation
python generate_caption.py \
    --video_dir Daily-Omni/video \
    --model_path "$MODEL_PATH" \
    --result_dir "$OUTPUT_DIR" \
    --num_gpus 8


# Step 2: merge generated caption files
echo "Merging all generated caption files..."
python merge_qa.py \
  --qa-data "grouped_data.json" \
  --input "$OUTPUT_DIR/model_caption.jsonl" \
  --output "$OUTPUT_DIR/captions_with_qa.jsonl"


# Step 3: evaluation
python evaluation.py "$OUTPUT_DIR/captions_with_qa.jsonl"





