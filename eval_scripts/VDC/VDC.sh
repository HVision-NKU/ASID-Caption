#!/bin/bash
MODEL_PATH="Model/ASID-Captioner-3B" 
OUTPUT_DIR="./output/"

mkdir -p "$OUTPUT_DIR"


python generate_caption.py \
    --video_dir VDC/videos \
    --model_path "$MODEL_PATH" \
    --result_dir "$OUTPUT_DIR" \
    --num_gpus 8


python score_seed1.6.py y "$OUTPUT_DIR/model_caption.json" "$OUTPUT_DIR/detailed.jsonl" detailed.jsonl

python score_seed1.6.py y "$OUTPUT_DIR/model_caption.json" "$OUTPUT_DIR/background.jsonl" background.jsonl

python score_seed1.6.py y "$OUTPUT_DIR/model_caption.json" "$OUTPUT_DIR/camera.jsonl" camera.jsonl

python score_seed1.6.py y "$OUTPUT_DIR/model_caption.json" "$OUTPUT_DIR/main_object.jsonl" main_object.jsonl

python score_seed1.6.py y "$OUTPUT_DIR/model_caption.json" "$OUTPUT_DIR/short.jsonl" short.jsonl

