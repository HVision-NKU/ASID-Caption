#!/bin/bash
MODEL_PATH="Model/ASID-Captioner-3B" 
OUTPUT_DIR="./output/"

mkdir -p "$OUTPUT_DIR"

python generate_caption.py \
    --video_dir UGC-VideoCap/video \
    --model_path "$MODEL_PATH" \
    --result_dir "$OUTPUT_DIR" \
    --num_gpus 8

python evaluation.py "${OUTPUT_DIR}/model_caption.jsonl" "${OUTPUT_DIR}/eval_results.json"
