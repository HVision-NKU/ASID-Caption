import os
import json
import argparse

def merge_captions(original_file_path, caption_file_path, merged_file_path):
    print(f"Now processing {caption_file_path} ")
    jsonl_data = {}
    file_key = os.path.splitext(os.path.basename(caption_file_path))[0]
    with open(caption_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            vid = item["video_id"]
            caption = item["caption"]
            jsonl_data[vid] = caption

    with open(original_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for vid, entry in data.items():
        if vid in jsonl_data:
            if file_key not in entry:
                entry[file_key] = jsonl_data[vid]

    with open(merged_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge caption files into a single JSON file.")
    parser.add_argument("--original_file", type=str, required=True, help="Path to the original JSON file.")
    parser.add_argument("--caption_files", type=str, nargs='+', required=True, help="List of caption files to merge.")
    parser.add_argument("--merged_file", type=str, required=True, help="Path to save the merged JSON file.")

    args = parser.parse_args()

    for cf in args.caption_files:
        merge_captions(args.original_file, cf, args.merged_file)
