import json
import os
import argparse  

def load_qa_mapping(file_path: str) -> dict:
    """
    Load QA data and build mapping from video_id to questions and duration
    
    Args:
        file_path: Path to grouped QA data JSON file
        
    Returns:
        Dictionary with video_id as key and dict(questions, video_duration) as value
    """
    print("Loading QA data source...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            grouped_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] QA data file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON format in QA data file: {e}")
        raise

    qa_mapping = {}
    for item in grouped_data:
        video_id = item.get("video_id")
        if not video_id:
            continue
        
        qa_mapping[video_id] = {
            "questions": item.get("questions", []),
            "video_duration": item.get("video_duration", "")
        }
    
    print(f"Loaded QA data for {len(qa_mapping)} videos")
    return qa_mapping


def process_caption_file(input_path: str, output_path: str, qa_mapping: dict) -> None:
    """
    Process caption JSONL file and supplement with QA data
    
    Args:
        input_path: Path to input caption JSONL file
        output_path: Path to output JSONL file
        qa_mapping: Pre-built video_id to QA data mapping
    """
    print("Matching and supplementing QA data...")
    processed_count = 0
    missing_qa_count = 0

    with open(input_path, 'r', encoding='utf-8') as in_file, \
         open(output_path, 'w', encoding='utf-8') as out_file:
        
        for line_num, line in enumerate(in_file, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse single line JSON object
                caption_item = json.loads(line)
                
                # Extract pure video ID (remove "_video" suffix if exists)
                raw_video_id = caption_item.get("video_id", "")
                video_id = raw_video_id[:-6] if raw_video_id.endswith("_video") else raw_video_id

                # Supplement QA data if available
                if video_id in qa_mapping:
                    caption_item["questions"] = qa_mapping[video_id]["questions"]
                    caption_item["video_duration"] = qa_mapping[video_id]["video_duration"]
                    processed_count += 1
                else:
                    # Set empty values for missing QA data to avoid runtime errors
                    caption_item["questions"] = []
                    caption_item["video_duration"] = ""
                    missing_qa_count += 1
                    print(f"[WARN] Line {line_num}: Video ID '{raw_video_id}' (pure ID: {video_id}) has no matching QA data")

                # Write processed item to output file
                out_file.write(json.dumps(caption_item, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num} parsing failed: {e} - skipping this line")
                continue

    # Print processing statistics
    print("\nProcessing completed!")
    print(f"- Videos with QA data supplemented: {processed_count}")
    print(f"- Videos with missing QA data: {missing_qa_count}")
    print(f"- Output file path: {output_path}")


def main():
    """Main function to orchestrate the QA data supplementation process"""
    # Parse command line arguments (all required)
    parser = argparse.ArgumentParser(description='caption JSONL file with QA data')
    parser.add_argument('--qa-data', type=str, required=True,
                        help='Path to QA data file (grouped_data.json)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input caption JSONL file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSONL file with supplemented QA data')
    
    args = parser.parse_args()

    # Step 1: Load and build QA data mapping
    qa_mapping = load_qa_mapping(args.qa_data)
    
    # Step 2: Process caption file and supplement QA data
    process_caption_file(args.input, args.output, qa_mapping)


if __name__ == "__main__":
    main()