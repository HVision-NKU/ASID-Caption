"""
This script demonstrates how to use ASID-Captioner (based on Qwen2.5-Omni) for audiovisual video captioning.
"""

import json
import os
import random
from tqdm import tqdm
from qwen_omni_utils import process_mm_info
import sys
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# --- Constants Definition ---
VIDEO_MAX_PIXELS = 401408  # 512*28*28
VIDEO_TOTAL_PIXELS = 20070400  # 512*28*28*50
USE_AUDIO_IN_VIDEO = True
MODEL_PATH = "Model/ASID-Captioner-3B"

os.environ['VIDEO_MAX_PIXELS'] = str(VIDEO_TOTAL_PIXELS)

def load_model_and_processor(model_path: str):
    print(f"Loading model from: {model_path}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


def generate_caption(model, processor, file_path, prompt):
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                }
            ],
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
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        },
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            use_audio_in_video=USE_AUDIO_IN_VIDEO, 
            return_audio=False, 
            max_new_tokens=4096,
            repetition_penalty=1.1,
        )

    text = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    model_generation = text.split("\nassistant\n")[-1]

    return model_generation


if __name__ == "__main__":

    video_path = sys.argv[1]

    # prompt_list = [
    # "Provide a comprehensive description of all the content in the video, leaving out no details, and naturally covering the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in a single coherent account.",
    # "Thoroughly describe everything in the video, capturing every detail across the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in a clear and unified description.",
    # "Please describe all the information in the video without sparing any detail, seamlessly incorporating the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in one coherent paragraph.",
    # "Offer a detailed description of the video that covers every aspect of its content, including the scene, characters, objects, actions, narrative elements, speech, camera, and emotions, presented as a single coherent paragraph.",
    # "Describe every aspect of the video in full detail, covering the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in a unified and coherent manner.",
    # "Please provide a thorough description of all the content in the video, naturally addressing the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in one coherent paragraph.",
    # "Give a detailed account of everything in the video, capturing all specifics related to the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in a single, coherent description.",
    # ]

    prompt_list = [
    "Provide a comprehensive description of all the content in the video, leaving out no details. Be sure to include as much of the audio information as possible, and ensure that your descriptions of the audio and video are closely aligned.",
    "Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.",
    "Please describe all the information in the video without sparing every detail in it. As you describe, you should also describe as much of the information in the audio as possible, and pay attention to the synchronization between the audio and video descriptions.",
    "Offer a detailed description of the video, making sure to include every detail. Also, incorporate as much information from the audio as you can, and ensure that your descriptions of the audio and video are in sync.",
    "Describe every aspect of the video in full detail, covering all the information it contains. Additionally, include as much of the audio content as you can, and make sure your descriptions of the audio and video are synchronized.",
    "Please provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so.",
    "Give a detailed account of everything in the video, capturing all the specifics. While doing so, also include as much information from the audio as possible, ensuring that the descriptions of audio and video are well-synchronized."
    ]

    prompt = random.choice(prompt_list)

    model, processor = load_model_and_processor(MODEL_PATH)
    print("\n--- Prompt ---\n")
    print(prompt)
    
    model_generation = generate_caption(model, processor, video_path, prompt)
    print("\n--- Model Generation ---\n")
    print(model_generation)

