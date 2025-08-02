#!/usr/bin/env python3
"""
Script to automatically generate captions for images using BLIP or other vision-language models.
"""

import os
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm


class CaptionGenerator:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """Initialize the caption generation model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model {model_name} on {self.device}")
        
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
    def generate_caption(self, image_path, max_length=50):
        """Generate a caption for a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=max_length, num_beams=5)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""
    
    def process_directory(self, input_dir, output_file, image_extensions=None):
        """Process all images in a directory and generate captions."""
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        input_path = Path(input_dir)
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        captions = {}
        
        print(f"Processing {len(image_files)} images...")
        for image_file in tqdm(image_files):
            caption = self.generate_caption(str(image_file))
            if caption:
                # Store relative path as key
                relative_path = image_file.name
                captions[relative_path] = caption
        
        # Save captions to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(captions)} captions to {output_file}")
        return captions


def main():
    parser = argparse.ArgumentParser(description="Generate captions for images")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing images")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSON file for captions")
    parser.add_argument("--model", type=str, default="Salesforce/blip-image-captioning-base",
                       help="Caption generation model")
    parser.add_argument("--max_length", type=int, default=50,
                       help="Maximum caption length")
    
    args = parser.parse_args()
    
    generator = CaptionGenerator(args.model)
    generator.process_directory(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
