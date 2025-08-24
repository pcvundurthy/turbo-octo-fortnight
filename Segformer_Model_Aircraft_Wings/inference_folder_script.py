
import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from pathlib import Path

# ==========================
# Folder Image Segmentation
# ==========================

def segment_images_in_folder(model_path: Path, input_folder: Path, output_folder: Path):
    start_time = time.time()

    print("======================================================================")
    print("Image Folder Segmentation Inference")
    print(f"Model: {model_path}")
    print(f"Input Folder: {input_folder}")
    print(f"Output Folder: {output_folder}")
    print("======================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b3")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3", num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    os.makedirs(output_folder, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in sorted(input_folder.iterdir()) if f.suffix.lower() in image_extensions]

    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],  # (height, width)
                mode="bilinear",
                align_corners=False,
            )
            pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # Save the segmented output
        output_path = output_folder / f"{img_path.stem}_seg.png"
        cv2.imwrite(str(output_path), pred_seg)

        print(f"Saved: {output_path}")

    print(f"Inference completed in {time.time() - start_time:.2f} seconds")

# Example usage:
segment_images_in_folder(Path("best_model.pth"), Path("test_data"), Path("output_blades_folder"))
