import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from pathlib import Path

# ==========================
# Video Segmentation Script
# ==========================

def segment_video(model_path: Path, input_video_path: Path, output_video_path: Path):
    start_time = time.time()

    print("======================================================================")
    print("Video Segmentation Inference")
    print(f"Model: {model_path}")
    print(f"Input Video: {input_video_path}")
    print("======================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b3")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3", num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    print(f"Video Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=pil_image, return_tensors="pt").to(device)

            outputs = model(**inputs)
            logits = outputs.logits

            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(frame.shape[0], frame.shape[1]),
                mode="bilinear",
                align_corners=False,
            )

            pred = torch.argmax(upsampled_logits, dim=1)[0].cpu().numpy()

            color_mask = np.zeros_like(frame)
            color_mask[pred == 1] = [0, 255, 0]  # Green mask for class 1

            blended = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
            out.write(blended)

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()

    end_time = time.time()
    print("======================================================================")
    print(f"Finished. Output saved to: {output_video_path}")
    print(f"Total Processing Time: {end_time - start_time:.2f} seconds")
    print("======================================================================")


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def run(
        model_path: Path = typer.Argument(..., help="Path to .pth model weights"),
        input_video_path: Path = typer.Argument(..., help="Path to input video"),
        output_video_path: Path = typer.Argument(..., help="Path to save output video")
    ):
        segment_video(model_path, input_video_path, output_video_path)

    app()

