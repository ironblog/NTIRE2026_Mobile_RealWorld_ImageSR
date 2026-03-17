"""
test.py — NTIRE 2026 Mobile Real-World Image SR Challenge
Team: team12_SNOWVision

Usage:
    python test.py
"""

import os
import sys
import glob
import math

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Model import ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "team12_SNOWVision"))
from model import mobilehgsr


# ── Paths ──
MODEL_PATH = os.path.join("model_zoo", "team12_SNOWVision", "best.pth")
INPUT_DIR  = os.path.join("data", "test", "lr")
OUTPUT_DIR = os.path.join("data", "test", "sr")


# ── Tiled inference ──
class TiledInference:
    def __init__(self, model, tile_size=128, overlap=24, device="cuda"):
        self.model = model
        self.tile = tile_size
        self.overlap = overlap
        self.device = device
        self.scale = 4
        hr_tile = tile_size * self.scale
        w1d = torch.hann_window(hr_tile, periodic=False) + 1e-8
        self.window = (w1d[None, :] * w1d[:, None]).to(device)

    @torch.no_grad()
    def infer(self, lr_img):
        _, _, H, W = lr_img.shape
        pad = self.overlap
        tile = self.tile
        s = self.scale
        lr_padded = F.pad(lr_img, (pad, pad, pad, pad), mode="reflect")
        _, _, pH, pW = lr_padded.shape
        out_h, out_w = pH * s, pW * s
        output = torch.zeros(1, 3, out_h, out_w, device=self.device)
        weight = torch.zeros(1, 1, out_h, out_w, device=self.device)
        step = tile - self.overlap
        rows = list(range(0, pH - tile + 1, step))
        cols = list(range(0, pW - tile + 1, step))
        if rows[-1] + tile < pH:
            rows.append(pH - tile)
        if cols[-1] + tile < pW:
            cols.append(pW - tile)
        for y in rows:
            for x in cols:
                lr_tile = lr_padded[:, :, y:y+tile, x:x+tile]
                sr_tile = self.model(lr_tile)
                oy, ox = y * s, x * s
                th, tw = sr_tile.shape[2], sr_tile.shape[3]
                output[:, :, oy:oy+th, ox:ox+tw] += sr_tile * self.window
                weight[:, :, oy:oy+th, ox:ox+tw] += self.window
        sr_full = output / weight.clamp(min=1e-8)
        crop = pad * s
        sr_full = sr_full[:, :, crop:crop + H * s, crop:crop + W * s]
        return sr_full.clamp(0, 1)


# ── Load model ──
def load_model(ckpt_path, device="cuda"):
    model = mobilehgsr().to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "params_ema" in ckpt:
        model.load_state_dict(ckpt["params_ema"], strict=True)
    elif "params" in ckpt:
        model.load_state_dict(ckpt["params"], strict=True)
    elif "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    return model


# ── Main ──
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_PATH, device)
    tiler = TiledInference(model, tile_size=128, overlap=24, device=device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    input_files = []
    for ext in extensions:
        input_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    input_files = sorted(input_files)

    if not input_files:
        print("No images found in %s" % INPUT_DIR)
        return

    print("Processing %d images from %s" % (len(input_files), INPUT_DIR))
    for i, path in enumerate(input_files):
        name = os.path.basename(path)
        out_name = os.path.splitext(name)[0] + "x4.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        lr_pil = Image.open(path).convert("RGB")
        lr_np = np.array(lr_pil).astype(np.float32) / 255.0
        lr_t = torch.from_numpy(lr_np).permute(2, 0, 1).unsqueeze(0).to(device)

        sr_t = tiler.infer(lr_t)

        sr_np = (sr_t[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(sr_np).save(out_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(input_files):
            print("  [%d/%d] %s" % (i + 1, len(input_files), out_name))

    print("Done. Output saved to %s" % OUTPUT_DIR)


if __name__ == "__main__":
    main()
