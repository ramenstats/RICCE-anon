"""Generate captions from a base Qwen2.5-VL model + a LoRA checkpoint saved
by ricce/training/{sft,tcpo}.py. Mirrors ricce vlm-infer but loads
the fine-tuned adapter instead of the plain base model, so trained
checkpoints can be scored with the same E08 pipeline as the frozen baseline.
"""
import argparse
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from ricce.utils import read_jsonl, load_yaml, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    vc = cfg["vlm"]
    base_name = cfg["training"]["model_name_or_path"]
    processor = AutoProcessor.from_pretrained(base_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_name, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    device = next(model.parameters()).device

    rows = [r for r in read_jsonl(a.manifest) if a.split == "all" or r["split"] == a.split]
    if a.limit > 0:
        rows = rows[: a.limit]

    out = []
    for i, r in enumerate(rows, 1):
        img1 = Image.open(r["t1_path"]).convert("RGB")
        img2 = Image.open(r["t2_path"]).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": img1}, {"type": "image", "image": img2}, {"type": "text", "text": vc["prompt"]}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img1, img2], padding=True, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=vc["max_new_tokens"], do_sample=False)
        pred = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        out.append({"image_id": r["image_id"], "prediction": pred})
        if i % 100 == 0:
            write_jsonl(out, a.output)
            print(f"{i}/{len(rows)}")
    write_jsonl(out, a.output)
    print(f"Wrote {len(out)} predictions to {a.output}")


if __name__ == "__main__":
    main()
