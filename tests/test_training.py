"""Masking unit test for the DPO/TCPO log-prob computation (design-review
critical review, item 4 / Section 5.2 H4): chosen/rejected log
probabilities must cover only assistant-answer tokens, never image/prompt/
padding tokens. Requires only the processor (no model weights), so it stays
fast enough to run every time rather than being a manual, easy-to-skip check.
"""
import numpy as np
from PIL import Image
import pytest
from ricce.training.common import build_pair_inputs


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


def test_labels_mask_covers_only_answer_tokens(processor, tmp_path):
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    p1 = tmp_path / "a.png"; p2 = tmp_path / "b.png"
    img.save(p1); img.save(p2)
    answer = "a new building was constructed in the area ."
    enc = build_pair_inputs(processor, str(p1), str(p2), "Describe the change.", answer, max_length=1024)
    labels = enc["labels"][0]
    unmasked = labels[labels != -100]
    # The unmasked span must be non-empty, shorter than the full sequence
    # (i.e. image + prompt tokens really were excluded), and decode back to
    # (approximately) the answer text -- not to prompt or image-placeholder
    # content.
    assert 0 < len(unmasked) < labels.shape[0]
    decoded = processor.tokenizer.decode(unmasked, skip_special_tokens=True).strip().lower()
    for word in ["building", "constructed", "area"]:
        assert word in decoded
    # No prompt vocabulary should leak into the unmasked span.
    assert "describe" not in decoded and "change" not in decoded
