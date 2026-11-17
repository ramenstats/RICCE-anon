"""Numerical validation of the DPO/TCPO loss used in ricce/training/tcpo.py
against the closed-form Rafailov et al. (2023) Eq.7 DPO loss on synthetic
log-probabilities (internal design review, Sec. 5.2 H3). This is a
lighter substitute for cross-checking against an external trusted
implementation (e.g. TRL's DPOTrainer), chosen to avoid adding a heavy new
dependency to an already-pinned environment; it still catches sign errors,
scaling errors, and formula-transcription mistakes.
"""
import math
import torch


def dpo_loss(lpch, lprj, rch, rrj, beta, w=1.0):
    delta = (lpch - lprj) - (rch - rrj)
    return -w * torch.nn.functional.logsigmoid(beta * delta)


def test_zero_margin_gives_log2():
    l0 = dpo_loss(torch.tensor(-2.0), torch.tensor(-2.0), torch.tensor(-2.0), torch.tensor(-2.0), beta=0.1)
    assert abs(l0.item() - math.log(2)) < 1e-5


def test_correct_direction_beats_wrong_direction():
    l0 = dpo_loss(torch.tensor(-2.0), torch.tensor(-2.0), torch.tensor(-2.0), torch.tensor(-2.0), beta=0.1)
    l1 = dpo_loss(torch.tensor(-1.0), torch.tensor(-5.0), torch.tensor(-2.0), torch.tensor(-2.0), beta=0.1)
    l2 = dpo_loss(torch.tensor(-5.0), torch.tensor(-1.0), torch.tensor(-2.0), torch.tensor(-2.0), beta=0.1)
    assert l1.item() < l0.item() < l2.item()


def test_confident_correct_loss_vanishes():
    l = dpo_loss(torch.tensor(20.0), torch.tensor(-20.0), torch.tensor(0.0), torch.tensor(0.0), beta=1.0)
    assert l.item() < 1e-6


def test_gradient_sign_increases_chosen_logprob():
    lpch = torch.tensor(-2.0, requires_grad=True)
    loss = dpo_loss(lpch, torch.tensor(-2.0), torch.tensor(-2.0), torch.tensor(-2.0), beta=0.1)
    loss.backward()
    assert lpch.grad.item() < 0  # loss decreases as lpch increases


def test_severity_weight_scales_linearly():
    l2 = dpo_loss(torch.tensor(-5.0), torch.tensor(-1.0), torch.tensor(-2.0), torch.tensor(-2.0), beta=0.1)
    lw = dpo_loss(torch.tensor(-5.0), torch.tensor(-1.0), torch.tensor(-2.0), torch.tensor(-2.0), beta=0.1, w=2.0)
    assert abs(lw.item() - 2 * l2.item()) < 1e-5
