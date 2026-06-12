"""Toy layerwise distillation loop for binary/ternary compression.

The production-scale idea:
1. quantize one student layer;
2. match the teacher layer output on public calibration text;
3. freeze it;
4. move to the next layer;
5. only then run public/dev PPL.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


if torch is not None:

    class TinyBlock(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.fc1 = nn.Linear(width, width)
            self.fc2 = nn.Linear(width, width)

        def forward(self, x):
            return x + self.fc2(torch.tanh(self.fc1(x)))


    class TinyStack(nn.Module):
        def __init__(self, width: int = 16, layers: int = 3):
            super().__init__()
            self.blocks = nn.ModuleList([TinyBlock(width) for _ in range(layers)])

        def forward(self, x, stop_after: int | None = None):
            for i, block in enumerate(self.blocks):
                x = block(x)
                if stop_after is not None and i == stop_after:
                    break
            return x


def ternary_weight_(param) -> None:
    with torch.no_grad():
        scale = param.abs().mean().clamp_min(1e-8)
        threshold = 0.7 * scale
        q = torch.where(
            param.abs() >= threshold,
            torch.sign(param) * scale,
            torch.zeros_like(param),
        )
        param.copy_(q)


def quantize_one_block_ternary(block) -> None:
    for name, param in block.named_parameters():
        if "weight" in name:
            ternary_weight_(param)


def distill_one_block(teacher, student, block_index: int, steps: int = 80) -> float:
    for p in student.parameters():
        p.requires_grad_(False)
    for p in student.blocks[block_index].parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(student.blocks[block_index].parameters(), lr=2e-2)
    loss = None
    for _ in range(steps):
        x0 = torch.randn(32, 16)
        with torch.no_grad():
            prefix = x0 if block_index == 0 else student(x0, stop_after=block_index - 1)
            target = teacher(x0, stop_after=block_index)
        pred = student.blocks[block_index](prefix)
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())


def toy_check() -> None:
    if torch is None:
        print("Install torch to run this toy distillation check.")
        return

    torch.manual_seed(0)
    teacher = TinyStack()
    student = TinyStack()
    student.load_state_dict(teacher.state_dict())

    for layer in range(len(student.blocks)):
        quantize_one_block_ternary(student.blocks[layer])
        loss = distill_one_block(teacher, student, layer)
        print(f"layer {layer} final fit mse: {loss:.6f}")


if __name__ == "__main__":
    toy_check()
