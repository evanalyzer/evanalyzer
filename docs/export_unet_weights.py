#!/usr/bin/env python3
# Copyright 2026 Joachim Danmayr.
# Licensed under the PolyForm Noncommercial License 1.0.0.
"""Export a bioimage.io U-Net's weights for convert_unet (dev tool).

The first half of converting a torch-em `UNet2d` (e.g. bioimage.io's
romantic-shark) into EVAnalyzer's model file: the parameters of the
package's TorchScript (`weights-torchscript.pt`) or PyTorch state dict
(`weights.pt`) as safetensors with PyTorch's names, f32. crates/nn's example
`convert_unet` then adds the architecture from the package's `rdf.yaml` and
writes the model file EVAnalyzer reads.

    pip install torch safetensors
    python export_unet_weights.py tests/romantic-shark/weights.pt weights.safetensors
"""

import argparse

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights", help="TorchScript (.pt) or state dict (.pt)")
    parser.add_argument("out", help="safetensors with PyTorch's names")
    args = parser.parse_args()

    try:
        state = torch.jit.load(args.weights, map_location="cpu").state_dict()
    except RuntimeError:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
    tensors = {
        name: t.detach().to(torch.float32).contiguous()
        for name, t in state.items()
        if t.is_floating_point()
    }
    save_file(tensors, args.out)
    print(f"{len(tensors)} tensors -> {args.out}")


if __name__ == "__main__":
    main()
