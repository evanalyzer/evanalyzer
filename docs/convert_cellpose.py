#!/usr/bin/env python3
"""Convert Cellpose-SAM weights into a TorchScript model EVAnalyzer can load.

EVAnalyzer's "AI Cellpose Segmentation" command runs models through libtorch,
which can only execute a **TorchScript** module (a serialized computation graph
*plus* weights) - a raw Cellpose-SAM checkpoint (what `cellpose` downloads to
``~/.cellpose/models/``, or what you'd download by hand from
https://huggingface.co/mouseland/cellpose-sam) contains only the *weights*,
with no graph to run, so it cannot be loaded directly. This script rebuilds
the Cellpose-SAM network (using the ``cellpose`` Python package), loads the
weights into it, and re-saves it with ``torch.jit`` as a ``.pt`` file you can
pick in the UI.

Cellpose-SAM's ViT (SAM) encoder bakes its positional embeddings for a
**fixed 256x256 token grid** at construction time - the network cannot run at
any other spatial size (Cellpose's own Python code enforces the same limit:
"bsize != 256 is not supported for cpsam"). This script always traces at
exactly 256x256; EVAnalyzer's Cellpose command tiles larger images into
overlapping 256x256 blocks internally before calling the exported model, for
exactly this reason - you never need to (and cannot) export at another size.

The exported model:
  * takes a ``[1, C, 256, 256]`` float tensor (``C`` = --channels, default 2:
    the grayscale image in channel 0, the rest zero-filled by EVAnalyzer;
    Cellpose-SAM's patch-embedding convolution only has weights for up to 3
    input channels, so ``C`` must be 1-3), and
  * returns a ``[1, 3, 256, 256]`` tensor: ``[dY, dX, cellprob]`` (Cellpose
    flows + cell-probability logits), which is exactly what the Rust side
    expects.

Usage
-----
    pip install "cellpose>=4" torch

    # downloads the default Cellpose-SAM weights ("cpsam_v2") automatically:
    python convert_cellpose.py --output cellpose_sam.pt

    # a specific built-in Cellpose-SAM checkpoint, or a compatible weights
    # file/path (must be the SAM backbone - see "Notes" below):
    python convert_cellpose.py --pretrained cpsam --output cellpose_sam.pt

Notes
-----
* Only the Cellpose-SAM ("cpsam"/"cpsam_v2", backbone ``sam_vitl``) family is
  supported. Classic Cellpose (v3.x and earlier, the convolutional CPnet
  architecture) has a different architecture and output layout and is not
  handled by this script.
* Cellpose-SAM's weights are ``bfloat16`` by default; this script casts them
  to ``float32`` before tracing (matching the ``Kind::Float`` tensors
  EVAnalyzer builds) rather than exporting in ``bfloat16``, because CPU
  ``bfloat16`` execution is dramatically slower without server-class hardware
  (AVX512-BF16) - the exported file is larger (~1.2GB) but runs reliably
  everywhere.
* Cellpose normally applies per-channel percentile normalization *before* the
  network. That step is NOT part of the TorchScript graph, so feed EVAnalyzer
  a reasonably normalized image (e.g. via the pipeline's contrast/normalize
  steps) for results comparable to the Cellpose GUI.
"""

import argparse
import sys

import torch

TILE_SIZE = 256


class FlowsOnly(torch.nn.Module):
    """Wraps a Cellpose-SAM net so the scripted module returns only the flow
    tensor.

    ``CPSAM.forward`` returns ``(y, style)``; EVAnalyzer accepts that tuple
    directly, but exporting just ``y`` keeps the model output unambiguous.
    """

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def build_net(pretrained_model: str) -> torch.nn.Module:
    """Load the Cellpose-SAM architecture with the requested weights."""
    from cellpose import models

    model = models.CellposeModel(gpu=False, pretrained_model=pretrained_model)
    if model.backbone != "sam_vitl":
        raise ValueError(
            f"'{pretrained_model}' is not a Cellpose-SAM checkpoint (backbone="
            f"{model.backbone!r}, expected 'sam_vitl') - this script only supports "
            f"the Cellpose-SAM family (cpsam/cpsam_v2)."
        )

    net = model.net.float()  # bfloat16 -> float32, see "Notes" above
    net.eval()
    return net


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained", default="cpsam_v2",
                        help="Built-in Cellpose-SAM model name ('cpsam_v2', 'cpsam') or a "
                             "path to a compatible checkpoint. Downloads automatically if "
                             "not already cached. Default: cpsam_v2.")
    parser.add_argument("--output", required=True, help="Destination TorchScript file (.pt).")
    parser.add_argument("--channels", type=int, default=2, choices=[1, 2, 3],
                        help="Input channel count to trace with (must match the "
                             "Cellpose command's 'input_channels'; Cellpose-SAM supports "
                             "1-3. Default 2.")
    args = parser.parse_args()

    try:
        net = build_net(args.pretrained)
    except ImportError:
        print("error: the 'cellpose' package is required: pip install 'cellpose>=4'",
              file=sys.stderr)
        return 1

    wrapped = FlowsOnly(net)
    wrapped.eval()

    example = torch.zeros(1, args.channels, TILE_SIZE, TILE_SIZE, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(wrapped, example)
        # Sanity check against a non-zero input: the output must be
        # [1, 3, 256, 256], and tracing must not have baked in the all-zero
        # example's values.
        probe = torch.randn(1, args.channels, TILE_SIZE, TILE_SIZE, dtype=torch.float32)
        out = scripted(probe)
    if out.shape != (1, 3, TILE_SIZE, TILE_SIZE):
        print(f"warning: traced model output shape {tuple(out.shape)} is not the "
              f"expected [1, 3, {TILE_SIZE}, {TILE_SIZE}]; the Cellpose command will "
              f"reject it.", file=sys.stderr)

    scripted.save(args.output)
    print(f"Wrote TorchScript model to {args.output} "
          f"(input [1, {args.channels}, {TILE_SIZE}, {TILE_SIZE}] -> output {tuple(out.shape)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
