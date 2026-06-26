#!/usr/bin/env python3
"""Convert a Cellpose weights file into a TorchScript model EVAnalyzer can load.

EVAnalyzer's "AI Cellpose Segmentation" command runs models through libtorch,
which can only execute a **TorchScript** module (a serialized computation graph
*plus* weights). A bioimage.io ``pytorch_state_dict`` (the ``.pth`` you download
from e.g. https://bioimage.io) contains only the *weights* — there is no graph to
run — so it cannot be loaded directly. This script rebuilds the Cellpose network
architecture (using the ``cellpose`` Python package), loads the weights into it,
and re-saves it with ``torch.jit`` as a ``.pt`` file you can pick in the UI.

The exported model:
  * takes a ``[1, C, H, W]`` float tensor (``C`` = --channels, default 2: the
    grayscale image in channel 0, the rest zero-filled by EVAnalyzer), and
  * returns a ``[1, 3, H, W]`` tensor: ``[dY, dX, cellprob]`` (Cellpose flows +
    cell-probability logits), which is exactly what the Rust side expects.

Usage
-----
    pip install "cellpose>=3,<4" torch
    python convert_cellpose.py --weights happy-elephant.pth --output happy-elephant.pt

    # built-in pretrained model instead of a downloaded file:
    python convert_cellpose.py --pretrained cyto3 --output cyto3.pt

Notes
-----
* Cellpose v3.x exports the classic flow+cellprob network this command targets.
  Cellpose v4 ("cellpose-SAM") has a different architecture and output layout and
  is not supported here.
* Cellpose normally applies per-channel percentile normalization *before* the
  network. That step is NOT part of the TorchScript graph, so feed EVAnalyzer a
  reasonably normalized image (e.g. via the pipeline's contrast/normalize steps)
  for results comparable to the Cellpose GUI.
"""

import argparse
import sys

import torch


class FlowsOnly(torch.nn.Module):
    """Wraps a Cellpose net so the scripted module returns only the flow tensor.

    Cellpose's ``CPnet.forward`` returns ``(y, style, ...)``; EVAnalyzer accepts
    that tuple, but exporting just ``y`` keeps the model output unambiguous.
    """

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def build_net(args) -> torch.nn.Module:
    """Load the Cellpose architecture with the requested weights."""
    from cellpose import models

    if args.pretrained:
        model = models.CellposeModel(gpu=False, model_type=args.pretrained)
    else:
        # `pretrained_model` accepts an arbitrary weights path and builds the
        # matching CPnet architecture for it.
        model = models.CellposeModel(gpu=False, pretrained_model=args.weights)

    net = model.net
    net.eval()
    net.mkldnn = False  # mkldnn tensors cannot be traced/scripted
    return net


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", help="Path to a Cellpose state_dict (.pth).")
    src.add_argument("--pretrained", help="Built-in model name, e.g. cyto3, cyto2, nuclei.")
    parser.add_argument("--output", required=True, help="Destination TorchScript file (.pt).")
    parser.add_argument("--channels", type=int, default=2,
                        help="Input channel count to trace with (must match the "
                             "Cellpose command's 'input_channels'; default 2).")
    parser.add_argument("--size", type=int, default=256,
                        help="Spatial size of the dummy tensor used for tracing "
                             "(the exported model still works at any H/W; default 256).")
    args = parser.parse_args()

    try:
        net = build_net(args)
    except ImportError:
        print("error: the 'cellpose' package is required: pip install 'cellpose>=3,<4'",
              file=sys.stderr)
        return 1

    wrapped = FlowsOnly(net)
    wrapped.eval()

    example = torch.zeros(1, args.channels, args.size, args.size, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(wrapped, example)
        # Sanity check: the output must be [1, 3, H, W].
        out = scripted(example)
    if out.ndim != 4 or out.shape[1] < 3:
        print(f"warning: traced model output shape {tuple(out.shape)} is not the "
              f"expected [1, 3, H, W]; the Cellpose command may reject it.",
              file=sys.stderr)

    scripted.save(args.output)
    print(f"Wrote TorchScript model to {args.output} "
          f"(input [1, {args.channels}, H, W] -> output {tuple(out.shape)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
