# Pretrained model candidates (burn migration, D2)

Researched 2026-11-25 for the decision which models EVAnalyzer converts and ships in its own
safetensors format (`docs/nn_burn_migration_plan.md`, D2 refined). Only models of the
architectures nn implements: torch-em `UNet2d`, StarDist 2D, Cellpose-SAM. Not decided yet.

Licence of the weights decides whether a converted file may be redistributed (CC-BY, MIT,
BSD, Apache: yes, with attribution). A second risk is the **training data**: several models
were trained on non-commercial (NC) data - whether that restricts the weights is a legal
grey area, flagged below.

Download counts: "Hypha / legacy" - Hypha (bioimage.io's live backend) counts since the 2025
migration, legacy is the older cumulative count (only the 44 models of the old collection).

## U-Net: torch-em `UNet2d` (2D)

All verified in their `rdf.yaml` (callable `UNet2d` from torch-em's `unet.py`; the two
`unet.py` versions differ only in an optional shape check). Default norm InstanceNorm,
final activation Sigmoid; unless noted depth 4, gain 2, 1 in, 2 out (foreground + boundary).

| id | task / modality | config | weights | licence | redistribution | DL | notes |
|---|---|---|---|---|---|---|---|
| affable-shark | nuclei, fluorescence (DAPI) | 64 features, zero-mean-unit-variance | state dict, torchscript | CC-BY-4.0 | yes (data DSB2018/BBBC038 CC0) | 7658 / 71879 | most used UNet2d |
| romantic-shark | same | same | same | CC-BY-4.0 | yes | 0 / - | re-upload, weights identical to affable-shark |
| hiding-tiger | cells, phase contrast (LIVECell) | 64 features | state dict, torchscript | CC-BY-4.0 | model yes, **data LIVECell CC BY-NC 4.0** | 1248 / 45966 | only label-free option |
| joyful-pig | same | - | torchscript only | CC-BY-4.0 | same NC flag | 0 / - | torchscript identical to hiding-tiger |
| powerful-chipmunk | cells, immunofluorescence (Covid-IF) | 64 features | state dict, torchscript | CC-BY-4.0 | yes (data CC-BY-4.0) | 1093 / 35536 | |
| shivering-raccoon | mitochondria, EM (ssTEM) | 64 features | state dict, torchscript | CC-BY-4.0 | yes (data CC-BY-4.0) | 1108 / 34805 | raw input |
| hiding-blowfish | mitochondria, EM (shallow2deep enhancer) | 64 features | state dict, torchscript | CC-BY-4.0 | yes (MitoEM data licence not checked) | 1843 / 41876 | input is an ilastik probability map |
| dazzling-crab | same | - | torchscript only | CC-BY-4.0 | yes | 0 / - | duplicate of hiding-blowfish |
| amiable-crocodile | cell boundaries, EM (enhancer) | 32 features, 1 out, **BatchNorm** | state dict only | MIT | yes | 499 / 2537 | needs BatchNorm loading; enhancer input |
| wild-whale | epithelial cells, affinities | 32 features, **8 out** | state dict, torchscript | CC-BY-4.0 | yes (data not documented) | 1079 / 29580 | needs affinity post-processing |
| charismatic-whale | E. coli, brightfield (DeepBacs) | fine-tune of affable-shark (arch not verified) | several | **CC-BY-SA-4.0** | share-alike: our file CC-BY-SA too | 169 / - | |

Popular 2D U-Nets of **other** architectures (not supported): plantseg `UNet2D`
(pioneering-rhino, laid-back-lobster; MIT), Keras U-Nets (discreet-rooster, creative-panda,
placid-llama, committed-turkey), HPA DPNUnet (loyal-parrot, conscientious-seashell; torchscript
only), the BiaPy family, careamics `UNet` (N2V), MONAI UNet (noisy-hedgehog), FRUNet. 3D
torch-em models are out of scope.

**Recommended start:** affable-shark (nuclei), powerful-chipmunk (cells IF),
shivering-raccoon (mitochondria EM); hiding-tiger (phase contrast) only after accepting the
LIVECell NC-data risk. romantic-shark (our test model) = affable-shark's weights.

## StarDist 2D

Official models of the stardist package (`github.com/stardist/stardist-models` releases
v0.1, zip = `config.json` + `thresholds.json` + `.h5`, ~5.7 MB), BSD-3-Clause.

| id | task / modality | config | thresholds | redistribution | notes |
|---|---|---|---|---|---|
| 2D_versatile_fluo | nuclei, fluorescence | 32 rays, grid (2,2), 1 channel, U-Net depth 3, 32 filters, conv after U-Net 128 | prob 0.479, nms 0.3 | yes (DSB2018 subset, CC0) | the standard choice |
| 2D_versatile_he | nuclei, H&E (RGB) | as above, **3 channels** | prob 0.692, nms 0.3 | weights yes, **data MoNuSeg CC BY-NC-SA 4.0** + TNBC CC-BY | standard for H&E slides |
| 2D_paper_dsb2018 | nuclei, fluorescence | 32 rays, grid (1,1), depth 3, 32 filters | prob 0.418, nms 0.5 | yes | older; tests grid 1 |
| 2D_demo | nuclei, fluorescence (demo) | as versatile_fluo | prob 0.486, nms 0.5 | yes | low value |

On bioimage.io as TF SavedModel only (same configs): fearless-crab (versatile fluo),
chatty-frog / beautiful-sheep (H&E). Convert from the official `.h5`.

**Recommended start:** 2D_versatile_fluo, 2D_versatile_he (NC-data flag), optionally
2D_paper_dsb2018.

## Cellpose-SAM

Code BSD-3 (MouseLand/cellpose). Weights on `huggingface.co/mouseland/cellpose-sam`, model
card `bsd-3-clause`. **But** the Cellpose README: "All Cellpose models are trained on data
that is licensed under CC-BY-NC." - needs a decision or legal advice (asking MouseLand
would settle it). SAM base weights: Apache-2.0.

| id | size | licence | redistribution | notes |
|---|---|---|---|---|
| cpsam_v2 | 1.23 GB | BSD-3 | probably, with attribution; **NC training-data flag** | default since v4.2 (June 2026), same architecture as cpsam |
| cpsam | 1.23 GB | BSD-3 | same flag | original (April 2025) |
| cpdino / cpdino-vitb | 1.21 GB / 344 MB | DINOv3 licence | **no** (custom Meta licence) | other architecture anyway |

Cellpose 3 models (cyto3, nuclei, tissuenet_cp3, livecell_cp3, ...) are another
architecture (residual U-Net CPnet) - out of scope.

**Recommended start:** cpsam_v2 (+ cpsam for older results), once the NC question is
settled.

## Uncertain

MitoEM data licence (hiding-blowfish), wild-whale's training data, charismatic-whale's
architecture, whether the BSD-3 card on Hugging Face overrides Cellpose's NC-data note.

## Sources

- bioimage.io live manifests: `https://hypha.aicell.io/bioimage-io/artifacts/bioimage.io/children?limit=2000&filters={"type":"model"}`
- legacy collection: https://bioimage-io.github.io/collection-bioimage-io/collection.json
- https://github.com/stardist/stardist (models/__init__.py, LICENSE.txt), https://github.com/stardist/stardist-models/releases/tag/v0.1
- https://monuseg.grand-challenge.org/Data/, https://zenodo.org/records/1175282 (TNBC), https://bbbc.broadinstitute.org/BBBC038
- https://github.com/sartorius-research/LIVECell, https://zenodo.org/records/5092850 (Covid-IF), https://api.figshare.com/v2/articles/856713 (VNC), https://zenodo.org/records/5764892 (affable-shark)
- https://github.com/MouseLand/cellpose, https://huggingface.co/mouseland/cellpose-sam
- https://github.com/facebookresearch/segment-anything, https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
