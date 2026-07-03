# SuctionNet learned-scorer setup (H7 — the real-data suction backend)

The suction quality seam (`grasping/suction/scorer.py`) has two backends behind one `SuctionScorer` Protocol:

- **`AnalyticalSuctionScorer`** — the sim-validatable default (seal × wrench physics, pure numpy, no
  dependency). This is what runs in sim and IRL by default; it already produces good, measured values
  (flat 1.0 vs curved/edge 0.0).
- **`LearnedSuctionScorer`** — the real-data backend: the published **SuctionNet-1Billion** DeepLabV3+
  RGB-D network (Cao et al., *RA-L 2021*). Its per-pixel output is `seal × center`, which maps 1:1 onto our
  `seal × wrench`. This doc sets it up.

> ⚠ **Honesty (sim2real).** SuctionNet is trained on **real** RealSense RGB-D (the graspnet dataset). On
> Isaac's **rendered** depth (and on synthetic scenes) it is out of distribution, so the **absolute** scores
> carry a sim2real gap — a real-hardware calibration concern. In sim, use it as a *running* learned scorer
> and a **cross-check** against the analytical model, not as calibrated ground truth. The real value lands
> when it runs on a real camera. Verified: on a synthetic flat box it scores low (out-of-distribution); the
> analytical scorer scores 1.0. That is expected.

The model code + weights live **OUTSIDE this repo** (like cuRobo/Coal), bridged by env vars — nothing in the
repo imports torch or the external code at module top, so `grasping/suction/` stays pure-numpy / CI-safe
unless the learned backend is actually used. This keeps it cleanly **removable** (delete the dirs + unset the
env vars) and avoids vendoring an unlicensed third-party repo.

## Install

The graspnet SuctionNet repo targets `torch 1.4.0 / CUDA 10.1` (2020), but the model is a standard
DeepLabV3+ — the pretrained **weights are a plain `model_state_dict`** that our `torch 2.7.1` (cu128, the
Blackwell RTX 5080) loads directly. So NO ancient sidecar env is needed; the driver builds the net on our
torch and loads the weights. The only compat shim: `torchvision.models.utils` (removed in torchvision 0.13+)
is monkeypatched to `torch.hub.load_state_dict_from_url` (only used with `pretrained_backbone=True`, which we
do not set). All handled in `grasping/suction/learned_backend.py`.

```bash
# 1) The model code (external, not vendored). Cloned into ext_deps/; the env var points at neural_network/.
git clone --depth 1 https://github.com/graspnet/suctionnet-baseline.git ext_deps/suctionnet

# 2) The pretrained realsense RGB-D weights (~192 MB) from the graspnet Google Drive.
#    On Windows behind a corporate MITM proxy, Python's requests can't verify the SSL cert
#    ("unable to get local issuer certificate"); pip-system-certs makes it use the Windows cert store.
pip install gdown pip-system-certs
mkdir -p ext_deps/suctionnet/weights && cd ext_deps/suctionnet/weights
python -m gdown 18TbctdhpNXEKLYDWFzI9cT1Wnhe-tn9h        # realsense DeepLabV3+ RGB-D
unzip -o realsense-deeplabplus-RGBD.zip                   # -> the checkpoint file (a model_state_dict, epoch 71)

# 3) Point the env vars at them (the LearnedSuctionScorer / SuctionNetModel read these).
export WILLY_SUCTIONNET_PREFIX="ext_deps/suctionnet/neural_network"
export WILLY_SUCTIONNET_WEIGHTS="ext_deps/suctionnet/weights/realsense-deeplabplus-RGBD"
```

Other pretrained variants (kinect RGB-D, depth-only, the FCN) are linked in the repo README; the env var
just points at the checkpoint you want. The kinect/depth models use the same loader.

## Use

```python
from backend.src.robot.grasping.suction.scorer import LearnedSuctionScorer
from backend.src.robot.grasping.suction.synthesis import SuctionConfig, synthesize_suction_grasps

cfg = SuctionConfig(scorer=LearnedSuctionScorer())          # reads WILLY_SUCTIONNET_* by default
grasps = synthesize_suction_grasps(mask, depth_mm, intrinsics, rgb=rgb, payload_mass_g=mass, config=cfg)
```

The scorer runs SuctionNet ONCE per scene (`prepare_scene`, inside the synthesis) and caches the heatmap;
each candidate is then a per-pixel lookup. **RGB is required** (SuctionNet is RGB-D) — the ground-truth Isaac
perception has no usable RGB on Isaac 5.1, so the learned scorer needs the **real-vision** perception path
(M2-style GroundingDINO + SAM2 RGB), not the GT-mask path. Without the env vars (or with `rgb=None`) the
learned scorer **honest-abstains** (`SuctionModelNotWired`), never a fabricated number.

## Remove

Delete `ext_deps/suctionnet` + `ext_deps/suctionnet/weights` and unset the two env vars. The repo is
byte-identical without them (the analytical scorer is the default; the integration test skips when unset).

See also [external-deps.md](external-deps.md), [curobo-setup.md](curobo-setup.md), [coal-setup.md](coal-setup.md).
