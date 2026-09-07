"""Convert a pathology foundation model to ONNX + a PathLearn descriptor.

Ported from the macOS `convert_phikon.py` / `convert_to_coreml.py`, with the
final step changed from `coremltools.convert(...)` to `torch.onnx.export(...)`.
The model *loaders* are kept verbatim from those scripts — they encode which
embedding each model returns, which is the easiest thing to get silently wrong.

Run in a conversion venv, which needs torch/timm/transformers — see
`tools/requirements-convert.txt`.  The app itself never needs any of them,
only onnxruntime.

    python convert_extractor.py phikon-v1 --out <dir>
    python convert_extractor.py uni2-h    --out <dir>

BEFORE YOU RUN THIS: UNI and UNI2-h are **gated**
===================================================
MahmoodLab/UNI and MahmoodLab/UNI2-h require you to request access on
HuggingFace, be approved, and then authenticate — `huggingface-cli login`
or `HF_TOKEN=...`.  Phikon is ungated.  Weights are never redistributed
with PathLearn: each model carries its own licence, and you convert your
own approved copy.

Downloads are cached by huggingface_hub, so a second run is offline anyway.
Pass --offline to forbid network access outright and use only what is
already cached.

Each run writes `<name>.onnx` plus `<name>.pathlearn-extractor.json`, and verifies
the export against PyTorch (cosine must clear --min-cosine, default 0.9999).

DIFFERENCES FROM THE macOS EXPORT, DELIBERATE
=============================================
* **Dynamic batch axis.** Core ML was pinned to `(1,3,224,224)`; ONNX takes a
  dynamic batch so patch extraction can run batched.  Spatial dims stay fixed at
  224x224 so the export matches what produced the existing banks.
* **Named I/O.** The two macOS scripts disagreed (`"input"` vs `"image"`) and the
  descriptor never recorded it, so the runtime hard-coded a guess.  Here the
  names are fixed and written into the descriptor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

# torch.onnx prints status lines containing emoji. On a stock Windows console
# (cp1252) that raises UnicodeEncodeError *after* a successful export, which
# looks exactly like a conversion failure. Force UTF-8 on our streams.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402

#: Set by --offline. Downloads are allowed by default: forcing cache-only made
#: this script work on the machine it was written on and fail everywhere else,
#: with "no checkpoint in the local cache" — which reads as a bug rather than
#: as "request access and log in".
_OFFLINE = False


def _set_offline(offline: bool) -> None:
    """Module-level because the per-model loaders take no arguments."""
    global _OFFLINE
    _OFFLINE = offline
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"


def _access_hint(repo_id: str, exc: Exception) -> str:
    """Turn an opaque hub failure into the thing the user has to go do."""
    url = f"https://huggingface.co/{repo_id}"
    text = str(exc)
    lowered = text.lower()
    if _OFFLINE or "local_files_only" in lowered or "offline" in lowered:
        return (f"{repo_id} is not in the local HuggingFace cache and this run "
                f"is offline. Drop --offline to download it, or pre-populate "
                f"the cache.")
    if any(k in lowered for k in ("gated", "401", "403", "awaiting a review",
                                  "unauthorized", "forbidden",
                                  "access to model")):
        return (f"{repo_id} is gated. Open {url}, request access and wait for "
                f"approval, then authenticate with `huggingface-cli login` or "
                f"by setting HF_TOKEN. Approval is per-account and is not "
                f"instant.")
    if "not found" in lowered or "404" in lowered:
        return (f"{repo_id} was not found. Check the name, or whether the repo "
                f"was renamed: {url}")
    return (f"Could not fetch {repo_id} from HuggingFace ({exc.__class__.__name__}). "
            f"If it is gated, request access at {url} and log in.")


class AccessError(RuntimeError):
    """A model could not be fetched, with an explanation of what to do."""


INPUT_NAME = "input"
OUTPUT_NAME = "features"

#: Opset 18, not the 17 that `03-MODELS-AND-ML.md` suggests.
#:
#: torch's exporter emits at 18 and then down-converts. For UNI2-h that
#: conversion silently produces an INVALID graph: SwiGLUPacked's `Split` keeps
#: its opset-18 `num_outputs` attribute, and the model only fails when ONNX
#: Runtime tries to load it ("Unrecognized attribute: num_outputs"), long after
#: the export appears to have succeeded. Exporting at 18 avoids the lossy
#: round-trip entirely. Phikon and UNI are fine either way.
OPSET = 18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PIXEL_SCALE = 1.0 / 255.0


@dataclass(frozen=True)
class ModelSpec:
    """How to build one extractor and what it produces."""

    name: str
    repo_id: str
    feature_dim: int
    build: Callable[[], torch.nn.Module]
    input_size: int = 224
    revision: int = 1
    notes: str = ""


def _load_phikon() -> torch.nn.Module:
    """Owkin Phikon (iBOT ViT-B/16).  Embedding = CLS token, 768-d."""
    from transformers import AutoModel

    def build(**extra):
        return AutoModel.from_pretrained("owkin/phikon",
                                         local_files_only=_OFFLINE, **extra)

    try:
        backbone = build(add_pooling_layer=False)
    except TypeError:
        # Newer transformers dropped the kwarg for some ViT classes.
        backbone = build()
    except Exception as exc:                      # noqa: BLE001 - re-raised
        raise AccessError(_access_hint("owkin/phikon", exc)) from exc

    class PhikonEmbedding(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            out = self.model(pixel_values=pixel_values)
            # CLS token, exactly as convert_phikon.py did.
            return out.last_hidden_state[:, 0]

    return PhikonEmbedding(backbone)


def _load_timm_with_state_dict(model: torch.nn.Module, repo_id: str) -> torch.nn.Module:
    """Fetch a gated timm checkpoint and load it strictly.

    The two candidate filenames are tried in turn because the repos do not
    agree on which they publish. A miss on the first is not an error; a miss
    on both is, and it is almost always about access rather than the file.
    """
    from huggingface_hub import hf_hub_download

    last_error: Exception | None = None
    for filename in ("pytorch_model.bin", "model.safetensors"):
        try:
            path = hf_hub_download(repo_id, filename,
                                   local_files_only=_OFFLINE)
        except Exception as exc:  # wrong name, or no access; try the other
            last_error = exc
            continue
        if filename.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(path)
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        # strict=True is the guard against timm construction drift: the exact
        # timm version used on the Mac is unrecoverable, so a silent shape or
        # naming mismatch must fail loudly here rather than produce garbage.
        model.load_state_dict(state, strict=True)
        return model
    raise AccessError(_access_hint(repo_id, last_error)
                      + f"{os.linesep}{os.linesep}Underlying error: {last_error}")


def _load_uni() -> torch.nn.Module:
    """UNI v1 (ViT-L/16).  timm pooled output, 1024-d."""
    import timm

    model = timm.create_model("vit_large_patch16_224", pretrained=False,
                              init_values=1e-5, num_classes=0,
                              dynamic_img_size=False)
    return _load_timm_with_state_dict(model, "MahmoodLab/UNI")


def _load_uni2h() -> torch.nn.Module:
    """UNI2-h (custom ViT-H, SwiGLU, 8 register tokens).  1536-d.

    The kwargs block is the authoritative architecture record from the macOS
    `convert_to_coreml.py` — reproduce it exactly.
    """
    import timm

    model = timm.create_model("vit_huge_patch14_224", pretrained=False,
                              img_size=224, patch_size=14, init_values=1e-5,
                              embed_dim=1536, depth=24, num_heads=24,
                              mlp_ratio=2.66667 * 2, num_classes=0,
                              no_embed_class=True,
                              mlp_layer=timm.layers.SwiGLUPacked,
                              act_layer=torch.nn.SiLU, reg_tokens=8,
                              dynamic_img_size=False)
    return _load_timm_with_state_dict(model, "MahmoodLab/UNI2-h")


SPECS: dict[str, ModelSpec] = {
    "phikon-v1": ModelSpec("phikon-v1", "owkin/phikon", 768, _load_phikon,
                           notes="CLS token of iBOT ViT-B/16"),
    "uni-v1": ModelSpec("uni-v1", "MahmoodLab/UNI", 1024, _load_uni,
                        notes="timm pooled, ViT-L/16"),
    "uni2-h": ModelSpec("uni2-h", "MahmoodLab/UNI2-h", 1536, _load_uni2h,
                        notes="timm pooled, ViT-H SwiGLU + 8 reg tokens"),
}


# -- arbitrary repositories --------------------------------------------------
#
# The three specs above carry hand-written architecture recipes, kept because
# they are verified against the banks the macOS build produced. They are not
# the only thing convertible: timm and transformers can both build a model
# from a repository's own config, which covers most pathology encoders.
#
# What is NOT convertible is a bare weights file. A state dict records tensor
# names and shapes, not the architecture that consumes them, so a lone
# `.safetensors` with no config beside it cannot be rebuilt. Point at the
# directory (or the repo id), not the file.


class _CLSEmbedding(torch.nn.Module):
    """Wrap a transformers backbone so it returns one vector per image."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values)
        pooled = getattr(out, "pooler_output", None)
        if pooled is not None and pooled.ndim == 2:
            return pooled
        hidden = out.last_hidden_state
        # CLS token. The alternative — mean over patch tokens — is a different
        # embedding, so it is not silently substituted.
        return hidden[:, 0]


def _load_generic(source: str) -> torch.nn.Module:
    """Build a model from a HuggingFace repo id or a local directory.

    timm is tried first because most pathology encoders are timm ViTs and its
    hub integration reproduces the training-time config (``init_values`` and
    friends) that a hand-written recipe has to guess. transformers is the
    fallback. Failures from both are reported together: knowing that *neither*
    could build it is what tells you the directory lacks a config.
    """
    local = Path(source)
    if local.is_dir():
        timm_source, hf_source = f"local-dir:{local.as_posix()}", str(local)
    elif local.exists():
        raise AccessError(
            f"{source} is a file. A weights file alone does not record the "
            f"architecture that consumes it, so it cannot be converted. Point "
            f"at the directory containing it — the one with config.json — or "
            f"give the HuggingFace repository id instead.")
    else:
        timm_source, hf_source = f"hf-hub:{source}", source

    errors: list[str] = []
    try:
        import timm
        model = timm.create_model(timm_source, pretrained=True, num_classes=0)
        print(f"  built with timm from {timm_source}")
        return model
    except Exception as exc:                          # noqa: BLE001 - collected
        errors.append(f"timm: {exc}")

    try:
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(hf_source, local_files_only=_OFFLINE)
        print(f"  built with transformers from {hf_source}")
        return _CLSEmbedding(backbone)
    except Exception as exc:                          # noqa: BLE001 - collected
        errors.append(f"transformers: {exc}")

    joined = os.linesep.join(f"  {e}" for e in errors)
    raise AccessError(
        f"Could not build a model from {source}.{os.linesep}{joined}"
        f"{os.linesep}{os.linesep}{_access_hint(source, Exception(errors[-1]))}")


def spec_for(source: str, name: str | None = None, revision: int = 1,
             input_size: int = 224) -> ModelSpec:
    """A known spec by key, or one built for an arbitrary repo/directory."""
    if source in SPECS and name is None:
        return SPECS[source]
    slug = name or _slugify(source)
    return ModelSpec(slug, source, feature_dim=0,
                     build=lambda: _load_generic(source),
                     input_size=input_size, revision=revision,
                     notes=f"auto-converted from {source}")


def _slugify(source: str) -> str:
    """`MahmoodLab/UNI2-h` -> `uni2-h`; a path -> its folder name."""
    tail = Path(source).name if Path(source).is_dir() else source.rsplit("/", 1)[-1]
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in tail)
    return cleaned.strip("-").lower() or "extractor"


#: ONNX is protobuf, which cannot exceed 2 GB. Stay under it with headroom.
INLINE_LIMIT_BYTES = 1_900_000_000


def _consolidate(onnx_path: Path) -> bool:
    """Fold external weight data back into the .onnx file when it fits.

    torch's exporter writes weights to a sidecar ``<name>.onnx.data``. That is
    mandatory above 2 GB (protobuf's hard ceiling) but awkward below it: the two
    files must never be separated, and a missing sidecar fails at load time with
    an unhelpful error. Returns True if the model still uses external data.
    """
    import onnx

    data_path = onnx_path.with_name(onnx_path.name + ".data")
    if not data_path.exists():
        return False

    total = onnx_path.stat().st_size + data_path.stat().st_size
    if total > INLINE_LIMIT_BYTES:
        print(f"  weights stay external ({total / 1e9:.2f} GB > 2 GB protobuf limit); "
              f"keep {data_path.name} beside the .onnx")
        return True

    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(model, str(onnx_path), save_as_external_data=False)
    data_path.unlink()
    print(f"  inlined weights into a single {onnx_path.stat().st_size / 1e6:.0f} MB file")
    return False


def descriptor(spec: ModelSpec, filename: str, external_data: bool = False) -> dict:
    """The PathLearn extractor descriptor (`02-DATA-FORMATS.md` §4).

    `kind` becomes "onnx" so a Windows-extracted patch can never be silently
    pooled with a macOS `coreml:` one.  Whether the two are numerically
    equivalent is a separate, measurable question.
    """
    return {
        "identity": {"kind": "onnx", "name": spec.name, "revision": spec.revision},
        "kind": "onnx",
        "inputWidth": spec.input_size,
        "inputHeight": spec.input_size,
        "featureDim": spec.feature_dim,
        "pixelNormalization": {
            "meanRGB": IMAGENET_MEAN,
            "stdRGB": IMAGENET_STD,
            "scale": PIXEL_SCALE,
        },
        "modelFilename": filename,
        # Not in the macOS schema: the two conversion scripts disagreed on the
        # input name and the runtime had to hard-code it.  Record it instead.
        "inputName": INPUT_NAME,
        "outputName": OUTPUT_NAME,
        "sourceRepo": spec.repo_id,
        "notes": spec.notes,
        # True when weights live in <name>.onnx.data beside the graph; that
        # file must never be separated from the .onnx.
        "externalData": external_data,
    }


def convert(spec: ModelSpec, out_dir: Path, min_cosine: float,
            opset: int = OPSET) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{spec.name}.onnx"
    json_path = out_dir / f"{spec.name}.pathlearn-extractor.json"

    where = "cache only" if _OFFLINE else "cache, downloading if needed"
    print(f"[{spec.name}] loading from {spec.repo_id} ({where})…")
    model = spec.build()
    model.eval()

    size = spec.input_size
    example = torch.randn(1, 3, size, size)

    with torch.no_grad():
        reference = model(example)
    if reference.ndim != 2:
        raise RuntimeError(f"Expected a (batch, dim) embedding, got {tuple(reference.shape)}")
    actual_dim = reference.shape[1]
    if spec.feature_dim == 0:
        # An arbitrary repo has no expected width to check against, so the
        # measured one is recorded. The descriptor still cannot disagree with
        # the model, which is what the check below protects for known specs.
        spec = replace(spec, feature_dim=actual_dim)
        print(f"[{spec.name}] measured feature dimension: {actual_dim}")
    elif actual_dim != spec.feature_dim:
        raise RuntimeError(
            f"[{spec.name}] featureDim mismatch: descriptor says {spec.feature_dim}, "
            f"the model returns {actual_dim}. Refusing to write a wrong descriptor — "
            f"this is exactly the failure mode that makes two feature spaces "
            f"silently incomparable."
        )
    print(f"[{spec.name}] PyTorch embedding OK: {tuple(reference.shape)}")

    print(f"[{spec.name}] exporting ONNX (opset {opset}, dynamic batch)…")
    torch.onnx.export(
        model, (example,), str(onnx_path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=opset, do_constant_folding=True,
    )

    external = _consolidate(onnx_path)

    print(f"[{spec.name}] verifying against PyTorch…")
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 1.0
    # Several inputs, including a batch, so the dynamic axis is exercised too.
    for batch in (1, 1, 4):
        probe = torch.randn(batch, 3, size, size)
        with torch.no_grad():
            expected = model(probe).numpy()
        got = session.run([OUTPUT_NAME], {INPUT_NAME: probe.numpy()})[0]
        if got.shape != expected.shape:
            raise RuntimeError(f"shape mismatch: onnx {got.shape} vs torch {expected.shape}")
        for e, g in zip(expected, got):
            cos = float(np.dot(e, g) / (np.linalg.norm(e) * np.linalg.norm(g) + 1e-12))
            worst = min(worst, cos)

    print(f"[{spec.name}] worst cosine vs PyTorch: {worst:.8f}")
    if worst < min_cosine:
        raise RuntimeError(
            f"[{spec.name}] cosine {worst:.8f} below threshold {min_cosine}. "
            f"Export is not faithful — not writing the descriptor."
        )

    json_path.write_text(json.dumps(descriptor(spec, onnx_path.name, external), indent=2),
                         encoding="utf-8")
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"[{spec.name}] wrote {onnx_path.name} ({size_mb:.1f} MB) and "
          f"{json_path.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("models", nargs="*", default=[],
                        help=f"known extractor(s) to convert: "
                             f"{', '.join(SPECS)}, or 'all'")
    parser.add_argument("--repo", help="any HuggingFace repo id, or a local "
                                       "directory containing config.json")
    parser.add_argument("--name", help="descriptor name for --repo "
                                       "(default: derived from the repo)")
    parser.add_argument("--revision", type=int, default=1,
                        help="revision number for --repo; bump it when the "
                             "weights change, so old patches stay identifiable")
    parser.add_argument("--input-size", type=int, default=224,
                        help="square input edge for --repo")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--min-cosine", type=float, default=0.9999,
                        help="fail if ONNX/PyTorch cosine falls below this")
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument("--offline", action="store_true",
                        help="forbid network access; use only the local "
                             "HuggingFace cache")
    args = parser.parse_args(argv)
    _set_offline(args.offline)

    if args.repo:
        targets = [spec_for(args.repo, args.name, args.revision, args.input_size)]
    elif args.models:
        unknown = [m for m in args.models if m not in SPECS and m != "all"]
        if unknown:
            parser.error(f"unknown model(s): {', '.join(unknown)}. "
                         f"Use --repo for anything not in {list(SPECS)}.")
        names = list(SPECS) if "all" in args.models else args.models
        targets = [SPECS[n] for n in names]
    else:
        parser.error("name a known model, or pass --repo")

    for spec in targets:
        try:
            convert(spec, args.out, args.min_cosine, args.opset)
        except AccessError as exc:
            # Not a traceback: there is nothing to debug, only something
            # for the user to go and do.
            print(f"[{spec.name}] {exc}", file=sys.stderr)
            return 2
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
