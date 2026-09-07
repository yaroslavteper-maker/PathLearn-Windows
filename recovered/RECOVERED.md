# Recovered from the macOS handoff bundle

The bundle at `from-mac/pathlearn-handoff/` was read in full on 2026-08-22 and
then **disappeared from disk** before its contents were consumed (the parent
`from-mac/README.md` survived; the `pathlearn-handoff/` subtree did not, and the
Recycle Bin was empty). The HuggingFace cache had already been rebuilt to
`~/.cache/huggingface/hub` and **survived intact** because it is hardlinked.

This file preserves everything that was read from the bundle before it vanished,
so the port is not blocked on getting it back. Values here were transcribed from
the actual files, not inferred.

---

## Recovered verbatim

### Extractor descriptors → `descriptors/`

All three reconstructed byte-for-byte from their originals. Common to all:
224×224, `scale = 1/255`, ImageNet mean `[0.485, 0.456, 0.406]`, std
`[0.229, 0.224, 0.225]`, `kind: "coreml"`, `revision: 1`.

| Name | featureDim | modelFilename |
|---|---:|---|
| `phikon-v1` | 768 | `phikon.mlpackage` |
| `uni-v1` | 1024 | `uni.mlpackage` |
| `uni2-h` | 1536 | `uni2.mlpackage` |

### Model construction (from `scripts/convert_to_coreml.py` and `convert_phikon.py`)

Quoted in the bundle's MANIFEST.md and reproduced here. These are the
authoritative architecture records — the `timm` version originally used is
unrecoverable, so `strict=True` loading plus the cosine check is what catches
construction drift.

```python
# phikon-v1  -> 768-d, CLS token
model = transformers.AutoModel.from_pretrained("owkin/phikon",
                                               add_pooling_layer=False)
# forward returns last_hidden_state[:, 0]

# uni-v1  -> 1024-d, timm pooled
model = timm.create_model("vit_large_patch16_224", pretrained=False,
                          init_values=1e-5, num_classes=0,
                          dynamic_img_size=False)
# load MahmoodLab/UNI state dict with strict=True

# uni2-h  -> 1536-d, timm pooled
model = timm.create_model("vit_huge_patch14_224", pretrained=False,
                          img_size=224, patch_size=14, init_values=1e-5,
                          embed_dim=1536, depth=24, num_heads=24,
                          mlp_ratio=2.66667 * 2, num_classes=0,
                          no_embed_class=True,
                          mlp_layer=timm.layers.SwiGLUPacked,
                          act_layer=torch.nn.SiLU, reg_tokens=8,
                          dynamic_img_size=False)
# load MahmoodLab/UNI2-h state dict with strict=True
```

`dynamic_img_size=False` was required on macOS because dynamic sizing emitted
`int()` position-embedding ops coremltools could not lower. Not a constraint for
ONNX, but keep 224×224 fixed so the export matches what produced the banks.

Input tensor names differed between the two scripts (`"input"` vs `"image"`);
both output `"features"`. Descriptors never recorded the input name. **Standardise
on ONNX and write the name into the descriptor.**

### Conversion environments

- Phikon (torch path): Python 3.11.15, torch 2.13.0, transformers 5.14.1,
  huggingface-hub 1.26.0, safetensors 0.8.0, coremltools 9.0.
- VISTA (TF path): Python 3.11.15, tensorflow 2.15.0, keras 2.15.0,
  scikit-image 0.26.0, numpy 1.26.4, coremltools 9.0.
- `timm` was in neither surviving venv — UNI/UNI2 were converted in an
  environment that no longer exists.

### VISTA facts

- The three `.h5` are **byte-for-byte the published files** from
  `github.com/gelatinfrogs/MicePan-Segmentation`, each exactly 5,994,424 bytes,
  never re-saved. Re-downloadable.
- Thresholds confirmed against `vista_model.py`: neoplasia **0.7**,
  metaplasia **0.5**, normal **0.3**; combine normal > metaplasia > neoplasia.
- The corrected Reinhard constants and the two-step normalisation are recorded
  as Corrections 5 and 6 in `../PORTING-NOTES.md`, verified independently
  against `reference-swift-source/Models/VISTAColor.swift`.

### Extractor usage on the Mac

- **uni2-h is the workhorse**: ~10 banks, >30k patches; `centroid3.cl` is
  stamped with it.
- **phikon-v1**: `PhikonPatches224.bank`, 1684 patches over 14 slides.
- **uni-v1**: descriptor and package existed, no bank ever produced.
- **Virchow: never set up.** No descriptor, no weights, no bank. The
  1280-vs-2560 embedding ambiguity is moot for this project.

---

## Not recovered — and what it costs

| Lost | Cost | Recovery |
|---|---|---|
| `scripts/VISTA/vista_model.py` | **Highest.** The UNet rebuild plus the `load_weights(by_name=True)` trick that sidesteps the Python-3.7 marshalled-`Lambda` breaking `load_model`. The manifest explicitly flagged this as the part not to re-derive. | Re-derivable from the MicePan repo, but the by-name loading would need working out again. |
| `scripts/VISTA/{vista_norm,target_stats,run_vista,convert_coreml}.py` | Moderate. | `VISTAColor.swift` carries the exact normalisation math (Corrections 5–6), so the important content survives. |
| `vista/*.h5` (3 × 5,994,424 B) | Low. | Unmodified published files — re-download from the MicePan repo. |
| `data/centroid3.cl` | Moderate — the trained uni2-h classifier. | Not on drive E:. Only recoverable from the Mac. |
| `data/Pancreatic Pathology.panin-profile.json` | Low. | Class names/colours are re-creatable; `ClassificationProfile.default()` already matches the shipped Swift default. |
| `data/224model.paninmodel.json` | Low. | Legacy-format classifier. |
| `verification/Uni2.bank` | Low. | Its primary slide was already missing; drive E: has other uni2-h banks. |
| Full `MANIFEST.md` text | Low. | Its substance is captured here and in `../PORTING-NOTES.md`. |

## What drive E: provides instead

Drive `E:` (a macOS-formatted data disk) turned out to hold far more than the
bundle did, and covers the verification need completely:

- **14 slides, ~20 GB** across `EiblClassifier1/` and `EiblClassifier2/`,
  including `slide-A.svs` — the exact slide the bundle's
  verification README recommended pairing with the Phikon bank.
- **8 `.bank` files**, including `EiblClassifier2/PhikonPatches224.bank`.
- **3 `.cl` classifiers**: `128_eibl_teper_classifier.cl`,
  `fwdozempicstudylivelink/Eibl.cl`, `fwdozempicstudylivelink/eibl2.cl`.
- ~30 `.geojson` sidecars and several `.predictions.json`.

Note `SomeClassifier/slide-A.geojson.macos-mirrored.bak`
— this Windows build already migrated that sidecar during the real-slide test,
which means the legacy-mirror migration has now run against genuine data.
