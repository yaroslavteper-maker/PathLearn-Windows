# PathLearn (Windows)

Whole-slide-image annotation and machine-learning workbench for digital
pathology. Windows rewrite of the macOS/Swift original
([yaroslavteper-maker/PathLearn](https://github.com/yaroslavteper-maker/PathLearn)) —
same data formats, same algorithms, one clean coordinate system.

The two are separate codebases with no shared history: that one is a Swift/Xcode
app for macOS, this one is Python/PySide6 for Windows. Annotations written by
either open in the other, with one caveat about Y — see
[Coordinates](#coordinates--read-this-before-touching-pixels).

> Research tooling. **Not a medical device; not for diagnostic use.**

## Install

Python 3.12+. Keep the venv **outside** any synced folder (OneDrive, Dropbox) —
a venv there causes constant sync churn and slow imports.

```bash
python -m venv %USERPROFILE%\.venvs\pathlearn
%USERPROFILE%\.venvs\pathlearn\Scripts\pip install -e ".[ml,dev]"
```

Add `excel` (e.g. `.[ml,excel,dev]`) if you want **Project ▸ Export Excel…**;
without it, CSV export still works and opens in Excel directly.

Use `.[gpu,dev]` instead of `.[ml,dev]` for CUDA inference. **Pick one:** `ml`
installs `onnxruntime`, `gpu` installs `onnxruntime-gpu`, and both provide the
same module. `gpu` also needs a matching CUDA runtime on the machine — without
it, sessions are created successfully and silently run on CPU at roughly 20x
the latency. **Machine Learning ▸ Installed Extractors…** reports the provider
actually in use, which is the only reliable way to notice.

OpenSlide comes from the `openslide-bin` wheel, so there is nothing to install
separately.

The install is editable, and the launcher's path finder records an absolute
path — if you move the checkout, re-run `pip install -e .` from the new
location or `pathlearn.exe` stops importing.

## Run it

```bash
%USERPROFILE%\.venvs\pathlearn\Scripts\pathlearn.exe
```

Or with a slide to open straight away:

```bash
%USERPROFILE%\.venvs\pathlearn\Scripts\pathlearn.exe "D:\slides\case-01.svs"
```

## Tests

```bash
%USERPROFILE%\.venvs\pathlearn\Scripts\python.exe -m pytest
```

The suite generates its own synthetic pyramidal slide, so it needs no data and
no extractors.

## Using it

| | |
|---|---|
| **Open a slide** | File ▸ Open Slide (`Ctrl+O`) — `.svs`, `.tif`, `.ndpi`, `.mrxs`, `.scn`, … |
| **Pan** | `V` — drag; or hold Shift / middle-drag in any tool |
| **Lasso** | `L` — drag to trace a region, release to close |
| **Polygon** | `P` — click vertices; `Enter` or click the first point to close, `Backspace` undo, `Esc` cancel |
| **Zoom** | wheel, `Ctrl +` / `Ctrl -`, `Ctrl 0` to fit |
| **Select** | click a region in Pan mode — it highlights and scrolls into view in the Annotations list — or click its row |
| **Zoom to a region** | double-click its row in the sidebar |
| **Hide overlay** | `Ctrl+H` |
| **Rename a class** | Annotations panel ▸ Rename Class… — renames it in the palette, the annotations, both banks and other slides' sidecars |
| **Class breakdown** | Annotations panel ▸ Class Breakdown… — % of each class by region count and by area |

Annotations auto-save to `<slide>.geojson` beside the slide on every change.
A finished prediction run auto-saves the same way, to
`<slide>.predictions.json.gz`, and is reloaded when the slide is reopened —
threshold and hidden classes included. Untick **Save heatmap beside the
slide** in the Heatmap panel to stop that.

## Coordinates — read this before touching pixels

Everything is **level-0 slide pixels, top-left origin, Y down** — OpenSlide's
space and QuPath's space. There is no mirroring anywhere; the macOS build's
`slideH − y` flip does not exist here and must not be reintroduced.
`pathlearn/coords.py` is the authority.

**Annotations drawn on macOS are mirrored.** The first time you open a slide
whose sidecar came from the Mac, PathLearn flips it once, backs the original up
as `<name>.geojson.macos-mirrored.bak`, and tells you it did. See
[PORTING-NOTES.md](PORTING-NOTES.md) — this contradicts the handoff docs, and
the reason is documented there.

## Layout

```
pathlearn/
  coords.py            the coordinate-space policy  <- read first
  io/     slide.py     OpenSlide wrapper (SlideImage)
          geojson.py   QuPath-dialect codec
  models/ annotation.py     Annotation, colours, geometry
          classification.py classes + profiles
          store.py          sidecar persistence + legacy migration
  ui/     canvas.py         tiled pan/zoom + drawing tools
          tiles.py          background tile cache
          main_window.py    menus, dialogs, status
          panels/           sidebar
  core/   sampler.py        patch grid, centre-inside rule, subtractive carve-out
          logistic.py       multinomial softmax, stratified split, z-scoring
          pooling.py        mean || max || std bag pooling
          tsne.py           symmetric t-SNE (seed-compatible with the Swift)
          centroids.py      class centroids + promotion to a softmax model
          stain.py          Ruifrok-Johnston H&E deconvolution
          components.py     4-connected blobs: area, centroid, perimeter
          nucleus.py        classical segmenter behind a swappable protocol
          geometry.py       the 14-D descriptor at fixed ~1 um/px
          shape.py          12 scale-invariant outline descriptors
  extractors/ registry.py   descriptor discovery; the plug-in point
              onnx_extractor.py   ONNX Runtime session + batching
  data/   bank.py           SQLite patch bank
          geometry_bank.py  per-annotation descriptors
  project.py           a cohort: one composition row per slide  <- Project menu
  pipeline/ extract.py train.py predict.py
            geometry_train.py geometry_predict.py
            stitch.py         predicted tiles -> one annotation per clump
            composition.py    per-class share of the predicted tissue
            annotation_stats.py per-class share of what you drew
            export_images.py  annotations -> JPEGs
tools/  convert_extractor.py  HuggingFace -> ONNX + descriptor
tests/                 1,320 tests, incl. two synthetic slide generators
```

Portable algorithms stay pure and tested; platform pieces (UI, slide I/O, model
runtime, DB) sit behind thin interfaces.

## Feature extractors

PathLearn ships **no model weights**. Phikon, UNI and UNI2-h each carry their
own licence and none permit redistribution, so you convert your own copy. The
app itself only ever loads ONNX — it does not import torch.

An extractor is two files in one folder:

```
%LOCALAPPDATA%\PathLearn\Extractors\
    phikon-v1.onnx
    phikon-v1.pathlearn-extractor.json
```

The descriptor records the identity (`onnx:phikon-v1:r1`), input size, feature
dimension, pixel normalisation and the ONNX tensor names. Nothing else in the
app changes when you add one; `PATHLEARN_EXTRACTOR_DIRS` (os.pathsep-separated)
adds more search paths. `recovered/descriptors/` holds the three macOS
descriptors as a **schema reference only** — they are CoreML (`.mlpackage`)
and will not load here; `convert_extractor.py` writes the ONNX ones you use.

**Identity is a guard, not a label.** It is stamped on every patch and every
trained classifier, and prediction refuses a model whose extractor identity does
not match — two feature spaces that get mixed produce confident nonsense.

### Converting one

`MahmoodLab/UNI` and `MahmoodLab/UNI2-h` are **gated**: request access on
HuggingFace, wait for approval, then authenticate. `owkin/phikon` is ungated.

```
python -m venv ~/.venvs/pathlearn-convert
~/.venvs/pathlearn-convert/Scripts/pip install -r tools/requirements-convert.txt
huggingface-cli login          # or set HF_TOKEN; not needed for Phikon
python tools/convert_extractor.py phikon-v1 --out "%LOCALAPPDATA%\PathLearn\Extractors"
```

Add `--offline` to forbid downloads and use only the local HuggingFace cache.
Each run exports the ONNX, verifies it against PyTorch (cosine >= 0.9999 by
default) and only then writes the descriptor — a bad export never produces a
loadable extractor. Check what the app can see with **Machine Learning >
Installed Extractors…**.

Geometry models need no extractor at all: they read the traced outline, so they
work before any of this is set up.

## Status

**Working:** viewer and annotation workflow, the pure algorithm core, the patch
bank, ONNX extractors, training, prediction/heatmaps, t-SNE, the geometry
descriptor + trainer + per-annotation grading, stitching predictions back into
annotations, the per-class tissue composition breakdown, and projects that
accumulate per-slide rows — the model's composition, your annotation
breakdown, or both — and export the cohort as CSV or a three-sheet Excel
workbook.

**Not yet built:** VISTA, and the detect-then-grade region-proposal cascade.
See [PORTING-NOTES.md](PORTING-NOTES.md) for the ordered plan and the measured
results — including the batch effect that makes leave-one-slide-out the only
score worth quoting.

The tests need no data of their own: they generate synthetic pyramidal slides,
including one with two regions of deliberately different architecture used to
check that the geometry descriptors separate them in the expected direction.

## Licence

MIT — see [LICENSE](LICENSE).

The licence covers **the source code only**. PathLearn ships no model weights:
Phikon, UNI and UNI2-h each carry their own licence, none of which permit
redistribution, and each must be obtained and converted by you under its own
terms. See [NOTICE](NOTICE) and the *Feature extractors* section above.
