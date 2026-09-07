"""The user manual, as data.

Separate from the window that displays it so the text can be tested — a claim
about a control that no longer exists is worse than no manual, and a test can
check the named widgets are still there.

Each topic is (title, keywords, html). Keywords only widen the search; the
title and body text are searched too.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Topic:
    title: str
    keywords: str
    html: str


_STYLE = """
<style>
  body { font-family: sans-serif; line-height: 1.45; }
  h2 { margin-top: 0; }
  h3 { margin-top: 18px; margin-bottom: 4px; }
  code { background: rgba(128,128,128,0.18); padding: 1px 4px; }
  th { text-align: left; padding-right: 14px; }
  td { padding-right: 14px; vertical-align: top; }
  .warn { color: #d0762a; }
</style>
"""


TOPICS: list[Topic] = [

Topic("Getting started", "overview workflow first steps begin", """
<h2>Getting started</h2>
<p>PathLearn is a whole-slide annotation and machine-learning workbench. The
usual path through it:</p>
<ol>
  <li><b>Open a slide</b> — File ▸ Open Slide, or drop a path on the command
      line.</li>
  <li><b>Draw regions</b> with the Lasso or Polygon tool, each in a class.</li>
  <li>Then either:
    <ul>
      <li><b>Patch route</b> — Extract Patches ▸ Train Model ▸ Predict. This
          learns from 224-pixel tiles and paints a heatmap.</li>
      <li><b>Geometry route</b> — Describe ▸ Train ▸ Grade, in the Geometry
          panel. This learns from the <i>shape of the outline you traced</i>
          and gives one verdict per region. It needs no extractor.</li>
    </ul>
  </li>
</ol>
<p>Annotations save themselves to a <code>.geojson</code> file beside the
slide on every change. There is no Save command and no unsaved state.</p>
<p class="warn">This is research tooling. Not a medical device, not for
diagnostic use.</p>
"""),

Topic("Opening and viewing slides", "open close zoom pan level pyramid mpp svs ndpi", """
<h2>Opening and viewing slides</h2>
<p><b>File ▸ Open Slide…</b> (Ctrl+O). Formats come from OpenSlide:
<code>.svs</code>, <code>.tif</code>, <code>.ndpi</code>, <code>.mrxs</code>,
<code>.scn</code> and others.</p>

<h3>Moving around</h3>
<table>
<tr><td>Pan</td><td>Drag with the Pan tool, or hold Shift / middle-drag with any tool</td></tr>
<tr><td>Zoom</td><td>Mouse wheel, or Ctrl+= / Ctrl+-</td></tr>
<tr><td>Fit slide</td><td>Ctrl+0</td></tr>
<tr><td>Select a region</td><td>Double-click it with the Pan tool</td></tr>
</table>
<p>Tiles load in the background, so a fast pan shows blank squares briefly.
The status bar reports the slide dimensions, the cursor position in level-0
pixels, and the current magnification and pyramid level.</p>

<h3>Other slide commands</h3>
<ul>
  <li><b>File ▸ Show Slide in Explorer</b> — reveal the file.</li>
  <li><b>Help ▸ Slide Properties</b> — dimensions, level downsamples, µm/px,
      and the raw OpenSlide property list.</li>
  <li><b>File ▸ Close Slide</b> (Ctrl+W) — flushes the sidecar and clears the
      view. Predictions belong to one slide and are dropped.</li>
</ul>
"""),

Topic("Drawing annotations", "lasso polygon trace draw outline close loop subtractive", """
<h2>Drawing annotations</h2>
<p>Pick a class in the Annotations panel first — new regions take the class
that is showing under <b>Draw as</b>.</p>

<h3>Lasso (L)</h3>
<p>Two ways to trace, and you can mix them freely:</p>
<ul>
  <li><b>Drag</b> — press, trace with the button held, release. The loop closes
      on release.</li>
  <li><b>Click and trace</b> — click once and let go, then move the pointer with
      <i>no button held</i>. <b>Double-click to close the loop</b> on its
      starting point. Better for a long outline, since nothing has to stay
      pressed.</li>
</ul>
<p>Which one you get is decided by whether the pointer moved while the button
was down.</p>

<h3>Polygon (P)</h3>
<ul>
  <li>Click each vertex.</li>
  <li><b>Double-click to close</b>, anywhere. Enter also closes, and so does
      clicking back on the first vertex.</li>
  <li><b>Backspace</b> removes the last vertex.</li>
</ul>

<h3>Both tools</h3>
<ul>
  <li><b>Esc</b> abandons the outline in progress.</li>
  <li>A region needs at least 3 points; fewer is refused.</li>
  <li>The dashed preview shows the edge that will close the loop, so the final
      shape is never a surprise.</li>
</ul>

<h3>Subtractive regions</h3>
<p>Tick <b>Selected region is subtractive</b> to turn a region into a hole.
Patches whose centre falls inside it are cancelled before extraction — useful
for empty lumen inside a lesion. Subtractive regions are drawn dashed, and are
never themselves extracted or graded.</p>
"""),

Topic("The Annotations panel", "sidebar use checkbox delete rename reclassify list area", """
<h2>The Annotations panel</h2>
<p>One row per region, with four columns:</p>
<table>
<tr><td><b>Use</b></td><td>Whether pipelines act on this region — extraction,
    geometry, and batch delete all read this tick and nothing else.</td></tr>
<tr><td><b>Class</b></td><td>Its class, with the class colour.</td></tr>
<tr><td><b>Name</b></td><td>Optional label, e.g. <code>P2 - 6</code>.</td></tr>
<tr><td><b>Area</b></td><td>Area in level-0 px², abbreviated.</td></tr>
</table>
<p>Unticked regions are still drawn, as a faint dotted outline with no fill, so
excluding something stays visible rather than making it vanish.</p>

<h3>Buttons</h3>
<ul>
  <li><b>Use All / Use None</b> — bulk tick. The usual repair after importing a
      file where most regions arrived unticked.</li>
  <li><b>Delete Checked (N)</b> — deletes every ticked region at once. The count
      is in the label because Use defaults to <i>everything</i>: on a freshly
      opened slide this button will say <code>Delete Checked (16)</code> and
      mean it. The confirmation lists the batch by class.</li>
  <li><b>Rename… / Reclassify… / Delete</b> — act on the row you have selected,
      not on the ticks. <i>Rename</i> changes this one region's label;
      <i>Reclassify</i> moves it to another class.</li>
  <li><b>Class Breakdown…</b> — what percentage of your annotations each class
      accounts for, by region count and by area.</li>
  <li><b>Rename Class…</b> — above the list, next to <b>Add Class…</b>.
      Renames the class itself everywhere it is recorded, which is a
      different thing from Reclassify.</li>
</ul>

<h3>Finding a region</h3>
<p><b>Click a region on the slide</b> in Pan mode and its row is highlighted
and scrolled into view — which is how you identify something you can see but
cannot find in a long list. Clicking blank slide clears the selection.
<b>Double-click a row</b> to do the reverse and zoom the canvas to that
region.</p>
<p>Double-click a row to zoom the canvas to that region.</p>
<p>Some sidecars written by the macOS build carry an <code>isVisible</code>
flag. It is preserved when the file is written back but has no effect here —
every region is drawn and every region is clickable.</p>
"""),

Topic("Saving your work as a state", "state save open session restore everything", """
<h2>Saving your work as a state</h2>
<p><b>File &#9656; Save State…</b> (Ctrl+S) writes everything the app is
holding into one <code>.pathlearn</code> file:</p>
<ul>
  <li>the class palette — names, colours, null flags;</li>
  <li>the patch bank and the geometry bank;</li>
  <li>the trained models, embedded rather than referenced, so the state does
      not depend on a <code>.cl</code> staying where it was;</li>
  <li>a copy of the annotations for every slide you have opened;</li>
  <li>the geometry settings you were working with.</li>
</ul>
<p>A real state measured 6.3 MB for 12 slides, 212 annotations, 1,596 patches
and 69 descriptors — small enough to email or drop in a backup.</p>

<h3>What is deliberately not in it</h3>
<p><b>Slides.</b> They are gigabytes each. A state records their <i>paths</i>
so it can put annotations back beside them, and says plainly when a slide is
not where it remembers rather than pretending it restored.</p>
<p><b>Extractors.</b> Installed models, not working state — 4 GB, shared by
every project, and separately licensed. Use <code>tools/transfer.py</code> for
those.</p>

<h3>Opening one</h3>
<p><b>File &#9656; Open State…</b> replaces the live data, so it asks first and
spells out what goes: the patch bank, the geometry bank, the palette, and the
<code>.geojson</code> beside every slide the state carries. <b>Each sidecar is
backed up</b> as <code>&lt;name&gt;.geojson.before-restore.bak</code> before
being overwritten.</p>
<p class="warn">Restoring is not a merge. A state is a snapshot of a whole
working session, and opening one puts you back at that snapshot — anything done
since is replaced, not combined.</p>

<h3>There is no longer a Profile menu</h3>
<p>A profile only ever held the class list, which was never a useful unit of
work on its own; it is now one part of a state. Classes and colours still
survive a restart — they are kept in settings — so nothing needs saving just to
keep your palette.</p>
"""),

Topic("Classes and profiles", "class colour color palette null exclude profile", """
<h2>Classes and profiles</h2>
<p><b>Draw as</b> picks the class for new regions, and <b>Add Class…</b>
creates one.</p>
<p>The <b>colour swatch beside the class picker</b> is the colour of that
class — click it to change it. Every region already drawn in the class is
recoloured with it, because the colour is stored on each annotation as well as
on the class, and leaving them behind would make the picker and the canvas
disagree.</p>

<h3>Null classes</h3>
<p>Tick <b>Exclude / null class</b> to mark the current class as null. Patches
of a null class never train as a real class; instead they become a reference,
and any candidate resembling them is dropped from training and prediction.
Use it for background, tearing, pen marks, and out-of-focus tissue.</p>

<h3>Where the palette lives</h3>
<p>There is no Profile menu any more. The class list is part of a saved state,
and is also kept in settings so it survives a restart on its own — see
<i>Saving your work as a state</i>.</p>
"""),

Topic("Import and export (GeoJSON)", "geojson import export qupath sidecar mirrored migrate", """
<h2>Import and export (GeoJSON)</h2>
<p>Annotations live in a <code>&lt;slide&gt;.geojson</code> sidecar next to the
slide, written atomically on every change. It is QuPath-compatible: you can
open it there, or bring QuPath annotations here.</p>

<h3>Importing</h3>
<p><b>File ▸ Import Annotations (GeoJSON)…</b>. You will be asked two things:</p>
<ul>
  <li><b>Coordinate origin</b> — answer Yes for a QuPath-style file (top-left
      origin). Answer <b>No</b> only for a file hand-copied from the macOS
      PathLearn build, whose Y coordinates are mirrored.</li>
  <li><b>Replace or append</b> — if the slide already has annotations.
      Appending gives the incoming regions fresh IDs so they cannot collide.</li>
</ul>
<p>Imported regions arrive ticked under <b>Use</b> unless the file explicitly
says otherwise.</p>

<h3>Legacy sidecars</h3>
<p>A sidecar written by the macOS build is detected on load, flipped upright
once, and rewritten with a marker so it is never flipped twice. The original is
backed up as <code>&lt;name&gt;.macos-mirrored.bak</code> first, and the app
tells you it happened. If the result looks upside-down relative to the tissue,
restore the backup and report it.</p>

<h3>Exporting</h3>
<p><b>File ▸ Export Annotations (GeoJSON)…</b> writes a copy, by default named
<code>&lt;slide&gt;-qupath.geojson</code>. The sidecar itself is already in the
same format, so this is only for sending a copy elsewhere.</p>
"""),

Topic("Saving annotations as JPEGs", "jpeg jpg image export picture crop figure", """
<h2>Saving annotations as JPEGs</h2>
<p><b>File &#9656; Export Annotations as JPEG…</b> writes one image per ticked
annotation, cut from the slide.</p>
<p>The pixels come from the slide itself, not from the canvas, so the result
does not depend on the zoom you happened to be at, and no outline or heatmap
appears unless you ask for it.</p>

<table>
<tr><td><b>Destination</b></td><td>Defaults to
    <code>&lt;slide&gt;-annotations</code> beside the slide.</td></tr>
<tr><td><b>One subfolder per class</b></td><td>Writes
    <code>PanIN-2/&lt;image&gt;.jpg</code>. This is the layout most image
    classifiers expect, and usually the reason for exporting.</td></tr>
<tr><td><b>Resolution</b></td><td><i>Automatic</i> picks the finest pyramid
    level whose crop still fits the size cap — full detail for a small duct,
    a manageable file for a large lesion. Or pin a level.</td></tr>
<tr><td><b>Max edge</b></td><td>Longest side of the written image; anything
    larger is downscaled.</td></tr>
<tr><td><b>Margin</b></td><td>Extra slide around the region, in level-0
    pixels. Context often makes a duct readable that is ambiguous when
    cropped tight.</td></tr>
<tr><td><b>Draw the outline</b></td><td>Paints your trace onto the image in
    the class colour — for figures rather than for training.</td></tr>
<tr><td><b>Blank outside the outline</b></td><td>Fills the surrounding tissue
    with white, so only the traced region remains. Off by default.</td></tr>
</table>

<p>Files are named <code>&lt;slide&gt;_&lt;class&gt;_&lt;name&gt;.jpg</code>,
falling back to a number when a region has no name. <b>Nothing is ever
overwritten</b> — a repeat export numbers the new files alongside the old.
Characters Windows forbids in filenames are replaced, so a class called
<code>grade 2/3</code> does not silently become a folder.</p>
<p>Only <b>ticked</b> annotations are exported, same as everywhere else.</p>
"""),

Topic("Coordinates", "coordinate origin mirror y flip level0 top-left", """
<h2>Coordinates</h2>
<p>One space, everywhere: <b>level-0 slide pixels, top-left origin, Y
increasing downward</b>. The same convention as OpenSlide and QuPath.</p>
<p>The macOS build rendered mirrored and stored mirrored view-Y in its
sidecars. This build does not mirror anywhere. The only place a flip happens is
the one-time migration of a legacy sidecar, or an import you explicitly tell to
flip.</p>
<p>Files written here carry a <code>"pathlearn"</code> marker recording the
coordinate space, so their origin is never guessed again.</p>
"""),

Topic("Feature extractors", "extractor onnx uni phikon install identity convert model", """
<h2>Feature extractors</h2>
<p>An extractor turns a 224-pixel patch into a feature vector. PathLearn ships
none — the models are licensed separately — so you install your own.</p>

<h3>Installing one</h3>
<p>Two files go in one folder:</p>
<pre>%LOCALAPPDATA%\\PathLearn\\Extractors\\
    phikon-v1.onnx
    phikon-v1.pathlearn-extractor.json</pre>
<p><b>Machine Learning ▸ Installed Extractors…</b> shows what was found,
reports any descriptor it had to skip and why, and has <b>Open Folder</b>
(which creates the folder if it does not exist) and <b>Rescan</b>.
<code>PATHLEARN_EXTRACTOR_DIRS</code> adds more search paths.</p>

<h3>Building one, from inside the app</h3>
<p><b>Machine Learning &#9656; Convert a Model to ONNX…</b> does the whole job:
pick a model, watch the log, and the result installs itself.</p>
<ul>
  <li><b>Built-in</b> — Phikon v1, UNI v1 or UNI2-h. These carry verified
      architecture recipes, so their output matches the feature space existing
      banks were built in.</li>
  <li><b>Other</b> — any HuggingFace repository id, or a local folder. timm and
      transformers build the model from the repository&#8217;s own config, so
      most pathology encoders convert without a recipe. The feature dimension
      is measured rather than assumed.</li>
</ul>
<p><b>What cannot be converted:</b> a bare weights file. A
<code>.safetensors</code> or <code>.bin</code> on its own records tensor names
and shapes but not the architecture that consumes them. Point at the folder
containing <code>config.json</code>, or give the repository id.</p>
<p>Conversion needs torch, timm and transformers — about 1.1 GB — which the app
does not otherwise depend on. The first time, <b>Set Up…</b> builds a separate
environment for them; nothing is added to PathLearn&#8217;s own. If you already
have a Python with torch, <b>Use Existing Python…</b> points at it instead.</p>
<p>UNI and UNI2-h are gated: request access on HuggingFace, wait for approval,
then <code>huggingface-cli login</code> or set <code>HF_TOKEN</code>. The log
says exactly this when access is the problem.</p>
<p>Every export is verified against PyTorch (cosine &#8805; 0.9999) before a
descriptor is written, so a bad conversion cannot produce a loadable
extractor. The same job runs from the command line as
<code>tools/convert_extractor.py</code>.</p>

<h3>Identity</h3>
<p>Each extractor has an identity like <code>onnx:phikon-v1:r1</code>, stamped
on every patch and every trained model. Prediction refuses a model whose
identity does not match the extractor in use. This is a guard, not a label:
mixing two feature spaces produces confident nonsense rather than an error.</p>

<h3>GPU</h3>
<p>The Installed Extractors dialog reports the execution provider actually in
use. If it says <code>CPUExecutionProvider</code>, extraction runs roughly 20x
slower — install <code>onnxruntime-gpu</code> with a matching CUDA runtime.
This is worth checking, because a missing CUDA runtime does not error; it just
quietly uses the CPU.</p>
"""),

Topic("Extracting patches", "extract patches grid stride white nuclei sampling bank", """
<h2>Extracting patches</h2>
<p><b>Machine Learning ▸ Extract Patches…</b> (Ctrl+E). Samples a grid inside
each ticked annotation, runs every patch through an extractor, and stores the
feature vectors in the patch bank.</p>

<table>
<tr><td><b>Model</b></td><td>Which installed extractor to use. Patch size
    defaults to its input size.</td></tr>
<tr><td><b>Pyramid level</b></td><td>Level 0 is full resolution. A higher level
    covers more tissue per patch at lower detail.</td></tr>
<tr><td><b>Patch size</b></td><td>In pixels <i>at the chosen level</i>.</td></tr>
<tr><td><b>Stride</b></td><td>Step between patch origins, also at that level.
    Stride below patch size overlaps them.</td></tr>
<tr><td><b>Max white fraction</b></td><td>Default 0.75. Patches whiter than
    this are background and are dropped.</td></tr>
<tr><td><b>Min nuclei per patch</b></td><td>0 disables the check. Above 0 each
    surviving patch is stain-deconvolved and segmented, which is noticeably
    slower.</td></tr>
</table>

<p>A patch belongs to an annotation if its <b>centre</b> falls inside — that
rule, not overlap. Ticked subtractive regions cancel any patch whose centre
lands in them.</p>
<p>Only <b>ticked</b> annotations are offered, and the class list lets you
narrow further. The sheet previews the patch count before you commit, which is
worth reading: a small stride on a large region produces tens of thousands.</p>
<p>Every patch records the slide path and the annotation it came from, so you
can find or delete it later.</p>
"""),

Topic("The patch bank", "bank sqlite browse export import clear delete patches", """
<h2>The patch bank</h2>
<p>A SQLite database of extracted feature vectors, shared across slides and
sessions. The <b>Patch Bank</b> panel shows totals per class and per extractor
identity, and warns when the bank holds more than one feature space.</p>

<h3>Buttons</h3>
<ul>
  <li><b>Browse…</b> — a filterable table of every patch: source slide, class,
      position, white fraction, nucleus count. Filter by slide or class, select
      rows and delete them, or delete everything from one slide.</li>
  <li><b>Export…</b> — a <code>.bank</code> JSON snapshot. This is the recovery
      path; it also reads banks written by the macOS build.</li>
  <li><b>Import…</b> — merge a <code>.bank</code> file in.</li>
  <li><b>Clear…</b> — delete every patch.</li>
</ul>
<p>Mixing feature spaces in one bank is allowed but never silently trained on:
the training sheet makes you pick one identity.</p>
"""),

Topic("Training a patch classifier", "train logistic pooled aggregation null validation l2", """
<h2>Training a patch classifier</h2>
<p><b>Machine Learning ▸ Train Model…</b> (Ctrl+T). Fits a multinomial logistic
regression over the banked feature vectors.</p>

<table>
<tr><td><b>Trained on</b></td><td>Which extractor identity to use. Patches from
    other identities are ignored.</td></tr>
<tr><td><b>Aggregation</b></td><td><i>Per patch</i> gives one training example
    per patch. <i>Pooled</i> concatenates mean, max and std over each
    annotation's patches into one example — one verdict per region, which suits
    a label that is a property of the lesion rather than of a tile.</td></tr>
<tr><td><b>Null threshold</b></td><td>Default 0.85. Cosine similarity to the
    nearest null patch above which a candidate is dropped. Cosine compares
    direction and ignores magnitude, so a patch can match despite very
    different brightness.</td></tr>
<tr><td><b>Validation fraction</b></td><td>Share held out for scoring.</td></tr>
<tr><td><b>Iterations, L2</b></td><td>Optimiser settings. L2 defaults to
    0.001.</td></tr>
<tr><td><b>Stop</b></td><td>Abandons the run. Cancellation is cooperative —
    it takes effect at the next checkpoint, including inside the descent loop,
    so it is prompt but not instant. A cancelled run produces no model.</td></tr>
<tr><td><b>Max white fraction</b></td><td>A second filter applied at training
    time, over what extraction already stored.</td></tr>
</table>

<p>Results show per-class precision, recall, F1 and support. The trained model
becomes the active one for Predict, and <b>Save Model…</b> writes a
<code>.cl</code> file.</p>
<p class="warn">This sheet splits <i>patches</i>, so patches from one
annotation — and one slide — land on both sides of the split. Expect the score
to be optimistic. See the note on the batch effect.</p>
"""),

Topic("t-SNE and the batch effect", "tsne umap embedding cluster batch slide leakage probe", """
<h2>t-SNE and the batch effect</h2>
<p><b>Machine Learning ▸ t-SNE Plot…</b> projects the banked features to 2-D.
Set the feature space, max points, perplexity and iterations, then Run.</p>

<h3>Colour by</h3>
<p>The important control. <b>Class</b> asks whether your classes separate.
<b>Slide (batch effect)</b> asks whether the <i>slides</i> separate — and if
they do, a classifier can score well by recognising staining and scanner
rather than biology.</p>
<p>Because eyeballing a scatter plot is unreliable, the window also fits a
linear probe on the features and reports how recoverable class and slide
identity each are. <b>Run it on your own bank and compare the two numbers.</b>
If slide identity is the more recoverable of the pair — which it was on the
data this was developed against — the features carry more about staining and
scanner than about biology.</p>
<p class="warn">The consequence is severe and easy to miss. A per-patch
accuracy is inflated whenever patches from one slide land in both the training
and the test half: the model recognises the slide, not the grade. Splitting by
whole slide instead can cut that score to a fraction of itself, sometimes to
the majority baseline. <b>Only a leave-one-slide-out score is worth
quoting.</b> And if a class appears on just one slide, no model can learn that
class — only that slide.</p>
<p>Mitigations: annotate each class across several slides, and consider stain
normalisation before extraction.</p>
"""),

Topic("Predicting and heatmaps", "predict heatmap inference regions confidence overlay", """
<h2>Predicting and heatmaps</h2>
<p><b>Machine Learning ▸ Predict…</b> (Ctrl+R). Tiles the regions you choose,
classifies each tile, and paints the result over the slide.</p>
<p>You need a slide open and at least one region drawn — prediction samples
tiles inside annotations, not the whole slide.</p>

<table>
<tr><td><b>Active</b></td><td>The model in use, with <b>Load Model…</b> to open
    a saved <code>.cl</code>.</td></tr>
<tr><td><b>Pyramid level, Patch size, Stride</b></td><td>As in extraction.
    Matching what the model was trained at matters.</td></tr>
<tr><td><b>Aggregation</b></td><td><i>Per patch</i> keeps each tile's own
    verdict. <i>Mean probability</i> gives one verdict per region. <i>Max
    probability</i> lets the worst focus win the region.</td></tr>
<tr><td><b>Max white fraction, Min nuclei</b></td><td>Skip background tiles.</td></tr>
<tr><td><b>Min confidence</b></td><td>Tiles below this are discarded rather
    than painted.</td></tr>
</table>
<p>Every region starts selected. <b>Select None</b> then tick the two or three
you care about is usually quicker than unticking the rest, and the count beside
the buttons says how many will be predicted over.</p>

<h3>Extractor substitution</h3>
<p>If the model's extractor is not installed but a comparable one is — the same
model in a different runtime — the sheet offers a substitution and asks you to
approve it once. Approvals are remembered. Measured on one such pair, the only
disagreements were low-confidence grade calls.</p>

<h3>Heatmap panel</h3>
<p>Toggle the overlay, choose which classes to paint, and raise the minimum
confidence to hide marginal tiles. Ctrl+H hides annotations if the outlines get
in the way.</p>
<p>A geometry model cannot run here — it reads a traced outline, and a tile has
no outline. Use Grade Annotations instead.</p>

<h3>The heatmap is kept beside the slide</h3>
<p>With <b>Save heatmap beside the slide</b> ticked — it is, by default — a
finished prediction run is written to
<code>&lt;slide&gt;.predictions.json.gz</code> next to the slide, and loaded
again the next time you open that slide. The confidence threshold and any
classes you hid travel with it, so the heatmap comes back exactly as you left
it and a figure you quoted still reproduces.</p>
<p>It is gzipped because it is written unattended: a large slide is around
fifteen megabytes of JSON and about one compressed. <b>Save…</b> still writes
a plain readable <code>.json</code> wherever you choose, and either form loads
— including a sidecar written by an older build or by the macOS app.</p>
<p>Unticking the box stops new saves. It does not delete a file already
written, and an existing one is still loaded.</p>
<p class="warn">Predictions are not part of <b>Save State</b>. They live next
to the slide, so they travel with the slide rather than with your working
state.</p>
"""),

Topic("Renaming a class", "rename class typo relabel merge palette classes fix name", """
<h2>Renaming a class</h2>
<p>A class name is not only a palette entry. The same text is stamped on every
annotation, on every extracted patch in the bank, and on every geometry
record. <b>Rename Class…</b> in the Annotations panel changes all of them
together.</p>

<h3 class="warn">Why not just retype it in the palette</h3>
<p>Because the rest would keep the old name. The bank would still hold patches
labelled the old way while every new extraction used the new one — so training
would quietly treat one class as two, and the model simply would not learn the
thing you thought you were teaching it. That failure is invisible until you
look at the class counts.</p>

<h3>What it offers</h3>
<p>The sheet counts everything <i>before</i> changing anything, and each part
is a tick you can clear:</p>
<table>
  <tr><th>This slide</th><td>Always. Every annotation on the open slide, and
      the class you are currently drawing with.</td></tr>
  <tr><th>Patch bank</th><td>Relabels the patches rather than deleting them —
      those features cost real time to extract.</td></tr>
  <tr><th>Geometry bank</th><td>Relabels the records. The descriptors are
      untouched: a rename is a relabel, not a re-describe.</td></tr>
  <tr><th>Other slides</th><td>Rewrites the <code>.geojson</code> sidecars of
      slides you have opened before. Hover the tick to see which, and how
      many annotations each holds.</td></tr>
</table>
<p>Slides you have <b>never opened</b> in PathLearn are not in that list and
keep the old name. Open one and rename again, or rename before you have spread
the class around.</p>

<h3 class="warn">Renaming onto an existing class merges them</h3>
<p>If you type a name that already exists, the two classes become one
everywhere — annotations, patches, geometry records. The sheet says so before
you confirm. It cannot be undone by renaming back, because nothing records
which rows came from which class.</p>
"""),

Topic("Annotation class breakdown", "breakdown percentage class annotations count area proportion how much each class tally", """
<h2>Annotation class breakdown — how much of each class you drew</h2>
<p><b>Class Breakdown…</b> in the Annotations panel answers "of everything I
have outlined on this slide, how much is PanIN-2?" — about <i>your tracing</i>,
not about anything a model predicted. (For the model's answer, see
<b>Tissue composition</b>.)</p>

<h3 class="warn">Two answers, and they disagree on purpose</h3>
<p>The sheet always shows both, as two bars and two columns:</p>
<table>
  <tr><th>Count %</th><td>How many regions carry each class. The literal
      reading of "percentage of each class", and the right number when each
      traced region is one lesion and you want to know how the lesions are
      distributed.</td></tr>
  <tr><th>Area %</th><td>How much slide each class covers. The right number
      when you want to know how much tissue is involved.</td></tr>
</table>
<p>These separate sharply in PanIN work, and the reason is measured rather
than theoretical: region area rises <b>steeply</b> with PanIN grade, by close
to an order of magnitude across the range. A slide with twenty small 1a ducts
and two large grade-3 lesions is over 90% 1a by count and can be mostly grade 3
by area. Quoting one without the other is how the same slide ends up described
two opposite ways — so decide which question you are asking before you quote a
figure.</p>
<p><b>Mean size</b> is in the table for the same reason: it is what makes the
gap between the two percentages legible at a glance.</p>

<h3>Scope</h3>
<p>The breakdown covers every annotation by default. Tick <b>Only annotations
checked under Use</b> to restrict it to the ones currently ticked — useful when
that is the set you are about to extract from or delete. The scope you chose is
written into the exported CSV, so a saved table always says what it counted.</p>

<h3>Subtractive polygons are holes, not regions</h3>
<p>A subtractive polygon carves space <i>out</i> of the region enclosing it —
empty lumen inside a lesion, say — and is never itself extracted. So the
breakdown:</p>
<ul>
  <li><b>does not count it</b> as a region of its own class, and</li>
  <li><b>deducts its area</b> from the class of the region enclosing it.</li>
</ul>
<p>Which region that is comes from the innermost one containing the hole's
centre, so a hole inside a lesion inside a block carves the lesion. A hole
sitting inside no region at all is reported and deducts from nothing — the
sheet says so, because it usually means the polygon is not where you meant
it.</p>

<h3 class="warn">Overlapping traces are counted twice</h3>
<p>Area is the sum of polygon areas, so two regions that overlap contribute
their overlap to both and the total exceeds the tissue actually covered. The
sheet warns when regions look like they overlap and the area percentages should
then be read as upper bounds. Hand-drawn regions rarely overlap; stitched ones
never do, by construction.</p>

<h3>Getting it out</h3>
<p><b>Copy Table</b> and <b>Save CSV…</b> both produce the same table, with the
scope, the subtractive counts and the overlap warning in a footer.</p>
"""),

Topic("Projects — comparing slides", "project cohort multiple slides accumulate excel csv export compare batch", """
<h2>Projects — one row per slide, built up over time</h2>
<p>A <b>project</b> collects the tissue-composition breakdown of many slides
into one table. The workflow is one slide at a time:</p>
<ol>
  <li><b>Project ▸ New Project…</b> — name it and pick where it lives.</li>
  <li>Open a slide and draw your regions.</li>
  <li>File either or both:
    <ul>
      <li><b>Project ▸ Add Annotation Breakdown</b> — what <i>you</i> drew.
          Also the <b>Add to…</b> button in the Class Breakdown sheet. Needs
          no model.</li>
      <li><b>Predict</b>, then <b>Project ▸ Add Predicted Composition</b> —
          what the <i>model</i> called it. Also the <b>Add to…</b> button in
          the Tissue Composition sheet.</li>
    </ul>
  </li>
  <li>Close the slide, open the next one, repeat.</li>
  <li><b>Project ▸ Project Table…</b> to see them side by side, and export.</li>
</ol>

<h3>Two sources, one slide</h3>
<p>A slide can carry <b>one predicted row and one annotated row at once</b> —
the <b>Source</b> column says which is which, and filing one never overwrites
the other. That is the point: you can put the model's answer next to your own
for the same slide.</p>
<table>
  <tr><th>predicted</th><td>What the model called the tissue, counted in
      <b>tiles</b>, out of the tissue it predicted over.</td></tr>
  <tr><th>annotated</th><td>What you drew, counted in <b>regions</b>, out of
      the regions you drew.</td></tr>
</table>
<p class="warn">Those are different denominators. Read down a column within one
source; do not average a predicted row into an annotated one. Confidence
columns are empty on annotated rows because a region you drew was never
scored — empty, not zero.</p>
<p><b>Remove Row</b> takes only the row you selected. The other source for the
same slide stays.</p>
<p>The project reopens by itself next time you start PathLearn, and every
addition is written to disk immediately — there is no Save Project command and
nothing to lose if the app stops unexpectedly.</p>

<h3>The table</h3>
<p>One row per slide; one <b>%</b> column per class across every model used.
The Project Table window stays open while you work, so you can keep adding to
it without closing anything.</p>
<table>
  <tr><th>Copy Table</th><td>The wide table as CSV, on the clipboard.</td></tr>
  <tr><th>Export CSV…</th><td>Writes two files: the wide table you named, and
      a <code>-long.csv</code> beside it with one row per slide <i>and</i>
      class — the shape statistics packages want.</td></tr>
  <tr><th>Export Excel…</th><td>A workbook with <b>Composition</b> (wide),
      <b>Detail</b> (long, with every per-run setting) and <b>About</b> (the
      caveats). Numbers go in as numbers, so you can chart and average them
      directly. Needs <code>openpyxl</code>; if it is missing the app tells
      you the one command that installs it.</td></tr>
  <tr><th>Remove Row</th><td>Drops the selected row only. The slide, its
      annotations, its predictions and the slide's other source row are all
      untouched.</td></tr>
</table>

<h3 class="warn">A blank cell is not a zero</h3>
<p>This is the one thing to get right when reading the table.</p>
<ul>
  <li>A <b>number</b> — including <code>0.0</code> — means that slide's model
      had that class and reported this much. Zero means it looked and found
      none.</li>
  <li>A <b>blank</b> means that slide's model had no such class at all, so the
      question was never asked.</li>
</ul>
<p>Filling blanks with zeros turns "not measured" into "measured as absent",
which is exactly the error that makes a cohort table lie. The exports keep the
cells empty for the same reason — do not fill them in downstream.</p>
<p>For an <b>annotated</b> row the class palette in force plays the part the
model's class list plays for a predicted one: a palette class you drew none of
is a real <b>0</b>, and a class that was never in your palette stays
<b>blank</b>.</p>

<h3>Adding a slide that is already in the project</h3>
<p>PathLearn asks. <b>Yes</b> replaces the old row — the usual case, when you
have re-run a prediction and the earlier number is superseded. <b>No</b> keeps
both rows, which is what you want when deliberately comparing two models on
one slide.</p>

<h3 class="warn">When rows are not comparable</h3>
<ul>
  <li><b>Different extractors.</b> If entries came from different feature
      extractors the table says so in orange, and the Excel About sheet
      repeats it. Two feature spaces are not comparable — re-run the odd ones
      out rather than reading across the rows.</li>
  <li><b>Different regions.</b> Each percentage is of the tissue predicted
      over <i>on that slide</i>. Predicting over a whole section on one slide
      and one hand-picked duct on another gives two numbers that do not belong
      in the same column. The <b>mm²</b> columns are the safer comparison, and
      exist for that reason.</li>
  <li><b>Different settings.</b> Confidence threshold, pixel size and grid
      size are recorded per row and appear on the Detail sheet. Rows made
      under different settings are not directly comparable either.</li>
</ul>

<h3>Where the project lives</h3>
<p>A single <code>.pathlearn-project.json</code> file wherever you saved it.
It stores the numbers, not the slides — moving or renaming a slide does not
break the project, and the project does not grow with your image data. It is
separate from <b>Save State</b>: a state holds banks, models and annotations,
while a project holds results. Both persist on their own.</p>
"""),

Topic("Tissue composition", "composition percentage proportion area breakdown burden quantify mm2 fraction", """
<h2>Tissue composition — how much of each class</h2>
<p><b>Tissue Composition…</b> in the Heatmap panel turns a prediction run into
the number you usually want out of it: what fraction of the tissue each class
occupies, as a percentage and as a real area.</p>

<h3>What you get</h3>
<table>
  <tr><th>Area %</th><td>The class's share of the predicted tissue. This is
      the headline figure.</td></tr>
  <tr><th>Area</th><td>In mm² when the slide records its pixel size, in
      pixels when it does not.</td></tr>
  <tr><th>Tiles / Tile %</th><td>The raw count and its share. Equal to the
      area share on a plain single-pass run; see below for when it is
      not.</td></tr>
  <tr><th>Mean confidence</th><td>How sure the model was, averaged over that
      class's tiles. A class holding 40% of the tissue at 0.52 confidence is
      a different result from 40% at 0.95.</td></tr>
</table>
<p>The coloured bar across the top is the same breakdown at a glance, in the
heatmap's own colours. <b>Copy Table</b> and <b>Save CSV…</b> both produce a
spreadsheet-ready table.</p>

<h3 class="warn">What the percentage is a percentage OF</h3>
<p>This is the part to get right before quoting a number.</p>
<ul>
  <li>The denominator is <b>the tissue that was predicted over</b> — the
      regions you chose in the Predict sheet. It is a whole-slide figure only
      if you predicted over the whole slide.</li>
  <li>Tiles the sampler rejected as too white or too nuclei-poor were never
      classified, and are <b>not</b> in the denominator. That is what you
      want: counting blank glass would deflate every class by however much
      background your region happened to enclose. But it does make this a
      percentage of <i>sampled tissue</i>, not of <i>slide area</i>.</li>
  <li>Tiles below the confidence slider are excluded, and the sheet says how
      many. Raising the slider changes the percentages — a run where a third
      of the tissue was too marginal to call is a different result from one
      where it was not, and the excluded count is how you tell.</li>
  <li>Classes you have <b>hidden</b> in the overlay are still counted.
      Hiding is a drawing choice; it does not move the numbers.</li>
</ul>
<p>The saved CSV repeats all of this in a footer, because a bare percentage
column outlives anyone's memory of what it was a percentage of.</p>

<h3>Why area share and tile share can differ</h3>
<p>On a single-pass run with no overlap they are identical. They separate when
tiles are not interchangeable:</p>
<ul>
  <li><b>Multi-pass runs</b> mix patch sizes. One large tile counts the same
      as one small tile by count, but covers more tissue by area.</li>
  <li><b>Overlapping tiles</b> (a stride below the patch size) cover the same
      tissue twice. A cell claimed by two classes is awarded to the more
      confident tile, so the areas still sum to 100% instead of
      double-counting the overlap.</li>
</ul>
<p>When the two columns disagree, <b>area share is the one to quote.</b></p>

<h3>Comparing slides</h3>
<p>Percentages are comparable between slides only if the regions were drawn
comparably — predicting over one whole section and over one hand-picked duct
gives two figures that do not belong in the same table. The mm² column is the
safer thing to compare, and it exists for that reason.</p>
"""),

Topic("Stitching predictions into annotations", "stitch merge tiles regions predicted adjacent connected", """
<h2>Stitching predictions into annotations</h2>
<p>After a prediction run, <b>Stitch to Annotations…</b> in the Heatmap panel
turns the tiles of one class into real annotations — so a heatmap becomes
something the geometry pipeline can describe.</p>
<p><b>Adjacent tiles become one annotation. Tiles that do not touch become
separate annotations of the same class</b>, because two lesions are two
lesions.</p>

<table>
<tr><td><b>Class</b></td><td>Which predicted class to stitch, with its tile
    count.</td></tr>
<tr><td><b>Min confidence</b></td><td>Weak tiles are ignored. Raising this
    shrinks regions and can split one into several — a weak tile in the middle
    of a lesion breaks it in two.</td></tr>
<tr><td><b>Adjacency</b></td><td>Edge-sharing only (4) by default. Two tiles
    meeting at a single corner are not one lesion in any sense a pathologist
    would accept, so diagonal joining is opt-in.</td></tr>
<tr><td><b>Min tiles per region</b></td><td>Discards specks. A lone
    high-confidence tile is usually noise.</td></tr>
<tr><td><b>Smooth staircase</b></td><td>Off by default — see below.</td></tr>
</table>
<p><b>Preview</b> reports what you would get before anything is created.</p>

<h3>Why the outline is blocky</h3>
<p>The region follows tile edges exactly, so its boundary is a staircase of
right angles. That is faithful: the area equals the tiles you selected, to the
pixel.</p>
<p class="warn">But the outline-shape descriptor reads a staircase as far more
irregular than the tissue is. Measured on regions stitched from tiles forming
perfect discs, circularity came out at 0.42-0.49 rather than near 1.0 — the
grid, not the biology. <b>Do not mix stitched regions and hand traces in one
training set</b> without checking that first.</p>
<p><b>Smooth staircase</b> removes the steps, but it moves the boundary: on an
L-shaped region a tolerance of 0.9 tiles shaved the inner corner and lost 10%
of the area. It is a trade, not a fix, which is why it is off unless you ask
for it.</p>

<h3>Afterwards</h3>
<p>The new annotations behave like any others: ticked under Use, drawn on the
canvas, saved to the sidecar, and named <code>&lt;class&gt; stitched N</code>.
Describe them in the Geometry panel to get their shape and PanIN descriptors.</p>
"""),

Topic("Geometry: describing annotations", "geometry describe descriptor shape texture nuclei mpp", """
<h2>Geometry: describing annotations</h2>
<p>The Geometry panel learns from the regions themselves rather than from
tiles. It needs no extractor and no GPU.</p>
<p><b>Describe N Checked Annotation(s)</b> computes two descriptor blocks per
region and stores them in the geometry bank:</p>
<ul>
  <li><b>Outline shape</b> — 12 scale-invariant numbers from the traced
      polygon: circularity, solidity, convexity, elongation, extent, radial
      variability and roughness, concavity fraction and depth, boundary
      complexity, lobe count, eccentricity. Computable for <i>any</i> valid
      polygon, and unaffected by staining or resolution. Size is deliberately
      excluded, so "bigger lesion ⇒ higher grade" cannot leak in.</li>
  <li><b>Interior texture</b> — 14 numbers about nuclei and lumina at a fixed
      physical resolution. Needs enough resolvable nuclei, so small regions
      often cannot supply it.</li>
</ul>

<table>
<tr><td><b>Target µm/px</b></td><td>Analysis resolution for the interior block.
    Every region is measured at the pyramid level nearest this.</td></tr>
<tr><td><b>Min nuclei</b></td><td>Default 10. A single duct of ~2,000 µm² holds
    only 10–20, so a higher value silently rejects most small regions. Outline
    shape ignores this.</td></tr>
<tr><td><b>Window px</b></td><td>Size of each interior analysis window.</td></tr>
</table>

<p>The panel reports how many records carry each block, so choosing a source
the bank cannot support is visible before training rather than an error after.
Records that cannot be computed are skipped with a reason, never stored as
zeros.</p>
"""),

Topic("The PanIN architecture descriptor", "panin polarity lumen nuclei grade descriptor 17", """
<h2>The PanIN architecture descriptor</h2>
<p>A 17-D block measuring two things nothing else in the app sees: <b>where the
nuclei sit relative to the lumen</b>, and <b>the shape of the space inside the
duct</b>. Choose it as <i>PanIN architecture</i> under Features in the Geometry
panel.</p>
<p>It came out of a study of 62 annotated regions across 8 slides
(unpublished, not part of this repository) that asked which
measurable features actually separate the grades.</p>

<h3>Polarity</h3>
<p>The strongest single axis found. In PanIN-1a about 4% of nuclei touch the
luminal surface; by PanIN-3 it is 41%, rising steadily through the grades. That
is the computable form of the loss of polarity and pseudostratification a
pathologist reads. Measured as <code>nucleiAtLumen</code>, the mean and
variability of nucleus-to-lumen distance, and the relative thickness of the
epithelial band.</p>

<h3>Inner-lumen shape</h3>
<p>The outline descriptor describes the boundary you traced. This describes the
space <i>inside</i> it — circularity, solidity, elongation, how much survives
erosion, how many arms it breaks into. PanIN-1a has a round open lumen and
PanIN-1b a branched slit while their nuclei are indistinguishable, so lumen
shape is the only available route to that distinction.</p>

<h3>Size is deliberately absent</h3>
<p class="warn">Every feature is a fraction, ratio or normalised shape.
Region area rises steeply with PanIN grade, so any raw count inherits that. In
development, a count of luminal debris looked like one of the strongest
features available until it was normalised per unit area — whereupon it fell to
nothing and reversed direction. It had been measuring how large a region was
traced, not what was in it. That is why no raw count appears in the descriptor,
and a test enforces it.</p>

<h3 class="warn">How well does it actually work?</h3>
<p><b>Train it on your own data and read the leave-one-slide-out score the
trainer reports.</b> That number, on your slides, is the only one worth
quoting — and the reason this section gives you no headline figure to lean on
instead. Development measurements were made on one small internal set and are
not a claim about your material.</p>
<p>Two things about its behaviour are worth knowing before you read a score,
because they change what a given accuracy means:</p>
<ul>
  <li><b>Read it as an ordinal score, not a verdict.</b> On four PanIN classes
      it confuses <i>neighbouring</i> grades far more often than distant ones,
      which is where human agreement is weakest too. Exact agreement
      understates it; the fraction landing within one grade is the fairer
      summary, and both are reported after training.</li>
  <li><b>Always compare against the majority baseline</b> the trainer prints
      beside the score. On an unbalanced set a classifier that names the
      commonest class every time can look respectable, and the gap between the
      two is the part that means anything.</li>
</ul>
<p>It describes more annotations than the interior-texture block does, because
it needs only a lumen and some nuclei rather than enough nuclei for texture
statistics — so on a given set it usually covers more of your regions.</p>
<p><b>Pooling helps a lot.</b> If four grades will not separate on your data,
<b>Machine Learning ▸ Convert Model</b> pools them — low versus high grade is
a markedly easier problem and often the one worth reporting.</p>
"""),

Topic("Geometry: training a shape model", "geometry train shape texture combined validation loso class", """
<h2>Geometry: training a shape model</h2>
<p>Once regions are described, <b>Train</b> fits a classifier on them.</p>

<table>
<tr><td><b>Features</b></td><td><i>Outline shape</i> (12-D) works on every
    region. <i>Interior texture</i> (14-D) needs nuclei. <i>Shape + texture</i>
    (26-D) requires both blocks present. <i>PanIN architecture</i> (17-D)
    measures polarity and inner-lumen shape — see the topic on it.</td></tr>
<tr><td><b>Classes</b></td><td>Tick only the classes to train. This matters:
    the bank accumulates across slides <i>and</i> across annotation schemes, and
    pooling two schemes makes one contradictory problem. An empty selection
    means nothing, not everything.</td></tr>
<tr><td><b>Validation</b></td><td><i>Stratified k-fold</i> splits regions at
    random, so every slide appears on both sides — optimistic.
    <i>Leave-one-slide-out</i> holds out whole slides and is the number that
    says whether this generalises.</td></tr>
<tr><td><b>CV folds, L2</b></td><td>Folds are capped by the smallest class.</td></tr>
</table>

<p>The result line always quotes a majority-class baseline, and says so
explicitly when the score is at or below it.</p>
<p><b>Run both splits and compare them.</b> In development the stratified
score was far higher than the leave-one-slide-out score on the same features
and the same data — the gap between those two numbers <i>is</i> the
measurement of how much the model is leaning on slide identity rather than
biology. Quote the lower one.</p>
<p><b>Save Model…</b> writes a <code>.cl</code> that can be reloaded later.</p>
"""),

Topic("Renaming and pooling model classes", "pool merge rename classes edit model regroup", """
<h2>Renaming and pooling model classes</h2>
<p><b>Edit Classes…</b> — beside Save Model in both the Geometry panel and the
Train Model sheet — renames a trained model&#8217;s classes, or pools several
into one, and saves the result as a separate model.</p>
<p>The use for it: a model trained to tell PanIN-1a from 1b from 2 from 3, used
on a slide where the only question is <i>is this PanIN at all</i>. Pooling the
four answers that with the model you already have.</p>

<h3>How</h3>
<p>One row per trained class. Type the name each should report as; give two rows
the same name and they pool. <b>Merge Selected</b> does it in bulk and suggests
the shared prefix, so selecting the four PanIN grades offers
<code>PanIN</code>. <b>Reset</b> returns to the original names.</p>

<h3>What it does and does not do</h3>
<p><b>Nothing is retrained.</b> The weights are untouched. Pooling a softmax
means summing <i>probabilities</i> — P(1a) + P(1b) — which is the exact
marginal, not an approximation and not the same as adding weight columns.</p>
<p>Scores are recomputed from the confusion matrix, so they change honestly: a
PanIN-2 called PanIN-3 stops being an error once the model only reports
&#8220;PanIN&#8221;, because it was never asked to make that call. In
development this was a large gain rather than a marginal one: a four-class
model that separated grades poorly became a usefully accurate low-grade versus
high-grade model on the same weights, because probability split between
PanIN-2 and PanIN-3 now adds up instead of being counted as two wrong
answers. Your own before-and-after scores are shown when you convert.</p>
<p class="warn">Pooling <i>everything</i> into one class scores 100% and tells
you nothing — there is no call left to get wrong. The sheet says so when you do
it. It is still useful as a detector, just not as an accuracy figure.</p>

<h3>The saved model</h3>
<p>Written as a normal <code>.cl</code> that loads anywhere in the app, and
described as &#8220;regrouped from 4&#8221; so it is never mistaken for one
trained on the pooled classes directly. The original is untouched — this always
saves a new file.</p>
"""),

Topic("Geometry: grading annotations", "grade verdict confidence apply reclassify geometry predict", """
<h2>Geometry: grading annotations</h2>
<p><b>Grade N Checked Annotation(s)</b> applies a geometry model to the regions
on the open slide — one verdict each, not a heatmap. <b>Load Model…</b> beside
it opens a saved geometry <code>.cl</code> without retraining; it refuses a
patch model rather than failing later.</p>
<p>Descriptors are computed exactly as they are for training, so a region
scores the same whether it went into the bank or through here. Nothing is
written back until you choose to write it.</p>

<h3>The results table</h3>
<p>One row per region: its current class, the predicted class, and the
confidence. Sort by confidence to find the marginal calls first; hover a row
for every class probability. Disagreements are highlighted, and
<b>Show only verdicts that differ</b> narrows to them. Regions that could not
be described show the reason instead of a guess.</p>
<p><b>Apply Predicted Classes…</b> reclassifies the disagreeing regions. It is
a separate, confirmed step because it overwrites the class you traced — the
ground truth — and saves the sidecar immediately, with no undo.</p>
<p class="warn">Agreement in this table is not accuracy. These regions were
most likely in the model's own training set, so agreement measures
memorisation. The honest number is the leave-one-slide-out score from
training.</p>
"""),

Topic("Keyboard shortcuts", "shortcuts keys hotkeys keyboard", """
<h2>Keyboard shortcuts</h2>
<table>
<tr><td>Ctrl+O</td><td>Open slide</td></tr>
<tr><td>Ctrl+W</td><td>Close slide</td></tr>
<tr><td>Ctrl+Q</td><td>Quit</td></tr>
<tr><td>V / L / P</td><td>Pan / Lasso / Polygon tool</td></tr>
<tr><td>Ctrl+= , Ctrl+-</td><td>Zoom in, zoom out</td></tr>
<tr><td>Ctrl+0</td><td>Zoom to fit</td></tr>
<tr><td>Ctrl+H</td><td>Show/hide all annotations</td></tr>
<tr><td>Ctrl+E</td><td>Extract patches</td></tr>
<tr><td>Ctrl+T</td><td>Train model</td></tr>
<tr><td>Ctrl+R</td><td>Predict</td></tr>
<tr><td>Enter</td><td>Close the outline being drawn</td></tr>
<tr><td>Backspace</td><td>Undo the last polygon vertex</td></tr>
<tr><td>Esc</td><td>Abandon the outline being drawn</td></tr>
</table>
<p>Mouse: wheel zooms about the pointer; Shift-drag or middle-drag pans with
any tool; double-click closes an outline, or selects a region in Pan mode.</p>
"""),

Topic("Where files live", "paths files storage location appdata bank profile", """
<h2>Where files live</h2>
<table>
<tr><td>Annotations</td><td><code>&lt;slide&gt;.geojson</code>, beside the
    slide</td></tr>
<tr><td>Legacy backup</td><td><code>&lt;slide&gt;.geojson.macos-mirrored.bak</code></td></tr>
<tr><td>Patch bank</td><td><code>%LOCALAPPDATA%\\PathLearn\\bank.db</code></td></tr>
<tr><td>Geometry bank</td><td><code>%LOCALAPPDATA%\\PathLearn\\geometry_bank.json</code></td></tr>
<tr><td>Extractors</td><td><code>%LOCALAPPDATA%\\PathLearn\\Extractors\\</code></td></tr>
<tr><td>Predictions</td><td><code>&lt;slide&gt;.predictions.json.gz</code>,
    beside the slide</td></tr>
<tr><td>Models</td><td>Wherever you save them (<code>.cl</code>)</td></tr>
<tr><td>States</td><td>Wherever you save them
    (<code>.pathlearn</code>, a zip archive)</td></tr>
<tr><td>Projects</td><td>Wherever you save them
    (<code>.pathlearn-project.json</code>)</td></tr>
<tr><td>Extractor descriptors</td><td>Beside each model
    (<code>.pathlearn-extractor.json</code>; the older
    <code>.paninextractor.json</code> is still read)</td></tr>
</table>

<p>Both banks accumulate across slides and sessions until you clear them.</p>

<h3>What is written without being asked</h3>
<ul>
  <li><b>Annotations</b>, on every change.</li>
  <li><b>The heatmap</b>, when a prediction run finishes — turn it off with
      <b>Save heatmap beside the slide</b> in the Heatmap panel.</li>
  <li><b>The open project</b>, whenever you add or remove a row.</li>
</ul>
<p>Everything else — states, models, exports — is written only when you ask
for it. Nothing is ever written inside the slide file itself.</p>
"""),

Topic("Troubleshooting", "problem error broken slow fails empty missing troubleshoot", """
<h2>Troubleshooting</h2>

<h3>The Describe or Grade button says 0, but I have annotations</h3>
<p>Check the <b>Use</b> column. Only ticked regions count. <b>Use All</b> fixes
it in one click.</p>

<h3>Extraction is extremely slow</h3>
<p>Open <b>Installed Extractors…</b> and read the execution provider. If it is
<code>CPUExecutionProvider</code> you are running on CPU, roughly 20x slower.
Install <code>onnxruntime-gpu</code> plus a matching CUDA runtime. A missing
CUDA runtime does not error — it silently falls back.</p>

<h3>"This model needs extractor …, which is not installed"</h3>
<p>The model was trained in a feature space you do not have. Install that
extractor, or accept the substitution the Predict sheet offers if it finds a
comparable one. Training a new model on your own bank also works.</p>

<h3>Predict will not run my geometry model</h3>
<p>By design. A geometry model reads a traced outline; the Predict pipeline
feeds 224-pixel tiles to an extractor and a tile has no outline. Use <b>Grade
Annotations</b> in the Geometry panel.</p>

<h3>My imported annotations are upside-down</h3>
<p>You answered the coordinate-origin question the wrong way. Delete them and
import again with the other answer. For a migrated sidecar, restore the
<code>.macos-mirrored.bak</code> file.</p>

<h3>The score looks great but predictions on a new slide are poor</h3>
<p>Almost certainly the batch effect. Re-score with leave-one-slide-out, and
check the t-SNE window's probe: if slide identity is more recoverable than
class, the model learned the slide. Annotate each class across several
slides.</p>

<h3>Interior texture is unavailable for most regions</h3>
<p>Small regions do not hold enough resolvable nuclei. Lower <b>Min nuclei</b>,
or use <b>Outline shape</b>, which works on any valid polygon.</p>
"""),
]


def as_html(topic: Topic) -> str:
    return _STYLE + topic.html


def plain_text(topic: Topic) -> str:
    """Body text with tags stripped, for searching."""
    out = []
    depth = 0
    for char in topic.html:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif depth == 0:
            out.append(char)
    return "".join(out)


def search(query: str) -> list[Topic]:
    """Topics matching *query*, title matches first. Empty query means all."""
    needle = query.strip().lower()
    if not needle:
        return list(TOPICS)
    titled = [t for t in TOPICS if needle in t.title.lower()]
    rest = [t for t in TOPICS
            if t not in titled
            and (needle in t.keywords.lower() or needle in plain_text(t).lower())]
    return titled + rest
