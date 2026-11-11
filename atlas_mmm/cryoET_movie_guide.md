# Cryo-ET data-collection movie: guideline

These scripts make a movie that shows where a tilt series was collected:

1. **Atlas**: the whole grid. A box marks the square, and the camera zooms in, rotating so the square map ends up upright.
2. **Square map**: the medium-mag map fades in during the zoom, so there's no blurry switch from one image to the other.
3. **Acquisition position**: a further zoom on the target (default 6×). The acquisition box appears only after the zoom stops, and the view holds for a few seconds.
4. **Tilt series**: the 0° image grows out of the box into an inset at half the frame size. Playback runs **0° → −60° → +60° → 0°**, with an angle readout, a scale bar and an animated stage-tilt schematic.

Rotation and scale between the images come from the SerialEM `.mdoc` files. Your manual coordinates set the positions, and image cross-correlation then refines the alignment.

---

## 1. Files

Keep both scripts in the same folder:

| Script | Purpose |
|---|---|
| `cryoet_movie.py` | Makes the movie |
| `locate_tilt_in_montage.py` | Reads mdoc files and fits stage positions. The movie script needs it, and it also works on its own (step 3). |

Inputs for one movie:

| Input | Example | Notes |
|---|---|---|
| Atlas image | `atlasgrid4_bin4_montage.mrc` | Stitched atlas montage, any binning |
| Atlas mdoc | `atlasgrid4.mrc.mdoc` | The montage mdoc from SerialEM |
| Square map image | `MMM_bin1_view8.mrc` | **One** stitched montage section (one square), any binning |
| Square map mdoc | `MMM.mrc.mdoc` | The full MMM mdoc. It can hold many squares; the script uses only yours. |
| Tilt series | `TS_001_bin8.mrc` | Binned stack, any order (see note below) |
| Tilt series mdoc | `TS_001.mrc.mdoc` | Gives the tilt angle of each image |

> **Stiched atlas montage & square map.** Using extractpieces and blendmont from IMOD.

```bash
extractpieces -mdoc atlasgrid4.mrc -output atlasgrid4.pl
blendmont -imin atlasgrid4.mrc -plin atlasgrid4.pl -imout atlasgrid4_bin4_montage.mrc -rootname atlasgrid4_bin4 -sloppy -shift -bin 4

extractpieces -mdoc MMM.mrc -output MMM.pl
blendmont -imin MMM.mrc -plin MMM.pl -imout MMM_bin1_montage.mrc -rootname MMM_bin1 -sloppy -shift -bin 1
```
> **Extract a single view from montage.** Using newstack.

```bash
newstack -input MMM_bin1_montage.mrc -output MMM_bin1_view8.mrc -secs 8
```

> **Tilt stack order.** SerialEM writes the mdoc in acquisition order (0°, +3°, −3°, …), but processed stacks are usually sorted by angle (−60° → +60°). The script compares the brightness of each image with the mdoc. If the stack is sorted and the mdoc isn't, it switches to angle order on its own and says so in the log. An mdoc sorted to match the stack is still the cleanest option.

---

## 2. Setup (once)

```bash
conda create -n cryoet-movie -c conda-forge python=3.11 numpy pandas opencv pillow mrcfile ffmpeg -y
conda activate cryoet-movie
pip install mdocfile
```

Check it:

```bash
python -c "import numpy, pandas, cv2, PIL, mrcfile, mdocfile; print('OK')"
ffmpeg -hide_banner -encoders | grep -E "libx264|h264_videotoolbox"
```

- The conda-forge ffmpeg may lack `libx264`. That's fine: the script picks the best encoder ffmpeg has (on a Mac, `h264_videotoolbox`) and prints which one it used.
- If you'd prefer `libx264`, run `conda install -c conda-forge "ffmpeg=*=gpl*"`, or use Homebrew's ffmpeg (`brew install ffmpeg` then `conda remove ffmpeg`).
- Keep this in its own environment rather than your Anaconda `base`, where NumPy 2 can break older packages.

---

## 3. (Optional) Locate the tilt series on the maps from the mdocs

```bash
python locate_tilt_in_montage.py MMM.mrc.mdoc TS_001.mrc --atlas-mdoc atlasgrid4.mrc.mdoc
```

From stage positions alone, this reports:

- **Which montage section and piece** of the square-map file contains the tilt series. Sections count from 0 in the mdoc; SerialEM's "view N" is section **N − 1**.
- **Where the target sits** in that piece, in the full montage, and in a bin-4 overview (`--overview-bin`).
- **With `--atlas-mdoc`:** where the square and the target sit on the atlas, plus the rotation and scale between the square map and the atlas.

Useful options:

| Option | Use |
|---|---|
| `--section N` | Force a montage section |
| `--atlas-bin 4` | Binning of your atlas image (default 4) |
| `--atlas-ref MMM.mrc.mdoc 17 1521 3205` | Correct the atlas offset using one square whose position you know. It prints a correction you can reuse with `--atlas-shift DX DY`. |
| `--is-sign -1` | Reverse the sign of the image shift (`SpecimenShift`) |
| `--mrc MMM.mrc --png piece.png` | Save the nearest piece with the target marked |
| `--json positions.json` | Save every position and transform |

> Treat these as **starting points**. Positions from stage coordinates alone can be off by several µm:
> - between magnification modes (atlas versus square map);
> - when the grid was reloaded between sessions;
> - when the tilt series used image shift.
>
> For the movie, use coordinates you read off the images (step 4).

---

## 4. Get the two manual coordinates

Open the images in 3dmod and note pixel coordinates in **IMOD convention** (origin at the bottom left, y up), in the image file's own binning:

| Coordinate | Example (bin 4) | Same point at bin 1 |
|---|---|---|
| Square map centre on the atlas, `--square-in-atlas` | `737 2203` | `2948 8812` |
| Tilt-series position on the square map, `--target-in-square` | `1027 732` | `4122 2916` |

To convert between binnings, multiply by the binning ratio. The script works out each file's binning from the MRC header and the mdoc pixel size.

---

## 5. QC first (always)

**All files available to download as example.**
Run with `--qc-only`. It aligns everything and writes the check images without rendering the movie:

```bash
python cryoet_movie.py \
  --atlas atlasgrid4_bin4_montage.mrc  --atlas-mdoc atlasgrid4.mrc.mdoc \
  --square MMM_bin4_view8.mrc         --square-mdoc MMM.mrc.mdoc \
  --tilt TS_001_bin8.mrc         --tilt-mdoc TS_001.mrc.mdoc \
  --square-in-atlas 737 2203 \
  --target-in-square 1027 732 \
  --search-um 0.3 \
  --tilt-flip180 \
  --zoom 12 --hold 3 \
  --out cryoET_movie_zoom12.mp4 \
  --qc-only
```

Check these outputs, named after `--out`:

| File | What to look for |
|---|---|
| `*_qc_atlas_square.jpg` | Checkerboard of the atlas and the square map: fibres, holes and edges should run **continuously** across the tiles |
| `*_qc_tilt_candidates.jpg` | One row per candidate position: **map with the box outlined**, **0° tilt turned to the map's orientation**, and **0° tilt as in 3dmod**. In the correct row, the first two panels show the same structures. |
| `*_qc_square_tilt.jpg` | Close-up of the selected placement |
| `*_alignment.json` | Final refined coordinates (IMOD, file pixels), rotations and scales |

The log also lists the best distinct candidate positions, with their scores and distance from your coordinate.

**If the tilt placement is wrong:**

| What you see | Fix |
|---|---|
| Right place, but the fibre/feature direction is reversed | Add `--tilt-flip180`, or remove it if it's already there |
| Right place, angle a little off | `--tilt-rotation <deg>`, starting from the angle in the log; it then fine-tunes ±2° |
| Box too large or too small | `--tilt-scale 1.05` or similar (the default comes from the mdoc's `AdjustedPixelSpacing`) |
| Box in the wrong place | Correct `--target-in-square` and keep `--search-um 0.3` |
| Another candidate row is right | `--tilt-candidate 2` (or 3) |

> **Why `--search-um 0.3`:** the tilt image covers only about 1.2 µm. If you trust your coordinate, a small search radius stops the match drifting to look-alike fibres nearby. Use larger values (the default is 3 µm) only when you don't know the position well, for example when the tilt series was moved there by image shift.

---

## 6. Render the movie

Use the same command without `--qc-only`:

```bash
python cryoet_movie.py \
  --atlas atlasgrid4_bin4_montage.mrc  --atlas-mdoc atlasgrid4.mrc.mdoc \
  --square MMM_bin4_view8.mrc         --square-mdoc MMM.mrc.mdoc \
  --tilt TS_001_bin8.mrc         --tilt-mdoc TS_001.mrc.mdoc \
  --square-in-atlas 737 2203 \
  --target-in-square 1027 732 \
  --search-um 0.3 \
  --tilt-flip180 \
  --zoom 12 --hold 3 \
  --out cryoET_movie_zoom12.mp4 \
```

The bin-1 version (better files, the deep zoom looks sharper):

```bash
python cryoet_movie.py \
  --atlas atlasgrid4_bin4_montage.mrc  --atlas-mdoc atlasgrid4.mrc.mdoc \
  --square MMM_bin1_view8.mrc         --square-mdoc MMM.mrc.mdoc \
  --tilt TS_001_bin3.mrc         --tilt-mdoc TS_001_bin3.mrc.mdoc \
  --square-in-atlas  2948 8812 \
  --target-in-square 4122 2916 \
  --search-um 0.3 \
  --tilt-flip180 \
  --zoom 12 --hold 3 \
  --out cryoET_movie_zoom12.mp4
```

To look at a few frames before rendering everything, add `--preview`. It saves key frames as `*_preview_*.jpg`.

---

## 7. Options

**Inputs and coordinates**

| Option | Default | Meaning |
|---|---|---|
| `--square-section N` | from `view<N>` in the file name, minus 1 | 0-based montage section of the square map in its mdoc |
| `--atlas-bin`, `--square-bin` | from header ÷ mdoc pixel size | Override the binning if the MRC header lacks a pixel size |
| `--tilt-pixel Å` | from header | Tilt-stack pixel size if the header lacks it |
| `--atlas-max`, `--square-max`, `--tilt-max-px` | 6000 / 8000 / 2048 px | Images larger than this are binned on load to save memory. A bin-1 atlas loads as roughly bin 4, which is all the movie displays. |

**Alignment**

| Option | Default | Meaning |
|---|---|---|
| `--search-um` | 3.0 | Search radius (µm) for the tilt image around `--target-in-square` |
| `--tilt-rotation deg` | full 360° search | Fix the tilt rotation; only fine-tuning ±2° remains |
| `--tilt-flip180` | off | Turn the tilt placement 180° from the automatic choice |
| `--tilt-scale` | from `AdjustedPixelSpacing` | Scale of the tilt footprint relative to nominal pixel sizes |
| `--tilt-candidate N` | 1 | Use the N-th best distinct position |
| `--search-atlas px` | about 16 µm | Search radius on the atlas |
| `--no-refine` | off | Use your coordinates and the mdoc rotation exactly as given |
| `--min-score A T` | 0.45 0.25 | Minimum correlation to accept the atlas and tilt refinements |

**Movie**

| Option | Default | Meaning |
|---|---|---|
| `--zoom` | 6 | Magnification of the final zoom on the acquisition position (10–15 for closer) |
| `--zoom-time s` | scales with `--zoom` | Duration of that zoom |
| `--hold s` | 3 | Seconds on the acquisition box before the tilt series |
| `--tilt-fps` | 10 | Tilt images per second during playback |
| `--tilt-max` | 60 | Play 0 → −max → +max → 0 |
| `--transition` | `sharp` | `sharp` fades the square map in during the zoom; `focus` uses a blur-to-sharp focus pull at the end of the zoom |
| `--tilt-as-file` | off | End the tilt inset in its 3dmod orientation instead of matching the square map on screen |
| `--no-schematic` | off | Leave out the stage-tilt animation next to the tilt series |
| `--font path.ttf` | system font | Font for labels |
| `--encoder` | auto | ffmpeg encoder: `libx264`, `h264_videotoolbox`, `libopenh264`, `mpeg4` |
| `--out` | `cryoET_movie.mp4` | Output file. QC images and JSON are named after it. |

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `_ARRAY_API not found` / "compiled using NumPy 1.x" | Old packages in Anaconda `base`. Use the `cryoet-movie` environment from step 2. |
| `Unrecognized option 'preset'` / `BrokenPipeError` | Your ffmpeg has no `libx264`. Use the current script, which picks another encoder automatically, or install a GPL ffmpeg. |
| `ValidationError ... XedgeDxy ... at most 2 items` | Newer SerialEM writes 3 values there. The current `locate_tilt_in_montage.py` handles it, so update the script. |
| `specimen shift 0.000, 0.000` though the mdoc has `SpecimenShift` | Older `mdocfile` drops unknown fields. The current script reads them directly. |
| Boxes instead of → or − in labels | The font lacks those characters. The script swaps them for plain ones; for nicer text use `--font`, e.g. `/System/Library/Fonts/Supplemental/Arial Unicode.ttf`. |
| "binned x4 on load" for the atlas | Expected (see `--atlas-max`). Coordinates still refer to the original file. |
| Tilt rotation ambiguous by 180° | A straight fibre looks the same both ways. The script breaks ties with the mdoc `RotationAngle`; override with `--tilt-flip180` after checking the QC image. |
| Tilt box clearly in the wrong place | Check the order of the tilt mdoc and stack (log line "tilt stack vs mdoc order"), use `--search-um 0.3`, and correct `--target-in-square`. |
| Atlas QC checkerboard not continuous | Check `--square-in-atlas` and `--square-section` (view N = section N − 1). |

---

## 9. Notes and good practice

- **Use bin 1 for the square map when you can.** The tilt image then covers about 160 × 110 map pixels instead of about 40 × 28, so the match is much more reliable and the 12× zoom stays sharp. The atlas never needs bin 1.
- **Use bin 4 for the tilt series.** The tilt image then covers about 1000 × 1000 map pixels, which is good for a 2K movie.
- **Always run `--qc-only` first.** Alignment takes seconds; rendering takes about a minute.
- **Square maps taken in View mode with a large defocus offset are slightly rotated and magnified relative to nominal.** The script applies `AdjustedPixelSpacing` from the mdoc for the tilt footprint and searches the rotation, so you don't need to correct for this yourself.
- **Every refined position is saved in `*_alignment.json`.** Reuse those values as manual coordinates with `--no-refine` to make the movie exactly reproducible.
- **Orientation:** in the movie, the square map (and by default the tilt inset) is shown turned to match the atlas. Add `--tilt-as-file` if the inset should end in its 3dmod orientation.
