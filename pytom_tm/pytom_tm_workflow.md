# 📌 PyTom Match Pick — Template Matching Workflow

**Repository:** [PyTom Match Pick GitHub](https://github.com/SBC-Utrecht/pytom-match-pick/)\
**Tutorial:** [Official Tutorial](https://sbc-utrecht.github.io/pytom-match-pick/tutorials/Tutorial/)

---

## 1️⃣ Set up on **Paris**

First, activate the `pytom_tm` environment:

```bash
source ~/activate_anaconda.sh
conda activate pytom_tm
```

---

## 2️⃣ Prepare working directory

Navigate to your project folder:

```bash
cd /storage/builab/Thibault/20250213_TetraCCDC147C_TS
```

Create a new working directory for PyTom:

```bash
mkdir pytom_tm
cd pytom_tm
```

---

## 3️⃣ Link tomograms

Link the reconstructed tomograms:

```bash
ln -s /storage/builab/Thibault/20240905_SPEF1_MT_TS/warp_tiltseries/reconstruction .
```

Your tomogram files should look like this (downsampled for speed):

```
CCDC147C_001_14.00Apx.mrc
...
CCDC147C_050_14.00Apx.mrc
```

---

## 4️⃣ Prepare templates and masks

You need:

- A **template** MRC file
- A **mask** MRC file to cover the template

Example files:

```
doublet_8nm_14.00Apx.mrc
dmt_mask.mrc
```

---

## 5️⃣ Extract CTF information

PyTom needs **defocus, tilt angle, and dose** information.\
This is stored in `warp_tiltseries/*.xml`.

Create a folder and link the XML files:

```bash
mkdir xml
cd xml
ln -s ../../warp_tiltseries/*.xml .
```

Extract the information:

```bash
python extract_info_from_warpxml.py xml/*.xml
```

---

## 6️⃣ Limit the tilt angles

Return to the main directory:

```bash
cd ..
```

Generate an angle list with a ±15° limit for filaments, step size 4°:

```bash
python generate_pytom_angle_list.py --a 4 --tilt_limit 15 --o angle_list_filament4.txt
```

---

## 7️⃣ Create the results directory

```bash
mkdir results
```

---

## 8️⃣ Run template matching (single tomogram)

Example command for one tomogram:

```bash
pytom_match_template.py \
  -t templates/doublet_8nm_14.00Apx.mrc \
  -m templates/dmt_mask.mrc \
  -v reconstruction/CCDC147C_001_14.00Apx.mrc \
  -d results/ \
  -a xml/CCDC147C_001.tlt \
  --low-pass 40 \
  --defocus xml/CCDC147C_001_defocus.txt \
  --amplitude 0.07 \
  --spherical 2.7 \
  --voltage 300 \
  --tomogram-ctf-model phase-flip \
  -g 0 \
  --volume-split 2 2 1 \
  --random-phase-correction \
  --dose-accumulation xml/CCDC147C_001_dose.txt \
  --angular-search angle_list_filament4.txt \
  --per-tilt-weighting
```

---

## 9️⃣ Run template matching (batch)

For multiple tomograms, run your batch script:

```bash
bash run_pytom_tm_batch.sh
```

---

## 🔟 Extract candidates (single tomogram)

Example extraction for one tomogram:

```bash
pytom_extract_candidates.py \
  -j results/CCDC147C_001_14.00Apx_job.json \
  -n 1500 \
  --particle-diameter 80
```

---

## ✅ Extract candidates (batch)

Run the batch extraction:

```bash
bash run_pytom_extract_batch.sh
```

---

## 🎉 **Done!**

You now have candidate coordinates ready for subtomogram averaging and downstream analysis.

