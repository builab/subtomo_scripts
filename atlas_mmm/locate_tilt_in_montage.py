#!/usr/bin/env python3
"""
Locate a SerialEM tilt series on a medium-mag montage map (MMM) and, optionally, on the grid atlas,
using the mdoc stage positions. Reads .mdoc files with the `mdocfile` package (teamtomo).

Required : square-map (MMM) montage mdoc  +  tilt-series mdoc
Optional : atlas montage mdoc (--atlas-mdoc)

  python locate_tilt_in_montage.py MMM4.mrc.mdoc T_202_ts_002.mrc.mdoc
  python locate_tilt_in_montage.py MMM4.mrc.mdoc T_202_ts_002.mrc.mdoc \
         --atlas-mdoc atlas.mrc.mdoc --atlas-bin 4 --json positions.json
  ... --mrc MMM4.mrc --png piece.png        (marked image of the closest square-map piece)

How it works
  * mdocfile.read() gives one row per [ZValue] block (one montage piece = one image in the .mrc) plus
    one row per [MontSection] block. PieceCoordinates = (x, y, z): z = montage section, so grouping by z
    gives each montage in the file (e.g. one per grid square).
  * Per montage, an affine  stage (um) -> montage pixel  is least-squares fitted from its pieces
    (StagePosition vs piece centre); pixel size, rotation and mirroring come out of the fit.
  * Tilt-series target = StagePosition + sign * SpecimenShift of the image closest to 0 deg.
  * Square map: the montage/piece nearest the target; target placed in the piece and in the blended
    (bin-N) overview.
  * Atlas (optional): the square-map centre and the tilt target are mapped into the atlas montage, and
    the relative rotation / scale / mirror between square map and atlas is reported (for registration).

Pixel coordinates: IMOD/3dmod convention (origin bottom-left, y up); y-down is also given where useful.
Requires numpy, pandas, mdocfile  (+ mrcfile, Pillow for --png)
NOTE: Image Shift not read properly so it is off.
"""
import argparse, json, logging, sys
import numpy as np
import mdocfile

logging.getLogger('mdocfile').setLevel(logging.ERROR)      # silence "unknown field" notices


TRUNCATE_2 = ('XedgeDxy', 'YedgeDxy')          # newer SerialEM writes "dx dy cc"; mdocfile accepts only 2 values
RAW_KEYS = ('SpecimenShift',)                   # fields some mdocfile versions drop -> read from the raw text


def read_mdoc(path):
    """mdocfile.read() made robust: sanitise known-incompatible lines, then restore fields mdocfile dropped."""
    import os, re, tempfile
    with open(path, errors='replace') as f:
        lines = f.read().splitlines()
    clean, raw, z = [], {k: {} for k in RAW_KEYS}, None
    for ln in lines:
        m = re.match(r'^\s*\[ZValue\s*=\s*(\d+)\s*\]', ln)
        if m: z = int(m.group(1))
        elif re.match(r'^\s*\[', ln): z = None                 # MontSection / title
        key = ln.split('=', 1)[0].strip() if '=' in ln else ''
        if key in TRUNCATE_2:
            vals = ln.split('=', 1)[1].split()
            ln = f'{key} = {" ".join(vals[:2])}'
        if key in raw and z is not None:
            raw[key][z] = ln.split('=', 1)[1].strip()
        clean.append(ln)
    fd, tmp = tempfile.mkstemp(suffix='.mdoc')
    try:
        with os.fdopen(fd, 'w') as f: f.write('\n'.join(clean) + '\n')
        df = mdocfile.read(tmp)
    finally:
        os.remove(tmp)
    for key, vals in raw.items():
        if vals and (key not in df or df[key].isna().all()):
            df[key] = df['ZValue'].map(lambda v: vals.get(int(v)) if v == v else None)
    return df


def vec(v, n=None):
    """mdocfile returns known fields as tuples and unknown ones as strings -> numpy array."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    a = np.array([float(x) for x in v.split()] if isinstance(v, str) else list(v), float)
    return a[:n] if n else a


# ================================================================ montage handling
class MontageFile:
    def __init__(self, path, name):
        self.path, self.name = path, name
        df = read_mdoc(path)
        if 'PieceCoordinates' not in df:
            sys.exit(f'{name}: {path} has no PieceCoordinates - not a montage mdoc')
        p = df[df['PieceCoordinates'].notna() & df['ZValue'].notna()].copy()
        pc = np.stack(p['PieceCoordinates'].map(lambda v: vec(v, 3)))
        p['px'], p['py'], p['sec'] = pc[:, 0], pc[:, 1], pc[:, 2].astype(int)
        st = np.stack(p['StagePosition'].map(lambda v: vec(v, 2)))
        p['sx'], p['sy'] = st[:, 0], st[:, 1]
        if 'AlignedPieceCoords' in p and p['AlignedPieceCoords'].notna().all():
            al = np.stack(p['AlignedPieceCoords'].map(lambda v: vec(v, 2)))
            p['ax'], p['ay'] = al[:, 0], al[:, 1]
        p['z'] = p['ZValue'].astype(int)
        self.df, self.pieces = df, p
        self.sections = (df[df['MontSection'].notna()].assign(sec=lambda d: d['MontSection'].astype(int))
                         .set_index('sec') if 'MontSection' in df else None)
        self.W, self.H = [int(x) for x in vec(df['ImageSize'].iloc[0], 2)]
        self.pixel_A = float(df['PixelSpacing'].iloc[0])
        self.file = df['ImageFile'].iloc[0] if 'ImageFile' in df else ''

    def nearest(self, xy, section=None):
        p = self.pieces if section is None else self.pieces[self.pieces['sec'] == section]
        d = np.hypot(p['sx'] - xy[0], p['sy'] - xy[1])
        return p.loc[d.idxmin()], float(d.min())

    def full_size(self, sec):
        if self.sections is not None and sec in self.sections.index and 'FullMontSize' in self.sections:
            v = vec(self.sections.loc[sec, 'FullMontSize'], 2)
            if v is not None: return v
        p = self.pieces[self.pieces['sec'] == sec]                 # fallback: nominal extent
        return np.array([p['px'].max() + self.W, p['py'].max() + self.H])

    def fit(self, sec, use_aligned=True):
        """Affine [px,py] = A @ [sx,sy] + t over the pieces of one montage (piece centres)."""
        p = self.pieces[self.pieces['sec'] == sec]
        key = ('ax', 'ay') if use_aligned and 'ax' in p else ('px', 'py')
        S = p[['sx', 'sy']].to_numpy()
        P = p[list(key)].to_numpy() + [self.W / 2, self.H / 2]
        if len(S) < 3 or np.linalg.matrix_rank(S - S.mean(0)) < 2:
            sys.exit(f'{self.name} section {sec}: piece stage positions are degenerate '
                     '(image-shift montage?) - cannot fit stage->pixel')
        X = np.c_[S, np.ones(len(S))]
        coef, *_ = np.linalg.lstsq(X, P, rcond=None)
        A, t = coef[:2].T, coef[2]
        u, sv, vt = np.linalg.svd(A); R = u @ vt
        return dict(sec=sec, A=A, t=t, key='aligned' if key[0] == 'ax' else 'nominal',
                    rms=float(np.sqrt(np.mean(np.sum((X @ coef - P) ** 2, axis=1)))),
                    px_per_um=float(sv.mean()), rot=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                    mirrored=bool(np.linalg.det(A) < 0), n=len(p))

    def place(self, xy, piece, fit):
        """Position of stage point xy inside `piece` (local linear map) and in the nominal montage frame."""
        local = np.array([self.W / 2, self.H / 2]) + fit['A'] @ (np.asarray(xy) - piece[['sx', 'sy']].to_numpy(float))
        return local, piece[['px', 'py']].to_numpy(float) + local

    def describe(self, fit):
        return (f"{fit['px_per_um']:.2f} px/um ({1e4 / fit['px_per_um']:.1f} A/px), rotation {fit['rot']:.1f} deg"
                f"{', MIRRORED' if fit['mirrored'] else ''}, rms {fit['rms']:.1f} px [{fit['key']}, {fit['n']} pieces]")


# ================================================================ tilt series
def tilt_target(path, sign, which='zero'):
    df = read_mdoc(path)
    df = df[df['StagePosition'].notna()]
    row = df.loc[df['TiltAngle'].abs().idxmin()] if which == 'zero' else df[df['ZValue'] == int(which)].iloc[0]
    st = vec(row['StagePosition'], 2)
    ss = vec(row.get('SpecimenShift'), 2) if 'SpecimenShift' in df else None
    ss = np.zeros(2) if ss is None else ss
    return dict(z=int(row['ZValue']), tilt=float(row['TiltAngle']), stage=st, ss=ss, target=st + sign * ss,
                label=row.get('NavigatorLabel', ''), date=row.get('DateTime', ''), file=row.get('ImageFile', ''))


def inside(local, W, H): return 0 <= local[0] < W and 0 <= local[1] < H
def r(v, n=1): return [round(float(x), n) for x in v]


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('square_mdoc', help='medium-mag montage (square map) mdoc  [required]')
    ap.add_argument('tilt_mdoc', help='tilt-series mdoc  [required]')
    ap.add_argument('--atlas-mdoc', help='atlas montage mdoc  [optional]')
    ap.add_argument('--atlas-bin', type=float, default=4, help='binning of your blended atlas image (default 4)')
    ap.add_argument('--atlas-section', type=int, default=None, help='atlas montage section (default: nearest)')
    ap.add_argument('--atlas-ref', nargs=4, metavar=('MDOC', 'SECTION', 'X', 'Y'),
                    help='calibrate the atlas with one square of known position: square-map mdoc (or "same"), '
                         '0-based section, and its centre X Y in your bin-N atlas image (IMOD coords)')
    ap.add_argument('--atlas-shift', nargs=2, type=float, metavar=('DX', 'DY'),
                    help='apply a known stage correction (um) to atlas positions (printed by --atlas-ref)')
    ap.add_argument('--is-sign', type=float, default=1, choices=[1, -1, 0],
                    help='target = stage + sign*SpecimenShift (default +1; 0 ignores image shift)')
    ap.add_argument('--tilt-z', default='zero', help="tilt image to use: 'zero' (closest to 0 deg) or a ZValue")
    ap.add_argument('--section', type=int, help='force the square-map montage section (default: nearest)')
    ap.add_argument('--nominal', action='store_true', help='fit with PieceCoordinates instead of AlignedPieceCoords')
    ap.add_argument('--overview-bin', type=float, default=4, help='binning of your blended square map (default 4)')
    ap.add_argument('--mrc', help='square-map .mrc stack (with --png: save closest piece with target marked)')
    ap.add_argument('--png')
    ap.add_argument('--csv', help='save every square-map piece with its distance to the target')
    ap.add_argument('--json', help='save all positions/transforms as JSON (e.g. for the movie script)')
    a = ap.parse_args()
    out = {}

    # ---------------- tilt series
    ts = tilt_target(a.tilt_mdoc, a.is_sign, a.tilt_z)
    tgt = ts['target']
    print(f"TILT SERIES      : {ts['file']}  (Z={ts['z']}, tilt {ts['tilt']:.2f} deg, nav label {ts['label']}, {ts['date']})")
    print(f"  stage {ts['stage'][0]:.3f}, {ts['stage'][1]:.3f} um  + {a.is_sign:+g} x specimen shift "
          f"{ts['ss'][0]:.3f}, {ts['ss'][1]:.3f} um  ->  target {tgt[0]:.3f}, {tgt[1]:.3f} um")
    out['tilt'] = dict(file=ts['file'], z=ts['z'], tilt=ts['tilt'], stage_um=r(ts['stage'], 3),
                       specimen_shift_um=r(ts['ss'], 3), is_sign=a.is_sign, target_um=r(tgt, 3))

    # ---------------- square map (required)
    sq = MontageFile(a.square_mdoc, 'square map')
    piece, dist = sq.nearest(tgt, a.section)
    sec = int(piece['sec'])
    fs = sq.fit(sec, not a.nominal)
    local, glob = sq.place(tgt, piece, fs)
    full = sq.full_size(sec); b = a.overview_bin
    print(f"\nSQUARE MAP       : {sq.file}  ({sq.pieces['sec'].nunique()} montages, piece {sq.W}x{sq.H} px, {sq.pixel_A} A/px)")
    print(f"  section {sec} (0-based; SerialEM view {sec + 1})  (nav label {piece['NavigatorLabel']}, FullMontSize {full[0]:.0f}x{full[1]:.0f})")
    print(f"  stage->px fit  : {sq.describe(fs)}")
    print(f"  closest piece  : ZValue {piece['z']} (image #{piece['z']} in the .mrc, 0-based), "
          f"PieceCoordinates {piece['px']:.0f} {piece['py']:.0f}, {dist:.2f} um from target")
    print(f"  target in piece    : x={local[0]:.1f}  y={local[1]:.1f}  (IMOD)  | y-down {sq.H - 1 - local[1]:.1f}  "
          f"{'INSIDE' if inside(local, sq.W, sq.H) else 'OUTSIDE piece - check section/sign'}")
    print(f"  target in montage  : x={glob[0]:.1f}  y={glob[1]:.1f}  (IMOD, full-size blended frame)")
    print(f"  target in bin-{b:g} map ({full[0] / b:.0f}x{full[1] / b:.0f}): x={glob[0] / b:.1f}  y={glob[1] / b:.1f}  (IMOD)")
    if a.is_sign != 0 and np.any(ts['ss']):
        la, _ = sq.place(ts['stage'] - a.is_sign * ts['ss'], piece, fs)
        print(f"  (opposite image-shift sign: x={la[0]:.1f} y={la[1]:.1f} in piece, {np.hypot(*(la - local)):.0f} px away - verify once)")
    # square-map centre in stage coords (inverse fit at the montage centre)
    sq_centre_stage = np.linalg.solve(fs['A'], full / 2 - fs['t'])
    out['square'] = dict(file=sq.file, section=sec, nav_label=str(piece['NavigatorLabel']), pixel_A=sq.pixel_A,
                         full_size=r(full, 0), fit=dict(px_per_um=fs['px_per_um'], rot_deg=fs['rot'],
                         mirrored=fs['mirrored'], rms_px=fs['rms'], coords=fs['key']),
                         closest_piece=dict(z=int(piece['z']), piece_xy=r(piece[['px', 'py']], 0), dist_um=round(dist, 3)),
                         target_in_piece=r(local), target_in_montage=r(glob),
                         overview_bin=b, target_in_overview=r(glob / b), centre_stage_um=r(sq_centre_stage, 3))

    # ---------------- atlas (optional)
    if a.atlas_mdoc:
        at = MontageFile(a.atlas_mdoc, 'atlas')
        apiece, adist = at.nearest(sq_centre_stage, a.atlas_section)
        asec = int(apiece['sec'])
        fa = at.fit(asec, not a.nominal)
        afull = at.full_size(asec); ab = a.atlas_bin

        # ---- optional calibration from one square whose atlas position you know
        shift = np.zeros(2)                      # stage-um correction added before mapping into the atlas
        if a.atlas_ref:
            rmdoc, rsec, rx, ry = a.atlas_ref[0], int(a.atlas_ref[1]), float(a.atlas_ref[2]), float(a.atlas_ref[3])
            ref = sq if rmdoc in ('same', a.square_mdoc) else MontageFile(rmdoc, 'reference square map')
            rf = ref.fit(rsec, not a.nominal)
            rc = np.linalg.solve(rf['A'], ref.full_size(rsec) / 2 - rf['t'])
            rp, _ = at.nearest(rc, asec)
            _, rpred = at.place(rc, rp, fa)
            dpx = np.array([rx, ry]) * ab - rpred                    # full-res atlas px
            shift = np.linalg.solve(fa['A'], dpx)                    # -> stage um
            print(f"\nATLAS CALIBRATION: {ref.file} section {rsec} (SerialEM view {rsec + 1}) measured at "
                  f"({rx:.0f}, {ry:.0f}) bin-{ab:g}, predicted ({rpred[0] / ab:.1f}, {rpred[1] / ab:.1f})")
            print(f"  correction: {dpx[0] / ab:+.1f}, {dpx[1] / ab:+.1f} bin-{ab:g} px  =  "
                  f"{shift[0]:+.2f}, {shift[1]:+.2f} um stage  (applied to all atlas positions below)")
            if str(ref.df['DateTime'].iloc[0])[:11] != str(sq.df['DateTime'].iloc[0])[:11]:
                print("  note: reference and square map are from different days - a grid reload between them "
                      "would add an offset this calibration cannot see.")
        elif a.atlas_shift:
            shift = np.array(a.atlas_shift, float)

        _, a_sq = at.place(sq_centre_stage + shift, apiece, fa)
        tpiece, tdist = at.nearest(tgt + shift, asec)
        _, a_tg = at.place(tgt + shift, tpiece, fa)
        # relative transform square-map px -> atlas px (both via stage)
        M = fa['A'] @ np.linalg.inv(fs['A'])
        u, sv, vt = np.linalg.svd(M); R = u @ vt
        rel_rot = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        rel_mirror = bool(np.linalg.det(M) < 0)
        print(f"\nATLAS            : {at.file}  ({at.pieces['sec'].nunique()} montage(s), piece {at.W}x{at.H} px, {at.pixel_A} A/px)")
        print(f"  section {asec}, FullMontSize {afull[0]:.0f}x{afull[1]:.0f}")
        print(f"  stage->px fit  : {at.describe(fa)}")
        print(f"  square-map centre in atlas : x={a_sq[0]:.1f}  y={a_sq[1]:.1f}  (full)  |  bin-{ab:g}: "
              f"x={a_sq[0] / ab:.1f}  y={a_sq[1] / ab:.1f}  (IMOD)   nearest atlas piece Z={apiece['z']} ({adist:.1f} um)")
        print(f"  tilt target in atlas       : x={a_tg[0]:.1f}  y={a_tg[1]:.1f}  (full)  |  bin-{ab:g}: "
              f"x={a_tg[0] / ab:.1f}  y={a_tg[1] / ab:.1f}  (IMOD)")
        print(f"  square map -> atlas        : scale 1/{1 / sv.mean():.2f} (full-res px), rotation {rel_rot:.1f} deg"
              f"{', MIRRORED' if rel_mirror else ''}   [bin-{b:g} map -> bin-{ab:g} atlas: 1/{b / ab / sv.mean():.2f}]")
        if abs(dist) > 0 and str(ts['date'])[:11] != str(at.df['DateTime'].iloc[0])[:11]:
            print("  note: atlas and tilt series were recorded on different days - stage coordinates may shift "
                  "if the grid was reloaded.")
        out['atlas'] = dict(file=at.file, section=asec, pixel_A=at.pixel_A, full_size=r(afull, 0), bin=ab,
                            fit=dict(px_per_um=fa['px_per_um'], rot_deg=fa['rot'], mirrored=fa['mirrored'],
                                     rms_px=fa['rms'], coords=fa['key']),
                            stage_correction_um=r(shift, 3), square_centre=r(a_sq), square_centre_binned=r(a_sq / ab),
                            target=r(a_tg), target_binned=r(a_tg / ab),
                            square_to_atlas=dict(scale=float(sv.mean()), rot_deg=rel_rot, mirrored=rel_mirror,
                                                 scale_binned=float(sv.mean() * b / ab)))

    # ---------------- outputs
    if a.csv:
        p = sq.pieces.assign(dist_um=np.hypot(sq.pieces['sx'] - tgt[0], sq.pieces['sy'] - tgt[1]))
        p.sort_values('dist_um')[['z', 'sec', 'NavigatorLabel', 'px', 'py', 'sx', 'sy', 'dist_um']].to_csv(a.csv, index=False)
        print(f'\nSaved {a.csv}')
    if a.json:
        with open(a.json, 'w') as f: json.dump(out, f, indent=2, default=str)
        print(f'Saved {a.json}')
    if a.mrc and a.png:
        import mrcfile
        from PIL import Image, ImageDraw
        with mrcfile.mmap(a.mrc, permissive=True) as m:
            img = np.flipud(np.array(m.data[int(piece['z'])], np.float32))       # IMOD y-up -> display
        lo, hi = np.percentile(img, [1, 99])
        im = Image.fromarray((np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)).convert('RGB')
        d = ImageDraw.Draw(im); x, y = local[0], sq.H - 1 - local[1]; rr = max(sq.W, sq.H) // 60; c = (255, 60, 40)
        d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=c, width=max(2, rr // 5))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            d.line([x + 1.5 * rr * dx, y + 1.5 * rr * dy, x + 3 * rr * dx, y + 3 * rr * dy], fill=c, width=3)
        im.save(a.png); print(f'Saved {a.png}')


if __name__ == '__main__':
    main()
