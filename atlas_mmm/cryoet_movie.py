#!/usr/bin/env python3
"""
Cryo-ET data-collection movie:  atlas  ->  square map  ->  acquisition position  ->  tilt series.

Positions are given manually; rotation and scale between the images come from the SerialEM mdoc files
(stage-position fits of the montage pieces) and are then refined by image cross-correlation.

  python cryoet_movie.py \
      --atlas atlasgrid4_bin4_montage.mrc   --atlas-mdoc atlasgrid4.mrc.mdoc \
      --square MMM4_bin4_view8.mrc          --square-mdoc MMM4.mrc.mdoc  --square-section 7 \
      --tilt T_222_ts_002_bin3.mrc          --tilt-mdoc T_222_ts_002.mrc.mdoc \
      --square-in-atlas 737 2203  --target-in-square 1027 732 \
      --out cryoET_movie.mp4

Coordinates are pixel positions in the binned images as read in IMOD/3dmod (origin bottom-left, y up).
  --square-in-atlas   centre of the square map on the atlas image
  --target-in-square  tilt-series position on the square-map image
  --square-section    0-based montage section in the square-map mdoc (SerialEM "view 8" = section 7);
                      if omitted it is taken from "view<N>" in the file name (N-1).

Useful options
  --qc-only        stop after alignment and write the QC images (check them before a full render)
  --preview        render a few key frames as JPG instead of the movie
  --no-refine      use manual coordinates + mdoc rotation exactly as given
  --zoom 6         magnification of the final zoom on the acquisition position
  --tilt-fps 10    tilt images per second during playback

Needs: numpy, opencv-python, mrcfile, Pillow, pandas, mdocfile, ffmpeg on PATH,
       and locate_tilt_in_montage.py in the same folder (mdoc reader / stage fits).
"""
import argparse, os, re, shutil, subprocess, sys, tempfile, json
import numpy as np, cv2, mrcfile
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locate_tilt_in_montage import MontageFile, read_mdoc, vec   # noqa: E402

W, H, FPS = 1920, 1080, 30
F2 = np.diag([1.0, -1.0])                     # IMOD (y up) <-> display (y down)


# ============================================================ small helpers
FONT_PATH = None                               # set from --font
def font(size):
    cands = ([FONT_PATH] if FONT_PATH else []) + [
        '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf',
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf', '/Library/Fonts/Arial Unicode.ttf',
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf', '/System/Library/Fonts/SFNS.ttf',
        '/System/Library/Fonts/Helvetica.ttc', '/Library/Fonts/Arial.ttf', 'C:/Windows/Fonts/arialbd.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 'DejaVuSans-Bold.ttf', 'Arial.ttf']
    for p in cands:
        try: return ImageFont.truetype(p, size)
        except OSError: pass
    try: return ImageFont.load_default(size)
    except TypeError: return ImageFont.load_default()

def has_glyph(f, ch):
    """True if font f really draws ch (not the 'missing glyph' box)."""
    try:
        a, b = f.getmask(ch), f.getmask('\U0010FFFD')
        return a.size != b.size or bytes(a) != bytes(b)
    except Exception:
        return False

def safe_text(f, s):
    """Replace characters the font cannot draw with plain equivalents."""
    for ch, alt in (('→', '>'), ('−', '-'), ('µ', 'u'), ('°', ' deg')):
        if ch in s and not has_glyph(f, ch): s = s.replace(ch, alt)
    return s

def ease(t): t = float(np.clip(t, 0, 1)); return t * t * (3 - 2 * t)
def rotm(deg): a = np.radians(deg); return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
def aff(L, t): M = np.eye(3); M[:2, :2] = L; M[:2, 2] = t; return M
def ang_of(L): return float(np.degrees(np.arctan2(L[1, 0], L[0, 0])))
def scale_of(L): return float(np.sqrt(abs(np.linalg.det(L))))
def apply(M, p): return (M @ np.array([p[0], p[1], 1.0]))[:2]

def norm8(a, lo=0.5, hi=99.5):
    s = a[::max(1, a.shape[0] // 512), ::max(1, a.shape[1] // 512)]
    p0, p1 = np.percentile(s, [lo, hi])
    return np.clip((a - p0) / max(p1 - p0, 1e-6) * 255, 0, 255).astype(np.float32)

def load_mrc(path, z=0, max_dim=None):
    """Load one image (display orientation, y down). If larger than max_dim it is binned by an integer
    factor f (pixel size scaled accordingly) to keep memory and run time reasonable for bin-1 data."""
    with mrcfile.mmap(path, permissive=True) as m:
        d = m.data
        img = np.array(d[z] if d.ndim == 3 else d, np.float32)
        vs = float(m.voxel_size.x)
        nz = d.shape[0] if d.ndim == 3 else 1
    f = 1
    if max_dim and max(img.shape) > max_dim:
        f = int(np.ceil(max(img.shape) / max_dim))
        img = cv2.resize(img, (img.shape[1] // f, img.shape[0] // f), interpolation=cv2.INTER_AREA)
        vs = vs * f if vs > 1 else vs
    return np.flipud(img).copy(), vs, nz, f

def binning(vs, mdoc_px, override, name, f=1):
    if override: return float(override) * f
    if vs > 1.0 and mdoc_px:
        b = vs / mdoc_px
        print(f'  {name}: header {vs:.2f} A/px / mdoc {mdoc_px:.2f} A/px -> bin {b:.2f}')
        return b
    sys.exit(f'{name}: cannot determine binning from the MRC header - pass --{name}-bin')


# ============================================================ registration refinement
def refine(big, small, L0, p0, R, dthetas, dscales, inner=0.85, label=''):
    """Find small-image pose in big image. L0: 2x2 small px -> big px (display), p0: big coords of the
    small-image centre. Searches rotation/scale offsets and +/-R px translation with normalised
    cross-correlation. Returns dict(L, p, score, second, ...)."""
    hs, ws = small.shape
    cs = np.array([ws / 2, hs / 2])
    s0 = scale_of(L0)
    f = min(1.0, 1.5 * s0)                                    # pre-shrink small image to ~big scale
    sm = cv2.resize(small, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else small
    csm = cs * f
    # template canvas + inscribed box (from central pose, shrunk for safety)
    corners = np.array([[0, 0], [ws, 0], [ws, hs], [0, hs]]) - cs
    ext = np.abs(corners @ L0.T).max(0)
    cw, ch = int(2 * ext[0] + 8), int(2 * ext[1] + 8)
    cc = np.array([cw / 2, ch / 2])
    M = np.c_[L0 / f, cc - (L0 / f) @ csm].astype(np.float32)
    mask = cv2.warpAffine(np.ones_like(sm), M, (cw, ch), flags=cv2.INTER_NEAREST)
    k = 1.0
    while k > 0.05:
        bx, by = int(ext[0] * k), int(ext[1] * k)
        if mask[int(cc[1]) - by:int(cc[1]) + by, int(cc[0]) - bx:int(cc[0]) + bx].min() > 0: break
        k -= 0.02
    bx, by = max(4, int(bx * inner)), max(4, int(by * inner))
    x0, y0 = int(p0[0]) - bx - R, int(p0[1]) - by - R
    win = big[max(0, y0):y0 + 2 * (by + R), max(0, x0):x0 + 2 * (bx + R)]
    ox, oy = max(0, x0), max(0, y0)
    best, second = None, -1
    for dt in dthetas:
        for ds in dscales:
            L = L0 @ rotm(dt) * ds
            Mt = np.c_[L / f, cc - (L / f) @ csm].astype(np.float32)
            tpl = cv2.warpAffine(sm, Mt, (cw, ch), flags=cv2.INTER_LINEAR)
            tpl = tpl[int(cc[1]) - by:int(cc[1]) + by, int(cc[0]) - bx:int(cc[0]) + bx]
            if tpl.std() < 1e-3 or win.shape[0] <= tpl.shape[0] or win.shape[1] <= tpl.shape[1]: continue
            r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if best is None or mx > best['score']:
                rr = r.copy(); cv2.circle(rr, loc, max(3, min(bx, by) // 2), -1, -1)
                second = float(rr.max())
                sub = [0.0, 0.0]                                   # sub-pixel peak (parabola)
                for ax_, (i, j) in enumerate([(loc[1], loc[0]), (loc[1], loc[0])]):
                    if ax_ == 0 and 0 < j < r.shape[1] - 1: l_, c_, r_ = r[i, j - 1], r[i, j], r[i, j + 1]
                    elif ax_ == 1 and 0 < i < r.shape[0] - 1: l_, c_, r_ = r[i - 1, j], r[i, j], r[i + 1, j]
                    else: continue
                    den = l_ - 2 * c_ + r_
                    if den < 0: sub[ax_] = float(np.clip(0.5 * (l_ - r_) / den, -0.5, 0.5))
                p = np.array([ox + loc[0] + sub[0] + bx, oy + loc[1] + sub[1] + by], float)
                best = dict(L=L, p=p, score=float(mx), dtheta=dt, dscale=ds)
    best['second'] = second
    best['shift'] = best['p'] - np.asarray(p0, float)
    sec_ = f' (2nd peak {second:.3f})' if second > -1 else ''
    print(f'  {label}: score {best["score"]:.3f}{sec_}, shift {best["shift"][0]:+.1f},'
          f'{best["shift"][1]:+.1f} px, rotation {best["dtheta"]:+.1f} deg, scale x{best["dscale"]:.3f}')
    return best


def checkerboard(a, b, n=8):
    h, w = a.shape; yy, xx = np.mgrid[0:h, 0:w]
    return np.where(((yy * n // h + xx * n // w) % 2) == 0, a, b)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--atlas', required=True); ap.add_argument('--atlas-mdoc', required=True)
    ap.add_argument('--square', required=True); ap.add_argument('--square-mdoc', required=True)
    ap.add_argument('--square-section', type=int)
    ap.add_argument('--tilt', required=True); ap.add_argument('--tilt-mdoc', required=True)
    ap.add_argument('--square-in-atlas', nargs=2, type=float, required=True, metavar=('X', 'Y'))
    ap.add_argument('--target-in-square', nargs=2, type=float, required=True, metavar=('X', 'Y'))
    ap.add_argument('--atlas-bin', type=float); ap.add_argument('--square-bin', type=float)
    ap.add_argument('--tilt-pixel', type=float, help='tilt-stack pixel size (A) if the header lacks it')
    ap.add_argument('--zoom', type=float, default=6,
                    help='magnification of the final zoom into the square map (default 6; e.g. 10 or 15 for closer)')
    ap.add_argument('--zoom-time', type=float, default=None, help='duration of that zoom in s (default scales with --zoom)')
    ap.add_argument('--hold', type=float, default=3.0,
                    help='seconds to hold on the acquisition box before the tilt series starts (default 3)')
    ap.add_argument('--tilt-fps', type=float, default=10)
    ap.add_argument('--tilt-max', type=float, default=60, help='play 0 -> -max -> +max -> 0 (default 60)')
    ap.add_argument('--no-refine', action='store_true')
    ap.add_argument('--encoder', default='auto',
                    help='ffmpeg video encoder: auto (default: libx264, else h264_videotoolbox on Mac, else ...), '
                         'or e.g. libx264, h264_videotoolbox, libopenh264, mpeg4')
    ap.add_argument('--transition', choices=['sharp', 'focus'], default='sharp',
                    help='atlas -> square map: "sharp" fades the square map in during the zoom with contrast matching only '
                         '(default); "focus" swaps at the end of the zoom with a blur-to-sharp focus pull')
    ap.add_argument('--tilt-as-file', action='store_true',
                    help='end the tilt inset in its 3dmod orientation (default: keep it consistent with the square map on screen)')
    ap.add_argument('--tilt-rotation', type=float, help='rotation (deg, display) of the tilt image on the square map; '
                    'skips the 360-deg search and only fine-tunes around it')
    ap.add_argument('--search-atlas', type=int, help='atlas search radius in loaded atlas px (default ~16 um)')
    ap.add_argument('--search-square', type=int, help='target search radius in loaded square px (default ~0.9 um)')
    ap.add_argument('--search-um', type=float, default=3.0,
                    help='search radius (um) for the tilt image around --target-in-square (default 3; the tilt '
                         'series may be offset from the Navigator point by image shift)')
    ap.add_argument('--tilt-scale', type=float,
                    help='tilt footprint scale factor vs nominal pixel sizes (default: from AdjustedPixelSpacing)')
    ap.add_argument('--tilt-candidate', type=int, default=1,
                    help='use the N-th best distinct position for the tilt image (see the log and QC image)')
    ap.add_argument('--tilt-flip180', action='store_true',
                    help='place the tilt image rotated 180 deg from the automatic choice (or from --tilt-rotation)')
    ap.add_argument('--atlas-max', type=int, default=6000, help='bin the atlas on load if larger (default 6000 px)')
    ap.add_argument('--square-max', type=int, default=8000, help='bin the square map on load if larger (default 8000 px)')
    ap.add_argument('--tilt-max-px', type=int, default=2048, help='bin tilt images on load if larger (default 2048 px)')
    ap.add_argument('--min-score', type=float, nargs=2, default=[0.45, 0.25], metavar=('ATLAS', 'TILT'),
                    help='minimum correlation to accept each refinement (default 0.45 0.25)')
    ap.add_argument('--no-schematic', dest='schematic', action='store_false',
                    help='do not draw the stage-tilt schematic next to the tilt series')
    ap.add_argument('--font', help='path to a .ttf/.otf font for the labels (default: first suitable system font)')
    ap.add_argument('--qc-only', action='store_true'); ap.add_argument('--preview', action='store_true')
    ap.add_argument('--out', default='cryoET_movie.mp4')
    ap.add_argument('--qc-prefix', default=None, help='prefix for QC images (default: next to --out)')
    a = ap.parse_args()
    if shutil.which('ffmpeg') is None and not (a.qc_only or a.preview):
        sys.exit('ffmpeg not found on PATH')
    qc = a.qc_prefix or os.path.splitext(a.out)[0]
    global FONT_PATH; FONT_PATH = a.font

    # ---------------- images
    print('Loading images')
    A, vsA, _, fA = load_mrc(a.atlas, max_dim=a.atlas_max)
    S, vsS, _, fS = load_mrc(a.square, max_dim=a.square_max)
    HA, WA = A.shape; HS, WS = S.shape
    for nm_, f_, sh_ in (('atlas', fA, A.shape), ('square map', fS, S.shape)):
        if f_ > 1: print(f'  {nm_}: binned x{f_} on load to {sh_[1]}x{sh_[0]} (coordinates still refer to the file)')
    if a.square_section is None:
        m = re.search(r'view_?0*(\d+)', os.path.basename(a.square), re.I)
        if not m: sys.exit('give --square-section (0-based montage section of the square map)')
        a.square_section = int(m.group(1)) - 1
        print(f'  square-map section {a.square_section} (from "view{m.group(1)}" in the file name)')

    # ---------------- mdoc calibration (rotation / scale)
    print('Reading mdoc files')
    at = MontageFile(a.atlas_mdoc, 'atlas'); asec = int(at.pieces['sec'].value_counts().idxmax())
    sq = MontageFile(a.square_mdoc, 'square map')
    fa, fs = at.fit(asec), sq.fit(a.square_section)
    binA = binning(vsA, at.pixel_A, a.atlas_bin, 'atlas', fA)
    binS = binning(vsS, sq.pixel_A, a.square_bin, 'square', fS)
    pxA, pxS = vsA if vsA > 1 else at.pixel_A * binA, vsS if vsS > 1 else sq.pixel_A * binS   # A/px displayed
    L_sa_imod = fa['A'] @ np.linalg.inv(fs['A']) * (binS / binA)        # square bin px -> atlas bin px (IMOD)
    L_sa0 = F2 @ L_sa_imod @ F2                                          # display convention
    if np.linalg.det(L_sa0) < 0: print('  WARNING: square map is mirrored relative to the atlas')
    print(f'  square -> atlas (mdoc): scale 1/{1 / scale_of(L_sa0):.2f}, rotation {ang_of(L_sa0):.1f} deg (display)')

    tdf = read_mdoc(a.tilt_mdoc)
    tdf = tdf[tdf['TiltAngle'].notna() & tdf['ZValue'].notna()]
    angles = dict(zip(tdf['ZValue'].astype(int), tdf['TiltAngle'].astype(float)))
    with mrcfile.mmap(a.tilt, permissive=True) as m:
        nzT = m.data.shape[0]; vsT = float(m.voxel_size.x)
    if len(angles) != nzT:
        print(f'  WARNING: tilt mdoc has {len(angles)} images, stack has {nzT} - using the first {min(len(angles), nzT)}')
    # does the mdoc order match the stack? compare per-image means with mdoc MinMaxMean
    if 'MinMaxMean' in tdf and len(angles) == nzT:
        with mrcfile.mmap(a.tilt, permissive=True) as m:
            smean = np.array([float(m.data[i][::4, ::4].mean()) for i in range(nzT)])
        mm = np.array([vec(v)[2] for v in tdf['MinMaxMean']])
        zs = tdf['ZValue'].astype(int).to_numpy(); ang = tdf['TiltAngle'].astype(float).to_numpy()
        srt = np.argsort(ang)
        c_given = np.corrcoef(smean, mm[np.argsort(zs)])[0, 1]
        c_sorted = np.corrcoef(smean, mm[srt])[0, 1]
        print(f'  tilt stack vs mdoc order: correlation {c_given:.3f} as listed, {c_sorted:.3f} if sorted by angle')
        if c_sorted > c_given + 0.1:
            print('  -> stack is sorted by tilt angle but the mdoc is in acquisition order: using sorted angles')
            angles = {i: float(ang[srt][i]) for i in range(nzT)}
        elif c_given < 0.8:
            print('  WARNING: stack and mdoc image order do not seem to match - check the tilt angles')
    z0 = min((z for z in angles if z < nzT), key=lambda z: abs(angles[z]))
    T0, _, _, fT = load_mrc(a.tilt, z0, max_dim=a.tilt_max_px)
    HT, WT = T0.shape
    pxT = vsT if vsT > 1 else (a.tilt_pixel or sys.exit('tilt pixel size unknown - pass --tilt-pixel'))
    pxT *= fT
    if fT > 1: print(f'  tilt: binned x{fT} on load to {WT}x{HT}')
    rotT = float(tdf['RotationAngle'].iloc[0]) if 'RotationAngle' in tdf else 0.0
    rotS = float(sq.df['RotationAngle'].dropna().iloc[0]) if 'RotationAngle' in sq.df else 0.0
    rel = rotT - rotS
    print(f'  tilt: {nzT} images, 0-deg image Z={z0} ({angles[z0]:+.1f}), {pxT:.2f} A/px; '
          f'square {pxS:.1f} A/px; mdoc rotation difference {rel:+.1f} deg')

    # ---------------- initial transforms (display coords)
    # user coordinates are file pixels (IMOD, y up) -> loaded display pixels
    sc_a = np.array([a.square_in_atlas[0], HA * fA - 1 - a.square_in_atlas[1]]) / fA
    tg_s = np.array([a.target_in_square[0], HS * fS - 1 - a.target_in_square[1]]) / fS
    R_atlas = a.search_atlas if a.search_atlas else max(10, round(60 * 2736 / pxA))
    R_square = a.search_square if a.search_square else max(10, round(a.search_um * 1e4 / pxS))
    if a.tilt_flip180 and a.tilt_rotation is not None: a.tilt_rotation += 180
    cS, cT = np.array([WS / 2, HS / 2]), np.array([WT / 2, HT / 2])
    L_sa, p_sa = L_sa0, sc_a
    # square maps taken in View mode with a large defocus offset are magnified differently from nominal;
    # SerialEM records the corrected value as AdjustedPixelSpacing in the MontSection block
    adj = 1.0
    try:
        sec_info = sq.sections.loc[a.square_section]
        if 'AdjustedPixelSpacing' in sec_info and float(sec_info['AdjustedPixelSpacing']) > 0:
            adj = float(sec_info['AdjustedPixelSpacing']) / sq.pixel_A
    except Exception:
        pass
    if a.tilt_scale: adj = 1.0 / a.tilt_scale
    pxS_true = pxS * adj
    if abs(adj - 1) > 0.005:
        print(f'  square map true pixel size {pxS_true:.1f} A (AdjustedPixelSpacing; nominal {pxS:.1f} A) '
              f'-> tilt footprint x{1 / adj:.3f}')
    L_ts0 = np.eye(2) * (pxT / pxS_true)
    L_ts, p_ts = L_ts0 @ rotm(a.tilt_rotation if a.tilt_rotation is not None else rel + (180 if a.tilt_flip180 else 0)), tg_s

    An, Sn, Tn = norm8(A), norm8(S), norm8(cv2.GaussianBlur(T0, (0, 0), 2))
    tilt_groups = []
    if not a.no_refine:
        print('Refining alignment')
        r1 = refine(An, Sn, L_sa0, sc_a, R_atlas, np.arange(-3, 3.01, 0.5), np.arange(0.96, 1.041, 0.01),
                    inner=0.85, label='atlas  <- square map (coarse)')
        r1 = refine(An, Sn, r1['L'], r1['p'], 4, np.arange(-0.5, 0.51, 0.1), np.arange(0.99, 1.0101, 0.0025),
                    inner=0.85, label='atlas  <- square map (fine)  ')
        if r1['score'] >= a.min_score[0]: L_sa, p_sa = r1['L'], r1['p']
        else: print('    refinement rejected (low score) - using manual position + mdoc rotation')
        # The tilt image covers only ~40 square-map px, so: high-pass both images (fibres/edges drive the
        # match, not brightness gradients), search the full 360 deg unless --tilt-rotation is given, and
        # break near-ties with the plain-image correlation (which sees the thick/thin-ice gradient).
        import io, contextlib
        sT = pxT / pxS_true
        hp = lambda img, sg: img - cv2.GaussianBlur(img, (0, 0), sg)
        sg_hp = 1865.0 / pxS                                   # ~190 nm high-pass, in square px
        Sh = hp(norm8(cv2.GaussianBlur(S, (0, 0), 0.7)), sg_hp)
        Th = hp(norm8(cv2.GaussianBlur(T0, (0, 0), 0.7 / sT)), sg_hp / sT)
        bases = [a.tilt_rotation] if a.tilt_rotation is not None else list(range(0, 360, 10))
        cands = []
        for base in bases:
            with contextlib.redirect_stdout(io.StringIO()):
                r = refine(Sh, Th, L_ts0 @ rotm(base), tg_s, R_square, np.arange(-5, 5.01, 1.0),
                           np.arange(0.95, 1.051, 0.025), inner=0.9)
                rp = refine(Sn, Tn, r['L'], r['p'], 1, [0], [1.0], inner=0.9)        # plain score, same pose
            cands.append((r['score'] + 0.5 * rp['score'], base + r['dtheta'], r, rp['score']))
        cands.sort(key=lambda c: -c[0])
        angd = lambda x, y: abs(((x - y + 180) % 360) - 180)
        # group by location: each group = one possible acquisition position (with its rotation choices)
        rad = 3000.0 / pxS                                         # 0.3 um
        groups = []
        for c_ in cands:
            g = next((g for g in groups if np.hypot(*(g[0][2]['p'] - c_[2]['p'])) < rad), None)
            (g.append(c_) if g is not None else groups.append([c_]))
        print(f'  square <- tilt: searched +/-{R_square * pxS / 1e4:.1f} um around the given target; '
              f'best distinct positions:')
        for k, g in enumerate(groups[:3], 1):
            pg = g[0][2]['p']; d_um = np.hypot(*(pg - tg_s)) * pxS / 1e4
            print(f'    #{k}: score {g[0][0]:.3f}, rotation {g[0][1]:6.1f} deg, {d_um:.2f} um from the given target, '
                  f'at {pg[0] * fS:.1f}, {HS * fS - 1 - pg[1] * fS:.1f} (IMOD, file px)')
        if len(groups) > 1 and groups[0][0][0] - groups[1][0][0] < 0.02:
            print('    WARNING: positions #1 and #2 score almost the same - check *_qc_tilt_candidates.jpg and pick one '
                  'with --tilt-candidate N (bin-1 square maps separate them much better)')
        k_sel = min(max(1, a.tilt_candidate), len(groups))
        cands = groups[k_sel - 1]
        if k_sel != 1: print(f'    using position #{k_sel} (--tilt-candidate)')
        far = [c_ for c_ in cands[1:] if angd(c_[1], cands[0][1]) > 20]
        if a.tilt_rotation is None and far and cands[0][0] - far[0][0] < 0.03:
            # near-tie at the same position (typically 180 deg apart: a straight fibre looks the same both ways).
            # Both images come from the same camera, so the mdoc RotationAngle values say whether a flip is real.
            pick = min([cands[0], far[0]], key=lambda c_: min(angd(c_[1], rel), angd(c_[1], -rel)))
            print(f'    near-tie {cands[0][1]:.0f} vs {far[0][1]:.0f} deg -> choosing {pick[1]:.0f} deg '
                  f'(closest to the mdoc RotationAngle difference {rel:+.1f} deg); use --tilt-flip180 for the other')
            cands.insert(0, cands.pop(cands.index(pick)))
        if a.tilt_flip180 and a.tilt_rotation is None:
            want = cands[0][1] + 180
            near = [c_ for c_ in cands if angd(c_[1], want) <= 15]
            if near: cands.insert(0, cands.pop(cands.index(near[0])))
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    r = refine(Sh, Th, L_ts0 @ rotm(want), cands[0][2]['p'], max(3, int(rad)), np.arange(-5, 5.01, 1.0),
                               np.arange(0.9, 1.101, 0.05), inner=0.9)
                cands.insert(0, (r['score'], want + r['dtheta'], r, float('nan')))
            print(f'    --tilt-flip180: using {cands[0][1]:.0f} deg')
        tilt_groups = groups[:3]
        best = refine(Sh, Th, cands[0][2]['L'], cands[0][2]['p'], 3, np.arange(-2, 2.01, 0.5),
                      np.arange(0.95, 1.051, 0.01), inner=0.9, label='square <- tilt (fine)')
        if best['score'] >= a.min_score[1]: L_ts, p_ts = best['L'], best['p']
        else:
            print('    tilt refinement rejected (low score) - using manual target and rotation')
            L_ts = L_ts0 @ rotm(a.tilt_rotation if a.tilt_rotation is not None else cands[0][1])
    T_SA = aff(L_sa, p_sa - L_sa @ cS)                     # square display px -> atlas display px
    T_TS = aff(L_ts, p_ts - L_ts @ cT)                     # tilt display px   -> square display px
    tgt_s = apply(T_TS, cT)
    res = dict(square_in_atlas_imod=[float(p_sa[0] * fA), float(HA * fA - 1 - p_sa[1] * fA)],
               target_in_square_imod=[float(tgt_s[0] * fS), float(HS * fS - 1 - tgt_s[1] * fS)],
               square_to_atlas=dict(scale=scale_of(L_sa), rot_deg=ang_of(L_sa)),
               tilt_to_square=dict(scale=scale_of(L_ts), rot_deg=ang_of(L_ts)))
    print(f'  final: square centre on atlas {res["square_in_atlas_imod"][0]:.1f}, {res["square_in_atlas_imod"][1]:.1f}'
          f' | target on square {res["target_in_square_imod"][0]:.1f}, {res["target_in_square_imod"][1]:.1f} (IMOD)')
    json.dump(res, open(qc + '_alignment.json', 'w'), indent=2)

    # ---------------- QC images
    Ti = np.linalg.inv(T_SA)
    Aw = cv2.warpAffine(An, Ti[:2].astype(np.float32), (WS, HS), flags=cv2.INTER_LINEAR)
    cv2.imwrite(qc + '_qc_atlas_square.jpg', checkerboard(Aw, Sn).astype(np.uint8))
    k = 6; half = int(max(WT, HT) * scale_of(L_ts) * 0.9) + 10
    win = np.array([tgt_s[0] - half, tgt_s[1] - half])
    Mz = aff(np.eye(2) * k, -win * k)
    sq_zoom = cv2.warpAffine(Sn, Mz[:2].astype(np.float32), (2 * half * k, 2 * half * k), flags=cv2.INTER_CUBIC)
    t_in = cv2.warpAffine(Tn, (Mz @ T_TS)[:2].astype(np.float32), (2 * half * k, 2 * half * k), flags=cv2.INTER_AREA)
    t_m = cv2.warpAffine(np.ones_like(Tn), (Mz @ T_TS)[:2].astype(np.float32), (2 * half * k, 2 * half * k)) > 0.5
    if t_m.any():                                          # match tilt contrast to the square map under it
        t_in = (t_in - t_in[t_m].mean()) / (t_in[t_m].std() + 1e-6) * sq_zoom[t_m].std() + sq_zoom[t_m].mean()
    ys_, xs_ = np.where(t_m)
    cbx = checkerboard(sq_zoom, np.clip(t_in, 0, 255), n=max(2, int(2 * half * k / max(1, (np.ptp(xs_) + 1) / 4))))
    cb = np.where(t_m, cbx, sq_zoom * 0.6)
    p1_ = cv2.cvtColor(sq_zoom.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pts = np.array([apply(Mz @ T_TS, c) for c in [(0, 0), (WT, 0), (WT, HT), (0, HT)]], np.int32)
    cv2.polylines(p1_, [pts], True, (40, 200, 255), 2)
    p3_ = cv2.cvtColor(np.where(t_m, np.clip(t_in, 0, 255), sq_zoom * 0.6).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    p2_ = cv2.cvtColor(cb.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(qc + '_qc_square_tilt.jpg', np.hstack([p1_, p3_, p2_]))
    if tilt_groups:                                        # one panel per candidate position
        pan, half_c = [], int(max(WT, HT) * scale_of(L_ts) * 1.3) + 10
        kz = max(1, int(round(420 / (2 * half_c))))
        raw = cv2.resize(Tn, (420, int(420 * HT / WT)), interpolation=cv2.INTER_AREA)
        rawp = np.full((420, 420), 40, np.float32); y0_ = (420 - raw.shape[0]) // 2; rawp[y0_:y0_ + raw.shape[0]] = raw
        rawp = cv2.cvtColor(rawp.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(rawp, '0-deg tilt (as in 3dmod)', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 200, 255), 2)
        rows = []
        for k, g in enumerate(tilt_groups, 1):
            Lg, pg = g[0][2]['L'], g[0][2]['p']
            Tg = aff(Lg, pg - Lg @ cT); Mz_ = aff(np.eye(2) * kz, -(pg - half_c) * kz); sz = 2 * half_c * kz
            sqz = cv2.warpAffine(Sn, Mz_[:2].astype(np.float32), (sz, sz), flags=cv2.INTER_CUBIC)
            tz = cv2.warpAffine(Tn, (Mz_ @ Tg)[:2].astype(np.float32), (sz, sz), flags=cv2.INTER_AREA, borderValue=40)
            pts = np.array([apply(Mz_ @ Tg, c) for c in [(0, 0), (WT, 0), (WT, HT), (0, HT)]], np.int32)
            o1 = cv2.cvtColor(sqz.astype(np.uint8), cv2.COLOR_GRAY2BGR); cv2.polylines(o1, [pts], True, (40, 200, 255), 2)
            o2 = cv2.cvtColor(np.clip(tz, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR); cv2.polylines(o2, [pts], True, (40, 200, 255), 2)
            o1, o2 = cv2.resize(o1, (420, 420)), cv2.resize(o2, (420, 420))
            d_um = np.hypot(*(pg - tg_s)) * pxS / 1e4
            cv2.putText(o1, f'#{k} map  ({d_um:.1f} um from target)', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 200, 255), 2)
            cv2.putText(o2, f'#{k} tilt at rot {g[0][1]:.0f}, score {g[0][0]:.2f}', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 200, 255), 2)
            rows.append(np.hstack([o1, o2, rawp]))
        pan = [np.vstack(rows)]
        cv2.imwrite(qc + '_qc_tilt_candidates.jpg', pan[0])
        print(f'  candidate positions: {qc}_qc_tilt_candidates.jpg')
    print(f'  QC images: {qc}_qc_atlas_square.jpg, {qc}_qc_square_tilt.jpg')
    if a.qc_only: return

    # ---------------- morph model (square map blurred + contrast-mapped to look like the atlas)
    m_ = np.zeros(S.shape, bool); m_[HS // 8:-HS // 8, WS // 8:-WS // 8] = True
    q = np.linspace(0, 100, 101); best = None
    rr_ = 1 / scale_of(L_sa)
    for sig in [rr_ * k_ for k_ in (0.25, 0.35, 0.6, 0.85, 1.1, 1.3, 1.7, 2.15)]:
        b = cv2.GaussianBlur(Sn, (0, 0), sig)
        xs, ys = np.percentile(b[m_], q), np.percentile(Aw[m_], q)
        e = np.mean((np.interp(b[m_], xs, ys) - Aw[m_]) ** 2)
        if best is None or e < best[0]: best = (e, sig, xs, ys)
    SIG, LUT_X, LUT_Y = best[1], best[2], best[3]

    # ---------------- renderer
    def pyramid(img):
        lv = [img.astype(np.float32)]
        while min(lv[-1].shape) > 300:
            lv.append(cv2.resize(lv[-1], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
        return lv
    r_ = max(8, min(HS, WS) // 24)
    fy = np.minimum(np.arange(HS), HS - 1 - np.arange(HS)); fx = np.minimum(np.arange(WS), WS - 1 - np.arange(WS))
    feather = np.outer(0.5 - 0.5 * np.cos(np.pi * np.clip(fy / r_, 0, 1)), 0.5 - 0.5 * np.cos(np.pi * np.clip(fx / r_, 0, 1)))
    PA, PS, PM = pyramid(An), pyramid(Sn), pyramid(feather.astype(np.float32))

    def view(c, z, phi):
        Rz = cv2.getRotationMatrix2D((0, 0), phi, z)
        return aff(np.eye(2), [W / 2, H / 2]) @ np.vstack([Rz, [0, 0, 1]]) @ aff(np.eye(2), -np.asarray(c))

    def warp(pyr, M, scale, out=(W, H)):
        L = 0
        while L + 1 < len(pyr) and scale * 2 ** (L + 1) <= 1.0: L += 1
        f = 2 ** L
        U = np.array([[f, 0, (f - 1) / 2], [0, f, (f - 1) / 2], [0, 0, 1]])
        return cv2.warpAffine(pyr[L], (M @ U)[:2].astype(np.float32), out, flags=cv2.INTER_LINEAR, borderValue=0)

    fT, fS = font(40), font(28)
    def label(d, xy, text, f, fill, anchor):          # anchored, glyph-safe text
        d.text(xy, safe_text(f, text), font=f, fill=fill, anchor=anchor)
    def tlen(d, text, f): return d.textlength(safe_text(f, text), font=f)
    def overlays(im, title, sub, ta, bar_nm_px=None, bar_pos=None, extra=None):
        ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        if extra: extra(d)
        if title and ta > 0:
            tw = tlen(d, title, fT)
            d.rounded_rectangle([40, 36, 84 + tw, 106], 14, fill=(0, 0, 0, int(150 * ta)))
            label(d, (62, 71), title, fT, (255, 255, 255, int(255 * ta)), 'lm')
            if sub:
                sw = tlen(d, sub, fS)
                d.rounded_rectangle([40, 116, 84 + sw, 166], 12, fill=(0, 0, 0, int(150 * ta)))
                label(d, (62, 141), sub, fS, (230, 230, 230, int(255 * ta)), 'lm')
        if bar_nm_px:
            for Lb in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]:
                if Lb / bar_nm_px >= 140: break
            bl = Lb / bar_nm_px; x1, y = bar_pos or (W - 60, H - 60); x0 = x1 - bl
            lab = f'{Lb / 1000:g} µm' if Lb >= 1000 else f'{Lb} nm'
            d.rounded_rectangle([x0 - 20, y - 52, x1 + 20, y + 22], 12, fill=(0, 0, 0, 150))
            d.rectangle([x0, y - 4, x1, y + 6], fill=(255, 255, 255, 255))
            label(d, ((x0 + x1) / 2, y - 26), lab, fS, (255, 255, 255, 255), 'mm')
        return np.asarray(Image.alpha_composite(Image.fromarray(im).convert('RGBA'), ov).convert('RGB'))

    def poly(d, V, frame, corners, alpha, color=(255, 200, 40), width=5):
        if alpha <= 0: return
        Mx = {'a': V, 's': V @ T_SA, 't': V @ T_SA @ T_TS}[frame]
        pts = [tuple(apply(Mx, p)) for p in corners]
        d.line(pts + [pts[0]], fill=color + (int(255 * alpha),), width=width, joint='curve')
        P = np.array(pts); ext = np.ptp(P, axis=0).max()
        if ext < 40:                                   # tiny on screen: add a ring so it can be seen
            c = P.mean(0); rr = 34
            d.ellipse([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr], outline=color + (int(255 * alpha),), width=4)

    def background(c, z, phi, sq_alpha, morph):
        V = view(c, z, phi)
        out = warp(PA, V, z) if sq_alpha < 1 else np.zeros((H, W), np.float32)
        if sq_alpha > 0:
            Ms, zs = V @ T_SA, z * scale_of(L_sa)
            s, m = warp(PS, Ms, zs), warp(PM, Ms, zs) * sq_alpha
            if morph > 0:
                sg = morph * SIG * zs
                if a.transition == 'focus' and sg > 0.3: s = cv2.GaussianBlur(s, (0, 0), sg)
                s = s * (1 - morph) + np.interp(s, LUT_X, LUT_Y).astype(np.float32) * morph
            out = out * (1 - m) + s * m
        return V, np.clip(out, 0, 255).astype(np.uint8)

    def frame(c, z, phi, sq_alpha, morph, title, sub, ta, boxes):
        V, bg = background(c, z, phi, sq_alpha, morph)
        rgb = np.dstack([bg] * 3)
        return overlays(rgb, title, sub, ta, pxA / 10 / z,
                        extra=lambda d: [poly(d, V, fr, co, al) for fr, co, al in boxes])

    # ---------------- camera path
    th = ang_of(L_sa); k90 = round(th / 90); PHI_END = th - 90 * k90     # square ends upright (x 90k)
    z_full = min(H / HA, W / WA) * 0.98; c_full = np.array([WA / 2, HA / 2])
    SQ_C = apply(T_SA, cS); z_sq = 0.97 * H / (HS * scale_of(L_sa)) if k90 % 2 == 0 else 0.97 * H / (WS * scale_of(L_sa))
    T_A = apply(T_SA, tgt_s); z_tg = z_sq * a.zoom
    sq_c = [(0, 0), (WS, 0), (WS, HS), (0, HS)]; t_c = [(0, 0), (WT, 0), (WT, HT), (0, HT)]

    def zoom_cam(t, c0, z0, c1, z1):
        e = ease(t); w0, w1 = 1 / z0, 1 / z1; w = w0 * (w1 / w0) ** e
        k = (w - w1) / (w0 - w1) if w0 != w1 else 0
        return c1 + (np.asarray(c0) - c1) * k, 1 / w

    TA_, SA_ = 'Atlas', 'Grid overview'
    TS_, SS_ = 'Square map', 'Selecting the acquisition position'
    TT_, ST_ = 'Acquisition position', 'Tilt-series target'
    TL_ = 'Tilt series'

    def s1(t):
        if t < 3: return frame(c_full, z_full, 0, 0, 1, TA_, SA_, 1, [('s', sq_c, ease((t - 1) / .8))])
        u = (t - 3) / 4; c, z = zoom_cam(u, c_full, z_full, SQ_C, z_sq)
        if a.transition == 'sharp':            # fade in the sharp square map while the atlas is still near native resolution
            alpha = ease((np.log(z) - np.log(0.6)) / (np.log(1.8) - np.log(0.6)))
        else:
            alpha = ease((t - 6) / 1.0)
        return frame(c, z, PHI_END * ease(u), alpha, 1, TA_, SA_, 1, [('s', sq_c, 1 - ease((t - 5.6) / 1.0))])

    def s2(t):
        mo = 1 - ease(t / (1.2 if a.transition == 'sharp' else 2.2))
        ti, su, ta = (TA_, SA_, 1 - ease(t / .8)) if t < .8 else (TS_, SS_, ease((t - .8) / .8))
        return frame(SQ_C, z_sq, PHI_END, 1, mo, ti, su, ta, [])

    ZT = a.zoom_time or float(np.clip(4.0 + 1.2 * np.log2(max(a.zoom, 1) / 6), 3.0, 8.0))
    def s3(t):                                 # zoom, then the acquisition box appears, then hold
        c, z = zoom_cam(t / ZT, SQ_C, z_sq, T_A, z_tg)
        ti, su, ta = (TS_, SS_, 1 - ease((t - ZT + 1.4) / .6)) if t < ZT - 0.8 else (TT_, ST_, ease((t - ZT + 0.8) / .6))
        return frame(c, z, PHI_END, 1, 0, ti, su, ta, [('t', t_c, ease((t - ZT - 0.2) / 0.6))])

    # ---------------- tilt inset
    V_end = view(T_A, z_tg, PHI_END)
    M0 = V_end @ T_SA @ T_TS                                  # tilt px -> screen at the end of the zoom
    a0, s0 = ang_of(M0[:2, :2]), scale_of(M0[:2, :2])
    a1 = 0.0 if a.tilt_as_file else 90 * round(a0 / 90)   # end orientation of the inset
    if a.tilt_as_file: a1 = a0 + ((a1 - a0 + 180) % 360 - 180)                 # shortest turn to file orientation
    bw, bh = (WT, HT) if round(a1 / 90) % 2 == 0 else (HT, WT)
    s1_ = min(0.5 * W / bw, 0.5 * H / bh)                     # inset = half the frame size
    p0, p1 = apply(M0, cT), np.array([W / 2 - (190 if a.schematic else 0), H / 2 - 10])
    if a.tilt_max:
        order = sorted((z for z in angles if z < nzT), key=lambda z: angles[z])
        neg = [z for z in order if -a.tilt_max - 0.5 <= angles[z] <= 0.5][::-1]          # 0 -> -60
        allr = [z for z in order if abs(angles[z]) <= a.tilt_max + 0.5]                   # -60 -> +60
        pos = [z for z in order if -0.5 <= angles[z] <= a.tilt_max + 0.5][::-1]           # +60 -> 0
        seq = neg + allr[1:] + pos[1:]
    print(f'  tilt playback: {len(seq)} images ({angles[seq[0]]:+.0f} -> {min(angles[z] for z in seq):+.0f} -> '
          f'{max(angles[z] for z in seq):+.0f} -> {angles[seq[-1]]:+.0f}) at {a.tilt_fps:g}/s')

    tw_, th_ = max(2, int(round(WT * s1_))), max(2, int(round(HT * s1_)))
    tilt_cache = {}
    def tilt_img(z):
        if z not in tilt_cache:
            img, _, _, _ = load_mrc(a.tilt, z, max_dim=a.tilt_max_px)
            img = cv2.resize(img, (tw_, th_), interpolation=cv2.INTER_AREA)
            tilt_cache[z] = norm8(cv2.GaussianBlur(img, (0, 0), 0.7), 0.5, 99.5)
        return tilt_cache[z]
    T0_pyr = pyramid(norm8(cv2.GaussianBlur(T0, (0, 0), 1.0)))

    def inset_M(u):
        e = ease(u); s = s0 * (s1_ / s0) ** e; ang = a0 + (a1 - a0) * e; p = p0 + (p1 - p0) * e
        return aff(s * rotm(ang), p - s * rotm(ang) @ cT), s, ang

    _, bg_end = background(T_A, z_tg, PHI_END, 1, 0)
    bg_end = bg_end.astype(np.float32)

    def schematic(d, cx, cy, ang, al):
        """Beam coming down onto a specimen slab tilted by ang (deg) about an axis normal to the screen."""
        if al <= 0: return
        A_ = lambda v: int(255 * al * v)
        d.rounded_rectangle([cx - 150, cy - 150, cx + 150, cy + 135], 18, fill=(0, 0, 0, A_(0.55)))
        label(d, (cx, cy - 122), 'Stage tilt', fS, (230, 230, 230, A_(1)), 'mm')
        for bx_ in (-50, 0, 50):                                       # electron beam
            d.line([cx + bx_, cy - 98, cx + bx_, cy - 52], fill=(120, 190, 255, A_(0.9)), width=3)
            d.polygon([(cx + bx_ - 8, cy - 60), (cx + bx_ + 8, cy - 60), (cx + bx_, cy - 44)], fill=(120, 190, 255, A_(0.9)))
        d.line([cx - 120, cy + 10, cx + 120, cy + 10], fill=(255, 255, 255, A_(0.35)), width=2)   # 0-deg reference
        r_ = np.radians(-ang); c_, s_ = np.cos(r_), np.sin(r_)
        hl, ht = 92, 9                                                 # slab half-length / half-thickness
        corners = [(-hl, -ht), (hl, -ht), (hl, ht), (-hl, ht)]
        pts = [(cx + x * c_ - y * s_, cy + 10 + x * s_ + y * c_) for x, y in corners]
        d.polygon(pts, fill=(255, 200, 40, A_(0.95)), outline=(255, 255, 255, A_(1)))
        d.ellipse([cx - 6, cy + 4, cx + 6, cy + 16], fill=(30, 30, 30, A_(1)))       # tilt axis
        if abs(ang) > 0.5:                                              # angle arc
            a0_, a1_ = sorted([0.0, -ang])
            d.arc([cx - 60, cy + 10 - 60, cx + 60, cy + 10 + 60], a0_, a1_, fill=(255, 255, 255, A_(0.8)), width=3)
            d.arc([cx - 60, cy + 10 - 60, cx + 60, cy + 10 + 60], a0_ + 180, a1_ + 180, fill=(255, 255, 255, A_(0.8)), width=3)

    def tilt_frame(img_full_pyr, img_small, M, s, alpha, dim, label_ang, title_a, title, sub, bar=True,
                   sch_ang=None, sch_alpha=1.0):
        out = bg_end * dim
        if img_small is not None:                              # playback: pre-sized image, rotation only
            Ms = M @ aff(np.eye(2) / s1_, [0, 0])
            ti = cv2.warpAffine(img_small, Ms[:2].astype(np.float32), (W, H), flags=cv2.INTER_LINEAR)
            mm = cv2.warpAffine(np.ones_like(img_small), Ms[:2].astype(np.float32), (W, H))
        else:
            ti = warp(img_full_pyr, M, s); mm = warp([np.ones_like(l) for l in img_full_pyr], M, s)
        mm *= alpha
        out = out * (1 - mm) + ti * mm
        rgb = np.dstack([np.clip(out, 0, 255).astype(np.uint8)] * 3)
        def extra(d):
            pts = [tuple(apply(M, p)) for p in t_c]
            d.line(pts + [pts[0]], fill=(255, 255, 255, int(220 * alpha)), width=3, joint='curve')
            if label_ang is not None and alpha > 0.5:
                ys = max(p[1] for p in pts) + 22; xs = min(p[0] for p in pts)
                v = int(round(label_ang)); lab = '0°' if v == 0 else f'{v:+d}°'.replace('-', '−')
                lw = max(tlen(d, t_, fT) for t_ in ('−60°', '+60°', '−00°'))        # fixed width: no jumping
                d.rounded_rectangle([xs, ys, xs + lw + 36, ys + 62], 14, fill=(0, 0, 0, 170))
                label(d, (xs + lw + 18, ys + 31), lab, fT, (255, 255, 255, 255), 'rm')  # right-aligned digits
                tx0, tx1, ty = xs + lw + 80, xs + lw + 380, ys + 31          # angle track
                d.line([tx0, ty, tx1, ty], fill=(255, 255, 255, 140), width=3)
                for v in (-a.tilt_max, 0, a.tilt_max):
                    x = tx0 + (v + a.tilt_max) / (2 * a.tilt_max) * (tx1 - tx0)
                    d.line([x, ty - 9, x, ty + 9], fill=(255, 255, 255, 160), width=2)
                x = tx0 + (label_ang + a.tilt_max) / (2 * a.tilt_max) * (tx1 - tx0)
                d.ellipse([x - 9, ty - 9, x + 9, ty + 9], fill=(255, 200, 40, 255))
            if sch_ang is not None and a.schematic:
                xr = max(p[0] for p in pts); yc = sum(p[1] for p in pts) / 4
                cx = min(W - 170, xr + 190)
                schematic(d, cx, yc, sch_ang, sch_alpha)
        bar_nm = (pxT / 10 / s) if (bar and alpha > 0.5) else None
        bx = max(p[0] for p in [apply(M, q) for q in t_c]); by = max(p[1] for p in [apply(M, q) for q in t_c])
        return overlays(rgb, title, sub, title_a, bar_nm, (bx - 20, by - 24), extra)

    def s4(t):                                                 # inset grows out of the footprint
        u = (t - 0.2) / 1.6; M, s, _ = inset_M(u)
        dim = 1 - 0.6 * ease(t / 1.2)
        ti, su, ta = (TT_, ST_, 1 - ease(t / .6)) if t < .6 else (TL_, '0° → −60° → +60° → 0°', ease((t - .6) / .6))
        return tilt_frame(T0_pyr, None, M, s, 1.0, dim, angles[z0] if t > 1.6 else None, ta, ti, su, bar=t > 1.6,
                          sch_ang=angles[z0], sch_alpha=ease((t - 1.4) / 0.6))

    Mend, _, _ = inset_M(1)
    fpt = max(1, int(round(FPS / a.tilt_fps)))
    def s5(i):                                                 # playback, frame index based
        j = min(i // fpt, len(seq) - 1)
        jn = min(j + 1, len(seq) - 1); fr = (i % fpt) / fpt if j < len(seq) - 1 else 0.0
        sa = angles[seq[j]] + (angles[seq[jn]] - angles[seq[j]]) * fr       # schematic moves smoothly
        return tilt_frame(None, tilt_img(seq[j]), Mend, s1_, 1.0, 0.4, angles[seq[j]], 1, TL_, '0° → −60° → +60° → 0°',
                          sch_ang=sa)

    scenes = [('s1', 7.0, s1), ('s2', 3.5, s2), ('s3', ZT + 0.8 + a.hold, s3), ('s4', 2.0, s4)]
    n5 = len(seq) * fpt + int(1.5 * FPS)

    if a.preview:
        pre = os.path.splitext(a.out)[0]
        shots = [(s1, 2.5), (s1, 5.5), (s1, 6.6), (s2, 1.0), (s2, 3.0), (s3, ZT * 0.5), (s3, ZT - 0.1), (s3, ZT + 1.5), (s4, 0.9)]
        for i, (f, t) in enumerate(shots): Image.fromarray(f(t)).save(f'{pre}_preview_{i}.jpg', quality=90)
        for i, idx in enumerate([0, len(seq) // 4 * fpt, len(seq) // 2 * fpt]):
            Image.fromarray(s5(idx)).save(f'{pre}_preview_t{i}.jpg', quality=90)
        print(f'Saved preview frames {pre}_preview_*.jpg'); return

    tmpd = tempfile.mkdtemp(prefix='cryoet_movie_')
    # pick an H.264 encoder this ffmpeg actually has (conda-forge's LGPL build lacks libx264)
    try:
        enc_list = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    except OSError:
        enc_list = ''
    if a.encoder != 'auto':
        enc = a.encoder
    else:
        enc = next((e for e in ('libx264', 'h264_videotoolbox', 'libopenh264', 'h264_nvenc', 'h264_qsv')
                    if f' {e} ' in enc_list), 'mpeg4')
    ENC = {'libx264': ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18'],
           'h264_videotoolbox': ['-c:v', 'h264_videotoolbox', '-b:v', '20M', '-allow_sw', '1'],
           'libopenh264': ['-c:v', 'libopenh264', '-b:v', '20M'],
           'h264_nvenc': ['-c:v', 'h264_nvenc', '-cq', '19', '-b:v', '0'],
           'h264_qsv': ['-c:v', 'h264_qsv', '-global_quality', '20'],
           'mpeg4': ['-c:v', 'mpeg4', '-q:v', '2']}.get(enc, ['-c:v', enc])
    print(f'  video encoder: {enc}' + ('  (no H.264 encoder found - install a GPL ffmpeg for best quality)' if enc == 'mpeg4' else ''))

    def encode(name, n, fn):
        tmp, final = os.path.join(tmpd, name + '.tmp.mp4'), os.path.join(tmpd, name + '.mp4')
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                              '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-'] + ENC + ['-pix_fmt', 'yuv420p', tmp],
                             stdin=subprocess.PIPE)
        try:
            for i in range(n): p.stdin.write(np.ascontiguousarray(fn(i)).tobytes())
            p.stdin.close()
        except BrokenPipeError:
            pass
        if p.wait() != 0: sys.exit(f'ffmpeg failed on {name} with encoder {enc} (see the ffmpeg message above); '
                                   f'try --encoder mpeg4 or install a GPL ffmpeg')
        os.replace(tmp, final); print(f'  rendered {name} ({n / FPS:.1f} s)'); return final
    print('Rendering')
    files = [encode(nm, int(d * FPS), lambda i, f=f: f(i / FPS)) for nm, d, f in scenes]
    files.append(encode('s5', n5, s5))
    lst = os.path.join(tmpd, 'list.txt')
    open(lst, 'w').write(''.join(f"file '{f}'\n" for f in files))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst,
                    '-c', 'copy', '-movflags', '+faststart', a.out], check=True)
    shutil.rmtree(tmpd, ignore_errors=True)
    print(f'Saved {a.out}  ({sum(d for _, d, _ in scenes) + n5 / FPS:.1f} s)')


if __name__ == '__main__':
    main()
