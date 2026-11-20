#!/usr/bin/env python
"""
Fast in-plane angle (RELION psi) of the centriole / cilium axis in cryo-ET tomograms,
with a browser-based review that runs on the remote server (view it through an ssh tunnel).

Outputs (only these):
  psi_results.csv              one row per tomogram: psi, status, IMOD points, auto values
  imod_axis/<tomo>_axis.txt    2 points 'contour x y z' for point2model (always written,
                               rewritten whenever you correct the axis in the review;
                               removed if you reject the tomogram)
  imod_axis/<tomo>_axis.mod    only with --mod (runs point2model -open -image <tomo>)

Commands:
  python centriole_psi.py run    tomos/*.mrc            # batch, no GUI (skips done tomos)
  python centriole_psi.py review tomos/*.mrc            # web review on the server
      then on your laptop:  ssh -N -L 8765:localhost:8765 you@server
      and open the URL the server prints (contains an access token)
  (review also computes any tomograms not yet in the CSV, in the background)

Points: unbinned px, 0-based (IMOD model coords), middle Z of the tomogram, centred where the
axis passes closest to the tomogram XY centre. Order satisfies psi = -atan2(y2-y1, x2-x1)
(RELION; --convention math gives +atan2). Axis is non-polar: psi in (-90, 90].

Method: Z-slab projection -> mask border + strong low-frequency edges (carbon) -> multi-scale
Hessian ridge filter at microtubule width; psi = peak of the orientation histogram.
Cross-check: angular power spectrum (Fourier). Flag CHECK if they disagree > 10 deg.
"""
import argparse, csv, json, math, os, secrets, shutil, socket, subprocess, sys, threading, time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


# ---------------------------------------------------------------- I/O + projection
def load_bin(path, b, apix=None):
    if path.endswith(".npy"):                          # test hook: 2D image
        v = np.load(path)[None].astype(np.float32)
        return v, (apix or 1.0), (1, v.shape[1] * b, v.shape[2] * b)
    import mrcfile
    with mrcfile.mmap(path, permissive=True) as m:
        v = np.asarray(m.data, dtype=np.float32)
        shape0 = v.shape
        a = float(m.voxel_size.x)
    if not apix and a <= 0:
        print(f"WARNING {path}: no pixel size in header, assuming 10 A (use --apix)"); a = 10.0
    a = apix or a
    if b > 1:
        z, y, x = (s // b * b for s in v.shape)
        v = v[:z, :y, :x].reshape(z // b, b, y // b, b, x // b, b).mean((1, 3, 5))
    return v, a * b, shape0


def project(vol, slab_px, zc=None):
    nz = vol.shape[0]
    if nz == 1:
        return vol[0], 0
    if zc is None:  # slab with most high-frequency structure
        hp = vol - ndi.gaussian_filter(vol, (0, 4, 4))
        prof = ndi.uniform_filter1d(hp.reshape(nz, -1).std(1), max(3, slab_px // 2))
        zc = int(np.argmax(prof[slab_px // 4: nz - slab_px // 4]) + slab_px // 4) if nz > slab_px else nz // 2
    z0, z1 = max(0, zc - slab_px // 2), min(nz, zc + slab_px // 2)
    return vol[z0:z1].mean(0), zc


# ---------------------------------------------------------------- masks
def valid_mask(img, mt_px):
    """Exclude padded / flat reconstruction borders."""
    s = max(2.0, mt_px)
    hp = img - ndi.gaussian_filter(img, s)
    lstd = np.sqrt(ndi.gaussian_filter(hp ** 2, 4 * s))
    m = lstd > 0.35 * np.median(lstd)
    m[:2] = m[-2:] = False; m[:, :2] = m[:, -2:] = False
    m = ndi.binary_erosion(m, iterations=min(int(3 * s) + 2, min(img.shape) // 10))
    if m.sum() < 0.2 * m.size:                       # fallback: whole image minus margin
        m = np.zeros_like(m); k = min(img.shape) // 20; m[k:-k, k:-k] = True
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)
    return m


def edge_mask(img, valid, mt_px, k=4.0):
    """Strong low-frequency steps (carbon / ice edge). Returns pixels to EXCLUDE."""
    s = 6 * mt_px
    lp = ndi.gaussian_filter(np.where(valid, img, img[valid].mean()), s)
    g = np.hypot(ndi.sobel(lp, 0), ndi.sobel(lp, 1))
    v = g[valid]; med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-9
    e = (g > med + k * 1.4826 * mad) & valid
    return ndi.binary_dilation(e, iterations=int(1.5 * s))


# ---------------------------------------------------------------- estimators
def wrap90(a):
    a = (np.asarray(a, float) + 90.0) % 180.0 - 90.0
    a = np.where(a <= -90.0, 90.0, a)
    return float(a) if a.ndim == 0 else a


def hist_peak(theta, w, bins=180, smooth=2.0, win=8.0):
    """theta in deg (-90,90]; returns refined peak and fraction of weight within +-win."""
    h, edges = np.histogram(theta, bins=bins, range=(-90, 90), weights=w)
    h = ndi.gaussian_filter1d(h, smooth, mode="wrap")
    c = 0.5 * (edges[1:] + edges[:-1])
    p = c[np.argmax(h)]
    d = (theta - p + 90) % 180 - 90
    sel = np.abs(d) < win
    if w[sel].sum() == 0:
        return p, 0.0
    peak = wrap90(p + np.average(d[sel], weights=w[sel]))
    return peak, float(w[sel].sum() / w.sum())


def psi_ridge(img, mask, mt_px):
    """Dark-line (MT) orientation from multi-scale Hessian. Returns psi_math, conf, strength map."""
    S = np.zeros_like(img); TH = np.zeros_like(img)
    for s in (0.5 * mt_px, 0.75 * mt_px, 1.0 * mt_px):
        s = max(s, 1.0)
        hxx = ndi.gaussian_filter(img, s, order=(0, 2)) * s * s
        hyy = ndi.gaussian_filter(img, s, order=(2, 0)) * s * s
        hxy = ndi.gaussian_filter(img, s, order=(1, 1)) * s * s
        tr = hxx + hyy; dt = np.sqrt(((hxx - hyy) / 2) ** 2 + hxy ** 2)
        l1, l2 = tr / 2 + dt, tr / 2 - dt                 # l1 >= l2
        strength = np.clip(l1 - np.abs(l2), 0, None)      # dark ridge: big +curvature across
        # direction of l1 eigenvector (across the line); line is perpendicular
        ang_across = 0.5 * np.degrees(np.arctan2(2 * hxy, hxx - hyy))
        better = strength > S
        S[better] = strength[better]; TH[better] = ang_across[better] + 90.0
    TH = (TH + 90) % 180 - 90
    m = mask & (S > 0)
    if m.sum() < 50:
        return np.nan, 0.0, S
    w = S[m] ** 2
    psi, conf = hist_peak(TH[m], w)
    S = np.where(mask, S, 0); S[~m] = 0
    return psi, conf, (S, TH)


def psi_fourier(img, mask, mt_px):
    """Angular power in MT band; line orientation = spectral streak + 90."""
    x = img - ndi.gaussian_filter(img, 4 * mt_px)
    soft = ndi.gaussian_filter(mask.astype(float), mt_px)
    x = (x - x[mask].mean()) * soft
    F = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    ny, nx = x.shape
    ky, kx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(ny)), np.fft.fftshift(np.fft.fftfreq(nx)), indexing="ij")
    r = np.hypot(kx, ky)  # cycles/px
    band = (r > 1 / (8 * mt_px)) & (r < 1 / (0.8 * mt_px)) & (r < 0.45)
    phi = np.degrees(np.arctan2(ky, kx))[band]
    w = (F[band] * r[band] ** 2)                       # whiten
    psi, conf = hist_peak(wrap90(phi + 90.0), w, win=5)
    return psi, conf


def axis_center(SM, psi, mt_px, tol=8.0):
    """Centre of the bundle: peak of the across-axis profile of aligned ridge strength."""
    S, TH = SM
    d = np.abs((TH - psi + 90) % 180 - 90)
    pos = S[S > 0]
    if pos.size == 0:
        return S.shape[1] / 2, S.shape[0] / 2
    w = np.where(d < tol, np.minimum(S, np.percentile(pos, 90)), 0)   # count lines, don't let one edge win
    yy, xx = np.indices(S.shape)
    t = math.radians(psi); ux, uy = math.cos(t), math.sin(t); nx_, ny_ = -uy, ux
    off = xx * nx_ + yy * ny_
    o0 = off.min(); ob = np.round(off - o0).astype(int)
    prof = np.bincount(ob.ravel(), w.ravel())
    prof = ndi.gaussian_filter1d(prof, 3 * mt_px)          # one bump per bundle
    pk = int(np.argmax(prof)); thr = 0.4 * prof[pk]
    lo = pk;  hi = pk
    while lo > 0 and prof[lo - 1] > thr: lo -= 1
    while hi < len(prof) - 1 and prof[hi + 1] > thr: hi += 1
    o_c = 0.5 * (lo + hi) + o0
    sel = (ob >= lo) & (ob <= hi)
    along = ((xx * ux + yy * uy) * w * sel).sum() / max((w * sel).sum(), 1e-12)
    return along * ux + o_c * nx_, along * uy + o_c * ny_


# ---------------------------------------------------------------- drawing
def draw(ax, img, psi_math, cx, cy, title, color="#ff3b30", psi2=None, edge=None):
    lo, hi = np.percentile(img, [1, 99])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, origin="lower")
    if edge is not None and edge.any():
        ax.contour(edge, levels=[0.5], colors="#ffcc00", linewidths=0.6, alpha=0.6)
    L = 0.6 * max(img.shape)
    for p, st, c, lw in ((psi_math, "-", color, 2), (psi2, "--", "#30b0ff", 1)):
        if p is None or not np.isfinite(p):
            continue
        t = math.radians(p); dx, dy = L * math.cos(t), L * math.sin(t)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], st, color=c, lw=lw)
    ax.plot(cx, cy, "o", color=color, ms=4)
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(0, img.shape[0])
    ax.set_title(title, fontsize=8); ax.axis("off")


# ---------------------------------------------------------------- review page


def encode_jpeg(img, max_dim):
    import io
    from PIL import Image
    lo, hi = np.percentile(img, [0.5, 99.5])
    u8 = (np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(u8)                       # row 0 = y 0; page flips for Y-up view
    f = max_dim / max(im.size)
    if f < 1:
        im = im.resize((max(1, round(im.size[0] * f)), max(1, round(im.size[1] * f))), Image.BOX)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------- analysis -> row
FIELDS = ["tomo", "psi", "status", "edited", "psi_auto", "psi_fourier", "ridge_conf", "flag",
          "convention", "x1", "y1", "z1", "x2", "y2", "z2",
          "auto_x1", "auto_y1", "auto_x2", "auto_y2", "nx", "ny", "nz", "half_len", "zc", "path"]
NUM = {"psi", "psi_auto", "psi_fourier", "ridge_conf", "x1", "y1", "z1", "x2", "y2", "z2",
       "auto_x1", "auto_y1", "auto_x2", "auto_y2", "nx", "ny", "nz", "half_len", "zc"}


def tomo_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def axis_points(psi_math, cx_b, cy_b, b, shape0, half):
    """Two points on the axis, centred where it passes closest to the tomogram XY centre."""
    nz, ny, nx = shape0
    t = math.radians(psi_math); ux, uy = math.cos(t), math.sin(t)
    ax_, ay_ = cx_b * b + (b - 1) / 2, cy_b * b + (b - 1) / 2
    s = ((nx - 1) / 2 - ax_) * ux + ((ny - 1) / 2 - ay_) * uy
    px, py, z = ax_ + s * ux, ay_ + s * uy, (nz - 1) / 2
    return (px - half * ux, py - half * uy, z), (px + half * ux, py + half * uy, z)


def analyze(path, a):
    """Auto estimate. Returns (row dict, binned projection, edge mask)."""
    vol, apx, shape0 = load_bin(path, a.bin, a.apix)
    mt_px = a.mt_A / apx
    img, zc = project(vol, max(3, int(a.slab_A / apx)), None if a.zc is None else a.zc // a.bin)
    valid = valid_mask(img, mt_px)
    edge = edge_mask(img, valid, mt_px, a.edge_k)
    mask = valid & ~edge
    if mask.sum() < 0.1 * valid.sum():
        edge[:] = False; mask = valid
    psi_r, conf_r, SM = psi_ridge(img, mask, mt_px)
    psi_f, _ = psi_fourier(img, mask, mt_px)
    if np.isfinite(psi_r) and isinstance(SM, tuple):
        cx, cy = axis_center(SM, psi_r, mt_px)
    else:
        psi_r, cx, cy = (psi_f if np.isfinite(psi_f) else 0.0), img.shape[1] / 2, img.shape[0] / 2
    dis = abs(wrap90(psi_r - psi_f)) if np.isfinite(psi_f) else 99
    flag = "" if (dis < 10 and conf_r > 0.15) else "CHECK"
    sgn = -1.0 if a.convention == "relion" else 1.0
    half = a.axis_len_A / (apx / a.bin) / 2 if a.axis_len_A else min(shape0[1:]) / 4
    p1, p2 = axis_points(psi_r, cx, cy, a.bin, shape0, half)
    psi_out = wrap90(sgn * psi_r)
    row = dict(tomo=tomo_name(path), psi=psi_out, status="check" if flag else "auto", edited=0,
               psi_auto=psi_out, psi_fourier=wrap90(sgn * psi_f) if np.isfinite(psi_f) else "",
               ridge_conf=conf_r, flag=flag, convention=a.convention,
               x1=p1[0], y1=p1[1], z1=p1[2], x2=p2[0], y2=p2[1], z2=p2[2],
               auto_x1=p1[0], auto_y1=p1[1], auto_x2=p2[0], auto_y2=p2[1],
               nx=shape0[2], ny=shape0[1], nz=shape0[0], half_len=half, zc=zc * a.bin,
               path=os.path.abspath(path))
    return row, img, edge


# ---------------------------------------------------------------- outputs
class Store:
    """psi_results.csv + imod_axis/ folder. Thread-safe, atomic writes."""

    def __init__(self, csv_path, imod_dir, make_mod):
        self.csv_path, self.imod_dir, self.make_mod = csv_path, imod_dir, make_mod
        self.lock = threading.Lock()
        self.rows = OrderedDict()
        if os.path.exists(csv_path):
            with open(csv_path) as fh:
                for r in csv.DictReader(fh):
                    self.rows[r["tomo"]] = {k: (float(v) if k in NUM and v not in ("", None) else v)
                                            for k, v in r.items()}
        os.makedirs(imod_dir, exist_ok=True)
        if make_mod and not shutil.which("point2model"):
            print("WARNING: point2model not found (load IMOD); writing .txt only"); self.make_mod = False

    def _fmt(self, v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    def save_csv(self):
        tmp = self.csv_path + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader()
            for r in self.rows.values():
                w.writerow({k: self._fmt(r.get(k, "")) for k in FIELDS})
        os.replace(tmp, self.csv_path)

    def write_imod(self, r):
        base = os.path.join(self.imod_dir, f"{r['tomo']}_axis")
        if r["status"] == "rejected":
            for ext in (".txt", ".mod"):
                if os.path.exists(base + ext): os.remove(base + ext)
            return
        tmp = base + ".txt.tmp"
        with open(tmp, "w") as fh:
            fh.write(f"1 {r['x1']:.2f} {r['y1']:.2f} {r['z1']:.2f}\n1 {r['x2']:.2f} {r['y2']:.2f} {r['z2']:.2f}\n")
        os.replace(tmp, base + ".txt")
        if self.make_mod:
            p = subprocess.run(["point2model", "-open", "-circle", "5", "-image", r["path"],
                                base + ".txt", base + ".mod"], capture_output=True, text=True)
            if p.returncode:
                print(f"point2model failed for {r['tomo']}: {p.stderr.strip()[:200]}")

    def put(self, r):
        with self.lock:
            self.rows[r["tomo"]] = r
            self.write_imod(r)
            self.save_csv()


# ---------------------------------------------------------------- run (batch)
def cmd_run(a):
    st = Store(a.csv, a.imod_dir, a.mod)
    panels = []
    for f in a.tomos:
        n = tomo_name(f)
        old = st.rows.get(n)
        reviewed = old and (old.get("status") in ("approved", "rejected") or str(old.get("edited")) in ("1", "1.0"))
        if old and (not a.redo or reviewed):
            print(f"{n:28s} skip ({'reviewed' if reviewed else 'done'})"); continue
        t0 = time.time()
        try:
            r, img, edge = analyze(f, a)
        except Exception as e:
            print(f"{n:28s} FAILED: {e}"); continue
        st.put(r)
        print(f"{n:28s} psi={r['psi']:6.1f}  fourier={r['psi_fourier'] if r['psi_fourier'] == '' else round(r['psi_fourier'], 1)!s:>6}"
              f"  conf={r['ridge_conf']:.2f}  {r['flag']:5s} ({time.time() - t0:.1f}s)")
        if a.montage:
            panels.append((img, r, edge))
    if a.montage and panels:
        write_montage(a.montage, panels, a.bin)
    print(f"-> {a.csv}, {a.imod_dir}/")


def write_montage(path, panels, b):
    n = len(panels); nc = min(5, n); nr = math.ceil(n / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 4.2 * nr), squeeze=False)
    for ax in axs.flat: ax.axis("off")
    for ax, (img, r, edge) in zip(axs.flat, panels):
        pm = math.degrees(math.atan2(r["y2"] - r["y1"], r["x2"] - r["x1"]))
        cx, cy = ((r["x1"] + r["x2"]) / 2 - (b - 1) / 2) / b, ((r["y1"] + r["y2"]) / 2 - (b - 1) / 2) / b
        draw(ax, img, pm, cx, cy, f"{r['tomo']}\npsi={r['psi']:.1f}° {r['flag']}",
             color="#ffcc00" if r["flag"] else "#ff3b30", edge=edge)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------- review (server)
class Reviewer:
    def __init__(self, a):
        self.a = a
        self.paths = OrderedDict((tomo_name(f), f) for f in a.tomos)
        self.names = list(self.paths)
        self.store = Store(a.csv, a.imod_dir, a.mod)
        self.tlocks = {n: threading.Lock() for n in self.names}
        self.cache, self.clock = OrderedDict(), threading.Lock()
        self.pool = ThreadPoolExecutor(2)
        self.pending = {n for n in self.names if n not in self.store.rows}
        self.errors = {}

    def meta(self, i):
        n = self.names[i]; r = self.store.rows.get(n)
        if r is None:
            return dict(name=n, ready=False, error=self.errors.get(n))
        g = lambda k: float(r[k])
        return dict(name=n, ready=True, status=r["status"], edited=str(r["edited"]) in ("1", "1.0"),
                    p1=[g("x1"), g("y1")], p2=[g("x2"), g("y2")],
                    auto_p1=[g("auto_x1"), g("auto_y1")], auto_p2=[g("auto_x2"), g("auto_y2")],
                    psi_auto=g("psi_auto"),
                    psi_fourier=None if r["psi_fourier"] in ("", None) else g("psi_fourier"),
                    conf=g("ridge_conf"), flag=r["flag"] or "", z=g("z1"), half=g("half_len"),
                    shape=[int(g("nz")), int(g("ny")), int(g("nx"))])

    def image(self, n):
        """JPEG of the projection; runs the auto analysis first if the tomogram is new."""
        with self.clock:
            if n in self.cache:
                self.cache.move_to_end(n); return self.cache[n]
        with self.tlocks[n]:
            with self.clock:
                if n in self.cache: return self.cache[n]
            f, a = self.paths[n], self.a
            if n not in self.store.rows:
                try:
                    r, img, _ = analyze(f, a)
                except Exception as e:
                    self.errors[n] = str(e); self.pending.discard(n); raise
                self.store.put(r); self.pending.discard(n)
                print(f"  auto {n}: psi={r['psi']:.1f} {r['flag']}")
            else:
                vol, apx, _ = load_bin(f, a.bin, a.apix)
                img, _ = project(vol, max(3, int(a.slab_A / apx)), int(float(self.store.rows[n]["zc"])) // a.bin)
            jpg = encode_jpeg(img, a.img_max)
            with self.clock:
                self.cache[n] = jpg
                while len(self.cache) > 40: self.cache.popitem(last=False)
            return jpg

    def background(self):
        for n in list(self.names):
            if n in self.pending:
                try: self.image(n)
                except Exception as e: print(f"  FAILED {n}: {e}")

    def save(self, d):
        n = d["name"]; r = dict(self.store.rows[n])
        (x1, y1), (x2, y2) = d["p1"], d["p2"]
        sgn = -1.0 if r["convention"] == "relion" else 1.0
        r.update(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                 psi=wrap90(sgn * math.degrees(math.atan2(y2 - y1, x2 - x1))),
                 status=d["status"] if d["status"] in ("auto", "check", "approved", "rejected") else "auto",
                 edited=1 if d.get("edited") else 0)
        self.store.put(r)
        return r


def make_handler(rv, token, page):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def _ok(self, body, ctype="application/json", code=200):
            if isinstance(body, (dict, list)): body = json.dumps(body).encode()
            elif isinstance(body, str): body = body.encode()
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def _auth(self):
            from urllib.parse import urlparse, parse_qs
            u = urlparse(self.path)
            if parse_qs(u.query).get("t", [""])[0] != token:
                self._ok("forbidden: open the URL printed by the server", "text/plain", 403); return None
            return u.path

        def do_GET(self):
            if self.path.startswith("/favicon"):
                self.send_response(204); self.end_headers(); return
            p = self._auth()
            if p is None: return
            try:
                if p == "/":
                    self._ok(page, "text/html; charset=utf-8")
                elif p == "/api/list":
                    self._ok(dict(convention=rv.a.convention, csv=os.path.abspath(rv.a.csv),
                                  pending=len(rv.pending),
                                  tomos=[rv.meta(i) for i in range(len(rv.names))]))
                elif p.startswith("/api/tomo/"):
                    self._ok(rv.meta(int(p.rsplit("/", 1)[1])))
                elif p.startswith("/api/img/"):
                    i = int(p.rsplit("/", 1)[1])
                    jpg = rv.image(rv.names[i])
                    for j in (i + 1, i + 2):                       # prefetch
                        if j < len(rv.names): rv.pool.submit(rv.image, rv.names[j])
                    self._ok(jpg, "image/jpeg")
                else:
                    self._ok("not found", "text/plain", 404)
            except Exception as e:
                self._ok(f"error: {e}", "text/plain", 500)

        def do_POST(self):
            p = self._auth()
            if p is None: return
            try:
                if p == "/api/save":
                    d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    r = rv.save(d)
                    self._ok(dict(ok=True, psi=r["psi"]))
                else:
                    self._ok("not found", "text/plain", 404)
            except Exception as e:
                self._ok(f"error: {e}", "text/plain", 500)
    return H


def cmd_review(a):
    rv = Reviewer(a)
    token = secrets.token_urlsafe(12)
    srv = None
    for port in range(a.port, a.port + 20):
        try:
            srv = ThreadingHTTPServer((a.host, port), make_handler(rv, token, REVIEW_TEMPLATE)); break
        except OSError:
            continue
    if srv is None:
        sys.exit(f"no free port in {a.port}-{a.port + 19}")
    port = srv.server_address[1]
    host = socket.getfqdn()
    if host.startswith("localhost") or "." not in host:
        host = socket.gethostname()
    user = os.environ.get("USER", "you")
    print(f"\n{len(rv.names)} tomograms, {len(rv.pending)} not yet analysed (computing in background)")
    print(f"results: {os.path.abspath(a.csv)}   IMOD points: {os.path.abspath(a.imod_dir)}/\n")
    print(f"On your laptop run:\n    ssh -N -L {port}:localhost:{port} {user}@{host}")
    print(f"then open:\n    http://localhost:{port}/?t={token}\n\nCtrl-C to stop (everything is already saved).")
    if rv.pending:
        threading.Thread(target=rv.background, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


# ---------------------------------------------------------------- CLI
def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("tomos", nargs="+")
    common.add_argument("--csv", default="psi_results.csv")
    common.add_argument("--imod-dir", default="imod_axis", help="folder for <tomo>_axis.txt")
    common.add_argument("--mod", action="store_true", help="also run point2model -> <tomo>_axis.mod")
    common.add_argument("--bin", type=int, default=2)
    common.add_argument("--apix", type=float, default=None, help="override pixel size (A, unbinned)")
    common.add_argument("--slab-A", type=float, default=800, help="projection slab thickness (A)")
    common.add_argument("--zc", type=int, default=None, help="slab centre z (unbinned px); default auto")
    common.add_argument("--mt-A", type=float, default=250, help="microtubule width (A)")
    common.add_argument("--edge-k", type=float, default=4.0, help="carbon-edge mask sensitivity (lower = more)")
    common.add_argument("--axis-len-A", type=float, default=None,
                        help="distance between the 2 IMOD points (A); default 1/2 smaller XY size")
    common.add_argument("--convention", choices=["relion", "math"], default="relion")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", parents=[common], help="batch auto estimate (no GUI)")
    r.add_argument("--redo", action="store_true", help="recompute tomograms not yet reviewed")
    r.add_argument("--montage", default=None, metavar="PNG", help="optional overview image")
    v = sub.add_parser("review", parents=[common], help="web review via ssh tunnel")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--host", default="127.0.0.1", help="bind address (keep 127.0.0.1 + ssh tunnel)")
    v.add_argument("--img-max", type=int, default=1200, help="max projection size sent to browser (px)")
    argv = sys.argv[1:]
    if argv and argv[0] not in ("run", "review", "-h", "--help"):
        argv = ["run"] + argv                                   # old-style call
    a = p.parse_args(argv)
    (cmd_run if a.cmd == "run" else cmd_review)(a)


REVIEW_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Centriole axis review</title>
<style>
:root{--bg:#121417;--panel:#1b1e23;--line:#2a2f37;--text:#e6e8eb;--dim:#8b939e;--red:#ff4d42;--blue:#4ab3ff;--green:#2ee59d;--amber:#ffc53d;--violet:#b38cff}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
#app{display:grid;grid-template-columns:260px 1fr 270px;height:100vh}
aside,#side{background:var(--panel);display:flex;flex-direction:column;min-height:0}
aside{border-right:1px solid var(--line)} #side{border-left:1px solid var(--line);padding:14px;gap:14px;overflow:auto}
header{padding:12px 14px;border-bottom:1px solid var(--line)}
h1{font-size:14px;margin:0 0 6px;font-weight:600}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;display:flex}
.bar span{display:block;height:100%}
#prog{color:var(--dim);font-size:12px;margin-top:6px}
select,button,input[type=file]{font:inherit;color:var(--text);background:#252a31;border:1px solid var(--line);border-radius:6px;padding:5px 9px}
button{cursor:pointer} button:hover{border-color:#4a5260}
button.primary{background:#1f6f4c;border-color:#2a8f63} button.danger{background:#5a2323;border-color:#7a3030}
#filter{margin:10px 14px;width:calc(100% - 28px)}
#list{overflow:auto;flex:1}
.item{display:flex;align-items:center;gap:8px;padding:6px 14px;cursor:pointer;border-left:3px solid transparent}
.item:hover{background:#22262c} .item.cur{background:#262b33;border-left-color:var(--blue)}
.dot{width:9px;height:9px;border-radius:50%;flex:none}
.item .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,Menlo,monospace;font-size:12px}
.item .ps{color:var(--dim);font-variant-numeric:tabular-nums;font-size:12px}
main{position:relative;min-width:0;min-height:0}
canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
#title{position:absolute;left:12px;top:10px;font-family:ui-monospace,Menlo,monospace;background:#000a;padding:4px 8px;border-radius:6px;pointer-events:none}
.big{font-size:34px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.5px}
.lbl{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.6px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;font-variant-numeric:tabular-nums}
.kv div:nth-child(odd){color:var(--dim)}
.row{display:flex;gap:6px;flex-wrap:wrap}
.row button{flex:1}
.status{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600}
input[type=range]{width:100%}
.keys{color:var(--dim);font-size:12px;display:grid;grid-template-columns:auto 1fr;gap:2px 10px}
kbd{font:11px ui-monospace,Menlo,monospace;background:#2a2f37;border:1px solid #3a414c;border-bottom-width:2px;border-radius:4px;padding:0 5px;color:var(--text)}
label.ck{display:flex;gap:6px;align-items:center;color:var(--dim)}
hr{border:0;border-top:1px solid var(--line);margin:0}
#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#2a8f63;color:#fff;padding:8px 14px;border-radius:8px;opacity:0;transition:opacity .2s;pointer-events:none}
</style>
</head>
<body>
<div id="app">
  <aside>
    <header>
      <h1>Centriole axis review</h1>
      <div class="bar" id="bar"></div>
      <div id="prog"></div>
    </header>
    <select id="filter">
      <option value="all">All tomograms</option>
      <option value="todo">Not yet reviewed</option>
      <option value="check">Flagged CHECK</option>
      <option value="edited">Edited</option>
      <option value="rejected">Rejected</option>
    </select>
    <div id="list"></div>
  </aside>
  <main>
    <canvas id="cv"></canvas>
    <div id="title"></div>
  </main>
  <div id="side">
    <div>
      <div class="lbl">ψ (<span id="conv"></span>)</div>
      <div class="big" id="psi">–</div>
      <div id="st"></div>
    </div>
    <div class="kv" id="kv"></div>
    <div class="row">
      <button class="primary" id="bApprove" title="Enter">Approve ↵</button>
      <button class="danger" id="bReject" title="X">Reject</button>
    </div>
    <div class="row">
      <button id="bReset" title="R">Reset to auto</button>
      <button id="bUseF" title="U">Use Fourier ψ</button>
    </div>
    <hr>
    <div>
      <div class="lbl">Contrast</div>
      <input type="range" id="con" min="0.5" max="4" step="0.05" value="1.1">
      <div class="lbl">Brightness</div>
      <input type="range" id="bri" min="0.5" max="1.8" step="0.02" value="1">
      <label class="ck"><input type="checkbox" id="yup" checked> Y axis up (IMOD view)</label>
      <label class="ck"><input type="checkbox" id="showF" checked> Show Fourier estimate</label>
      <label class="ck"><input type="checkbox" id="showAuto" checked> Show auto line when edited</label>
    </div>
    <hr>
    <div id="save" class="lbl" style="text-transform:none;letter-spacing:0;font-size:12px">connected</div>
    <div class="keys">
      <kbd>drag ●</kbd><span>move endpoint</span>
      <kbd>drag line</kbd><span>shift the axis</span>
      <kbd>shift+drag</kbd><span>draw a new axis</span>
      <kbd>[ ]</kbd><span>rotate 1° (+alt: 0.2°)</span>
      <kbd>wheel</kbd><span>zoom · right-drag pan</span>
      <kbd>↵ / A</kbd><span>approve &amp; next</span>
      <kbd>X</kbd><span>reject &amp; next</span>
      <kbd>← →</kbd><span>previous / next</span>
      <kbd>R</kbd><span>reset · <kbd>U</kbd> use Fourier</span>
      <kbd>0</kbd><span>fit view</span>
    </div>
  </div>
</div>
<div id="toast"></div>
<script>
const TOKEN = new URLSearchParams(location.search).get("t") || "";
const api = p => p + (p.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(TOKEN);
let DATA = null, T = [], state = [], SGN = -1;
const COL = {auto:"#8b939e", check:"#ffc53d", edited:"#b38cff", approved:"#2ee59d", rejected:"#ff4d42", pending:"#3a414c"};
const ready = i => T[i] && T[i].ready;
function adopt(i, m) {        // merge server meta for tomogram i
  T[i] = Object.assign(T[i] || {}, m);
  if (m.ready) state[i] = {p1: m.p1, p2: m.p2, status: m.status, edited: !!m.edited};
}
async function fetchList() {
  const r = await fetch(api("/api/list")); if (!r.ok) throw new Error(r.status);
  DATA = await r.json(); SGN = DATA.convention === "relion" ? -1 : 1;
  DATA.tomos.forEach((m, i) => { if (!ready(i) || !T[i]) adopt(i, m); });
  return DATA.pending;
}
let saveTimer = null;
async function save(i = cur) {
  if (!ready(i)) return; const s = state[i];
  $("save").textContent = "saving…"; $("save").style.color = "var(--dim)";
  try {
    const r = await fetch(api("/api/save"), {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: T[i].name, p1: s.p1, p2: s.p2, status: s.status, edited: s.edited})});
    if (!r.ok) throw new Error(await r.text());
    $("save").textContent = "✓ saved to " + DATA.csv; $("save").style.color = "var(--green)";
  } catch (e) { $("save").textContent = "✗ NOT saved: " + e.message; $("save").style.color = "var(--red)"; }
}
const $ = id => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
let cur = 0, imgs = {}, view = null, drag = null;

// ---------- geometry (all stored points: unbinned px, 0-based, IMOD) ----------
const wrap90 = a => { a = ((a + 90) % 180 + 180) % 180 - 90; return a <= -90 ? 90 : a; };
const psiOf = s => wrap90(SGN * Math.atan2(s.p2[1] - s.p1[1], s.p2[0] - s.p1[0]) * 180 / Math.PI);
function normalize(t, s) {           // centre on the axis point closest to tomo centre, fixed length
  let ux = s.p2[0] - s.p1[0], uy = s.p2[1] - s.p1[1], L = Math.hypot(ux, uy);
  if (L < 1e-6) return; ux /= L; uy /= L;
  if (ux < 0 || (ux === 0 && uy > 0)) { ux = -ux; uy = -uy; }
  const cx = (t.shape[2] - 1) / 2, cy = (t.shape[1] - 1) / 2;
  const k = (cx - s.p1[0]) * ux + (cy - s.p1[1]) * uy, mx = s.p1[0] + k * ux, my = s.p1[1] + k * uy;
  const h = t.half;
  s.p1 = [mx - h * ux, my - h * uy]; s.p2 = [mx + h * ux, my + h * uy];
}
function setPsi(t, s, psi) {           // rotate about midpoint to a given output psi
  const m = [(s.p1[0] + s.p2[0]) / 2, (s.p1[1] + s.p2[1]) / 2], a = SGN * psi * Math.PI / 180;
  s.p1 = [m[0] - t.half * Math.cos(a), m[1] - t.half * Math.sin(a)];
  s.p2 = [m[0] + t.half * Math.cos(a), m[1] + t.half * Math.sin(a)];
  normalize(t, s);
}
// unbinned <-> display-image px  <-> canvas px
const u2d = (t, p) => [(p[0] + .5) / t.sx - .5, (p[1] + .5) / t.sy - .5];
const d2u = (t, p) => [(p[0] + .5) * t.sx - .5, (p[1] + .5) * t.sy - .5];
const yup = () => $("yup").checked;
function d2c(t, p) { return [view.ox + (p[0] + .5) * view.z, yup() ? view.oy + (t.h - p[1] - .5) * view.z : view.oy + (p[1] + .5) * view.z]; }
function c2d(t, q) { const x = (q[0] - view.ox) / view.z - .5, yy = (q[1] - view.oy) / view.z; return [x, yup() ? t.h - yy - .5 : yy - .5]; }
const u2c = (t, p) => d2c(t, u2d(t, p)), c2u = (t, q) => d2u(t, c2d(t, q));

function dims(t, im) { t.w = im.naturalWidth; t.h = im.naturalHeight; t.sx = t.shape[2] / t.w; t.sy = t.shape[1] / t.h; }
function fit() {
  const t = T[cur]; if (!t.w) { t.w = t.shape ? t.shape[2] : 1000; t.h = t.shape ? t.shape[1] : 1000; t.sx = t.sy = 1; }
  const W = cv.clientWidth, H = cv.clientHeight, z = Math.min(W / t.w, H / t.h) * 0.96;
  view = {z, ox: (W - t.w * z) / 2, oy: (H - t.h * z) / 2};
}

// ---------- drawing ----------
function lineAcross(t, p1, p2, color, width, dash) {
  const a = u2c(t, p1), b = u2c(t, p2), dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1;
  const E = 4000, ux = dx / L, uy = dy / L;
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
  ctx.beginPath(); ctx.moveTo(a[0] - ux * E, a[1] - uy * E); ctx.lineTo(b[0] + ux * E, b[1] + uy * E); ctx.stroke(); ctx.restore();
}
function draw() {
  const t = T[cur], s = state[cur], dpr = window.devicePixelRatio || 1;
  if (cv.width !== cv.clientWidth * dpr || cv.height !== cv.clientHeight * dpr) {
    cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0b0c0e"; ctx.fillRect(0, 0, cv.clientWidth, cv.clientHeight);
  const im = imgs[cur];
  if (!ready(cur) || !im || !im.complete || !im.naturalWidth) {
    ctx.fillStyle = "#8b939e"; ctx.font = "15px system-ui";
    ctx.fillText(im && im.failed ? "Failed to load: " + im.failed : "Computing projection on the server…", 24, 60); return;
  }
  if (im && im.complete) {
    ctx.save();
    ctx.filter = `contrast(${$("con").value}) brightness(${$("bri").value})`;
    ctx.imageSmoothingEnabled = view.z < 2;
    if (yup()) { ctx.translate(view.ox, view.oy + t.h * view.z); ctx.scale(view.z, -view.z); }
    else { ctx.translate(view.ox, view.oy); ctx.scale(view.z, view.z); }
    ctx.drawImage(im, 0, 0); ctx.restore();
  }
  // clip overlays to image area
  ctx.save(); ctx.beginPath(); ctx.rect(view.ox, view.oy, t.w * view.z, t.h * view.z); ctx.clip();
  if ($("showF").checked && t.psi_fourier !== null) {
    const ts = {p1:[...s.p1], p2:[...s.p2]}; setPsi(t, ts, t.psi_fourier);
    lineAcross(t, ts.p1, ts.p2, "#4ab3ff", 1.2, [6, 5]);
  }
  if (s.edited && $("showAuto").checked) lineAcross(t, t.auto_p1, t.auto_p2, "#ff4d4299", 1, [2, 4]);
  const c = s.status === "rejected" ? "#ff4d42" : s.edited ? "#b38cff" : s.status === "approved" ? "#2ee59d" : "#ff4d42";
  lineAcross(t, s.p1, s.p2, c, 2);
  if (drag && drag.mode === "draw") {
    ctx.save(); ctx.strokeStyle = "#fff"; ctx.setLineDash([4, 3]); ctx.beginPath();
    ctx.moveTo(...drag.c0); ctx.lineTo(...drag.c1); ctx.stroke(); ctx.restore();
  }
  ctx.restore();
  [s.p1, s.p2].forEach((p, i) => {
    const q = u2c(t, p);
    ctx.beginPath(); ctx.arc(q[0], q[1], 8, 0, 7); ctx.fillStyle = "#0009"; ctx.fill();
    ctx.lineWidth = 2.5; ctx.strokeStyle = "#2ee59d"; ctx.stroke();
    ctx.fillStyle = "#2ee59d"; ctx.font = "bold 12px system-ui"; ctx.fillText("P" + (i + 1), q[0] + 11, q[1] - 9);
  });
  if (s.status === "rejected") {
    ctx.fillStyle = "#ff4d42"; ctx.font = "bold 28px system-ui"; ctx.fillText("REJECTED", view.ox + 14, view.oy + t.h * view.z - 16);
  }
}

// ---------- side panel / list ----------
function stLabel(s) { if (!s) return "pending"; return s.status === "approved" ? (s.edited ? "approved (edited)" : "approved") : s.edited ? "edited" : s.status; }
function stColor(s) { if (!s) return COL.pending; return s.status === "approved" ? COL.approved : s.status === "rejected" ? COL.rejected : s.edited ? COL.edited : COL[s.status]; }
function panel() {
  $("conv").textContent = SGN < 0 ? "RELION" : "atan2";
  if (!ready(cur)) { $("psi").textContent = "…"; $("st").innerHTML = ""; $("kv").innerHTML = ""; $("title").textContent = `${cur + 1}/${T.length}  ${T[cur].name}`; return; }
  const t = T[cur], s = state[cur], p = psiOf(s), d = wrap90(p - t.psi_auto);
  $("psi").textContent = (s.status === "rejected" ? "— " : "") + p.toFixed(1) + "°";
  $("st").innerHTML = `<span class="status" style="background:${stColor(s)}22;color:${stColor(s)}">${stLabel(s)}</span>`;
  $("kv").innerHTML = [
    ["auto ψ", t.psi_auto.toFixed(1) + "°"], ["Δ from auto", (d >= 0 ? "+" : "") + d.toFixed(1) + "°"],
    ["Fourier ψ", t.psi_fourier === null ? "–" : t.psi_fourier.toFixed(1) + "°"],
    ["ridge conf", t.conf.toFixed(2)], ["auto flag", t.flag || "ok"],
    ["P1", s.p1.map(v => v.toFixed(1)).join(", ") + ", " + t.z.toFixed(1)],
    ["P2", s.p2.map(v => v.toFixed(1)).join(", ") + ", " + t.z.toFixed(1)],
    ["size", `${t.shape[2]}×${t.shape[1]}×${t.shape[0]}`]
  ].map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join("");
  $("title").textContent = `${cur + 1}/${T.length}  ${t.name}`;
}
function visible(i) {
  const f = $("filter").value, s = state[i] || {status: "pending"};
  return f === "all" || (f === "todo" && s.status !== "approved" && s.status !== "rejected") ||
    (f === "check" && T[i].flag) || (f === "edited" && s.edited) || (f === "rejected" && s.status === "rejected");
}
function list() {
  const L = $("list"); L.innerHTML = "";
  T.forEach((t, i) => {
    if (!visible(i)) return;
    const s = state[i], el = document.createElement("div");
    el.className = "item" + (i === cur ? " cur" : "");
    el.innerHTML = `<span class="dot" style="background:${stColor(s)}"></span><span class="nm">${t.name}</span><span class="ps">${s ? psiOf(s).toFixed(1) + "°" : "…"}</span>`;
    el.onclick = () => go(i); L.appendChild(el);
  });
  const n = T.length, c = k => state.filter(x => x && k(x)).length;
  const ap = c(s => s.status === "approved"), rj = c(s => s.status === "rejected");
  $("bar").innerHTML = `<span style="width:${ap / n * 100}%;background:${COL.approved}"></span><span style="width:${rj / n * 100}%;background:${COL.rejected}"></span>`;
  $("prog").textContent = `${ap} approved · ${rj} rejected · ${n - ap - rj} to review · ${c(s => s.edited)} edited` + (DATA.pending ? ` · ${DATA.pending} still computing` : "");
  const ci = L.querySelector(".cur"); if (ci) ci.scrollIntoView({block: "nearest"});
}
function refresh() { draw(); panel(); list(); }
function load(i) {
  if (i < 0 || i >= T.length || imgs[i]) return;
  const im = new Image();
  im.onload = async () => {
    if (!ready(i)) { const r = await fetch(api(`/api/tomo/${i}`)); adopt(i, await r.json()); }
    dims(T[i], im); if (i === cur) { fit(); refresh(); } else list();
  };
  im.onerror = () => { im.failed = "see server log"; delete imgs[i]; if (i === cur) draw(); };
  im.src = api(`/api/img/${i}`); imgs[i] = im;
}
function go(i) {
  cur = Math.max(0, Math.min(T.length - 1, i)); fit(); load(cur); load(cur + 1); refresh();
}
function nextTodo() {
  for (let k = 1; k <= T.length; k++) { const i = (cur + k) % T.length;
    if (visible(i) && (!state[i] || (state[i].status !== "approved" && state[i].status !== "rejected"))) return go(i); }
  go(Math.min(cur + 1, T.length - 1));
}
function mark(st) { if (!ready(cur)) return; state[cur].status = st; save(); nextTodo(); }
function edited() { const s = state[cur]; s.edited = true; if (s.status === "approved" || s.status === "rejected") s.status = T[cur].flag ? "check" : "auto"; }

// ---------- mouse ----------
const pos = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
function distToLine(q, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1;
  return Math.abs((q[0] - a[0]) * dy - (q[1] - a[1]) * dx) / L;
}
cv.addEventListener("contextmenu", e => e.preventDefault());
cv.addEventListener("mousedown", e => {
  if (!ready(cur) || !T[cur].sx) return;
  const t = T[cur], s = state[cur], q = pos(e);
  if (e.button === 2 || e.button === 1) { drag = {mode: "pan", q, ox: view.ox, oy: view.oy}; return; }
  if (e.shiftKey) { drag = {mode: "draw", c0: q, c1: q}; return; }
  const a = u2c(t, s.p1), b = u2c(t, s.p2);
  if (Math.hypot(q[0] - a[0], q[1] - a[1]) < 12) drag = {mode: "p1"};
  else if (Math.hypot(q[0] - b[0], q[1] - b[1]) < 12) drag = {mode: "p2"};
  else if (distToLine(q, a, b) < 8) drag = {mode: "move", u0: c2u(t, q), p1: [...s.p1], p2: [...s.p2]};
  else drag = {mode: "pan", q, ox: view.ox, oy: view.oy};
});
window.addEventListener("mousemove", e => {
  if (!ready(cur) || !T[cur].sx) return;
  const t = T[cur], s = state[cur], q = pos(e);
  if (!drag) {
    const a = u2c(t, s.p1), b = u2c(t, s.p2);
    cv.style.cursor = e.shiftKey ? "crosshair" : (Math.hypot(q[0] - a[0], q[1] - a[1]) < 12 || Math.hypot(q[0] - b[0], q[1] - b[1]) < 12) ? "grab" : distToLine(q, a, b) < 8 ? "move" : "default";
    return;
  }
  if (drag.mode === "pan") { view.ox = drag.ox + q[0] - drag.q[0]; view.oy = drag.oy + q[1] - drag.q[1]; draw(); return; }
  if (drag.mode === "draw") { drag.c1 = q; draw(); return; }
  const u = c2u(t, q);
  if (drag.mode === "p1") s.p1 = u; else if (drag.mode === "p2") s.p2 = u;
  else { const dx = u[0] - drag.u0[0], dy = u[1] - drag.u0[1];
    s.p1 = [drag.p1[0] + dx, drag.p1[1] + dy]; s.p2 = [drag.p2[0] + dx, drag.p2[1] + dy]; }
  s.edited = true; draw(); panel();
});
window.addEventListener("mouseup", () => {
  if (!drag) return;
  const t = T[cur], s = state[cur];
  if (drag.mode === "draw") {
    if (Math.hypot(drag.c1[0] - drag.c0[0], drag.c1[1] - drag.c0[1]) > 10) { s.p1 = c2u(t, drag.c0); s.p2 = c2u(t, drag.c1); edited(); normalize(t, s); }
  } else if (drag.mode !== "pan") { edited(); normalize(t, s); }
  const m = drag.mode; drag = null; if (m !== "pan") save(); refresh();
});
cv.addEventListener("wheel", e => {
  e.preventDefault(); const q = pos(e), f = Math.exp(-e.deltaY * 0.0015);
  view.ox = q[0] - (q[0] - view.ox) * f; view.oy = q[1] - (q[1] - view.oy) * f; view.z *= f; draw();
}, {passive: false});

// ---------- keys / buttons ----------
function rotate(d) { if (!ready(cur)) return; const t = T[cur], s = state[cur]; setPsi(t, s, psiOf(s) + d); edited(); refresh();
  clearTimeout(saveTimer); saveTimer = setTimeout(() => save(), 400); }
function reset() { if (!ready(cur)) return; const t = T[cur]; state[cur] = {p1: [...t.auto_p1], p2: [...t.auto_p2], status: t.flag ? "check" : "auto", edited: false}; save(); refresh(); }
function useF() { const t = T[cur]; if (!ready(cur) || t.psi_fourier === null) return; setPsi(t, state[cur], t.psi_fourier); edited(); save(); refresh(); }
window.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox") return;
  const k = e.key;
  if (k === "ArrowRight" || k === "j") go(cur + 1);
  else if (k === "ArrowLeft" || k === "k") go(cur - 1);
  else if (k === "Enter" || k === "a") mark("approved");
  else if (k === "x") mark("rejected");
  else if (k === "r") reset();
  else if (k === "u") useF();
  else if (k === "0") { fit(); draw(); }
  else if (k === "[" || k === "]") rotate((k === "]" ? 1 : -1) * (e.altKey ? 0.2 : 1));
  else return;
  e.preventDefault();
});
$("bApprove").onclick = () => mark("approved"); $("bReject").onclick = () => mark("rejected");
$("bReset").onclick = reset; $("bUseF").onclick = useF;
["con", "bri", "showF", "showAuto"].forEach(id => $(id).oninput = draw);
$("yup").onchange = () => { draw(); };
$("filter").onchange = list;
window.addEventListener("resize", () => { fit(); draw(); });

function toast(m) { const t = $("toast"); t.textContent = m; t.style.opacity = 1; setTimeout(() => t.style.opacity = 0, 2200); }

(async () => {
  try { await fetchList(); } catch (e) { document.body.innerHTML = "<p style='padding:30px'>Cannot reach server (wrong or missing token?). Use the URL printed by the server.</p>"; return; }
  go(0);
  const poll = async () => {               // pick up tomograms finished by the background worker
    if (!DATA.pending) return;
    try { await fetchList(); list(); if (!ready(cur)) { delete imgs[cur]; load(cur); } } catch (e) {}
    setTimeout(poll, 3000);
  };
  setTimeout(poll, 3000);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
