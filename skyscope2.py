"""
skyscope2.py -- terapixel-class imagery: view it, search it, find what changed.
Pure NumPy. Synthetic sky (no Roman data exists in this sandbox).

UPGRADES over v1, each grounded in what real pipelines do:

  A. VIEWING   tile pyramid (planar analog of astronomy's HiPS scheme, which
               maps surveys onto hierarchical HEALPix tiles; full HiPS needs
               spherical geometry -- out of scope for a flat synthetic sky).

  B. SEARCH    morphology features (multi-scale DoG, image moments,
               concentration index, gradient-orientation histogram, thumbnail)
               instead of random projections; quantized with REAL quantvec
               machinery: random orthogonal rotation -> per-dim Lloyd-Max
               codebooks -> 4-bit packed codes -> ADC lookup-table search.

  C. ANOMALY   kNN-isolation in embedding space (rare morphology = far from
               neighbours), tested on planted dim streaks that brightness
               ranking cannot find.

  D. CHANGE    ZOGY-lite difference imaging: two epochs, subtract, matched-
               filter with the PSF, threshold the S/N map at 5 sigma. This is
               the de-facto standard for transient detection. Transient
               brightness is deliberately made to OVERLAP bright stars so the
               v1 brightness hack fails and the real technique has to win.
"""

import numpy as np

RNG = np.random.default_rng(0)
H = W = 4096
PATCH = 64
NP_SIDE = H // PATCH                       # 64x64 grid of patches
N_PATCH = NP_SIDE * NP_SIDE
PSF_SIGMA = 1.3

# ============================================================ synthetic sky
def gauss_kernel(sigma, half):
    ax = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return k

def stamp(img, y, x, kernel, amp):
    h = kernel.shape[0] // 2
    y0, y1 = max(0, y - h), min(img.shape[0], y + h + 1)
    x0, x1 = max(0, x - h), min(img.shape[1], x + h + 1)
    ky0 = y0 - (y - h); kx0 = x0 - (x - h)
    img[y0:y1, x0:x1] += amp * kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]

def add_streak(img, y, x, length, amp, angle):
    """Dim diagonal streak: satellite-trail / cosmic-ray analog."""
    t = np.arange(-length // 2, length // 2)
    ys = np.clip((y + t * np.sin(angle)).astype(int), 0, img.shape[0] - 1)
    xs = np.clip((x + t * np.cos(angle)).astype(int), 0, img.shape[1] - 1)
    for dy in (-1, 0, 1):                       # ~2-3 px wide
        yy = np.clip(ys + dy, 0, img.shape[0] - 1)
        img[yy, xs] += amp * (0.5 if dy else 1.0)

def make_sky(seed=0):
    rng = np.random.default_rng(seed)
    static = np.zeros((H, W), np.float32)
    star_k, gal_k = gauss_kernel(PSF_SIGMA, 5), gauss_kernel(5.0, 16)

    star_amps = rng.uniform(0.15, 0.80, 4000)              # bright tail!
    for a in star_amps:
        stamp(static, rng.integers(0, H), rng.integers(0, W), star_k, a)

    gal_c = []
    for _ in range(120):
        y, x = int(rng.integers(24, H - 24)), int(rng.integers(24, W - 24))
        stamp(static, y, x, gal_k, rng.uniform(0.20, 0.50))
        gal_c.append((y, x))

    streak_c = []
    for _ in range(5):
        y, x = int(rng.integers(40, H - 40)), int(rng.integers(40, W - 40))
        add_streak(static, y, x, 40, 0.30, rng.uniform(0, np.pi))
        streak_c.append((y, x))

    # two epochs: same statics, independent noise; transients ONLY in epoch 2,
    # amplitudes 0.5-0.9 -> overlap the bright-star range (0.15-0.80).
    noise = lambda: rng.normal(0.015, 0.008, (H, W)).astype(np.float32)
    ref = np.clip(static + noise(), 0, None)
    sci = static.copy()
    tr_c = []
    for _ in range(6):
        y, x = int(rng.integers(24, H - 24)), int(rng.integers(24, W - 24))
        stamp(sci, y, x, star_k, rng.uniform(0.50, 0.90))
        tr_c.append((y, x))
    sci = np.clip(sci + noise(), 0, None)
    return ref, sci, np.array(gal_c), np.array(streak_c), np.array(tr_c)

# ============================================================ A. pyramid
def build_pyramid(img, tile=256):
    levels = [img]
    while min(levels[-1].shape) > tile:
        a = levels[-1]
        h2, w2 = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
        levels.append(a[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2)
                      .mean((1, 3)).astype(np.float32))
    return levels

# ============================================================ B. features
def fft_blur(img, sigma):
    """Gaussian blur of the WHOLE image via FFT -- one pass, all patches."""
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.rfftfreq(img.shape[1])[None, :]
    g = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (fy ** 2 + fx ** 2))
    return np.fft.irfft2(np.fft.rfft2(img) * g, s=img.shape).astype(np.float32)

def to_patches(img):
    return img.reshape(NP_SIDE, PATCH, NP_SIDE, PATCH).transpose(0, 2, 1, 3) \
              .reshape(N_PATCH, PATCH, PATCH)

def patch_coords():
    g = np.arange(NP_SIDE) * PATCH
    yy, xx = np.meshgrid(g, g, indexing="ij")
    return np.stack([yy.ravel(), xx.ravel()], 1).astype(np.int32)

def morphology_features(img):
    """Per-patch features that encode SHAPE, not just brightness."""
    P = to_patches(img)                                   # (N,64,64)
    med = np.median(P.reshape(N_PATCH, -1), 1)[:, None, None]
    Pb = P - med                                          # background-subtract
    feats = []

    # multi-scale DoG band energies (point-like vs extended vs streak scale)
    blurs = {s: to_patches(fft_blur(img, s)) for s in (1, 2, 4, 8)}
    for a, b in ((1, 2), (2, 4), (4, 8)):
        dog = blurs[a] - blurs[b]
        feats += [np.abs(dog).mean((1, 2)), dog.max((1, 2))]

    # intensity stats
    flat = Pb.reshape(N_PATCH, -1)
    feats += [flat.mean(1), flat.std(1), flat.max(1)]

    # concentration: central 16x16 flux / total positive flux
    pos = np.clip(Pb, 0, None)
    c = PATCH // 2
    feats.append(pos[:, c-8:c+8, c-8:c+8].sum((1, 2)) / (pos.sum((1, 2)) + 1e-6))

    # central second moments around patch centre -> size + anisotropy
    ax = np.arange(PATCH) - (PATCH - 1) / 2
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    m = pos.sum((1, 2)) + 1e-6
    mu20 = (pos * yy ** 2).sum((1, 2)) / m
    mu02 = (pos * xx ** 2).sum((1, 2)) / m
    mu11 = (pos * yy * xx).sum((1, 2)) / m
    feats += [mu20, mu02, mu11]

    # gradient-orientation histogram (8 bins, magnitude weighted) - streaks
    gy, gx = np.gradient(img.astype(np.float32))
    mag, ang = np.hypot(gy, gx), np.arctan2(gy, gx) % np.pi
    bins = np.minimum((ang / np.pi * 8).astype(np.int32), 7)
    Pm, Pbn = to_patches(mag), to_patches(bins.astype(np.float32)).astype(np.int32)
    hog = np.zeros((N_PATCH, 8), np.float32)
    for b in range(8):
        hog[:, b] = (Pm * (Pbn == b)).sum((1, 2))
    hog /= hog.sum(1, keepdims=True) + 1e-6
    feats.append(hog.T)

    # 8x8 thumbnail of the background-subtracted patch (coarse shape)
    thumb = Pb.reshape(N_PATCH, 8, 8, 8, 8).mean((2, 4)).reshape(N_PATCH, 64)
    feats.append(thumb.T)

    # orientation anisotropy: streaks light up one HOG bin
    feats.append(hog.max(1) / (hog.mean(1) + 1e-6))

    F = np.vstack([f if f.ndim == 2 else f[None] for f in feats]).T.astype(np.float32)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)               # z-score per dim
    # Block equalization: each feature block contributes equally to squared
    # distance, so the 64-dim thumbnail can't swamp the morphology dims.
    blocks = [6, 3, 1, 3, 8, 64, 1]   # dog, stats, conc, moments, hog, thumb, aniso
    i = 0
    for b in blocks:
        F[:, i:i + b] /= np.sqrt(b); i += b
    assert i == F.shape[1]
    # NOTE: deliberately NO per-row L2 norm. Normalizing pure-noise (empty)
    # patches turns them into random unit directions, which made them look
    # mutually 'isolated' and inverted anomaly detection (v2 bug). Euclidean
    # on z-scored features keeps empty patches clustered near the origin.
    return F

# ====================================== B. quantvec: rotate -> Lloyd-Max -> 4-bit ADC
def lloyd_max_1d(x, k=16, iters=12):
    cb = np.quantile(x, np.linspace(0, 1, k + 2)[1:-1]).astype(np.float32)
    for _ in range(iters):
        a = np.argmin(np.abs(x[:, None] - cb[None]), 1)
        for j in range(k):
            sel = x[a == j]
            if len(sel):
                cb[j] = sel.mean()
        cb.sort()
    return cb

class QuantIndex:
    """Rotation + per-dim Lloyd-Max codebooks + 4-bit packing + ADC LUT search."""
    def __init__(self, emb, bits=4, seed=3):
        if emb.shape[1] % 2:                              # pad to even for packing
            emb = np.hstack([emb, np.zeros((len(emb), 1), np.float32)])
        D = emb.shape[1]
        self.emb_dim = D
        q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, D)))
        self.R = q.astype(np.float32)
        rot = emb @ self.R
        k = 2 ** bits
        self.cb = np.stack([lloyd_max_1d(rot[:, d], k) for d in range(D)])  # (D,k)
        codes = np.argmin(np.abs(rot[:, :, None] - self.cb[None]), 2).astype(np.uint8)
        self.packed = (codes[:, 0::2] << 4) | codes[:, 1::2]               # 2 codes/byte
        self.D, self.bits = D, bits

    def unpack(self):
        if not hasattr(self, "_codes"):
            c = np.empty((self.packed.shape[0], self.D), np.uint8)
            c[:, 0::2] = self.packed >> 4
            c[:, 1::2] = self.packed & 0x0F
            self._codes = c
        return self._codes

    def dequant(self):
        """Reconstruct DB vectors from codes (for batch ops; same ADC math)."""
        return self.cb[np.arange(self.D)[None, :], self.unpack()]

    def index_bytes(self):
        return self.packed.nbytes + self.cb.nbytes + self.R.nbytes

    def search(self, query, k=10, exclude=None, rerank_with=None, shortlist=50):
        """Euclidean ADC: LUT[d,j] = (q_rot[d]-cb[d,j])^2; dist = sum of lookups.
        If rerank_with (float32 emb matrix) is given, ADC retrieves a shortlist
        which is re-ranked exactly -- the quantvec store_full=True tier."""
        if len(query) < self.D:
            query = np.concatenate([query, np.zeros(self.D - len(query), np.float32)])
        qr = query @ self.R
        lut = (qr[:, None] - self.cb) ** 2                # (D, 2^bits)
        dists = lut[np.arange(self.D)[None, :], self.unpack()].sum(1)
        if exclude is not None:
            dists[exclude] = np.inf
        if rerank_with is None:
            return np.argsort(dists)[:k], dists
        cand = np.argsort(dists)[:shortlist]
        exact = ((rerank_with[cand] - query[None, :rerank_with.shape[1]]) ** 2).sum(1)
        return cand[np.argsort(exact)][:k], dists

# ============================================================ C. anomaly: kNN isolation
def knn_isolation(qidx, emb, k=8):
    """Mean Euclidean distance to k nearest neighbours in the QUANTIZED index.
    Batched via dequantized codes -- numerically identical to per-query ADC,
    just expressed as one matmul. High = isolated = rare morphology."""
    if emb.shape[1] < qidx.D:
        emb = np.hstack([emb, np.zeros((len(emb), qidx.D - emb.shape[1]), np.float32)])
    Q = (emb @ qidx.R).astype(np.float32)
    V = qidx.dequant().astype(np.float32)
    d2 = (Q ** 2).sum(1)[:, None] - 2 * Q @ V.T + (V ** 2).sum(1)[None]
    np.fill_diagonal(d2, np.inf)
    part = np.partition(d2, k, axis=1)[:, :k]
    return np.sqrt(np.clip(part, 0, None)).mean(1).astype(np.float32)

# ============================================================ D. ZOGY-lite
def difference_detect(ref, sci, nsigma=5.0):
    D = sci - ref                                          # statics cancel
    p = gauss_kernel(PSF_SIGMA, 6); p /= p.sum()
    # matched filter via FFT correlation (symmetric kernel -> conv == corr)
    pf = np.zeros_like(D); ph = p.shape[0]
    pf[:ph, :ph] = p
    pf = np.roll(pf, (-(ph // 2), -(ph // 2)), (0, 1))
    S = np.fft.irfft2(np.fft.rfft2(D) * np.fft.rfft2(pf), s=D.shape)
    sigD = 1.4826 * np.median(np.abs(D - np.median(D)))    # robust noise of D
    sigS = sigD * np.sqrt((p ** 2).sum())                  # filtered-noise sigma
    snr = S / sigS
    # local maxima above threshold, greedy non-max suppression
    cand = np.argwhere(snr > nsigma)
    if len(cand) == 0:
        return np.empty((0, 2), int), snr
    vals = snr[cand[:, 0], cand[:, 1]]
    order = np.argsort(-vals); cand = cand[order]
    keep = []
    for y, x in cand:
        if all((y - ky) ** 2 + (x - kx) ** 2 > 36 for ky, kx in keep):
            keep.append((y, x))
    return np.array(keep), snr

def match(dets, truth, r=5):
    hits = 0
    for ty, tx in truth:
        if len(dets) and np.min((dets[:, 0]-ty)**2 + (dets[:, 1]-tx)**2) <= r*r:
            hits += 1
    return hits

# ============================================================ run + verify
def label(coords, centers, half=0):
    lab = np.zeros(len(coords), bool)
    for cy, cx in centers:
        lab |= (coords[:, 0] - half <= cy) & (cy < coords[:, 0] + PATCH + half) & \
               (coords[:, 1] - half <= cx) & (cx < coords[:, 1] + PATCH + half)
    return lab

def main():
    import time
    rep, t0 = [], time.time()
    ref, sci, gal_c, streak_c, tr_c = make_sky()
    rep.append(f"SKY  {H}x{W} synthetic, 2 epochs | 4000 stars (amp<=0.80), "
               f"{len(gal_c)} galaxies, {len(streak_c)} dim streaks, "
               f"{len(tr_c)} transients (amp 0.50-0.90, epoch-2 only)")

    # A
    lv = build_pyramid(sci)
    rep.append(f"\nA. PYRAMID  {' -> '.join(f'{l.shape[0]}' for l in lv)}  "
               f"overhead {100*(sum(l.size for l in lv)/sci.size-1):.1f}% "
               f"(planar analog of HiPS tiling)")

    # B
    coords = patch_coords()
    emb = morphology_features(sci)
    qidx = QuantIndex(emb, bits=4)
    f_kb, q_kb = emb.nbytes / 1024, qidx.index_bytes() / 1024
    rep.append(f"\nB. SEARCH  {N_PATCH} patches, {emb.shape[1]}-d morphology features")
    rep.append(f"   index: float32 {f_kb:.0f} KB -> 4-bit packed {q_kb:.0f} KB "
               f"= {f_kb/q_kb:.1f}x smaller")

    gal_lab = label(coords, gal_c)
    gidx = np.where(gal_lab)[0]
    rng = np.random.default_rng(7)
    qs = rng.choice(gidx, min(30, len(gidx)), replace=False)
    rec, rec_rr, prec = [], [], []
    for qi in qs:
        ex = np.argsort(((emb - emb[qi]) ** 2).sum(1))
        ex = set(ex[ex != qi][:10].tolist())
        got, _ = qidx.search(emb[qi], 10, exclude=np.array([qi]))
        got_rr, _ = qidx.search(emb[qi], 10, exclude=np.array([qi]), rerank_with=emb)
        rec.append(len(ex & set(got.tolist())) / 10)
        rec_rr.append(len(ex & set(got_rr.tolist())) / 10)
        prec.append(gal_lab[got_rr].mean())
    rep.append(f"   recall@10: 4-bit ADC alone {np.mean(rec):.2f}  ->  "
               f"+fp32 rerank of top-50 (store_full tier): {np.mean(rec_rr):.2f}")
    rep.append(f"   'galaxies like this' precision@10: {np.mean(prec):.2f}  "
               f"(v1 was 0.42; random = {gal_lab.mean():.3f})")

    # C
    iso = knn_isolation(qidx, emb, k=8)
    st_lab = label(coords, streak_c)
    topN = np.argsort(-iso)[:3 * len(streak_c)]
    rep.append(f"\nC. ANOMALY (kNN isolation on quantized index)")
    rep.append(f"   dim streaks in top-{3*len(streak_c)} isolation scores: "
               f"{st_lab[topN].sum()}/{len(streak_c)}")
    # brightness baseline on streaks
    bright = np.argsort(-to_patches(sci).max((1, 2)))[:3 * len(streak_c)]
    rep.append(f"   brightness baseline finds: {st_lab[bright].sum()}/{len(streak_c)} "
               f"(streaks are dim -- brightness can't see them)")

    # D
    dets, snr = difference_detect(ref, sci, nsigma=5.0)
    hits = match(dets, tr_c)
    rep.append(f"\nD. CHANGE (ZOGY-lite difference imaging, 5-sigma)")
    rep.append(f"   transients recovered: {hits}/{len(tr_c)}   "
               f"false positives: {len(dets)-hits}")
    tr_lab = label(coords, tr_c)
    bright2 = np.argsort(-to_patches(sci).max((1, 2)))[:3 * len(tr_c)]
    rep.append(f"   v1 brightness hack on this sky: {tr_lab[bright2].sum()}/{len(tr_c)} "
               f"(bright stars now contaminate the top ranks)")

    rep.append(f"\ntotal runtime: {time.time()-t0:.1f}s")
    print("\n".join(rep))

    # figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    g = lambda a: np.clip(a / (np.percentile(a, 99.9) + 1e-9), 0, 1) ** 0.4
    fig, ax = plt.subplots(2, 3, figsize=(16, 10.5))

    ax[0,0].imshow(g(lv[2]), cmap="magma"); ax[0,0].set_title("overview (pyramid L2)")
    qi = qs[0]
    ax[0,1].imshow(g(sci), cmap="magma"); ax[0,1].set_title("morphology search: galaxies like cyan")
    y, x = coords[qi]
    ax[0,1].add_patch(Rectangle((x, y), PATCH, PATCH, ec="cyan", fc="none", lw=2))
    got, _ = qidx.search(emb[qi], 10, exclude=np.array([qi]))
    for j in got:
        ry, rx = coords[j]
        ax[0,1].add_patch(Rectangle((rx, ry), PATCH, PATCH, ec="lime", fc="none", lw=1.2))
    ax[0,2].imshow(g(sci), cmap="magma"); ax[0,2].set_title("kNN-isolation anomalies (red) vs true streaks (yellow)")
    for j in topN:
        ry, rx = coords[j]
        ax[0,2].add_patch(Rectangle((rx, ry), PATCH, PATCH, ec="red", fc="none", lw=1.3))
    for sy, sx in streak_c:
        ax[0,2].add_patch(Rectangle((sx-32, sy-32), 64, 64, ec="yellow", fc="none", lw=1.0, ls=":"))
    D = sci - ref
    ax[1,0].imshow(np.clip(D, -.05, .5), cmap="coolwarm"); ax[1,0].set_title("difference image (statics cancel)")
    ax[1,1].imshow(np.clip(snr, 0, 10), cmap="viridis"); ax[1,1].set_title("matched-filter S/N map")
    ax[1,2].imshow(g(sci), cmap="magma"); ax[1,2].set_title(f"5-sigma detections (red) vs truth (yellow): {hits}/{len(tr_c)}")
    for dy, dx in dets:
        ax[1,2].add_patch(plt.Circle((dx, dy), 30, ec="red", fc="none", lw=1.5))
    for ty, tx in tr_c:
        ax[1,2].add_patch(plt.Circle((tx, ty), 50, ec="yellow", fc="none", lw=1.0, ls=":"))
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    import os; os.makedirs("/mnt/user-data/outputs", exist_ok=True)
    plt.savefig("/mnt/user-data/outputs/skyscope2_demo.png", dpi=100, bbox_inches="tight")
    print("figure -> /mnt/user-data/outputs/skyscope2_demo.png")

if __name__ == "__main__":
    main()
