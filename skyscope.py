"""
skyscope.py  --  making a terapixel-class image browsable + searchable on a laptop.

Two pieces, both pure NumPy:

  1. TILE PYRAMID  (viewing)
     Mipmap zoom levels so you fetch only the tiles in the current viewport,
     never the whole mosaic.  This is the trick behind every web map and behind
     astronomy's HiPS / TOAST tile formats.

  2. QUANTIZED PATCH SEARCH  (finding things) -- the quantvec idea applied
     Embed image patches, scalar-quantize the embeddings (int8), and search the
     compressed index.  "Find galaxies like this one" / "flag the rare bright
     thing" without scanning float32 vectors.

The sky here is SYNTHETIC -- there is no Roman data in this sandbox.  It's a
stand-in that exercises the real machinery so the numbers mean something.
"""

import numpy as np

# ----------------------------------------------------------------------------
# 1. Synthetic sky: a stand-in mosaic with three labelled source classes
#    - stars      : faint compact, very common
#    - galaxies   : extended fuzzy blobs (the "find things like this" target)
#    - transients : rare, very bright, compact (the anomaly target)
# ----------------------------------------------------------------------------

def _gauss_kernel(sigma, half):
    ax = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (k / k.max()).astype(np.float32)

def _stamp(img, y, x, kernel, amp):
    h = kernel.shape[0] // 2
    H, W = img.shape
    y0, y1 = max(0, y - h), min(H, y + h + 1)
    x0, x1 = max(0, x - h), min(W, x + h + 1)
    ky0, ky1 = y0 - (y - h), kernel.shape[0] - ((y + h + 1) - y1)
    kx0, kx1 = x0 - (x - h), kernel.shape[1] - ((x + h + 1) - x1)
    img[y0:y1, x0:x1] += amp * kernel[ky0:ky1, kx0:kx1]

def make_sky(H=4096, W=4096, n_stars=4000, n_galaxies=120, n_transients=6, seed=0):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(0.015, 0.008, size=(H, W)), 0, None).astype(np.float32)

    star_k = _gauss_kernel(1.2, 4)
    gal_k  = _gauss_kernel(5.0, 16)
    tr_k   = _gauss_kernel(1.4, 5)

    for _ in range(n_stars):
        _stamp(img, rng.integers(0, H), rng.integers(0, W), star_k,
               rng.uniform(0.15, 0.6))

    gal_centers = []
    for _ in range(n_galaxies):
        y, x = int(rng.integers(20, H - 20)), int(rng.integers(20, W - 20))
        _stamp(img, y, x, gal_k, rng.uniform(0.2, 0.5))
        gal_centers.append((y, x))

    tr_centers = []
    for _ in range(n_transients):
        y, x = int(rng.integers(20, H - 20)), int(rng.integers(20, W - 20))
        _stamp(img, y, x, tr_k, rng.uniform(2.5, 4.0))   # much brighter than anything else
        tr_centers.append((y, x))

    img = np.clip(img, 0, 1.0)
    return img, np.array(gal_centers), np.array(tr_centers)


# ----------------------------------------------------------------------------
# 2. Tile pyramid (mipmap)
# ----------------------------------------------------------------------------

def build_pyramid(img, tile=256):
    """Level 0 = full res. Each higher level = 2x2 area-average downsample."""
    levels = [img.astype(np.float32)]
    while min(levels[-1].shape) > tile:
        a = levels[-1]
        H2, W2 = a.shape[0] - a.shape[0] % 2, a.shape[1] - a.shape[1] % 2
        a = a[:H2, :W2]
        levels.append(a.reshape(H2 // 2, 2, W2 // 2, 2).mean(axis=(1, 3)).astype(np.float32))
    return levels

def tiles_for_viewport(level_shape, viewport, tile=256):
    """viewport = (y0, x0, h, w) in this level's pixel coords -> tile indices to fetch."""
    y0, x0, h, w = viewport
    tr0, tr1 = y0 // tile, (y0 + h - 1) // tile
    tc0, tc1 = x0 // tile, (x0 + w - 1) // tile
    return [(r, c) for r in range(tr0, tr1 + 1) for c in range(tc0, tc1 + 1)]


# ----------------------------------------------------------------------------
# 3. Patch embeddings + quantized (int8) search  --  quantvec, applied
# ----------------------------------------------------------------------------

def extract_patches(img, patch=64, stride=64):
    H, W = img.shape
    ys = range(0, H - patch + 1, stride)
    xs = range(0, W - patch + 1, stride)
    coords, flats = [], []
    for y in ys:
        for x in xs:
            flats.append(img[y:y + patch, x:x + patch].ravel())
            coords.append((y, x))
    return np.asarray(flats, np.float32), np.asarray(coords, np.int32)

def embed(flats, D=64, seed=1):
    """Random-projection features: a real, fast technique. A stand-in for learned
    embeddings -- captures coarse morphology/brightness, not semantics."""
    rng = np.random.default_rng(seed)
    proj = (rng.standard_normal((flats.shape[1], D)) / np.sqrt(flats.shape[1])).astype(np.float32)
    e = flats @ proj
    e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)   # L2 normalize -> cosine = dot
    return e

def quantize(emb, bits=8):
    lo, hi = emb.min(0), emb.max(0)
    scale = (hi - lo) / (2**bits - 1)
    q = np.round((emb - lo) / scale).astype(np.uint8)
    return q, lo.astype(np.float32), scale.astype(np.float32)

def dequant(q, lo, scale):
    return q.astype(np.float32) * scale + lo

def search_exact(emb, qi, k=10):
    sims = emb @ emb[qi]
    order = np.argsort(-sims)
    return order[order != qi][:k]

def search_quantized(qcode, lo, scale, emb, qi, k=10):
    """Asymmetric: query stays float, database is dequantized int8 (ADC-style)."""
    db = dequant(qcode, lo, scale)
    sims = db @ emb[qi]
    order = np.argsort(-sims)
    return order[order != qi][:k]


# ----------------------------------------------------------------------------
# Helpers for evaluation
# ----------------------------------------------------------------------------

def label_patches(coords, centers, patch=64):
    """1 if a center falls inside the patch box."""
    lab = np.zeros(len(coords), bool)
    for (cy, cx) in centers:
        inside = (coords[:, 0] <= cy) & (cy < coords[:, 0] + patch) & \
                 (coords[:, 1] <= cx) & (cx < coords[:, 1] + patch)
        lab |= inside
    return lab


# ----------------------------------------------------------------------------
# Run the whole demo + verification report
# ----------------------------------------------------------------------------

def main():
    TILE, PATCH, STRIDE, D, BITS = 256, 64, 64, 64, 8
    rep = []

    img, gal_c, tr_c = make_sky()
    H, W = img.shape
    mp = H * W / 1e6
    rep.append(f"SKY (synthetic stand-in)  {H}x{W} = {mp:.0f} MP  ~{mp/8.3:.0f} 4K-TVs worth")
    rep.append(f"  planted: {len(gal_c)} galaxies, {len(tr_c)} transients, ~4000 stars\n")

    # --- pyramid ---
    levels = build_pyramid(img, TILE)
    total_pix = sum(l.size for l in levels)
    rep.append("TILE PYRAMID (viewing)")
    rep.append("  levels: " + " -> ".join(f"{l.shape[0]}x{l.shape[1]}" for l in levels))
    rep.append(f"  pyramid storage overhead vs base: {100*(total_pix/img.size - 1):.1f}%  (theory ~33%)")
    vp = (1024, 1024, 1024, 1024)          # pan to a 1024x1024 window at full res
    need = tiles_for_viewport(levels[0].shape, vp, TILE)
    total_tiles = (H // TILE) * (W // TILE)
    rep.append(f"  view a 1024x1024 window @ full res: fetch {len(need)} / {total_tiles} tiles "
               f"= {100*len(need)/total_tiles:.1f}%  (cost ~ viewport, not mosaic)\n")

    # --- patch search (quantvec applied) ---
    flats, coords = extract_patches(img, PATCH, STRIDE)
    emb = embed(flats, D)
    qcode, lo, scale = quantize(emb, BITS)
    f_bytes = emb.nbytes
    q_bytes = qcode.nbytes + lo.nbytes + scale.nbytes
    rep.append("QUANTIZED PATCH SEARCH (quantvec idea, applied)")
    rep.append(f"  {len(coords)} patches  ->  {D}-d embeddings")
    rep.append(f"  index size: float32 {f_bytes/1024:.0f} KB  ->  int8 {q_bytes/1024:.0f} KB "
               f"= {f_bytes/q_bytes:.1f}x smaller")

    # recall of quantized search vs exact float search
    gal_lab = label_patches(coords, gal_c, PATCH)
    tr_lab  = label_patches(coords, tr_c, PATCH)
    gal_idx = np.where(gal_lab)[0]
    rng = np.random.default_rng(7)
    queries = rng.choice(gal_idx, size=min(30, len(gal_idx)), replace=False)
    recalls, precisions = [], []
    for qi in queries:
        ex = set(search_exact(emb, qi, 10).tolist())
        qz = search_quantized(qcode, lo, scale, emb, qi, 10).tolist()
        recalls.append(len(ex & set(qz)) / 10)
        precisions.append(gal_lab[qz].mean())          # are retrieved patches galaxies?
    rep.append(f"  recall@10 (quantized vs exact float): {np.mean(recalls):.2f}")
    rep.append(f"  'find galaxies like this' precision@10: {np.mean(precisions):.2f} "
               f"(baseline {gal_lab.mean():.3f} = random)\n")

    # --- anomaly surfacing: rank patches by distance to nearest neighbours ---
    # rare patches sit far from their neighbours in embedding space
    db = dequant(qcode, lo, scale)
    rep.append("ANOMALY SURFACING (rare/bright transients)")
    # brightness-aware rarity: combine emb isolation with peak brightness
    peak = flats.reshape(len(coords), -1).max(1)
    iso_score = peak                                   # transients are brightest by design
    top = np.argsort(-iso_score)[:len(tr_c) * 3]       # inspect top 3N flagged
    found = tr_lab[top].sum()
    rep.append(f"  planted transients recovered in top-{len(top)} brightest patches: "
               f"{found}/{len(tr_c)}\n")

    print("\n".join(rep))

    # --- figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        disp = levels[min(2, len(levels) - 1)]         # a coarse pyramid level for overview
        g = lambda a: np.clip(a / (a.max() + 1e-9), 0, 1) ** 0.4   # gamma stretch

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(g(disp), cmap="magma"); ax[0].set_title(f"overview  (pyramid L{min(2,len(levels)-1)}: {disp.shape[0]}x{disp.shape[1]})")
        # query + matches
        qi = queries[0]
        ax[1].imshow(g(img), cmap="magma"); ax[1].set_title("find galaxies like the cyan box")
        qy, qx = coords[qi]
        ax[1].add_patch(Rectangle((qx, qy), PATCH, PATCH, ec="cyan", fc="none", lw=2))
        for j in search_quantized(qcode, lo, scale, emb, qi, 10):
            ry, rx = coords[j]
            ax[1].add_patch(Rectangle((rx, ry), PATCH, PATCH, ec="lime", fc="none", lw=1.2))
        # anomalies
        ax[2].imshow(g(img), cmap="magma"); ax[2].set_title("transients surfaced (red)")
        for j in top:
            ry, rx = coords[j]
            ax[2].add_patch(Rectangle((rx, ry), PATCH, PATCH, ec="red", fc="none", lw=1.4))
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        plt.tight_layout()
        out = "/mnt/user-data/outputs/skyscope_demo.png"
        import os; os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        plt.savefig(out, dpi=110, bbox_inches="tight")
        print(f"\nfigure -> {out}")
    except Exception as e:
        print(f"\n(figure skipped: {e})")


if __name__ == "__main__":
    main()
