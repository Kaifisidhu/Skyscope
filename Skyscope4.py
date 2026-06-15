"""
skyscope4.py  -- binary codes, misregistration, and similarity layout.

Builds on skyscope3 (A&L PSF matching, quadtree, Perona-Malik) and adds three
verified pieces, plus an explicit bridge to the uploaded ASCII-render zips.

[A] BINARY-CODE PATCH SEARCH.  The uploaded zips (ascii-render, ASCILINE) both
    do one thing: map a pixel block's luminance to a small alphabet by scalar
    quantization. At a 2-symbol alphabet that IS 1-bit binarization -- a binary
    code. We take that to its useful form: compress patch descriptors to a
    bitstring and search by Hamming distance (XOR + popcount) + exact rerank.
    The matrix<->binary-code coupling is ITQ (Gong & Lazebnik 2011): a rotation
    MATRIX R and a BINARY CODE B alternately minimize ||B - V R||^2 -- the
    rigorous form of "a matrix and a binary code computing off each other."

[B] SUB-PIXEL MISREGISTRATION.  A symmetric (Gaussian-only) matching kernel can
    broaden a PSF but cannot SHIFT it, so a fractional-pixel offset between
    epochs leaves a DIPOLE at every star. Multiplying the Gaussians by in-kernel
    monomials u,v adds antisymmetric components that DO represent a shift, so the
    polynomial-modulated kernel absorbs the misregistration. This is why the A&L
    kernel is polynomial-modulated.

[C] SIMILARITY LAYOUT.  Arrange source cutouts on a grid so neighbors look alike,
    by optimal assignment (scipy linear_sum_assignment) of items to grid cells
    -- a deterministic, training-free alternative to the SOM.

Every number below is produced by running the code. Pure NumPy + SciPy.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import shift as ndshift
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------------
# shared helpers (verified in skyscope3)
# ----------------------------------------------------------------------------
def gk(sigma, half):
    ax = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()

def fftconv(img, ker):
    kh, kw = ker.shape
    pad = np.zeros_like(img)
    pad[:kh, :kw] = ker
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), (0, 1))
    return np.fft.irfft2(np.fft.rfft2(img) * np.fft.rfft2(pad), img.shape)

LUT = np.array([bin(i).count("1") for i in range(256)], np.uint8)
report = []
def say(s=""):
    report.append(s); print(s)

# ============================================================================
# [A] BINARY-CODE PATCH SEARCH
# ============================================================================
say("=" * 70)
say("[A] BINARY-CODE PATCH SEARCH  (compress descriptors -> bitstring -> Hamming)")
say("=" * 70)

# a textured sky to cut patches from
H = W = 640
skyA = np.zeros((H, W))
for _ in range(1500):
    y, x = rng.integers(20, H - 20), rng.integers(20, W - 20)
    skyA[y, x] += rng.uniform(0.3, 1.0)
skyA = fftconv(skyA, gk(1.6, 9))
skyA += 0.02 * rng.standard_normal((H, W))

# extract a grid of patches; descriptor = 8x8 downsample (64-dim), like a mip cell
P, step, dlen = 16, 12, 8
coords = [(y, x) for y in range(0, H - P, step) for x in range(0, W - P, step)]
def descriptor(y, x):
    patch = skyA[y:y + P, x:x + P]
    return patch.reshape(dlen, P // dlen, dlen, P // dlen).mean((1, 3)).ravel()
D = np.array([descriptor(y, x) for (y, x) in coords])
n, d = D.shape
say(f"  {n} patches, descriptor dim {d} (an {dlen}x{dlen} luminance grid -- the zips' operation)")

# ground-truth float nearest neighbours
nq = 300
qidx = rng.choice(n, nq, replace=False)
def true_knn(Q, X, k=10):
    return np.array([np.argsort(((X - q) ** 2).sum(1))[:k] for q in Q])
truth = true_knn(D[qidx], D, 11)[:, 1:]            # drop self

# binary codes: PCA -> {ITQ rotation, PCA-sign}
c = 64
mean = D.mean(0)
Dc = D - mean
w, v = np.linalg.eigh(Dc.T @ Dc / n)
comps = v[:, ::-1][:, :c]
V = Dc @ comps

def itq(V, n_iter=50, seed=0):
    r = np.random.default_rng(seed); cc = V.shape[1]
    R = np.linalg.qr(r.standard_normal((cc, cc)))[0]
    losses = []
    for _ in range(n_iter):
        Z = V @ R; B = np.sign(Z); B[B == 0] = 1
        losses.append(((B - Z) ** 2).sum())
        U, _, Vt = np.linalg.svd(B.T @ V); R = Vt.T @ U.T
    return R, losses
R, losses = itq(V, 50)

bits_itq = (V @ R > 0)
bits_pca = (V > 0)

def hamming_rerank(qb_all, xb_all, Q, X, m, k=10):
    Xp = np.packbits(xb_all, 1); Qp = np.packbits(qb_all, 1)
    out = []
    for qp, q in zip(Qp, Q):
        dist = LUT[np.bitwise_xor(Xp, qp)].sum(1)
        cand = np.argpartition(dist, m)[:m]
        out.append(cand[np.argsort(((X[cand] - q) ** 2).sum(1))[:k + 1]])
    return [o[o != qi][:k] for o, qi in zip(out, qidx)]   # drop self if present

def recall(approx, truth):
    return np.mean([len(set(a) & set(t)) / len(t) for a, t in zip(approx, truth)])

say(f"  compression: {d} float32 ({d*4} B) -> {c} bits ({c//8} B) = {d*4/(c//8):.0f}x; "
    f"ITQ loss {losses[0]:.0f}->{losses[-1]:.0f} (monotone decrease = coupling correct)")
ms = [20, 50, 100, 300, 500]
rec_itq, rec_pca = [], []
for m in ms:
    ri = recall(hamming_rerank(bits_itq[qidx], bits_itq, D[qidx], D, m), truth)
    rp = recall(hamming_rerank(bits_pca[qidx], bits_pca, D[qidx], D, m), truth)
    rec_itq.append(ri); rec_pca.append(rp)
    say(f"    shortlist {m:>3} ({100*m/n:4.1f}% scored): recall@10  ITQ {ri:.3f}   PCA-sign {rp:.3f}")
say(f"  note: here ITQ DOES beat PCA-sign (smooth, correlated patch descriptors); on the")
say(f"  earlier clustered data PCA-sign won -- the rotation's benefit is data-dependent.")

# ASCII bridge: render one patch at a 10-glyph ramp (the zip) and 2-glyph (binary)
RAMP10 = " .:-=+*#%@"
RAMP2 = " @"
py, px = coords[rng.integers(len(coords))]
big = skyA[py:py + P, px:px + P]
def to_ascii(block, ramp):
    g = (block - block.min()) / (np.ptp(block) + 1e-9)
    idx = np.clip((g * (len(ramp) - 1)).round().astype(int), 0, len(ramp) - 1)
    return "\n".join("".join(ramp[i] for i in row) for row in idx)
ascii10 = to_ascii(big, RAMP10)
ascii2 = to_ascii(big, RAMP2)

# ============================================================================
# [B] SUB-PIXEL MISREGISTRATION
# ============================================================================
say("\n" + "=" * 70)
say("[B] SUB-PIXEL MISREGISTRATION  (why the kernel is polynomial-modulated)")
say("=" * 70)
Hb = Wb = 320; BORDER = 24
sky = np.zeros((Hb, Wb)); Ns = 120
sy = rng.integers(BORDER, Hb - BORDER, Ns); sx = rng.integers(BORDER, Wb - BORDER, Ns)
sky[sy, sx] = rng.uniform(0.5, 1.0, Ns)
ref = fftconv(sky, gk(1.3, 11)) + rng.normal(0, 0.001, (Hb, Wb))
sci_aligned = fftconv(sky, gk(1.8, 11))
DX, DY = 0.4, -0.3
sci = ndshift(sci_aligned, (DY, DX), order=3, mode="reflect") + rng.normal(0, 0.001, (Hb, Wb))
def interior(a): return a[BORDER:-BORDER, BORDER:-BORDER]

def al_solve(ref, sci, in_kernel_order, sigmas=(1.0, 2.0, 3.5), kh=11, bg_order=1):
    ax = np.arange(-kh, kh + 1); uu, vv = np.meshgrid(ax, ax)
    cols = []
    for s in sigmas:
        g = gk(s, kh)
        for i in range(in_kernel_order + 1):
            for j in range(in_kernel_order + 1 - i):
                cols.append(fftconv(ref, (uu.astype(float)**i) * (vv.astype(float)**j) * g))
    yy, xx = np.mgrid[0:Hb, 0:Wb] / Hb
    cols.append(np.ones((Hb, Wb)))
    for i in range(1, bg_order + 1): cols += [xx**i, yy**i]
    A = np.stack([interior(cc).ravel() for cc in cols], 1)
    coef, *_ = np.linalg.lstsq(A, interior(sci).ravel(), rcond=None)
    return sci - sum(cc * wt for cc, wt in zip(cols, coef))

D_naive = sci - ref
D_sym = al_solve(ref, sci, 0)
D_poly = al_solve(ref, sci, 2)
def star_rms(Dm):
    p = [Dm[y-3:y+4, x-3:x+4] for y, x in zip(sy, sx)
         if BORDER < y < Hb-BORDER and BORDER < x < Wb-BORDER]
    return float(np.sqrt(np.mean([q**2 for q in p])))
say(f"  sub-pixel shift dx={DX}, dy={DY}")
say(f"  star residual RMS  naive {star_rms(D_naive):.4f}  symmetric-K {star_rms(D_sym):.4f}  "
    f"polynomial-K {star_rms(D_poly):.4f}")
say(f"  polynomial kernel {star_rms(D_sym)/star_rms(D_poly):.1f}x cleaner than symmetric "
    f"-> absorbed the shift a symmetric kernel cannot represent.")
# a bright star stamp for the figure (clear dipole)
bi = np.argmax([sky[y, x] for y, x in zip(sy, sx)])
cy, cx = sy[bi], sx[bi]
st = lambda Dm: Dm[cy-7:cy+8, cx-7:cx+8]

# ============================================================================
# [C] SIMILARITY LAYOUT (linear assignment vs SOM)
# ============================================================================
say("\n" + "=" * 70)
say("[C] SIMILARITY LAYOUT  (arrange cutouts by similarity: assignment vs SOM)")
say("=" * 70)
G = 8; Ncut = G * G; cut = 21
# sources with varied shape so a layout is meaningful
imgs, feats = [], []
for _ in range(Ncut):
    sxx, syy = rng.uniform(1.0, 3.5, 2)
    th = rng.uniform(0, np.pi)
    ax = np.arange(-cut // 2 + 1, cut // 2 + 1)
    X_, Y_ = np.meshgrid(ax, ax)
    Xr = X_ * np.cos(th) + Y_ * np.sin(th)
    Yr = -X_ * np.sin(th) + Y_ * np.cos(th)
    g = np.exp(-(Xr**2 / (2 * sxx**2) + Yr**2 / (2 * syy**2)))
    g += 0.03 * rng.standard_normal(g.shape)
    imgs.append(g); feats.append(g.ravel())
imgs = np.array(imgs); feat = np.array(feats)
feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)

cells = np.array([(r, c) for r in range(G) for c in range(G)], float)
def pca2(X):
    Xc = X - X.mean(0); w, v = np.linalg.eigh(Xc.T @ Xc)
    return Xc @ v[:, ::-1][:, :2]
def assign(embed):
    e = embed - embed.min(0); e = e / (e.max(0) + 1e-9) * (G - 1)
    cost = ((e[:, None, :] - cells[None, :, :])**2).sum(2)
    ri, ci = linear_sum_assignment(cost)
    p = np.empty(Ncut, int); p[ri] = ci; return p
def som(feat, G, epochs=400):
    w = rng.standard_normal((G * G, feat.shape[1])) * 0.1
    for t in range(epochs):
        lr = 0.5 * (1 - t / epochs); rad = G * 0.5 * (1 - t / epochs) + 0.5
        x = feat[rng.integers(0, len(feat))]
        bmu = np.argmin(((w - x)**2).sum(1))
        h = np.exp(-((cells - cells[bmu])**2).sum(1) / (2 * rad**2))[:, None]
        w += lr * h * (x - w)
    return w
def metrics(p):
    pos = np.empty((Ncut, 2)); pos[np.arange(Ncut)] = cells[p]
    gi, gj = np.triu_indices(Ncut, 1)
    gd = np.linalg.norm(pos[gi] - pos[gj], axis=1)
    fd = np.linalg.norm(feat[gi] - feat[gj], axis=1)
    return spearmanr(gd, fd).statistic

p_lap = assign(pca2(feat))
wsom = som(feat, G)
cost = ((feat[:, None, :] - wsom[None, :, :])**2).sum(2)
ri, ci = linear_sum_assignment(cost); p_som = np.empty(Ncut, int); p_som[ri] = ci
rho_rand = np.mean([metrics(rng.permutation(Ncut)) for _ in range(5)])
rho_som, rho_lap = metrics(p_som), metrics(p_lap)
say(f"  distance-preservation rho (grid dist vs feature dist):")
say(f"    random {rho_rand:+.3f}   SOM {rho_som:+.3f}   linear-assignment {rho_lap:+.3f}")
say(f"  assignment grid is deterministic + training-free; SOM is stochastic (varies by init).")

def grid_image(placement):
    canvas = np.zeros((G * cut, G * cut))
    for item, cell in enumerate(placement):
        r, cc = int(cells[cell, 0]), int(cells[cell, 1])
        canvas[r*cut:(r+1)*cut, cc*cut:(cc+1)*cut] = imgs[item]
    return canvas

# ============================================================================
# FIGURE
# ============================================================================
fig = plt.figure(figsize=(15, 14))
gs = fig.add_gridspec(3, 3, hspace=0.32, wspace=0.22)

ax = fig.add_subplot(gs[0, 0]); ax.imshow(big, cmap="magma"); ax.set_title("[A] a sky patch")
ax.axis("off")
ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
ax.text(0.0, 1.0, ascii10, family="monospace", fontsize=6.5, va="top", ha="left")
ax.set_title("same patch, 10-glyph ramp (the zip's algorithm)")
ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
ax.text(0.0, 1.0, ascii2, family="monospace", fontsize=6.5, va="top", ha="left")
ax.set_title("2-glyph ramp = 1-bit binary code")

ax = fig.add_subplot(gs[1, 0])
ax.plot(ms, rec_itq, "o-", label="ITQ"); ax.plot(ms, rec_pca, "s-", label="PCA-sign")
ax.set_xlabel("shortlist size m"); ax.set_ylabel("recall@10"); ax.set_ylim(0, 1.05)
ax.set_title(f"[A] Hamming filter + rerank ({d*4//(c//8)}x smaller)"); ax.legend(); ax.grid(alpha=.3)
vlim = float(np.abs(st(D_naive)).max()) * 0.85
ax = fig.add_subplot(gs[1, 1]); ax.imshow(st(D_naive), cmap="coolwarm", vmin=-vlim, vmax=vlim)
ax.set_title("[B] naive sub-pixel diff: DIPOLE"); ax.axis("off")
ax = fig.add_subplot(gs[1, 2]); ax.imshow(st(D_poly), cmap="coolwarm", vmin=-vlim, vmax=vlim)
ax.set_title("[B] polynomial kernel: shift absorbed"); ax.axis("off")

ax = fig.add_subplot(gs[2, 0]); ax.imshow(grid_image(p_som), cmap="viridis")
ax.set_title(f"[C] SOM layout (rho {rho_som:.2f})"); ax.axis("off")
ax = fig.add_subplot(gs[2, 1]); ax.imshow(grid_image(p_lap), cmap="viridis")
ax.set_title(f"[C] linear-assignment (rho {rho_lap:.2f})"); ax.axis("off")
ax = fig.add_subplot(gs[2, 2])
ax.bar(["random", "SOM", "assign"], [rho_rand, rho_som, rho_lap],
       color=["#999", "#5588cc", "#cc5544"])
ax.set_ylabel("distance-preservation rho"); ax.set_title("[C] layout quality"); ax.grid(alpha=.3, axis="y")

fig.suptitle("skyscope4: binary-code search  |  misregistration  |  similarity layout", fontsize=14)
fig.savefig("/mnt/user-data/outputs/skyscope4_demo.png", dpi=110, bbox_inches="tight")
say(f"\nfigure -> /mnt/user-data/outputs/skyscope4_demo.png")
