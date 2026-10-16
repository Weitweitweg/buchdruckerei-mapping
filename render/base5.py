import cv2, numpy as np, math, sys, subprocess

FPS = 30; DUR = 60.0
CW, CH = 3840, 2160
rng = np.random.default_rng(5)

import os
SRC = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'facade.jpg'))
X0, X1, Y0, Y1 = 495, 1605, 185, 1365
H = 1600
S = H / (Y1 - Y0)
W = int(round(H * (X1 - X0) / (Y1 - Y0) / 2)) * 2
crop = SRC[Y0:Y1, X0:X1]
F = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
def T(x, y): return ((x - X0) * S, (y - Y0) * S)

# ---------- masks ----------
hsv = cv2.cvtColor((F * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
sat, val = hsv[..., 1], hsv[..., 2]
bm = (cv2.cvtColor((F * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) > 14).astype(np.uint8)
bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
ff = bm.copy() * 255
fm = np.zeros((H + 2, W + 2), np.uint8)
cv2.floodFill(ff, fm, (0, 0), 128)
bm = ((ff != 128)).astype(np.uint8)
BLD = bm.astype(np.float32)
BLDs = cv2.GaussianBlur(BLD, (3, 3), 0)

# glass: rects + pointed arches (orig coords)
RECTS = [
    (905, 465, 940, 517), (995, 435, 1125, 518), (1175, 468, 1213, 518),
    (607, 585, 725, 695), (837, 588, 905, 695), (988, 588, 1133, 695), (1213, 590, 1280, 697), (1370, 598, 1487, 702),
    (583, 805, 712, 930), (828, 805, 898, 930), (985, 808, 1138, 930), (1220, 808, 1297, 935), (1385, 810, 1513, 935)]
ARCHES = [  # x0, x1, apex, spring, bottom
    (553, 692, 1052, 1110, 1225), (785, 918, 1055, 1110, 1225), (997, 1130, 1055, 1110, 1225),
    (1228, 1305, 1058, 1092, 1232), (1397, 1565, 1062, 1115, 1232)]
LABEL = np.zeros((H, W), np.int32)
WINFO = []  # (cx, cy, x0, y0, x1, y1, kind)
k = 1
for (a, b, c, d) in RECTS:
    (xa, ya), (xb, yb) = T(a, b), T(c, d)
    LABEL[int(ya):int(yb), int(xa):int(xb)] = k
    WINFO.append(((xa + xb) / 2, (ya + yb) / 2, xa, ya, xb, yb, 'r')); k += 1
for (a, c, ap, sp, bo) in ARCHES:
    (xa, ya), (xb, yb) = T(a, ap), T(c, bo)
    ys = T(0, sp)[1]
    w = xb - xa
    sc = (0.866 * w) / max(1, ys - ya)
    inside = (XX >= xa) & (XX <= xb) & (YY <= yb) & (YY >= ya)
    up = YY < ys
    dy = (ys - YY) * sc
    arch = (np.hypot(XX - xa, dy) <= w) & (np.hypot(XX - xb, dy) <= w)
    m = inside & (~up | arch)
    LABEL[m] = k
    WINFO.append(((xa + xb) / 2, (ya + yb) / 2, xa, ya, xb, yb, 'a')); k += 1
RC = T(1055, 355); RR = 21 * S
LABEL[np.hypot(XX - RC[0], YY - RC[1]) < RR] = k
WINFO.append((RC[0], RC[1], RC[0] - RR, RC[1] - RR, RC[0] + RR, RC[1] + RR, 'o')); k += 1
NWIN = k - 1
GLASS = (LABEL > 0).astype(np.float32)

pl = ((val > 165) & (sat < 45)).astype(np.uint8) * BLD.astype(np.uint8)
pl = cv2.morphologyEx(pl, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
pl = cv2.morphologyEx(pl, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
PLASTER = np.clip(pl.astype(np.float32) - GLASS, 0, 1)
eave = T(0, 548)[1]
ROOF = ((YY < eave) & ((XX < T(780, 0)[0]) | (XX > T(1310, 0)[0]))).astype(np.float32) * BLD * (1 - PLASTER)
SAND = np.clip(BLD - PLASTER - GLASS - ROOF, 0, 1)
dil = cv2.dilate(GLASS, np.ones((31, 31), np.uint8))
FRAMES = SAND * dil

# distance from glass (outside) for neon rings
DT = cv2.distanceTransform((1 - GLASS).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
DTin = cv2.distanceTransform(GLASS.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
# nearest window label for every pixel (for per-window ring colours)
_, NEAR = cv2.distanceTransformWithLabels((1 - GLASS).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
lab_at = np.zeros(NEAR.max() + 1, np.int32)
ys_, xs_ = np.nonzero(GLASS > 0)
lab_at[NEAR[ys_, xs_]] = LABEL[ys_, xs_]
NEARWIN = lab_at[NEAR]

# architectural line drawing
regions = (PLASTER * 1 + SAND * 2 + ROOF * 3).astype(np.uint8) + (LABEL > 0).astype(np.uint8) * 4
grad = cv2.morphologyEx(regions, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
sil = cv2.morphologyEx(BLD.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
LINES = (grad | sil).astype(np.float32) * 1.0
LINES = cv2.GaussianBlur(LINES, (3, 3), 0) * 1.6
G8 = cv2.cvtColor((F * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
CANNY = cv2.Canny(cv2.GaussianBlur(G8, (3, 3), 0), 50, 140).astype(np.float32) / 255 * BLD
LUMA = F.mean(axis=2)

def noise2d(scale, seed):
    r = np.random.default_rng(seed)
    sm = r.random((H // scale + 3, W // scale + 3)).astype(np.float32)
    return cv2.resize(sm, (W, H), interpolation=cv2.INTER_CUBIC)[:H, :W]
NZ = noise2d(40, 1) * 0.6 + noise2d(12, 2) * 0.4
NZ = (NZ - NZ.min()) / (NZ.max() - NZ.min())
ORDER = np.hypot(XX - W / 2, (YY - H) * 1.1) / np.hypot(W / 2, H * 1.1) * 0.85 + NZ * 0.15

# text mask
gx0, gy0, gx1, gy1 = 935, 712, 1180, 768
gl = cv2.cvtColor(SRC[gy0:gy1, gx0:gx1], cv2.COLOR_BGR2GRAY)
(ta, tb), (tc, td) = T(gx0, gy0), T(gx1, gy1)
TEXT = np.zeros((H, W), np.float32)
TEXT[int(tb):int(tb) + int(td - tb), int(ta):int(ta) + int(tc - ta)] = cv2.resize((gl < 110).astype(np.float32), (int(tc - ta), int(td - tb)))

# ---------- helpers ----------
def ss(a, b, t):
    x = np.clip((t - a) / (b - a), 0, 1); return x * x * (3 - 2 * x)
def env(t, a, b, f=0.5): return ss(a, a + f, t) * (1 - ss(b - f, b, t))
def C(*bgr): return np.array(bgr, np.float32)
def m3(m): return m[..., None]
PAL = [C(0.75, 0.15, 1.0), C(0.95, 0.85, 0.15), C(0.1, 0.55, 1.0), C(0.35, 0.95, 0.35), C(1.0, 0.35, 0.55), C(0.2, 0.25, 1.0), C(1.0, 0.75, 0.3)]
WARM = C(0.25, 0.62, 1.0)
GOLD = C(0.22, 0.72, 1.0)
WINORDER = sorted(range(NWIN), key=lambda i: abs(WINFO[i][0] - W / 2))
def win_mask(i): return (LABEL == i + 1).astype(np.float32)
WMASKS = [win_mask(i) for i in range(NWIN)]

# trees
def grow_tree(x, y, ang, length, depth, t0, segs):
    if depth == 0 or length < 6:
        segs.append(('tip', x, y, t0)); return
    x2 = x + math.cos(ang) * length; y2 = y + math.sin(ang) * length
    dur = 0.35 + 0.1 * depth
    segs.append(('seg', x, y, x2, y2, t0, dur, depth))
    n = 2 if depth > 1 else 3
    for j in range(n):
        da = rng.uniform(0.25, 0.6) * (1 if j % 2 == 0 else -1) + rng.uniform(-0.1, 0.1)
        grow_tree(x2, y2, ang + da, length * rng.uniform(0.62, 0.78), depth - 1, t0 + dur * 0.85, segs)
TREES = []
for (tx, delay) in [(T(1058, 0)[0], 0.0), (T(770, 0)[0], 0.5), (T(1340, 0)[0], 0.8)]:
    segs = []
    grow_tree(tx, H - 5, -math.pi / 2 + rng.uniform(-0.05, 0.05), (230 if delay == 0 else 190) * 2.1, 6, 28.0 + delay, segs)
    TREES.append(segs)
BAUBLES = []
for segs in TREES:
    for s in segs:
        if s[0] == 'tip' and rng.random() < 0.55:
            BAUBLES.append((s[1], s[2], s[3], PAL[int(rng.integers(len(PAL)))], rng.uniform(4, 9) * 2.1))

FORCE_SWIRL = False
SPARK = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0, 6.28), rng.uniform(1.5, 4)) for _ in range(160)]

# swirl curves for finale
SWIRLS = []
for i in range(14):
    cx, cy = rng.uniform(0.1, 0.9) * W, rng.uniform(0.15, 0.95) * H
    r0 = rng.uniform(40, 120); turns = rng.uniform(1.2, 2.5); d = 1 if rng.random() < 0.5 else -1
    th = np.linspace(0, turns * 2 * np.pi, 220)
    r = r0 * (1 - th / (turns * 2 * np.pi) * 0.85)
    pts = np.stack([cx + d * r * np.cos(th + i), cy + r * np.sin(th + i)], 1)
    SWIRLS.append((pts, PAL[i % len(PAL)], rng.uniform(0, 1)))

def sparkle(t, amt, col=C(1, 1, 1)):
    img = np.zeros((H, W), np.float32)
    for (x, y, ph, sp) in SPARK:
        v = max(0.0, math.sin(t * sp + ph)) ** 8 * amt
        if v > 0.02:
            cv2.circle(img, (int(x), int(y)), 2, float(v), -1, cv2.LINE_AA)
    g = cv2.GaussianBlur(img, (0, 0), 3) * 4 + img
    return m3(g) * col

def neon_rings(cols, inten, width=5.0, off=7.0):
    core = np.exp(-((DT - off) / (width * 0.5)) ** 2)
    glow = np.exp(-np.maximum(DT - off, 0) / 16.0) * 0.45 + np.exp(-np.maximum(off - DT, 0) / 6.0) * 0.25 * (DT < off) * (DT > 0)
    colmap = cols[NEARWIN]          # (H,W,3)
    im = inten[NEARWIN]
    return colmap * m3((core * 1.6 + glow) * im) * m3(BLD)

# ---------- scenes ----------
def scene(t):
    P = np.zeros((H, W, 3), np.float32)
    def mix(Cn, k):
        nonlocal P
        P = P * (1 - k) + Cn * k

    # A: line drawing intro 0-6.5
    if t < 7.0:
        prog = ss(0.6, 5.0, t) * 1.05
        rev = np.clip((prog - ORDER) / 0.02, 0, 1)
        head = np.exp(-np.abs(prog - ORDER) / 0.012) * (prog < 1.04)
        lines = (LINES * rev + LINES * head * 2.5 + CANNY * rev * 0.25)
        Cn = m3(lines) * C(0.95, 0.8, 1.0)
        fill = ss(4.6, 6.2, t)
        Cn = Cn + fill * (m3(PLASTER) * 0.18 * C(0.9, 0.7, 0.8) + m3(SAND) * 0.12 + m3(GLASS) * WARM * 0.9 * (0.6 + 0.4 * NZ[..., None]))
        mix(Cn, 1 - ss(6.5, 7.0, t))
    # B: neon arches 6.5-13.5
    if 6.4 < t < 13.6:
        k = env(t, 6.5, 13.5, 0.5); tt = t - 6.5
        cols = np.zeros((NWIN + 1, 3), np.float32); inten = np.zeros(NWIN + 1, np.float32)
        for rank, i in enumerate(WINORDER):
            x = WINFO[i][0] / W
            hue = C(1.0, 0.85, 0.1) * (1 - x) + C(1.0, 0.25, 0.65) * x
            hue = hue * (0.75 + 0.25 * math.sin(tt * 1.5 + x * 6))
            on = tt - 0.15 * rank
            fl = 0.0 if on < 0 else (1.0 if on > 0.5 else (1.0 if math.sin(on * 90) > 0 else 0.2))
            arch = WINFO[i][6] == 'a'
            cols[i + 1] = hue; inten[i + 1] = fl * (1.0 if arch else 0.55)
        Cn = neon_rings(cols, inten)
        base = m3(PLASTER) * 0.07 * C(1.0, 0.6, 0.7) + m3(SAND) * 0.05
        inner = np.zeros_like(P)
        for i in range(NWIN):
            wm = WMASKS[i]
            if WINFO[i][6] == 'a':
                yb = WINFO[i][5]; ya = WINFO[i][3]
                grad = np.clip((YY - ya) / (yb - ya), 0, 1)
                inner += m3(wm * (0.25 + 0.75 * grad)) * cols[i + 1] * inten[i + 1] * 0.8
            else:
                inner += m3(wm) * WARM * (0.55 if (i * 7) % 3 else 0.9) * ss(0.5, 1.5, tt)
        mix(base + Cn + inner + sparkle(t, 0.5) * 0.5, k)
    # C: painted facade 13.5-20.5
    if 13.4 < t < 20.6:
        k = env(t, 13.5, 20.5, 0.4); tt = t - 13.5
        wipe = np.clip((tt * 700 - XX - 120 * NZ) / 60, 0, 1)
        ochre = C(0.38, 0.78, 1.0) * (0.85 + 0.15 * NZ[..., None])
        terr = C(0.25, 0.38, 0.85)
        base = m3(PLASTER) * ochre + m3(SAND) * terr * 0.9 + m3(ROOF) * C(0.35, 0.3, 0.3)
        glass = np.zeros_like(P)
        for rank, i in enumerate(WINORDER):
            ft = tt - 0.8 - 0.12 * rank
            if WINFO[i][6] == 'a':
                col = C(0.45, 0.85, 1.0) * 1.2
                glass += m3(WMASKS[i]) * col
                continue
            ci = (i + int(max(0, ft) / 1.6)) % len(PAL)
            flip = abs(math.cos(math.pi * ((max(0, ft) % 1.6) / 1.6) ** 3)) if ft > 0 else 0
            x0, y0, x1, y1 = WINFO[i][2:6]
            cy = (y0 + y1) / 2; hh = (y1 - y0) / 2 * flip
            band = (np.abs(YY - cy) <= hh).astype(np.float32)
            glass += m3(WMASKS[i] * band) * PAL[ci] * 1.2 + m3(WMASKS[i] * (1 - band)) * 0.05
        sweep = np.exp(-(((XX + YY * 0.6) - (tt - 2.5) * 500) / 90) ** 2)
        Cn = (base + glass) * m3(wipe) + m3(LINES * (1 - wipe)) * C(0.95, 0.8, 1.0) + m3(sweep * BLD) * 0.5 + m3(LINES * wipe) * 0.25
        mix(Cn, k)
    # D: coral origami folds 20.5-27.5
    if 20.4 < t < 27.6:
        k = env(t, 20.5, 27.5, 0.4); tt = t - 20.5
        coral = C(0.5, 0.55, 1.0) * 1.2
        facet = np.ones((H, W), np.float32); fold = np.zeros((H, W), np.float32)
        for j, (ax, ay, sp) in enumerate([(0.3, 0.0, 0.5), (0.7, 1.0, -0.4), (0.5, 0.3, 0.7)]):
            px = (ax + 0.25 * math.sin(tt * sp + j)) * W; py = ay * H
            ang = 1.0 + j * 1.2 + tt * 0.25 * sp
            sd = ((XX - px) * math.sin(ang) - (YY - py) * math.cos(ang))
            facet *= np.where(sd > 0, 1.0, 0.55 + 0.1 * j)
            fold += np.exp(-np.abs(sd) / 2.5)
        sh = np.zeros((H, W), np.float32)
        d = int(18 + 10 * math.sin(tt * 0.8))
        src = np.clip(FRAMES + GLASS, 0, 1)
        dx = int(d * 0.7)
        sh[d:, dx:] = src[:-d, :-dx]
        sh = np.clip(cv2.GaussianBlur(sh, (0, 0), 4) - src, 0, 1)
        base = m3(PLASTER) * coral * m3(facet) * m3(1 - 0.8 * sh) + m3(SAND) * C(0.3, 0.32, 0.75) * m3(facet) + m3(ROOF) * C(0.3, 0.25, 0.35)
        glassc = m3(GLASS) * C(0.08, 0.05, 0.12) + m3(np.exp(-((DT - 3) / 2) ** 2) * BLD) * C(0.7, 0.8, 1.0) * 0.6
        motif = m3(CANNY * PLASTER) * C(1.0, 0.95, 1.0) * 0.4 * (0.5 + 0.5 * math.sin(tt * 2))
        Cn = base + glassc + motif
        Cn += m3(fold * BLD) * C(0.85, 0.9, 1.0) * 0.9 + m3((facet > 0.99) * PLASTER) * 0.12
        mix(Cn, k)
    # E: golden trees 27.5-35
    if 27.4 < t < 35.1:
        k = env(t, 27.5, 35.0, 0.4); tt = t - 27.5
        base = m3(PLASTER) * 0.28 * C(0.95, 0.9, 1.0) + m3(SAND) * 0.15 + m3(GLASS) * C(0.18, 0.12, 0.12)
        tr = np.zeros((H, W), np.float32)
        for segs in TREES:
            for s in segs:
                if s[0] != 'seg': continue
                _, x, y, x2, y2, t0, dur, depth = s
                g = (t - t0) / dur
                if g <= 0: continue
                g = min(1, g)
                xe, ye = x + (x2 - x) * g, y + (y2 - y) * g
                cv2.line(tr, (int(x), int(y)), (int(xe), int(ye)), 1.0, max(1, int(depth * 1.6)), cv2.LINE_AA)
        trg = cv2.GaussianBlur(tr, (0, 0), 6)
        Cn = base + m3(tr) * GOLD * 1.4 + m3(trg) * GOLD * 1.5
        bb = np.zeros_like(P)
        for (x, y, t0, col, r) in BAUBLES:
            g = (t - t0 - 0.3) / 0.35
            if g <= 0: continue
            rr = r * min(1.0, g) * (1 + 0.25 * math.exp(-max(0, g - 1) * 4))
            cv2.circle(bb, (int(x), int(y)), max(1, int(rr)), tuple(float(c) for c in col * 1.3), -1, cv2.LINE_AA)
        Cn = Cn + bb + cv2.GaussianBlur(bb, (0, 0), 6) * 1.2 + sparkle(t, ss(2, 4, tt), GOLD)
        mix(Cn * m3(BLDs), k)
    # F: glowing ornaments in windows 35-41
    if 34.9 < t < 41.1:
        k = env(t, 35.0, 41.0, 0.4); tt = t - 35.0
        base = m3(PLASTER) * 0.1 * C(1.0, 0.75, 0.85) + m3(SAND) * 0.06
        orn = np.zeros((H, W), np.float32)
        colimg = np.zeros_like(P)
        for rank, i in enumerate(WINORDER):
            cx, cy, x0, y0, x1, y1, kind = WINFO[i]
            on = ss(0.2 + rank * 0.12, 0.8 + rank * 0.12, tt)
            if on <= 0: continue
            R = min(x1 - x0, y1 - y0) * 0.42
            th = np.linspace(0, 2 * np.pi, 200)
            kk = 3 + (i % 4)
            r = R * (0.55 + 0.45 * np.cos(kk * th + tt * (0.6 if i % 2 else -0.6))) * on
            pts = np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1).astype(np.int32)
            tmp = np.zeros((H, W), np.float32)
            cv2.polylines(tmp, [pts], True, 1.0, 2, cv2.LINE_AA)
            cv2.circle(tmp, (int(cx), int(cy)), max(1, int(R * 0.15 * on)), 1.0, -1, cv2.LINE_AA)
            colimg += m3(tmp) * PAL[(i * 3) % len(PAL)]
            orn += tmp
        glow = cv2.GaussianBlur(colimg, (0, 0), 5)
        warmw = m3(GLASS) * WARM * 0.35
        fil = m3(LINES * ss(1.5, 3.0, tt) * (1 - GLASS)) * C(0.6, 0.75, 1.0) * 0.5
        mix(base + warmw + colimg * 1.3 + glow * 1.5 + fil + sparkle(t, 0.8, C(0.8, 0.9, 1.0)), k)
    # G: light blocks + wireframe 41-47
    if 40.9 < t < 47.1:
        k = env(t, 41.0, 47.0, 0.3); tt = t - 41.0
        gx_l, gx_r = T(760, 0)[0], T(1340, 0)[0]; gy = T(0, 990)[1]
        regs = [(XX < gx_l) & (YY < gy), (XX > gx_r) & (YY < gy), (XX >= gx_l) & (XX <= gx_r) & (YY < gy), YY >= gy]
        Cn = m3(CANNY) * 0.2 * C(0.9, 0.7, 1.0)
        for j, rgn in enumerate([2, 0, 1, 3, 2, 3, 0, 1]):
            bt = tt - j * 0.4
            if bt < 0: continue
            v = math.exp(-bt * 1.4)
            Cn = Cn + m3(regs[rgn].astype(np.float32) * BLD) * (0.35 + 1.0 * LUMA[..., None]) * v * C(0.85, 0.95, 1.0)
        wire = ss(3.2, 3.8, tt)
        scan = (np.sin(YY / 6 + tt * 8) > 0.7).astype(np.float32) * 0.15
        Cn = Cn + wire * (m3(LINES) * C(1.0, 0.7, 1.0) * 1.3 + m3(CANNY * 0.7) * C(1.0, 0.6, 0.9) + m3(scan * BLD) * C(1.0, 0.5, 0.8))
        mix(Cn, k)
    # H: glitch 47-50
    if 46.9 < t < 50.1:
        k = env(t, 47.0, 50.0, 0.2); tt = t - 47.0
        base = m3(PLASTER) * C(0.38, 0.78, 1.0) + m3(SAND) * C(0.25, 0.38, 0.85) + m3(GLASS) * C(1.0, 0.4, 0.8)
        fr = int(t * 12)
        rr = np.random.default_rng(fr)
        out = base.copy()
        for _ in range(14):
            y0 = int(rr.uniform(0, H)); hh = int(rr.uniform(5, 60)); sx = int(rr.uniform(-80, 80))
            out[y0:y0 + hh] = np.roll(base[y0:y0 + hh], sx, axis=1)
        b, g, r = out[..., 0], out[..., 1], out[..., 2]
        sp = int(6 + 6 * math.sin(tt * 9))
        out = np.stack([np.roll(b, -sp, 1), g, np.roll(r, sp, 1)], -1)
        if rr.random() < 0.3: out = 1.2 - out
        if rr.random() < 0.25: out = m3(LINES) * C(1, 1, 1) * 1.5
        mix(out * m3(BLD) * (0.8 + 0.4 * rr.random()), k)
    # I: neon swirl finale 50-60
    if t >= 49.9:
        k = ss(50.0, 50.4, t); tt = t - 50.0
        grad = np.clip(YY / H, 0, 1)[..., None]
        base = m3(PLASTER) * (C(0.75, 0.35, 0.9) * (1 - grad) + C(0.6, 0.25, 1.0) * grad) * 0.55 + m3(SAND) * C(0.4, 0.2, 0.35)
        cols = np.zeros((NWIN + 1, 3), np.float32); inten = np.ones(NWIN + 1, np.float32) * 0.8
        glass = np.zeros_like(P)
        for i in range(NWIN):
            cols[i + 1] = PAL[(i + int(tt * 2)) % len(PAL)]
            glass += m3(WMASKS[i]) * PAL[(i * 2 + int(tt * 2.5)) % len(PAL)] * 0.9
        rings = neon_rings(cols, inten, 4.0, 6.0)
        sw = np.zeros_like(P)
        for (pts, col, ph) in SWIRLS:
            n = len(pts)
            grow = 1.0 if FORCE_SWIRL else ss(ph * 2, ph * 2 + 2.0, tt)
            m = int(n * grow)
            if m < 2: continue
            cv2.polylines(sw, [pts[:m].astype(np.int32)], False, tuple(float(c) for c in col * 1.4), 3, cv2.LINE_AA)
        sw = sw + cv2.GaussianBlur(sw, (0, 0), 6) * 1.5
        pul = 0.85 + 0.15 * math.sin(tt * 6.3)
        Cn = (base + glass + rings + sw * m3(BLD)) * pul + sparkle(t, 1.0)
        fl = math.exp(-max(0, t - 57.6) * 3) * (t > 57.6)
        Cn = Cn + fl * 1.2
        Cn = Cn * (1 - ss(58.0, 59.4, t)) + m3(LINES) * C(0.95, 0.8, 1.0) * env(t, 58.4, 59.8, 0.4) * 0.8
        mix(Cn, k)
    return P * m3(BLDs)

# ---------- preview composite ----------
NIGHT = F * C(0.11, 0.08, 0.09) * m3(BLDs)
GAIN = 0.4 + 0.6 * np.clip(np.power(F, 0.75) * 1.2, 0, 1)
OX, OY = (CW - W) // 2 + 60, 30
yv, xv = np.mgrid[0:CH, 0:CW].astype(np.float32)
VIG = np.clip(1 - 0.35 * (((xv - CW / 2) / (CW / 2)) ** 2 + ((yv - CH / 2) / (CH / 2)) ** 2), 0, 1)[..., None]
FLOOR = np.clip((yv - (OY + H - 2)) / 60, 0, 1)[..., None]
del yv, xv

def frame(t):
    P = scene(t)
    tex = NIGHT + GAIN * P
    can = np.zeros((CH, CW, 3), np.float32)
    can[OY:OY + H, OX:OX + W] = tex
    # floor + reflection
    fy0 = OY + H
    refl = cv2.flip(tex[-(CH - fy0):], 0) if CH - fy0 > 0 else None
    if refl is not None:
        hgt = refl.shape[0]
        fade = np.linspace(0.22, 0.0, hgt)[:, None, None]
        can[fy0:fy0 + hgt, OX:OX + W] += cv2.GaussianBlur(refl, (0, 0), 3) * fade
    can[fy0:] += np.array([0.03, 0.022, 0.025], np.float32)
    sm = cv2.resize(can, (CW // 4, CH // 4), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(np.maximum(sm - 0.55, 0), (0, 0), 5)
    can += cv2.resize(sm, (CW, CH)) * 0.8
    can *= VIG
    can *= ss(0, 0.6, t)
    out = np.clip(np.power(np.clip(can, 0, 1), 1 / 1.05) * 255, 0, 255).astype(np.uint8)
    return out, P

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stills':
        for tm in [float(x) for x in sys.argv[2].split(',')]:
            cv2.imwrite(f'/home/claude/mapping/s3_{tm:05.1f}.jpg', frame(tm)[0])
    elif mode == 'masks':
        vis = np.zeros((H, W, 3), np.float32)
        vis += m3(PLASTER) * C(0.9, 0.9, 0.9) + m3(SAND) * C(0.2, 0.3, 0.8) + m3(ROOF) * C(0.5, 0.4, 0.3) + m3(GLASS) * C(1.0, 0.7, 0.1)
        vis += m3(FRAMES) * C(0.0, 0.5, 0.0)
        cv2.imwrite('/home/claude/mapping/masks.jpg', np.hstack([np.clip(vis, 0, 1) * 255, F * 255, m3(LINES) * 255 * np.ones(3)]).astype(np.uint8))
    else:
        f0, f1, tag = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        pv = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{CW}x{CH}', '-r', str(FPS), '-i', '-',
                               '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p', f'prev_{tag}.mp4'], stdin=subprocess.PIPE)
        pc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                               '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', f'cont_{tag}.mp4'], stdin=subprocess.PIPE)
        for i in range(f0, f1):
            o, P = frame(i / FPS)
            pv.stdin.write(o.tobytes())
            pc.stdin.write(np.clip(P * 255, 0, 255).astype(np.uint8).tobytes())
            if i % 60 == 0: print(tag, i, flush=True)
        pv.stdin.close(); pc.stdin.close(); pv.wait(); pc.wait()
