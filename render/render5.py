import sys, os, math, subprocess, cv2, numpy as np
import base5 as R
from base5 import (F, W, H, XX, YY, BLD, BLDs, PLASTER, SAND, ROOF, GLASS, FRAMES, LINES, CANNY, NZ, ORDER,
                   WINFO, WMASKS, NWIN, ss, C, m3, T, FPS, CW, CH)

HERE = os.path.dirname(os.path.abspath(__file__))
DUR = 120.0
rng = np.random.default_rng(21)
NEG = -20000.0
K = H / 1000.0
REL = 1.7 * K
DEP = 2.1 * K
CUR_T = [0.0]

def rmap(img, x, y, border=0.0):
    return cv2.remap(np.asarray(img, np.float32), np.asarray(x, np.float32), np.asarray(y, np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

# ---------- base relief ----------
joints = CANNY * SAND
HB = (FRAMES * 9.0 + SAND * 2.5 - GLASS * 16.0 - joints * 1.5) * REL
gy_line = int(T(0, 990)[1])
HB[gy_line:gy_line + int(6 * K)] += 5.0 * REL * BLD[gy_line:gy_line + int(6 * K)]
HB = cv2.GaussianBlur(HB.astype(np.float32), (0, 0), 1.0 * K)
GRAYF = F.mean(axis=2, keepdims=True)
LUMA = GRAYF[..., 0]

H2, W2 = H // 2, W // 2
XXh, YYh = np.mgrid[0:H2, 0:W2].astype(np.float32)[::-1]
YYh, XXh = np.mgrid[0:H2, 0:W2].astype(np.float32)

# tiles
TSZ = int(36 * K)
ty_n, tx_n = H // TSZ + 1, W // TSZ + 1
TR1 = rng.random((ty_n, tx_n)).astype(np.float32)
TR2 = rng.random((ty_n, tx_n)).astype(np.float32)
tyc, txc = np.mgrid[0:ty_n, 0:tx_n].astype(np.float32)
TDIAG = ((txc * TSZ) + (H - tyc * TSZ)) / (W + H)
lx_ = (XX % TSZ); ly_ = (YY % TSZ)
TBEVEL = np.clip(np.minimum(np.minimum(lx_, TSZ - 1 - lx_), np.minimum(ly_, TSZ - 1 - ly_)) / (2.5 * K), 0, 1).astype(np.float32)
def up(a, ts=TSZ, nx=tx_n, ny=ty_n):
    return cv2.resize(a, (nx * ts, ny * ts), interpolation=cv2.INTER_NEAREST)[:H, :W]

# pins (for Ethnik 101)
PSZ = int(22 * K)
py_n, px_n = H // PSZ + 1, W // PSZ + 1
plx = (XX % PSZ); ply = (YY % PSZ)
PBEVEL = np.clip(np.minimum(np.minimum(plx, PSZ - 1 - plx), np.minimum(ply, PSZ - 1 - ply)) / (2.0 * K), 0, 1).astype(np.float32)

# voronoi shards (computed at half res, upscaled nearest)
NPTS = 64
PTS = np.stack([rng.uniform(0, W, NPTS), rng.uniform(0, H, NPTS)], 1).astype(np.float32)
d2 = np.full((H2, W2), 1e12, np.float32); VORh = np.zeros((H2, W2), np.int32)
for i, (px, py) in enumerate(PTS):
    d = (XXh * 2 - px) ** 2 + (YYh * 2 - py) ** 2
    m = d < d2; d2[m] = d[m]; VORh[m] = i
VOR = cv2.resize(VORh.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
VEDGE = (cv2.morphologyEx(VOR.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0).astype(np.float32) * BLD
VDEL = rng.uniform(0, 1, NPTS).astype(np.float32)
VORD = (np.hypot(PTS[:, 0] - W / 2, PTS[:, 1] - H * 0.45) / (W * 0.7)).astype(np.float32)

def interior_tex():
    img = np.zeros((H, W, 3), np.float32)
    for i in range(NWIN):
        cx, cy, x0, y0, x1, y1, k = WINFO[i]
        u = np.clip((XX - x0) / max(1, x1 - x0), 0, 1); v = np.clip((YY - y0) / max(1, y1 - y0), 0, 1)
        lamp = np.exp(-(((u - 0.5) / 0.35) ** 2 + ((v - 0.25) / 0.3) ** 2))
        shelves = (np.sin(v * 40) > 0.85).astype(np.float32) * 0.25
        img += m3(WMASKS[i]) * (C(0.25, 0.6, 1.0) * m3(0.35 + 1.2 * lamp) + m3(shelves))
    return img
INTERIOR = interior_tex()

# ---------- video sources ----------
class Clip:
    def __init__(self, name):
        self.path = os.path.join(HERE, name); self.cap = None; self.last = None; self.lastf = None
    def get(self, sec):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.path)
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)); self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        f = int(sec * self.fps) % max(1, self.n)
        if f == self.lastf: return self.last
        if self.lastf is None or f != self.lastf + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, fr = self.cap.read()
        self.last = fr.astype(np.float32) / 255 if ok else np.zeros((1080, 1920, 3), np.float32); self.lastf = f
        return self.last

def afterglow(fr, decay=0.80):
    """strobe-free but fluid: each frame = max(current, previous*decay), run twice around the loop"""
    n = len(fr); out = np.empty_like(fr); prev = fr[-1].astype(np.float32) * 0
    for rnd in range(2):
        for i in range(n):
            cur = np.maximum(fr[i].astype(np.float32), prev * decay)
            prev = cur
            if rnd == 1: out[i] = cur.astype(fr.dtype) if fr.dtype != np.float32 else cur
    sm = np.empty_like(out)
    for i in range(n):
        sm[i] = ((out[i - 1].astype(np.float32) + 2 * out[i].astype(np.float32) + out[(i + 1) % n].astype(np.float32)) / 4).astype(out.dtype)
    return sm

def _e07_stable():
    cap = cv2.VideoCapture(os.path.join(HERE, 'Ethnik3_07.mov')); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (960, 540)).astype(np.float32) / 255)
    fr = np.stack(fr); n = len(fr)
    return afterglow(fr, 0.86)
E07S = _e07_stable()
def e07_frame(sec):
    return E07S[int(sec * 30) % len(E07S)]
_E84 = None
def e84_frame(sec):
    global _E84
    if _E84 is None:
        cap = cv2.VideoCapture(os.path.join(HERE, 'Ethnik3_84.mov')); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(cv2.cvtColor(cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
        fr = np.stack(fr); n = len(fr)
        mx = afterglow(fr, 0.84)
        means = mx.reshape(n, -1).mean(1) / 255
        sm = np.array([means[max(0, i - 12):i + 13].mean() for i in range(n)])
        _E84 = (mx, (0.16 / np.maximum(sm, 0.02)).astype(np.float32))
    mx, gain = _E84
    i = int(sec * 30) % len(mx)
    v = np.clip(mx[i].astype(np.float32) / 255 * gain[i], 0, 1)
    return np.repeat(v[..., None], 3, axis=2)
_E101 = None
def e101_small(sec):
    global _E101
    if _E101 is None:
        cap = cv2.VideoCapture(os.path.join(HERE, 'Ethnik3_101.mov')); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            v = place(f.astype(np.float32) / 255, 1.0, W / 2, H * 0.47)
            fr.append(cv2.resize(v, (px_n, py_n), interpolation=cv2.INTER_AREA))
        fr = np.stack(fr); n = len(fr)
        mx = np.stack([fr[max(0, i - 2):i + 3].max(0) for i in range(n)])
        _E101 = np.stack([mx[max(0, i - 2):i + 3].mean(0) for i in range(n)])
    return _E101[int(sec * 30) % len(_E101)]
_E101F = None
def e101_full(sec):
    global _E101F
    if _E101F is None:
        cap = cv2.VideoCapture(os.path.join(HERE, 'Ethnik3_101.mov')); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA))
        fr = np.stack(fr); n = len(fr)
        _E101F = afterglow(fr, 0.80)
    return _E101F[int(sec * 30) % len(_E101F)].astype(np.float32) / 255 * 1.15
POLY = Clip('poly.mp4'); E07 = Clip('Ethnik3_07.mov'); E84 = Clip('Ethnik3_84.mov')
E101 = Clip('Ethnik3_101.mov'); E124 = Clip('Ethnik3_124.mov')

def place(img, scale_w, cx, cy):
    """place a video frame into tex space: width = scale_w * W, centred at (cx, cy) in tex px"""
    h, w = img.shape[:2]
    nw = int(W * scale_w); nh = int(nw * h / w)
    big = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((H, W) + img.shape[2:], np.float32)
    ox = int(cx - nw / 2); oy = int(cy - nh / 2)
    xa, ya = max(0, ox), max(0, oy); xb, yb = min(W, ox + nw), min(H, oy + nh)
    if xb > xa and yb > ya:
        out[ya:yb, xa:xb] = big[ya - oy:yb - oy, xa - ox:xb - ox]
    return out

# ---------- facade colour grading ----------
TC0 = C(0.035, 0.04, 0.085); TC1 = C(0.20, 0.27, 0.58); TC2 = C(0.80, 0.90, 1.0)
def tone(g):
    g3 = m3(np.clip(g, 0, 1)); a = np.clip(g3 * 2, 0, 1); b = np.clip(g3 * 2 - 1, 0, 1)
    return TC0 * (1 - a) + TC1 * a + (TC2 - TC1) * b
def lum(img): return img.mean(axis=2)
def grade(vid, keep=0.35):
    """pull a clip into the facade palette, keep a little of its own colour"""
    g = lum(vid)
    return tone(np.clip(g * 1.15, 0, 1)) * (1 - keep) + vid * keep * (0.6 + 0.4 * m3(g))

def gothic_hall():
    img = np.zeros((H, W, 3), np.float32)
    cx, base = W / 2, H * 0.93
    for k in range(14, 0, -1):
        s_ = 0.86 ** k
        w = W * 0.95 * s_; h = H * 1.25 * s_
        x0, x1 = cx - w / 2, cx + w / 2; ys = base - h * 0.55; ya = base - h
        sc = (0.866 * w) / max(1, ys - ya)
        inside = (XX >= x0) & (XX <= x1) & (YY <= base) & (YY >= ya)
        dy = (ys - YY) * sc
        arch = (np.hypot(XX - x0, dy) <= w) & (np.hypot(XX - x1, dy) <= w)
        m = (inside & ((YY >= ys) | arch)).astype(np.float32)
        ring = np.clip(m - cv2.erode(m, np.ones((int(18 * K * s_ + 3),) * 2, np.uint8)), 0, 1)
        shade = 0.55 * (1 - k / 15) + 0.08
        img = img * (1 - m3(ring)) + m3(ring) * TC1 * shade * 1.6
    glow = np.exp(-(((XX - cx) / (W * 0.28)) ** 2 + ((YY - H * 0.45) / (H * 0.32)) ** 2))
    img += m3(glow) * C(0.25, 0.5, 0.95) * 0.35
    floor = np.clip((YY - H * 0.80) / (H * 0.2), 0, 1)
    img += m3(floor * glow) * TC1 * 0.3
    return img
HALL = gothic_hall()

# ---------- renderer (half-res ray march, full-res shading) ----------
def render(Hm, alb, light, Lcol=C(1, 0.97, 0.92), amb=0.12, spec=0.0, shin=40.0, specCol=C(1, 1, 1),
           emis=None, layers=64, fog=300.0, back_alb=None, back_z=None, back_scale=1.0, mask=None, back_emis=False, thin_back=None, back_par=None):
    t = CUR_T[0]
    vx, vy, vz = (W / 2 + 140 * K * math.sin(t * 0.23), H * 0.92 - 60 * K * math.sin(t * 0.17), 1750.0 * K)
    bm = BLD if mask is None else mask
    Hm = np.where(bm > 0.5, Hm, NEG).astype(np.float32)
    inside = Hm > NEG + 1
    zmax = float(Hm[inside].max()); zmin = float(Hm[inside].min())
    # --- half-res march
    Hh = cv2.resize(Hm, (W2, H2), interpolation=cv2.INTER_NEAREST) * 0.5
    dxh = ((vx / 2 - XXh) / (vz / 2)).astype(np.float32); dyh = ((vy / 2 - YYh) / (vz / 2)).astype(np.float32)
    rng_z = max(1.0, zmax - zmin)
    N = int(np.clip(rng_z / 2 / 3.0, 6, layers))
    zs = np.linspace(zmax / 2, zmin / 2, N + 1)
    hz = np.full((H2, W2), zmin / 2, np.float32); found = np.zeros((H2, W2), bool)
    prev = None; zprev = None
    for z in zs:
        h = rmap(Hh, XXh + dxh * z, YYh + dyh * z, NEG)
        diff = h - z
        hit = (diff >= 0) & ~found
        if prev is None:
            hz[hit] = z
        else:
            tt = np.clip(prev / (prev - diff + 1e-6), 0, 1)
            hz[hit] = (zprev + (z - zprev) * tt)[hit]
        found |= hit; prev = diff; zprev = z
    if thin_back is not None:
        hz[~found] = thin_back / 2
    # half-res shadows
    lx, ly, lz = light; ll = math.sqrt(lx * lx + ly * ly + lz * lz); lx, ly, lz = lx / ll, ly / ll, lz / ll
    hxh = XXh + dxh * hz; hyh = YYh + dyh * hz
    sh = np.ones((H2, W2), np.float32)
    steps = 18; maxd = min(260.0 * DEP / 2, (rng_z / 2 + 10) / max(0.15, lz) * 1.1)
    for k in range(1, steps + 1):
        tt = maxd * k / steps
        h = rmap(Hh, hxh + lx * tt, hyh + ly * tt, NEG)
        occ = np.clip((h - (hz + lz * tt)) / 2.0, 0, 1)
        sh = np.minimum(sh, 1 - occ)
    sh = cv2.resize(cv2.GaussianBlur(sh, (0, 0), 1.0), (W, H), interpolation=cv2.INTER_LINEAR)
    hzf = cv2.resize(hz, (W, H), interpolation=cv2.INTER_LINEAR) * 2.0
    # --- full-res shading
    dx = (vx - XX) / vz; dy = (vy - YY) / vz
    hx = XX + dx * hzf; hy = YY + dy * hzf
    A = rmap(alb, hx, hy)
    if back_alb is not None and back_z is not None:
        deep = hzf <= back_z
        if back_par is not None:
            bx = XX + (hx - XX) * back_par; by = YY + (hy - YY) * back_par
        else:
            bx = W / 2 + (hx - W / 2) * back_scale; by = H / 2 + (hy - H / 2) * back_scale
        BA = rmap(back_alb, bx, by)
        A = np.where(m3(deep), BA, A)
    Hc = Hm.clip(zmin - 5, None)
    gx = cv2.Sobel(Hc, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(Hc, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gxs = rmap(gx, hx, hy); gys = rmap(gy, hx, hy)
    nlen = np.sqrt(gxs * gxs + gys * gys + 1)
    nx, ny, nz = -gxs / nlen, -gys / nlen, 1 / nlen
    dif = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
    hb = cv2.resize(cv2.GaussianBlur(cv2.resize(Hc, (W2, H2), interpolation=cv2.INTER_AREA), (0, 0), 5 * K), (W, H))
    ao = np.clip(1 - (rmap(hb, hx, hy) - hzf) / (45.0 * DEP), 0.25, 1.0)
    col = A * (m3(amb * ao) + m3(dif * sh) * Lcol)
    if spec > 0:
        vn = np.sqrt(dx * dx + dy * dy + 1)
        hxv, hyv, hzv = lx + dx / vn, ly + dy / vn, lz + 1 / vn
        hn = np.sqrt(hxv * hxv + hyv * hyv + hzv * hzv)
        sp = np.clip((nx * hxv + ny * hyv + nz * hzv) / hn, 0, 1) ** shin
        col += m3(sp * sh * spec) * specCol
    if emis is not None:
        col += rmap(emis, hx, hy)
    if back_emis and back_alb is not None:
        col = np.where(m3(deep), BA, col)
    col *= m3(np.exp(np.minimum(hzf, 0) / fog))
    col *= m3(rmap(cv2.GaussianBlur(bm.astype(np.float32), (0, 0), 1.0), hx, hy) * (hzf > NEG + 10))
    return col

# ---------- scenes ----------
def sc_lines(t, t0, rev_dir=False):
    tt = t - t0
    prog = ss(0.6, 4.6, tt) * 1.05
    rev = np.clip((prog - ORDER) / 0.02, 0, 1)
    head = np.exp(-np.abs(prog - ORDER) / 0.012) * (prog < 1.04)
    return m3(LINES * rev + LINES * head * 2.5 + CANNY * rev * 0.2) * C(0.95, 0.88, 1.0)

def sc_relight(t, t0, cool=False):
    tt = t - t0
    ang = math.pi * (1.05 - 0.6 * ss(0, 6.5, tt))
    light = (math.cos(ang), -0.55, 0.42 + 0.25 * ss(4, 6.5, tt))
    if cool: light = (math.cos(ang) * 0.6, -0.9, 0.35)
    depth = 1.6 + 1.6 * ss(1.5, 5.0, tt)
    Lc = C(1.0, 0.85, 0.7) * 1.3 if cool else C(1.0, 0.95, 0.88) * 1.35
    return render(HB * depth, F, light, Lcol=Lc, amb=0.06, spec=0.25, shin=20)

def sc_blocks(t, t0):
    tt = t - t0
    d = TDIAG * 1.8 + TR1 * 0.35
    ph = np.clip((tt - d) / 1.0, 0, 1)
    bump = np.sin(np.pi * ph) * (25 + 55 * TR2) * DEP
    d2 = 3.6 + TR1 * 1.8
    ph2 = np.clip((tt - d2) / 1.4, 0, 1)
    sink = -np.sin(np.pi * ph2) * (20 + 50 * TR2) * DEP
    Hm = HB * 1.5 + up(bump + sink) * TBEVEL * BLD
    light = (-0.55 + 0.3 * math.sin(tt * 0.4), -0.65, 0.5)
    return render(Hm, F * 1.05, light, Lcol=C(1.0, 0.95, 0.9) * 1.35, amb=0.07, spec=0.2, shin=25)

def sc_windows(t, t0):
    tt = t - t0
    pop = ss(0.3, 1.4, tt) * (1 - ss(2.6, 3.4, tt))
    deep = ss(3.0, 4.2, tt) * (1 - ss(6.0, 6.8, tt))
    box = np.clip(GLASS + FRAMES, 0, 1)
    Hm = HB * 1.5 + box * 55 * DEP * pop - GLASS * 240 * DEP * deep
    emis = INTERIOR * (0.3 + 1.2 * deep) * m3(GLASS)
    alb = np.where(m3(GLASS > 0) & (deep > 0.05), INTERIOR * 0.6, F)
    return render(Hm, alb, (-0.5, -0.6, 0.55), Lcol=C(1.0, 0.95, 0.9) * 1.25, amb=0.08, spec=0.2, emis=emis, fog=420 * DEP)

GABLE = ((XX > T(780, 0)[0]) & (XX < T(1330, 0)[0]) & (YY < T(0, 985)[1])).astype(np.float32) * BLD

def sc_portal84(t, t0):
    """Ethnik 84: seen through the windows, then the stone of the centre dissolves softly into the gothic hall"""
    tt = t - t0
    op = ss(0.4, 1.6, tt) * (1 - ss(9.4, 10.6, tt))
    big = ss(3.4, 5.6, tt) * (1 - ss(7.8, 9.6, tt))
    back_z = -700.0 * K
    fig = place(np.clip(e84_frame(tt) ** 0.9 * 1.25, 0, 1), 1.05, W / 2, H * 0.42)
    gv = lum(fig)
    alpha = m3(np.clip((gv - 0.04) / 0.20, 0, 1))
    zoom = 1.0 + 0.03 * tt
    hall = cv2.warpAffine(HALL, np.float32([[zoom, 0, (1 - zoom) * W / 2], [0, zoom, (1 - zoom) * H * 0.45]]), (W, H))
    vid = hall * (1 - alpha) + tone(np.clip(gv * 1.1, 0, 1)) * 1.15 * alpha
    alb = F * (0.55 - 0.2 * op)
    light = (-0.35, -0.8, 0.5)
    Hm1 = np.where((GLASS > 0.5) & (op > 0.5), NEG, HB * 1.6)
    kw = dict(Lcol=C(1.0, 0.97, 0.95), amb=0.06, spec=0.15, fog=1e9, back_alb=vid, back_z=back_z + 1.0,
              layers=40, back_emis=True, thin_back=back_z, back_par=0.12)
    R1 = render(Hm1, alb, light, **kw)
    if big < 0.005:
        return R1
    Hm2 = np.where(GABLE > 0.5, NEG, Hm1)
    R2 = render(Hm2, alb, light, **kw)
    rr = np.hypot((XX - W / 2) / (W * 0.30), (YY - H * 0.42) / (H * 0.45))
    grow = np.clip((big * 1.8 - rr) / 0.45 + (NZ - 0.5) * 0.6, 0, 1)
    grow = grow * grow * (3 - 2 * grow)
    a = m3(cv2.GaussianBlur(GABLE, (0, 0), 28 * K) * grow)
    return R1 * (1 - a) + R2 * a

def poly_frame(sec):
    g = POLY.get(sec).mean(axis=2)
    return place(g, 1.25, W / 2, T(0, 800)[1])

def sc_collapse(t, t0):
    tt = t - t0
    crack = ss(0.0, 1.2, tt) * (1 - ss(1.8, 2.3, tt))
    fall_t = np.clip(tt - (1.6 + VORD * 1.4 + VDEL * 0.5), 0, None)
    back_t = np.clip(tt - (4.4 + (1 - VORD) * 1.2 + VDEL * 0.3), 0, 1.0)
    zc = -np.minimum(fall_t ** 2 * 260 * DEP, 700 * DEP)
    zc = zc * (1 - ss(0, 1, back_t)) + 8 * DEP * np.sin(np.clip(back_t, 0, 1) * np.pi) * (back_t > 0) * (back_t < 1)
    zcell = zc[VOR]
    room_z = -420.0 * DEP
    pf = cv2.GaussianBlur(poly_frame(tt), (0, 0), 1.2 * K)
    back = room_z + pf * 160.0 * DEP
    Hm = np.maximum(HB * 1.4 + zcell - VEDGE * 6 * K * crack, back)
    emis = m3(VEDGE * crack) * C(0.7, 0.85, 1.0) * 2.0
    palb = m3(pf) * C(0.95, 0.97, 1.0) * 1.5
    return render(Hm, F, (-0.4, -0.7, 0.6), Lcol=C(1.0, 0.96, 0.92) * 1.3, amb=0.25, spec=0.3, emis=emis, fog=1400 * DEP,
                  back_alb=palb, back_z=-200.0 * DEP, layers=80)

def sc_ripple(t, t0):
    tt = t - t0
    amp = 9 * 1.5 * K * ss(0.2, 1.5, tt) * (1 - ss(5.8, 6.8, tt))
    c1 = T(1266, 1180); c2 = T(700, 500)
    r1 = np.hypot(XX - c1[0], YY - c1[1]); r2 = np.hypot(XX - c2[0], YY - c2[1])
    wav = amp * (np.sin(r1 / (30 * K) - tt * 5) * np.exp(-r1 / (900 * K)) + 0.7 * ss(2, 3, tt) * np.sin(r2 / (26 * K) - tt * 6) * np.exp(-r2 / (700 * K)))
    Hm = HB * 1.3 + wav * BLD
    alb = (0.4 * F + 0.6 * GRAYF) * C(1.0, 0.95, 0.9)
    light = (-0.6 + 0.4 * math.sin(tt * 0.5), -0.6, 0.5)
    return render(Hm, alb, light, Lcol=C(1.0, 0.9, 0.8) * 1.15, amb=0.08, spec=1.4, shin=60, specCol=C(1.0, 0.95, 0.9))

def sc_relief07(t, t0):
    """Ethnik 07: the drawn head grows out of the plaster as a white stone relief"""
    tt = t - t0
    amp = ss(0.3, 3.0, tt) * (1 - ss(8.0, 10.8, tt))
    v = place(e07_frame(tt), 1.0, W * 0.52, H * 0.66)
    feather = np.exp(-(((XX - W * 0.50) / (W * 0.40)) ** 2 + ((YY - H * 0.62) / (H * 0.36)) ** 4))
    reveal = np.clip((amp * 1.3 - (1 - feather)) / 0.3, 0, 1)
    v = np.clip(v * 1.15, 0, 1) ** 1.3 * feather * reveal
    rel = cv2.GaussianBlur(v, (0, 0), 9 * K) * 0.8 + cv2.GaussianBlur(v, (0, 0), 1.2 * K) * 0.2
    Hm = HB * 1.4 + rel * 16 * DEP * amp
    base = 0.5 * F + 0.5 * GRAYF * C(0.92, 0.96, 1.0)
    alb = F * 0.30 * (1 - amp) + base * m3(0.22 + 1.1 * v) * amp
    emis = tone(v) * m3(v * amp * (1 - 0.8 * GLASS)) * 0.3
    ang = math.pi * (1.0 - 0.6 * ss(0, 10, tt))
    return render(Hm, alb, (math.cos(ang), -0.5, 0.45), Lcol=C(1.0, 0.95, 0.9) * 1.2, amb=0.25, spec=0.3, shin=30, emis=emis)

def sc_monolith101(t, t0):
    """Ethnik 101: the figure swells softly out of the gable stone and sinks back"""
    tt = t - t0
    ext = ss(0.3, 2.8, tt) * (1 - ss(8.0, 10.8, tt))
    soft = cv2.GaussianBlur(GABLE, (0, 0), 45 * K) * BLD
    soft = np.clip(soft / soft.max(), 0, 1)
    fr = np.clip(e101_full(tt), 0, 1) ** 0.55
    vid = grade(place(fr, 1.0, W / 2, H * 0.40), keep=0.6)
    gv = lum(vid)
    breathe = 1 + 0.08 * math.sin(tt * 1.3)
    a = soft * np.clip((ext * 1.25 - (1 - soft)) / 0.35, 0, 1)
    Hm = HB * 1.3 + (a * 55 * DEP * breathe + cv2.GaussianBlur(gv, (0, 0), 4 * K) * a * 22 * DEP)
    alb = F * 0.35 * m3(1 - a) + vid * m3(a)
    emis = vid * m3(a) * (0.6 + 0.4 * m3(GLASS))
    return render(Hm, alb, (-0.45, -0.7, 0.55), Lcol=C(1.0, 0.95, 0.9) * 1.1, amb=0.15, spec=0.25, shin=40, emis=emis, layers=60)

def sc_temple124(t, t0):
    """Ethnik 124: the facade splits like a temple gate and reveals the stone mask deep inside"""
    tt = t - t0
    o = ss(0.6, 3.6, tt) * (1 - ss(8.8, 11.8, tt))
    sx = o * W * 0.36
    zhalf = -o * 70 * DEP
    cx = W / 2
    left = XX < (cx - sx); right = XX > (cx + sx); gap = ~(left | right)
    srcx = np.where(left, XX + sx, np.where(right, XX - sx, XX)).astype(np.float32)
    Hh = rmap(HB * 1.4, srcx, YY, NEG)
    Bh = rmap(BLD, srcx, YY)
    Fh = rmap(F, srcx, YY)
    vid = grade(place(E124.get(tt), 1.05, W / 2, H * 0.43), keep=0.15) * 1.15
    vg = np.clip(1.25 - np.hypot((XX - W / 2) / (W * 0.40), (YY - H * 0.43) / (H * 0.42)), 0, 1)
    vid = vid * m3(vg * vg * (3 - 2 * vg))
    lum = cv2.GaussianBlur(vid.mean(axis=2), (0, 0), 1.2 * K)
    zb = -380 * DEP + lum * 90 * DEP * (0.3 + 0.7 * o)
    Hm = np.where(gap, zb, Hh + zhalf)
    alb = np.where(m3(gap), vid * 0.6, Fh)
    mask = np.where(gap, BLD, Bh * BLD)
    emis = vid * 0.7 * m3(gap.astype(np.float32))
    return render(Hm, alb, (0.0, -0.75, 0.6), Lcol=C(1.0, 0.95, 0.9) * 1.5, amb=0.3, spec=0.5, shin=35, emis=emis,
                  fog=2500 * DEP, mask=mask, layers=90)

def sc_finale(t, t0):
    tt = t - t0
    beat = 0.6
    p = math.exp(-((tt % beat)) * 6) if tt > 0.6 else 0.0
    gable = cv2.GaussianBlur(GABLE, (0, 0), 1.0 * K)
    big = ss(3.6, 4.6, tt) * (1 - ss(5.4, 6.1, tt)) + ss(8.0, 8.8, tt) * (1 - ss(10.0, 10.8, tt))
    box = np.clip(GLASS + FRAMES, 0, 1)
    Hm = HB * 1.6 + box * 18 * DEP * p + gable * 70 * DEP * big
    alb = m3(PLASTER) * C(0.78, 0.9, 1.0) + m3(SAND) * C(0.28, 0.33, 0.72) + m3(ROOF) * C(0.3, 0.3, 0.32) + m3(GLASS) * C(0.1, 0.1, 0.1)
    alb = alb * (0.75 + 0.25 * m3(LUMA))
    emis = INTERIOR * m3(GLASS) * (0.9 + 0.5 * p)
    ang = math.pi * (0.95 - 0.55 * ss(0, 11, tt))
    return render(Hm, alb, (math.cos(ang), -0.6, 0.45), Lcol=C(0.8, 0.92, 1.0) * 1.4, amb=0.1, spec=0.3, shin=20, emis=emis)

TL = [  # start, end, fn, t0
    (5.5, 15.2, sc_relight, 6.5),
    (14.0, 23.2, sc_blocks, 14.0),
    (22.0, 31.2, sc_windows, 22.0),
    (30.0, 41.4, sc_portal84, 30.0),
    (40.2, 49.4, sc_collapse, 40.2),
    (48.2, 56.4, sc_ripple, 48.2),
    (55.2, 66.4, sc_relief07, 55.2),
    (65.2, 76.4, sc_monolith101, 65.2),
    (75.2, 87.8, sc_temple124, 75.2),
    (86.6, 96.2, lambda t, t0: sc_relight(t, t0, cool=True), 86.6),
    (95.0, 108.4, sc_finale, 95.0),
]
FADE = 1.2

def scene(t):
    CUR_T[0] = t
    P = np.zeros((H, W, 3), np.float32)
    if t < 7.0:
        P += sc_lines(t, 0.0) * (1 - ss(5.8, 7.0, t))
    for (a, b, fn, t0) in TL:
        if a <= t < b:
            k = ss(a, a + FADE, t) * (1 - ss(b - FADE, b, t))
            if a == 5.5: k = ss(5.5, 7.0, t) * (1 - ss(b - FADE, b, t))
            if k <= 0.001: continue
            P = P + fn(t, t0) * k
    fl = math.exp(-max(0, t - 107.6) * 4) * (t > 107.6)
    P = P + fl * 0.8 * m3(BLDs)
    if t >= 108.0:
        P = P + sc_lines(t, 108.0) * (1 - ss(113.5, 116.0, t)) * ss(108.0, 108.6, t)
    return P * m3(BLDs)

R.scene = scene
BH = 2100; BW = int(round(W * BH / H / 2)) * 2
OX4, OY4 = (CW - BW) // 2 + 60, 30
_yv = np.arange(CH, dtype=np.float32)[:, None]; _xv = np.arange(CW, dtype=np.float32)[None, :]
VIG4 = np.clip(1 - 0.35 * (((_xv - CW / 2) / (CW / 2)) ** 2 + ((_yv - CH / 2) / (CH / 2)) ** 2), 0, 1)[..., None].astype(np.float32)

def frame4k(t):
    P = scene(t)
    tex = R.NIGHT + R.GAIN * P
    big = cv2.resize(tex, (BW, BH), interpolation=cv2.INTER_CUBIC)
    can = np.zeros((CH, CW, 3), np.float32)
    can[OY4:OY4 + BH, OX4:OX4 + BW] = big
    fy0 = OY4 + BH; hgt = CH - fy0
    if hgt > 0:
        refl = cv2.flip(big[-hgt:], 0)
        can[fy0:, OX4:OX4 + BW] += cv2.GaussianBlur(refl, (0, 0), 4) * np.linspace(0.22, 0.0, hgt, dtype=np.float32)[:, None, None]
        can[fy0:] += np.array([0.03, 0.022, 0.025], np.float32)
    sm = cv2.resize(can, (CW // 8, CH // 8), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(np.maximum(sm - 0.55, 0), (0, 0), 3)
    can += cv2.resize(sm, (CW, CH)) * 0.8
    can *= VIG4
    can *= ss(0, 0.6, t)
    out = np.clip(np.power(np.clip(can, 0, 1), 1 / 1.05) * 255, 0, 255).astype(np.uint8)
    return out, P
R.frame = frame4k
R.GAIN = (0.85 + 0.15 * np.clip(np.power(F, 0.75) * 1.2, 0, 1)).astype(np.float32)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stills':
        for tm in [float(x) for x in sys.argv[2].split(',')]:
            cv2.imwrite(os.path.join(HERE, f'k5_{tm:06.1f}.jpg'), R.frame(tm)[0])
    elif mode == 'content':
        f0, f1, tag = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        pc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                               '-c:v', 'libx264', '-preset', 'fast', '-crf', '12', '-threads', '2', '-pix_fmt', 'yuv420p', os.path.join(HERE, f'cc_{tag}.mp4')], stdin=subprocess.PIPE)
        for i in range(f0, f1):
            CUR_T[0] = i / FPS
            P = scene(i / FPS)
            pc.stdin.write(np.clip(P * 255, 0, 255).astype(np.uint8).tobytes())
            if i % 15 == 0: print(tag, i, flush=True)
        pc.stdin.close(); pc.wait(); print(tag, 'done', flush=True)
    elif mode == 'dumpbase':
        np.save(os.path.join(HERE, 'night.npy'), R.NIGHT.astype(np.float32)); np.save(os.path.join(HERE, 'gain.npy'), R.GAIN.astype(np.float32))
    else:
        f0, f1, tag = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        pv = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{CW}x{CH}', '-r', str(FPS), '-i', '-',
                               '-c:v', 'libx264', '-preset', 'fast', '-crf', '17', '-threads', '2', '-x264-params', 'rc-lookahead=8:sync-lookahead=0', '-pix_fmt', 'yuv420p', os.path.join(HERE, f'k5p_{tag}.mp4')], stdin=subprocess.PIPE)
        pc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                               '-c:v', 'libx264', '-preset', 'fast', '-crf', '16', '-threads', '2', '-x264-params', 'rc-lookahead=8:sync-lookahead=0', '-pix_fmt', 'yuv420p', os.path.join(HERE, f'k5c_{tag}.mp4')], stdin=subprocess.PIPE)
        for i in range(f0, f1):
            o, P = R.frame(i / FPS)
            pv.stdin.write(o.tobytes()); pc.stdin.write(np.clip(P * 255, 0, 255).astype(np.uint8).tobytes())
            if i % 15 == 0: print(tag, i, flush=True)
        pv.stdin.close(); pc.stdin.close(); pv.wait(); pc.wait()
        print(tag, 'done', flush=True)
