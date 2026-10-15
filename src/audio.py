import numpy as np, wave
SR = 44100; DUR = 60.0; N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(3)
L = np.zeros(N, np.float32); Rr = np.zeros(N, np.float32)
def ss(a, b, x):
    v = np.clip((x - a) / (b - a), 0, 1); return v * v * (3 - 2 * v)
def add(sig, st, g=1.0, pan=0.0):
    i = int(st * SR); j = min(N, i + len(sig))
    if j <= i: return
    s = sig[:j - i] * g
    L[i:j] += s * min(1, 1 - pan); Rr[i:j] += s * min(1, 1 + pan)
def note(m): return 440.0 * 2 ** ((m - 69) / 12)

BPM = 100; B = 60 / BPM
# Am - F - C - G, one chord per bar (4 beats)
CH = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
BAR = 4 * B
def chord_at(x): return CH[int(x / BAR) % 4]

# pad
pad = np.zeros(N, np.float32)
nb = int(DUR / BAR) + 1
for b in range(nb):
    st = b * BAR; n = int(BAR * SR) + int(0.4 * SR)
    tt = np.arange(n) / SR
    e = np.minimum(1, tt / 0.4) * np.exp(-np.maximum(0, tt - BAR) * 6)
    sig = np.zeros(n)
    for m in CH[b % 4]:
        for h, a in [(1, 1), (2, 0.35), (3, 0.15), (0.5, 0.5)]:
            sig += a * np.sin(2 * np.pi * note(m) * h * tt + rng.uniform(0, 6)) * (1 + 0.002 * np.sin(tt * 3))
    i = int(st * SR); j = min(N, i + n)
    pad[i:j] += (sig * e)[:j - i].astype(np.float32)
padvol = 0.035 * (ss(0, 4, t) * (1 - 0.5 * ss(27.5, 28.5, t) + 0.5 * ss(35, 36, t)) * (1 - ss(58.5, 60, t)))
add(pad * padvol, 0, 1.0)

def pluck(f, d=0.35):
    tt = np.arange(int(d * SR)) / SR
    return ((np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 12)).astype(np.float32)
def bell(f, d=1.6):
    tt = np.arange(int(d * SR)) / SR
    return ((np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 2.76 * tt) * np.exp(-tt * 3) + 0.25 * np.sin(2 * np.pi * f * 5.4 * tt) * np.exp(-tt * 6)) * np.exp(-tt * 2.2)).astype(np.float32)
def kick():
    tt = np.arange(int(0.4 * SR)) / SR
    f = 48 + 100 * np.exp(-tt * 35)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 8)).astype(np.float32)
def hat(d=0.05):
    n = rng.standard_normal(int(d * SR)).astype(np.float32)
    n = n - np.convolve(n, np.ones(3) / 3, 'same')
    return n * np.exp(-np.arange(len(n)) / SR * 80)
def clap():
    n = rng.standard_normal(int(0.2 * SR)).astype(np.float32)
    n = np.convolve(n, np.ones(4) / 4, 'same')
    e = np.exp(-np.arange(len(n)) / SR * 18)
    return n * e
def swell(d):
    tt = np.arange(int(d * SR)) / SR
    n = rng.standard_normal(len(tt)).astype(np.float32)
    n = np.convolve(n, np.ones(6) / 6, 'same')
    return n * (tt / d) ** 3
def boom():
    tt = np.arange(int(2.0 * SR)) / SR
    return (np.sin(2 * np.pi * np.cumsum(35 + 40 * np.exp(-tt * 6)) / SR) * np.exp(-tt * 2)).astype(np.float32)

K = kick()
# intro shimmer: rising bells on line drawing
for i in range(24):
    tm = 0.8 + i * 0.18
    m = chord_at(tm)[i % 3] + 24
    add(bell(note(m), 1.2), tm, 0.05, pan=np.sin(i))
# neon flicker buzz 6.5-9
tt = np.arange(int(2.5 * SR)) / SR
buzz = np.sign(np.sin(2 * np.pi * 100 * tt)) * 0.3 * (np.sin(tt * 90) > 0) * np.exp(-tt * 0.8)
add(buzz.astype(np.float32), 6.5, 0.05)
# arps
for (a, b_, vol) in [(6.5, 27.5, 0.07), (41.0, 47.0, 0.05), (50.0, 57.6, 0.09)]:
    x = a
    i = 0
    while x < b_:
        ch = chord_at(x)
        m = ch[[0, 1, 2, 1][i % 4]] + 12 + (12 if (i // 4) % 2 else 0)
        add(pluck(note(m)), x, vol, pan=0.4 * np.sin(i * 0.7))
        x += B / 2; i += 1
# drums
for (a, b_) in [(13.5, 27.5), (41.0, 47.0), (50.0, 57.6)]:
    x = a; i = 0
    while x < b_:
        add(K, x, 0.55)
        add(hat(), x + B / 2, 0.12, pan=0.3)
        if i % 2 == 1: add(clap(), x, 0.18)
        x += B; i += 1
# trees + ornaments: bells
for i in range(40):
    tm = 28.0 + i * 0.32 + rng.uniform(-0.05, 0.05)
    m = chord_at(tm)[int(rng.integers(3))] + 24
    add(bell(note(m)), tm, 0.06, pan=rng.uniform(-0.8, 0.8))
for i in range(20):
    tm = 35.2 + i * 0.28
    m = chord_at(tm)[i % 3] + 19
    add(bell(note(m), 2.0), tm, 0.05, pan=rng.uniform(-0.8, 0.8))
# light blocks: hits
for j in range(8):
    add(boom(), 41.0 + j * 0.4, 0.25)
# glitch stutter
x = 47.0
while x < 50.0:
    seg = (np.sign(np.sin(2 * np.pi * rng.uniform(80, 800) * np.arange(int(0.06 * SR)) / SR)) * 0.4).astype(np.float32)
    add(seg, x, 0.12, pan=rng.uniform(-1, 1)); add(hat(0.03) * 2, x, 0.2)
    x += rng.choice([B / 4, B / 8, B / 2])
# transitions
for tr in [6.5, 13.5, 20.5, 27.5, 35.0, 41.0, 50.0]:
    add(swell(1.5), tr - 1.5, 0.12)
    add(boom(), tr, 0.35)
# finale chord
tt = np.arange(int(3.0 * SR)) / SR
fin = sum(np.sin(2 * np.pi * note(m) * tt) for m in [57, 60, 64, 69, 76]) * np.exp(-tt * 1.2)
add(fin.astype(np.float32), 57.6, 0.08)
add(boom(), 57.6, 0.5)

# simple reverb via FFT convolution
def reverb(x, d=1.8, mix=0.25):
    n = int(d * SR)
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-np.arange(n) / SR * 3.5)
    ir /= np.sqrt((ir ** 2).sum())
    m = len(x) + n
    y = np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[:len(x)]
    return x + mix * y.astype(np.float32)
L = reverb(L); Rr = reverb(Rr)
out = np.stack([L, Rr], 1)
out *= (1 - ss(59.0, 60, t))[:, None]
out = np.tanh(out * 2.2) * 0.8
with wave.open('sound.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((out * 32767).astype(np.int16).tobytes())
print('ok', np.abs(out).max())
