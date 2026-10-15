import sys, cv2, numpy as np, subprocess
CW, CH = 3840, 2160
NIGHT = np.load('night.npy'); GAIN = np.load('gain.npy')
H, W = NIGHT.shape[:2]
BH = 2100; BW = int(round(W * BH / H / 2)) * 2
OX4, OY4 = (CW - BW) // 2 + 60, 30
yv = np.arange(CH, dtype=np.float32)[:, None]; xv = np.arange(CW, dtype=np.float32)[None, :]
VIG4 = np.clip(1 - 0.35 * (((xv - CW / 2) / (CW / 2)) ** 2 + ((yv - CH / 2) / (CH / 2)) ** 2), 0, 1)[..., None].astype(np.float32)
def ss(a, b, t):
    x = min(1, max(0, (t - a) / (b - a))); return x * x * (3 - 2 * x)
src, out, f0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
cap = cv2.VideoCapture(src)
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{CW}x{CH}', '-r', '30', '-i', '-',
                      '-c:v', 'libx264', '-preset', 'fast', '-crf', '17', '-threads', '2', '-x264-params', 'rc-lookahead=8', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
i = f0
while True:
    ok, fr = cap.read()
    if not ok: break
    t = i / 30
    P = fr.astype(np.float32) / 255
    tex = NIGHT + GAIN * P
    big = cv2.resize(tex, (BW, BH), interpolation=cv2.INTER_CUBIC)
    can = np.zeros((CH, CW, 3), np.float32)
    can[OY4:OY4 + BH, OX4:OX4 + BW] = big
    fy0 = OY4 + BH; hgt = CH - fy0
    refl = cv2.flip(big[-hgt:], 0)
    can[fy0:, OX4:OX4 + BW] += cv2.GaussianBlur(refl, (0, 0), 4) * np.linspace(0.22, 0.0, hgt, dtype=np.float32)[:, None, None]
    can[fy0:] += np.array([0.03, 0.022, 0.025], np.float32)
    sm = cv2.resize(can, (CW // 8, CH // 8), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(np.maximum(sm - 0.55, 0), (0, 0), 3)
    can += cv2.resize(sm, (CW, CH)) * 0.8
    can *= VIG4 * ss(0, 0.6, t)
    p.stdin.write(np.clip(np.power(np.clip(can, 0, 1), 1 / 1.05) * 255, 0, 255).astype(np.uint8).tobytes())
    i += 1
p.stdin.close(); p.wait(); print('done', i - f0)
