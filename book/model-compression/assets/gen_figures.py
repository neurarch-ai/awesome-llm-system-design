"""Figures for the model-compression chapter. Run: python3 gen_figures.py (needs matplotlib).

The memory and throughput curves come from the same formulas as the chapter's
capstone (70B parameters, 80 layers, 8 KV heads, head dim 128, one 80GB accelerator
at 3.3 TB/s), so the plots and the printed table agree by construction. The error
bars in the third panel are measured on a synthetic weight block with a few outlier
channels, which is what makes group size matter.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
plt.rcParams.update({'figure.dpi': 130, 'font.size': 11, 'figure.autolayout': True})
BLUE, ORANGE, GREEN, GREY, RED = '#2563eb', '#ea7317', '#15803d', '#64748b', '#b91c1c'

N = 70e9
LAYERS, KV_HEADS, HEAD_DIM = 80, 8, 128
HBM_BYTES, HBM_BW, GROUP = 80e9, 3.3e12, 128

w_bytes = lambda bits: N * (bits + 16 / GROUP) / 8
kv_bytes = lambda seq, bits=16: 2 * LAYERS * KV_HEADS * HEAD_DIM * seq * bits / 8
tok_s = lambda w, kv, b: b * HBM_BW / (w + b * kv)

# --- 1. the speedup is a function of batch size -----------------------------
batch = np.arange(1, 65)
base = np.array([tok_s(w_bytes(16), kv_bytes(32768), b) for b in batch])
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for bits, color, label in [(8, BLUE, 'int8 weights'), (4, ORANGE, 'int4 group-wise')]:
    speed = np.array([tok_s(w_bytes(bits), kv_bytes(32768), b) for b in batch])
    ax.plot(batch, speed / base, lw=2, color=color, label=label)
kv4 = np.array([tok_s(w_bytes(4), kv_bytes(32768, 8), b) for b in batch])
ax.plot(batch, kv4 / base, lw=2, ls='--', color=GREEN, label='int4 weights + int8 KV')
ax.axhline(1, ls=':', lw=1.2, color=GREY)
ax.set_xlabel('batch size (concurrent sequences, 32K context)')
ax.set_ylabel('decode throughput vs fp16')
ax.set_title('The quantization win is a function of batch size')
ax.annotate('demos measure here', (1, 3.2), xytext=(6, 3.35), fontsize=9, color='#334155',
            arrowprops=dict(arrowstyle='->', color=GREY, lw=1))
ax.annotate('production lives here', (48, 1.28), xytext=(28, 2.2), fontsize=9,
            color='#334155', arrowprops=dict(arrowstyle='->', color=GREY, lw=1))
ax.set_xlim(1, 64)
ax.set_ylim(0.9, 3.6)
ax.grid(alpha=0.25)
ax.legend(fontsize=9, framealpha=0.9, loc='upper right')
fig.savefig(OUT + "fig-speedup-vs-batch.png")
plt.close(fig)

# --- 2. where the memory goes -----------------------------------------------
configs = [('fp16', 16, 16), ('int8 W', 8, 16), ('int4 W', 4, 16), ('int4 W\n+ int8 KV', 4, 8)]
labels = [c[0] for c in configs]
w = np.array([w_bytes(c[1]) / 1e9 for c in configs])
kv = np.array([kv_bytes(32768, c[2]) / 1e9 for c in configs])

fig, ax = plt.subplots(figsize=(6.2, 3.9))
ax.bar(labels, w, color=BLUE, label='weights')
ax.bar(labels, kv, bottom=w, color=ORANGE, label='KV cache, one 32K sequence')
ax.axhline(HBM_BYTES / 1e9, ls='--', lw=1.5, color=RED)
ax.text(2.55, HBM_BYTES / 1e9 + 3, 'one 80GB accelerator', fontsize=9, color=RED)
for i, (a, b) in enumerate(zip(w, kv)):
    ax.text(i, a + b + 3, f"{a + b:.0f} GB", ha='center', fontsize=9, color='#334155')
ax.set_ylabel('GB')
ax.set_title('int8 does not fit; int4 does. KV is not a footnote.')
ax.set_ylim(0, 175)
ax.grid(alpha=0.25, axis='y')
ax.legend(fontsize=9, framealpha=0.9)
fig.savefig(OUT + "fig-memory-breakdown.png")
plt.close(fig)

# --- 3. what the bits and the zeros cost ------------------------------------
rng = np.random.default_rng(11)
W = rng.normal(0, 0.02, 8192)
W[rng.choice(8192, 12, replace=False)] *= 25          # a few outlier channels


def quant(w, bits, group=GROUP):
    out = w.astype(float).copy()
    qmax = 2 ** (bits - 1) - 1
    for i in range(0, len(w), group):
        g = out[i:i + group]
        s = np.abs(g).max() / qmax or 1e-12
        out[i:i + group] = s * np.clip(np.round(g / s), -qmax - 1, qmax)
    return out


def prune_unstructured(w, keep=0.5):
    cutoff = np.sort(np.abs(w))[::-1][int(len(w) * keep) - 1]
    return np.where(np.abs(w) >= cutoff, w, 0.0)


def prune_2of4(w):
    g = w.reshape(-1, 4).copy()
    drop = np.argsort(np.abs(g), axis=1)[:, :2]
    for row, cols in enumerate(drop):
        g[row, cols] = 0.0
    return g.reshape(-1)


err = lambda a: np.linalg.norm(W - a) / np.linalg.norm(W)
bars = [('int8\ng=128', err(quant(W, 8)), BLUE), ('int4\ng=128', err(quant(W, 4)), BLUE),
        ('int4\ng=32', err(quant(W, 4, 32)), BLUE), ('int3\ng=128', err(quant(W, 3)), BLUE),
        ('50% pruned\nunstructured', err(prune_unstructured(W)), GREY),
        ('50% pruned\n2:4', err(prune_2of4(W)), ORANGE)]

fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
for i, b in enumerate(bars):
    ax.text(i, b[1] + 0.008, f"{b[1]:.3f}", ha='center', fontsize=9, color='#334155')
ax.set_ylabel('relative reconstruction error')
ax.set_title('Group size is the cheapest knob; 2:4 costs more than unstructured')
ax.set_ylim(0, 0.42)
ax.grid(alpha=0.25, axis='y')
fig.savefig(OUT + "fig-quant-prune-error.png")
plt.close(fig)

# --- 4. same accuracy, different model ---------------------------------------
n, correct, flip_each = 1000, 720, 60
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.bar(['baseline\naccuracy', 'compressed\naccuracy'], [correct / n, correct / n],
       color=BLUE, width=0.55)
ax.bar(['flip rate\n(disagreement)'], [2 * flip_each / n], color=RED, width=0.55)
for i, v in enumerate([correct / n, correct / n, 2 * flip_each / n]):
    ax.text(i, v + 0.015, f"{v:.3f}", ha='center', fontsize=10, color='#334155')
ax.set_ylim(0, 0.85)
ax.set_ylabel('fraction of items')
ax.set_title('Identical accuracy, 12% of items answered differently')
ax.grid(alpha=0.25, axis='y')
fig.savefig(OUT + "fig-accuracy-vs-flips.png")
plt.close(fig)

print("wrote 4 figures to", OUT)
