import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-llm-system-design/book/continued-pretraining/assets/"

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.autolayout': True
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

# --- 1. Catastrophic forgetting vs replay ratio ---
fig, ax = plt.subplots(figsize=(6.5, 4.0))
replay = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50])
# forgetting (drop in general benchmark %, higher is worse)
forget  = np.array([18.0, 9.5, 5.5, 3.2, 1.8, 1.2, 0.6])
# domain gain (improvement on domain benchmark %, higher is better)
gain    = np.array([14.0, 13.5, 12.8, 11.8, 10.0, 8.5, 5.5])

ax.plot(replay * 100, forget, 'o-', color=RED,    lw=2, label='general benchmark drop (forgetting)')
ax.plot(replay * 100, gain,   's--', color=GREEN,  lw=2, label='domain benchmark gain')
ax.axvline(10, color=GRAY, ls=':', lw=1.4)
ax.text(11, 15.5, '10% replay:\ngood balance', color=GRAY, fontsize=9)
ax.set_xlabel('replay fraction of general data (%)')
ax.set_ylabel('benchmark change (pp)')
ax.set_title('Forgetting falls sharply with a small replay fraction\n(illustrative)')
ax.legend(fontsize=9, frameon=False)
fig.savefig(OUT + 'fig-forgetting-vs-replay.png')
plt.close(fig)
print('wrote fig-forgetting-vs-replay.png')

# --- 2. Context length vs memory and compute cost ---
fig, ax = plt.subplots(figsize=(7.0, 4.2))
L = np.array([4, 8, 16, 32, 64, 128])  # in thousands of tokens
kv_mem   = L ** 1      # linear in L (relative units)
attn_flops = (L ** 2) / (L[0] ** 2)  # quadratic, normalized to 4K=1

ax.plot(L, kv_mem / kv_mem[0],    'o-',  color=BLUE,   lw=2, label='KV-cache memory (linear in L)')
ax.plot(L, attn_flops,            's--', color=RED,    lw=2, label='prefill attention FLOPs (quadratic in L)')

for lv, km, af in zip(L, kv_mem / kv_mem[0], attn_flops):
    ax.annotate(f'{af:.0f}x', (lv, af), xytext=(0, 6), textcoords='offset points',
                ha='center', fontsize=8, color=RED)

ax.set_xticks(L)
ax.set_xticklabels([f'{v}K' for v in L])
ax.set_xlabel('context length')
ax.set_ylabel('relative cost (4K = 1x)')
ax.set_title('KV-cache grows linearly; prefill attention grows quadratically\n(illustrative, both normalized to 4K = 1x)')
ax.legend(fontsize=9, frameon=False)
fig.savefig(OUT + 'fig-context-length-memory.png')
plt.close(fig)
print('wrote fig-context-length-memory.png')

# --- 3. Needle-in-a-haystack recall by depth and method ---
depths = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # % depth in context

def recall_curve(near_end_boost, mid_dip, scale):
    """Illustrative recall curve: high at ends, lower in the middle."""
    mid = np.abs(depths - 50) / 50.0
    r = scale * (0.5 + 0.4 * mid + near_end_boost * (depths < 5).astype(float) * 0.05
                 - mid_dip * np.exp(-((depths - 50)**2) / 600))
    return np.clip(r, 0.0, 1.0)

recall_naive  = recall_curve(0.1, 0.45, 0.70)   # naive extension, bad middle
recall_pi     = recall_curve(0.05, 0.25, 0.82)  # linear PI, better but still mid-dip
recall_yarn   = recall_curve(0.02, 0.10, 0.92)  # YaRN, good overall
recall_strong = recall_curve(0.01, 0.04, 0.97)  # strong model (e.g. Llama 3 128K)

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(depths, recall_naive,  'o-',  color=RED,    lw=2, label='naive extrapolation (no RoPE rescale)')
ax.plot(depths, recall_pi,     's--', color=ORANGE,  lw=2, label='linear position interpolation')
ax.plot(depths, recall_yarn,   'd-',  color=BLUE,   lw=2, label='YaRN (non-uniform + temp correction)')
ax.plot(depths, recall_strong, '^--', color=GREEN,  lw=2, label='staged extension (Llama 3 style)')
ax.axvspan(35, 65, alpha=0.07, color=GRAY)
ax.text(38, 0.30, 'lost-in-the-middle\nzone', color=GRAY, fontsize=8.5)
ax.set_xlabel('needle depth in context (%)')
ax.set_ylabel('recall')
ax.set_ylim(0.0, 1.05)
ax.set_title('Needle-in-a-haystack recall by position depth and method\n(illustrative; real models show sharper mid-dip)')
ax.legend(fontsize=9, frameon=False, loc='lower center')
fig.savefig(OUT + 'fig-niah-recall-by-method.png')
plt.close(fig)
print('wrote fig-niah-recall-by-method.png')

# --- 4. RoPE effective frequency scaling by dimension ---
d = 64  # head dimension
dims = np.arange(0, d // 2)  # 0 .. 31
b_orig = 10000
s = 16  # length scale factor (e.g., 8K -> 128K)

theta_orig = b_orig ** (-2 * dims / d)

# Linear PI: divide all by s
theta_pi = theta_orig / s

# NTK-ABF: raise base
b_abf = b_orig * (s ** (d / (d - 2)))
theta_abf = b_abf ** (-2 * dims / d)

# YaRN: ramp - low-freq (high i) interpolated, high-freq (low i) left near-unscaled
# simplified: gamma = 1 for i < threshold, 0 for i > threshold, ramp in between
alpha, beta = 1.0, 32.0  # typical YaRN params
r = 2 * np.pi * theta_orig / 1.0  # wavelength relative to L_orig
gamma = np.where(r < alpha, 0.0, np.where(r > beta, 1.0, (r - alpha) / (beta - alpha)))
theta_yarn = gamma * theta_orig + (1 - gamma) * theta_orig / s

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.semilogy(dims, theta_orig, '-',   color=GRAY,   lw=2, alpha=0.7, label='original (no extension)')
ax.semilogy(dims, theta_pi,   '--',  color=ORANGE,  lw=2, label='linear PI (uniform / s)')
ax.semilogy(dims, theta_abf,  '-.',  color=RED,    lw=2, label='NTK-ABF (raise base)')
ax.semilogy(dims, theta_yarn, '-',   color=BLUE,   lw=2.5, label='YaRN (spare high-freq dims)')

ax.axvspan(0, 8, alpha=0.07, color=GREEN)
ax.text(0.5, theta_orig[0] * 0.3, 'high-freq\n(local)', color=GREEN, fontsize=8.5)
ax.axvspan(22, 31, alpha=0.07, color=BLUE)
ax.text(23, theta_orig[-1] * 8, 'low-freq\n(global)', color=BLUE, fontsize=8.5)

ax.set_xlabel('RoPE dimension index i')
ax.set_ylabel('effective frequency (log scale)')
ax.set_title('How each method rescales RoPE frequencies across dimensions\n(illustrative, s=16, d=64 head dim)')
ax.legend(fontsize=9, frameon=False)
fig.savefig(OUT + 'fig-rope-frequency-scaling.png')
plt.close(fig)
print('wrote fig-rope-frequency-scaling.png')

print("all 4 figures written")
