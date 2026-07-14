import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-llm-system-design/book/kv-cache/assets/"
plt.rcParams.update({'figure.dpi': 130, 'font.size': 11, 'figure.autolayout': True})

# Context lengths (columns) and depth fractions (rows)
ctx_labels = ['4k', '8k', '16k', '32k', '64k', '128k']
ctx_vals   = [4096, 8192, 16384, 32768, 65536, 131072]
depth_pcts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # % into document

n_ctx   = len(ctx_vals)
n_depth = len(depth_pcts)

# Illustrative recall: degrades at long context; dips worst near 50% depth at long context
rng = np.random.default_rng(42)
acc = np.ones((n_depth, n_ctx))

for ci, cv in enumerate(ctx_vals):
    # Base decay with context length (strong decay beyond 16k)
    ctx_factor = max(0.0, 1.0 - 0.55 * max(0, (cv - 8192) / (131072 - 8192)) ** 0.6)
    for di, dp in enumerate(depth_pcts):
        # Mid-context dip: worst around 40-60% depth, worse at long context
        depth_factor = 1.0 - 0.30 * np.exp(-((dp - 50) ** 2) / (2 * 18 ** 2)) * (1 - ctx_factor)
        base = ctx_factor * depth_factor
        # Add small noise
        acc[di, ci] = np.clip(base + rng.normal(0, 0.02), 0, 1)

# Clamp edges: very short contexts are near-perfect everywhere
acc[:, 0] = np.clip(acc[:, 0] + 0.05, 0, 1)
acc[:, 1] = np.clip(acc[:, 1] + 0.03, 0, 1)

fig, ax = plt.subplots(figsize=(7.0, 4.4))
im = ax.imshow(acc, aspect='auto', cmap='RdYlGn', vmin=0.0, vmax=1.0, origin='upper')

ax.set_xticks(range(n_ctx))
ax.set_xticklabels(ctx_labels)
ax.set_yticks(range(n_depth))
ax.set_yticklabels([f'{d}%' for d in depth_pcts])
ax.set_xlabel('Context length')
ax.set_ylabel('Needle insertion depth')
ax.set_title('NIAH retrieval accuracy: context length vs. depth (illustrative)')

# Annotate cells with recall value
for di in range(n_depth):
    for ci in range(n_ctx):
        v = acc[di, ci]
        ax.text(ci, di, f'{v:.2f}', ha='center', va='center',
                fontsize=7.5, color='black' if 0.25 < v < 0.80 else 'white')

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Retrieval accuracy')

fig.savefig(OUT + "fig-niah-heatmap.png")
plt.close(fig)
print("wrote fig-niah-heatmap.png")
