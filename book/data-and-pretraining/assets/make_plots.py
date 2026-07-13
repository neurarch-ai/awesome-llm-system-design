import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-llm-system-design/book/data-and-pretraining/assets/"

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.autolayout': True
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

# ── 1. Web keep-rate funnel ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
stages = [
    "Raw Common Crawl\n(WARC)",
    "After text\nextraction",
    "After language\nID filter",
    "After quality\nfilter",
    "After MinHash\ndedup",
    "Clean token\nstream",
]
keep = [100.0, 58.0, 32.0, 14.0, 8.5, 5.2]
colors = [BLUE, BLUE, ORANGE, ORANGE, GREEN, GREEN]
bars = ax.bar(range(len(stages)), keep, color=colors, alpha=0.80, width=0.6, edgecolor='white')
for i, (bar, pct) in enumerate(zip(bars, keep)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{pct:.1f}%", ha='center', va='bottom', fontsize=9, color=GRAY)
    if i > 0:
        drop = keep[i - 1] - keep[i]
        ax.annotate("", xy=(i, keep[i] + 0.3), xytext=(i - 1, keep[i - 1] + 0.3),
                    arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel("Fraction of raw bytes kept (%)")
ax.set_title("Web crawl keep-rate funnel: single-digit yield is normal and correct\n"
             "(illustrative; exact rates vary by pipeline)")
legend_elements = [
    mpatches.Patch(facecolor=BLUE, alpha=0.80, label='extraction and routing'),
    mpatches.Patch(facecolor=ORANGE, alpha=0.80, label='quality filtering'),
    mpatches.Patch(facecolor=GREEN, alpha=0.80, label='deduplication'),
]
ax.legend(handles=legend_elements, fontsize=9, frameon=False)
fig.savefig(OUT + "fig-web-funnel.png")
plt.close(fig)

# ── 2. Tokenizer fertility by language ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
langs = ["English", "French", "German", "Arabic", "Chinese", "Thai"]
fertility_english_vocab = [1.3, 1.7, 1.9, 4.8, 5.1, 7.3]   # tokens/word with English-heavy vocab
fertility_multilingual  = [1.4, 1.6, 1.8, 2.4, 2.1, 2.9]   # tokens/word with multilingual vocab
x = np.arange(len(langs))
w = 0.38
b1 = ax.bar(x - w / 2, fertility_english_vocab, w, label='English-heavy vocab (32K)',
            color=RED, alpha=0.80)
b2 = ax.bar(x + w / 2, fertility_multilingual, w, label='Multilingual vocab (128K)',
            color=GREEN, alpha=0.80)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=8.5, color=GRAY)
ax.axhline(1.0, color=BLUE, ls='--', lw=1, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(langs)
ax.set_ylabel("Tokens per word (fertility)")
ax.set_title("Tokenizer fertility by language: non-English scripts are expensive\n"
             "with an English-heavy vocabulary (illustrative)")
ax.legend(fontsize=9, frameon=False)
fig.savefig(OUT + "fig-fertility.png")
plt.close(fig)

# ── 3. Loss vs compute (scaling law) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.2))
C = np.logspace(18, 24, 200)   # FLOPs

# Chinchilla-optimal frontier: L decreases as ~C^{-0.5} (equal scaling of N and D)
E = 1.69   # irreducible loss (illustrative)
A = 4.0e11
alpha = 0.34
L_opt = E + A / (C ** alpha)

# Undertrained (GPT-3-style, large N, few tokens): converges slower
L_under = E + A * 1.8 / (C ** (alpha * 0.72))

ax.loglog(C, L_opt,   color=BLUE,   lw=2.5, label='Chinchilla-optimal (scale N and D together)')
ax.loglog(C, L_under, color=ORANGE, lw=2.5, ls='--', label='Undertrained large model (too few tokens)')

# Annotate the crossover region
c_mid = 3e21
ax.annotate("smaller, well-trained\nmodel wins here",
            xy=(c_mid, E + A / (c_mid ** alpha)),
            xytext=(c_mid * 12, E + A / (c_mid ** alpha) * 1.18),
            fontsize=9, color=BLUE,
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))

ax.set_xlabel("Compute budget C  (FLOPs, log scale)")
ax.set_ylabel("Validation loss L  (log scale)")
ax.set_title("Scaling law: Chinchilla-optimal vs undertrained\n"
             r"$L(N,D) = E + A/N^{\alpha} + B/D^{\beta}$  (illustrative)")
ax.legend(fontsize=9, frameon=False)
fig.savefig(OUT + "fig-scaling-law.png")
plt.close(fig)

# ── 4. Dense vs MoE: total params vs active params ───────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.2))
models = [
    ("Dense 7B\n(Llama 3 8B)",    7,   7,   BLUE,  'o'),
    ("Dense 70B\n(Llama 3 70B)",  70,  70,  BLUE,  's'),
    ("Dense 405B\n(Llama 3 405B)",405, 405, BLUE,  'D'),
    ("MoE 8x7B\n(Mixtral)",       46,  12,  GREEN, 'o'),
    ("MoE 141B\n(DeepSeek-V2)",  141,  21,  GREEN, 's'),
    ("MoE 671B\n(DeepSeek-V3)",  671,  37,  GREEN, 'D'),
]
for name, total, active, color, marker in models:
    ax.scatter(total, active, s=220, color=color, marker=marker,
               alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
    dy = 2 if total < 200 else -5
    ax.annotate(name, (total, active), xytext=(3, dy + 2),
                textcoords='offset points', fontsize=8, color=color)

# y = x line (dense)
x_line = np.array([1, 800])
ax.plot(x_line, x_line, color=GRAY, ls='--', lw=1.2, label='Dense: active = total')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel("Total parameters (B, log)")
ax.set_ylabel("Active parameters per token (B, log)")
ax.set_title("Dense vs MoE: total capacity vs per-token compute cost\n"
             "(MoE buys more capacity at the same per-token FLOPs)")
dense_patch  = mpatches.Patch(color=BLUE,  alpha=0.85, label='Dense')
moe_patch    = mpatches.Patch(color=GREEN, alpha=0.85, label='MoE')
diag_line    = plt.Line2D([0], [0], color=GRAY, ls='--', lw=1.2, label='active = total (dense)')
ax.legend(handles=[dense_patch, moe_patch, diag_line], fontsize=9, frameon=False)
fig.savefig(OUT + "fig-dense-vs-moe.png")
plt.close(fig)

print("wrote 4 figures: fig-web-funnel.png, fig-fertility.png, fig-scaling-law.png, fig-dense-vs-moe.png")
