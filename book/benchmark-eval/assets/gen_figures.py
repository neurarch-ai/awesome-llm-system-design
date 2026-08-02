"""Figures for the benchmark-eval chapter. Run: python3 gen_figures.py (needs matplotlib).

Every curve here is computed from the formula it illustrates, not drawn by hand:
binomial standard errors, the paired-difference standard error, pass@k and pass^k,
and a prediction-powered estimate on simulated judge/human labels.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
plt.rcParams.update({'figure.dpi': 130, 'font.size': 11, 'figure.autolayout': True})
BLUE, ORANGE, GREY, RED = '#2563eb', '#ea7317', '#64748b', '#b91c1c'

# --- 1. how wide is a benchmark score, by item count ------------------------
n = np.arange(20, 2100, 5)
half = 1.96 * np.sqrt(0.25 / n) * 100          # 95% half-width at p = 0.5, in points

fig, ax = plt.subplots(figsize=(6.2, 3.9))
ax.plot(n, half, lw=2, color=BLUE)
for size, name, off in [(30, 'AIME-style (30 items)', (25, -18)),
                        (198, 'GPQA Diamond (198)', (14, 6)),
                        (500, 'SWE-bench Verified (500)', (14, 6)),
                        (2000, 'large suite (2000)', (-135, 16))]:
    h = 1.96 * np.sqrt(0.25 / size) * 100
    ax.plot([size], [h], 'o', color=ORANGE, ms=7)
    ax.annotate(f"{name}\n+/- {h:.1f} pts", (size, h), textcoords='offset points',
                xytext=off, fontsize=9, color='#334155')
ax.axhline(3, ls='--', lw=1, color=GREY)
ax.text(120, 1.1, 'below this line, a 3-point claim is discussable',
        fontsize=9, color=GREY)
ax.set_xlabel('benchmark size (items)')
ax.set_ylabel('95% CI half-width at p = 0.5 (points)')
ax.set_title('A score is an estimate: how wide, by item count')
ax.set_xlim(0, 2100)
ax.set_ylim(0, 20)
ax.grid(alpha=0.25)
fig.savefig(OUT + "fig-error-bars-vs-n.png")
plt.close(fig)

# --- 2. paired vs unpaired comparison ---------------------------------------
# 500 items, A = 71%, B = 69%, discordant: A wins 25, B wins 15.
N, pa, pb, b, c = 500, 0.71, 0.69, 25, 15
se_a = 1.96 * np.sqrt(pa * (1 - pa) / N) * 100
se_b = 1.96 * np.sqrt(pb * (1 - pb) / N) * 100
delta = (b - c) / N * 100
se_d = 1.96 * np.sqrt(b + c) / N * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.7))
ax1.errorbar([0, 1], [pa * 100, pb * 100], yerr=[se_a, se_b], fmt='o', ms=8,
             color=BLUE, ecolor=BLUE, capsize=6, lw=2)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['model A', 'model B'])
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylabel('score (%)')
ax1.set_title(f'Unpaired: +/- {se_a:.1f} pts each\n(intervals overlap heavily)', fontsize=10)
ax1.grid(alpha=0.25, axis='y')

ax2.errorbar([0], [delta], yerr=[se_d], fmt='o', ms=8, color=ORANGE,
             ecolor=ORANGE, capsize=6, lw=2)
ax2.axhline(0, ls='--', lw=1.2, color=RED)
ax2.set_xticks([0])
ax2.set_xticklabels(['A minus B\n(per item)'])
ax2.set_xlim(-0.6, 0.6)
ax2.set_ylabel('difference (points)')
ax2.set_title(f'Paired: {delta:+.1f} +/- {se_d:.1f} pts\n(still crosses zero, but you can size it)',
              fontsize=10)
ax2.grid(alpha=0.25, axis='y')
fig.savefig(OUT + "fig-paired-vs-unpaired.png")
plt.close(fig)

# --- 3. pass@k rises, pass^k falls ------------------------------------------
k = np.arange(1, 13)
p = 0.9
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(k, 1 - (1 - p) ** k, marker='o', ms=5, lw=2, color=BLUE,
        label=r'pass@$k$: at least one of $k$ succeeds (coverage)')
ax.plot(k, p ** k, marker='s', ms=5, lw=2, color=ORANGE,
        label=r'pass$^k$: all $k$ succeed (reliability)')
ax.axhline(p, ls='--', lw=1, color=GREY)
ax.text(8.2, p + 0.02, 'single-attempt rate = 0.90', fontsize=9, color=GREY)
ax.set_xlabel('k  (independent attempts)')
ax.set_ylabel('probability')
ax.set_title('Two metrics, same 90% agent, opposite directions')
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)
ax.legend(fontsize=9, framealpha=0.9, loc='lower left')
fig.savefig(OUT + "fig-passk-vs-passhatk.png")
plt.close(fig)

# --- 4. PPI: unbiased and narrow --------------------------------------------
rng = np.random.default_rng(7)
TRUTH, NBIG, NSMALL, BIAS = 0.68, 5000, 300, 0.06
judge_all = rng.random(NBIG) < (TRUTH + BIAS)          # judge runs 6 points high
idx = rng.choice(NBIG, NSMALL, replace=False)
judge_sub = judge_all[idx]
human_sub = np.where((judge_sub == 1) & (rng.random(NSMALL) < BIAS / (TRUTH + BIAS)),
                     0, judge_sub)
judge_only = judge_all.mean()
human_only = human_sub.mean()
ppi = judge_only + (human_sub - judge_sub).mean()
se_judge = 1.96 * np.sqrt(judge_only * (1 - judge_only) / NBIG)
se_human = 1.96 * np.sqrt(human_only * (1 - human_only) / NSMALL)
se_ppi = 1.96 * np.sqrt((human_sub - judge_sub).var() / NSMALL
                        + judge_only * (1 - judge_only) / NBIG)

fig, ax = plt.subplots(figsize=(6.6, 3.6))
labels = [f'judge only\n(n = {NBIG})', f'humans only\n(n = {NSMALL})',
          f'PPI\n(judge {NBIG} + humans {NSMALL})']
vals, errs, colors = [judge_only, human_only, ppi], [se_judge, se_human, se_ppi], [RED, GREY, BLUE]
for i, (v, e, col) in enumerate(zip(vals, errs, colors)):
    ax.errorbar([i], [v], yerr=[e], fmt='o', ms=8, capsize=7, lw=2, color=col)
ax.axhline(TRUTH, ls='--', lw=1.2, color='#15803d')
ax.text(2.15, TRUTH + 0.004, 'truth', fontsize=9, color='#15803d')
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlim(-0.5, 2.6)
ax.set_ylabel('estimated quality')
ax.set_title('Precise and biased, unbiased and wide, or both')
ax.grid(alpha=0.25, axis='y')
fig.savefig(OUT + "fig-ppi-correction.png")
plt.close(fig)

print("wrote 4 figures to", OUT)
