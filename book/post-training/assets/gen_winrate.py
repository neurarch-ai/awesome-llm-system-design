import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-llm-system-design/book/post-training/assets/"
plt.rcParams.update({'figure.dpi': 130, 'font.size': 11, 'figure.autolayout': True})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

# Illustrative win-rate data: (model name, win_rate, n_comparisons)
models = ['Model A\n(n=120)', 'Model B\n(n=400)', 'Model C\n(n=1000)', 'Model D\n(n=1000)']
win_rates = [0.57, 0.54, 0.54, 0.63]
ns = [120, 400, 1000, 1000]

# 95% CI half-width using normal approximation: 1.96 * sqrt(p*(1-p)/n)
cis = [1.96 * np.sqrt(p * (1 - p) / n) for p, n in zip(win_rates, ns)]

colors = []
for p, ci in zip(win_rates, cis):
    lo = p - ci
    if lo > 0.50:
        colors.append(GREEN)   # significant win
    else:
        colors.append(GRAY)    # CI overlaps 0.5

fig, ax = plt.subplots(figsize=(6.4, 4.0))
x = np.arange(len(models))
bars = ax.bar(x, win_rates, color=colors, width=0.5, zorder=3)
ax.errorbar(x, win_rates, yerr=cis, fmt='none', ecolor='black', elinewidth=1.8,
            capsize=6, capthick=1.8, zorder=4)

ax.axhline(0.50, ls='--', lw=1.5, color=RED, label='50% (no preference)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('Preference win rate vs. production baseline')
ax.set_title('Win rate with 95% CI: overlapping 0.5 is not a real win')
ax.set_ylim(0.30, 0.82)
ax.grid(axis='y', alpha=0.25, zorder=0)
ax.legend(framealpha=0.9)

# Annotate significance
for xi, p, ci, c in zip(x, win_rates, cis, colors):
    lo = p - ci
    note = 'significant' if lo > 0.50 else 'not significant'
    ax.text(xi, p + ci + 0.015, note, ha='center', va='bottom', fontsize=8.5,
            color=GREEN if note == 'significant' else GRAY)

fig.savefig(OUT + "fig-winrate-bar-ci.png")
plt.close(fig)
print("wrote fig-winrate-bar-ci.png")
