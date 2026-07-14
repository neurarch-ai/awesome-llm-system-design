import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-llm-system-design/book/evaluation/assets/"
plt.rcParams.update({'figure.dpi': 130, 'font.size': 11, 'figure.autolayout': True})
BLUE, ORANGE = '#2563eb', '#ea7317'

k = np.arange(1, 21)

fig, ax = plt.subplots(figsize=(6.0, 3.8))

for p, color, label in [(0.2, BLUE, 'p = 0.2  (harder problem)'),
                         (0.4, ORANGE, 'p = 0.4  (easier problem)')]:
    passk = 1 - (1 - p) ** k
    ax.plot(k, passk, marker='o', markersize=4, lw=2, color=color, label=label)

ax.axhline(1.0, ls='--', lw=1, color='#64748b')
ax.set_xlabel('k  (number of samples drawn per problem)')
ax.set_ylabel(r'$\mathrm{pass@}k$')
ax.set_title(r'$\mathrm{pass@}k = 1 - (1-p)^k$  for two per-sample pass rates')
ax.set_xlim(1, 20)
ax.set_ylim(0, 1.08)
ax.grid(alpha=0.25)
ax.legend(framealpha=0.9)

fig.savefig(OUT + "fig-passk-curve.png")
plt.close(fig)
print("wrote fig-passk-curve.png")
