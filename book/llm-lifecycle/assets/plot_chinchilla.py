import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-llm-system-design/book/llm-lifecycle/assets/"

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11, 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.autolayout': True
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

fig, ax = plt.subplots(figsize=(8, 5))

# x-axis: model size in billions of parameters (log scale)
N = np.logspace(np.log10(1), np.log10(1000), 300)

# Chinchilla-optimal line: 20 tokens per parameter
chinchilla_line = 20 * np.ones_like(N)
ax.semilogx(N, chinchilla_line, color=GRAY, lw=1.8, ls='--',
            label='Chinchilla-optimal (20 tok/param)')

# Known sourced points
# Llama 3 8B: ~15T tokens / 8B params = 1875 ~ 1800 tok/param (sourced)
ax.scatter([8], [1800], s=120, color=BLUE, zorder=5, marker='o')
ax.annotate('Llama 3 8B\n~1800 tok/param',
            xy=(8, 1800), xytext=(2.5, 1400),
            fontsize=9, color=BLUE,
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0))

# Llama 3 70B: ~15T tokens / 70B = ~214 ~ 200 tok/param (sourced, approximate)
ax.scatter([70], [200], s=120, color=BLUE, zorder=5, marker='s')
ax.annotate('Llama 3 70B\n~200 tok/param',
            xy=(70, 200), xytext=(25, 380),
            fontsize=9, color=BLUE,
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0))

# Llama 3 405B: ~15.6T / 405B ~ 38 tok/param (corrected from ~20)
ax.scatter([405], [38], s=120, color=BLUE, zorder=5, marker='D')
ax.annotate('Llama 3 405B\n~38 tok/param',
            xy=(405, 38), xytext=(150, 90),
            fontsize=9, color=BLUE,
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0))

# Mistral 7B: token count not publicly disclosed; shown as illustrative only
ax.scatter([7], [600], s=100, color=ORANGE, zorder=5, marker='o',
           alpha=0.7, edgecolors='none')
ax.annotate('Mistral 7B\n(undisclosed,\nillustr.)',
            xy=(7, 600), xytext=(1.4, 700),
            fontsize=8.5, color=ORANGE,
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.0))

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Model size (billions of parameters, log scale)')
ax.set_ylabel('Tokens per parameter (log scale)')
ax.set_title(
    'Chinchilla-optimal vs inference-aware overtraining\n'
    '(models below the dashed line are near-optimal for training compute;\n'
    ' models far above trade training FLOPs for cheaper inference)'
)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=GRAY, ls='--', lw=1.8, label='Chinchilla-optimal (20 tok/param)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=BLUE,
           markersize=9, label='Llama 3 (sourced)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=ORANGE,
           markersize=9, alpha=0.7, label='Mistral 7B (illustrative; token count undisclosed)'),
]
ax.legend(handles=legend_elements, fontsize=9, frameon=False)

fig.savefig(OUT + 'fig-chinchilla-vs-overtrained.png')
plt.close(fig)
print("wrote fig-chinchilla-vs-overtrained.png to", OUT)
