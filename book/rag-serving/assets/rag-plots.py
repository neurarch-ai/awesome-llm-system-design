import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-llm-system-design/book/rag-serving/assets/"

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11, 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.autolayout': True
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

# 1) Recall vs chunk size for structural vs fixed-size chunking
fig, ax = plt.subplots(figsize=(7, 4))
chunk_sizes = np.array([50, 100, 200, 400, 800, 1600])

# Structural chunking: peaks around 400 tokens, degrades gently on both sides
recall_structural = np.array([0.52, 0.63, 0.72, 0.79, 0.74, 0.66])

# Fixed-size chunking: peaks a bit lower around 200-300, more sensitive to size
recall_fixed = np.array([0.44, 0.56, 0.65, 0.67, 0.61, 0.54])

ax.plot(chunk_sizes, recall_structural, marker='o', color=BLUE, lw=2, label='Structural / recursive chunking')
ax.plot(chunk_sizes, recall_fixed, marker='s', color=ORANGE, lw=2, linestyle='--', label='Fixed-size chunking')

ax.axvspan(300, 500, alpha=0.08, color=BLUE, label='Sweet spot (~400 tok)')
ax.set_xscale('log')
ax.set_xticks(chunk_sizes)
ax.set_xticklabels(chunk_sizes)
ax.set_xlabel('Chunk size (tokens, log scale)')
ax.set_ylabel('recall@10')
ax.set_ylim(0.35, 0.90)
ax.set_title('Recall@10 vs chunk size\n(structural chunking peaks higher and is more robust)')
ax.legend(fontsize=9, frameon=False)
fig.savefig(OUT + 'fig-recall-vs-chunk-size.png')
plt.close(fig)

# 2) Dense-only vs hybrid (dense + BM25) recall@k
fig, ax = plt.subplots(figsize=(7, 4))
ks = np.array([5, 10, 20, 50, 100, 200, 500])

# Dense-only: saturates more slowly
recall_dense = 1.0 - np.exp(-ks / 130.0)

# Hybrid: roughly 3-5 pp higher throughout, especially at small k where exact terms matter
recall_hybrid = np.clip(1.0 - np.exp(-ks / 95.0), 0, 1.0)

ax.plot(ks, recall_dense, marker='o', color=ORANGE, lw=2, label='Dense-only (vector)')
ax.plot(ks, recall_hybrid, marker='s', color=BLUE, lw=2, label='Hybrid (vector + BM25 / RRF)')

ax.fill_between(ks, recall_dense, recall_hybrid, alpha=0.12, color=BLUE, label='Hybrid gain')
ax.set_xscale('log')
ax.set_xlabel('k (candidates retrieved)')
ax.set_ylabel('recall@k')
ax.set_ylim(0.0, 1.05)
ax.set_title('Dense-only vs hybrid retrieval recall@k\n(hybrid wins on exact terms, codes, and jargon)')
ax.legend(fontsize=9, frameon=False)
fig.savefig(OUT + 'fig-dense-vs-hybrid-recall.png')
plt.close(fig)

# 3) RAG pipeline latency breakdown (two scenarios)
fig, ax = plt.subplots(figsize=(8, 3.8))

components = ['Embed query', 'Vector search', 'Rerank', 'LLM prefill', 'LLM decode']
colors_comp = [GREEN, BLUE, ORANGE, RED, '#7c3aed']

# Scenario A: without reranker
no_rerank = np.array([20, 40, 0, 320, 700])
# Scenario B: with cross-encoder reranker (rerank adds ~80ms but shortens LLM prefill cost indirectly)
with_rerank = np.array([20, 40, 80, 260, 680])

scenarios = [no_rerank, with_rerank]
labels = ['Without reranker', 'With cross-encoder reranker']
ys = [0.6, 0.2]
height = 0.25

for idx, (vals, label, y) in enumerate(zip(scenarios, labels, ys)):
    left = 0
    for v, c in zip(vals, colors_comp):
        if v > 0:
            bar = ax.barh(y, v, height, left=left, color=c, alpha=0.85,
                          edgecolor='white', linewidth=0.5)
            if v >= 50:
                ax.text(left + v / 2, y, f'{v}ms', ha='center', va='center',
                        fontsize=8.5, color='white', fontweight='bold')
        left += v
    ax.text(-15, y, label, ha='right', va='center', fontsize=9.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=n, alpha=0.85)
                   for c, n in zip(colors_comp, components) if n != 'Rerank']
legend_elements.insert(2, Patch(facecolor=ORANGE, label='Rerank', alpha=0.85))
ax.legend(handles=legend_elements, loc='lower right', fontsize=8.5, frameon=False,
          ncol=2)
ax.set_xlabel('Cumulative latency (ms)')
ax.set_xlim(-200, 1200)
ax.set_yticks([])
ax.set_title('RAG online-path latency breakdown\n(reranker adds ~80ms but cuts LLM prefill tokens)')
fig.savefig(OUT + 'fig-rag-latency-breakdown.png')
plt.close(fig)

# 4) Retrieval-then-rerank funnel (log scale bar chart)
fig, ax = plt.subplots(figsize=(6.5, 4))

stages = ['Corpus\n(50M docs)', 'Retrieved\n(top 100)', 'Reranked\n(top 10)', 'LLM context\n(top 5)']
counts = [50_000_000, 100, 10, 5]
bar_colors = [GRAY, BLUE, ORANGE, GREEN]

bars = ax.bar(stages, counts, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.2)
for bar, count in zip(bars, counts):
    label = f'{count:,}' if count < 1000 else f'{count / 1e6:.0f}M'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
            label, ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_yscale('log')
ax.set_ylabel('Number of documents (log scale)')
ax.set_title('Retrieve-then-rerank funnel\n(each stage narrows the pool cheaply before generation)')
ax.set_ylim(1, 5e8)
fig.savefig(OUT + 'fig-rerank-funnel.png')
plt.close(fig)

print("wrote 4 figures to", OUT)
