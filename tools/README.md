# tools

Regenerate the aggregate docs from source. From the repo root:

```
node tools/build.mjs
```

This rebuilds four files from `topics/` plus the sources in this folder:

- **CASE-STUDIES.md**, per topic: the comparative block from `comparisons/` followed by the case list read from that topic's `## Seen in production` section.
- **CASE-TEARDOWNS.md**: the per-topic teardown sections in `teardowns/`, concatenated.
- **CASE-STUDIES-BY-COMPANY.md** and **CASE-STUDIES-BY-INDUSTRY.md**: pivots of the same case list (grouped by company / by industry).

## Sources you edit

- `comparisons/NN.md`: the visual-first comparative block for topic `NN` (what they share, a divergence Mermaid diagram, a choices side-by-side table, the math that separates the approaches, a tradeoff quadrant plot).
- `teardowns/NN.md`: the per-case teardowns for topic `NN` (design diagram, interview questions, tricks, common mistakes).

## Adding a case

1. Add its bullet to the topic's `## Seen in production` (verify the link is a live first-party writeup).
2. Add a teardown for it to `tools/teardowns/NN.md`.
3. Refresh `tools/comparisons/NN.md` so the diagram/table/quadrant include the new company.
4. Run `node tools/build.mjs`.

## Conventions

No em or en dashes anywhere. Math is GitHub-flavored LaTeX; do not use `\operatorname` (GitHub's KaTeX rejects it, use `\text{...}`) and never put `#` inside math (use `n_{\text{bins}}` style). Every Mermaid code fence must be closed; `quadrantChart` point labels with spaces must be quoted.
