# Contributing

Thanks for improving this book. A few house rules keep every chapter rendering
correctly on GitHub and reading consistently. CI runs the validator on every push
and pull request, so please run it locally before opening a PR.

## Before you push

```bash
node tools/validate-book.mjs
```

It must print `Book validation passed`. If it reports a problem, fix it and re-run.
The validator enforces the rules below (Node 24, no dependencies).

## House rules (enforced by CI)

- **No em or en dashes.** Use commas, periods, or parentheses. A `—` or `–` fails CI.
- **Math must be GitHub-KaTeX safe.** GitHub renders `$...$` and `$$...$$` with KaTeX,
  but its Markdown pass runs first, so a few characters need care:
  - Use `\ast` for a star/optimal superscript, never a literal `*` inside math (a
    bare `*` is eaten as Markdown emphasis and breaks the formula, e.g. write
    `B^{\ast}`, not `B^*`).
  - Use `\lt` / `\gt`, not a bare `<` or `>` before a letter.
  - Use `\text{...}`, not `\operatorname{...}`.
  - Escape literal dollar amounts as `\$` (for example `\$5,000`) so they do not pair
    with a real inline-math `$` and swallow the text between them.
- **Mermaid line breaks are `<br/>`, never `\n`.** A literal `\n` inside a mermaid
  block does not render.
- **Code fences balance, and a closing fence sits alone on its line.** Every
  ```` ``` ```` opens and closes, and a closing fence carries no text: ```` ``` The upfront ````
  closes nothing, so the block runs on and renders the rest of the file as code.
  The backtick count stays even, which is why this hid for so long.
- **Images live in the chapter's `assets/` folder**, and every
  `![...](assets/...)` reference must resolve to a file that exists.
- **Internal links resolve** to a file, a `file.md`, or a directory.
- **No duplicate section headings** within a single file.
- **Emphasis must be able to close.** `**Label.** Text` is fine; `**标签。**正文` is
  not. CommonMark will not close a `**` run that follows punctuation and precedes a
  letter, so both delimiters render as literal asterisks and the bold is lost. Put a
  space after the closing run, as the English does, or move the punctuation outside
  the emphasis. Only CJK text hits this, because English always has the space.

## Citations

- Cite **real primary sources only**: arXiv abstract pages
  (`https://arxiv.org/abs/XXXX.XXXXX`), first-party company engineering blogs, or
  official docs.
- **Never invent** an arXiv ID, a benchmark number, a latency, or a
  "Company X uses Y" adoption claim. If you are unsure of an arXiv ID, name the
  method by author or organization and year instead of guessing a number.
- Re-check external links (arXiv, blogs, docs) with:

  ```bash
  node tools/check-links.mjs
  ```

  This is not part of the blocking push CI (external sites rate-limit and bot-block,
  which would make a gate flaky). A scheduled job runs it monthly, and you should run
  it yourself whenever you add or change links. It reports each link as OK, BLOCKED
  (a bot-block on a page that almost certainly exists), or DEAD (a 404, DNS failure,
  or TLS error) and fails only on DEAD.

## The Chinese edition (`book-zh/`)

`book-zh/` mirrors `book/` file for file: same folder names, same file names, one
translation per source file. The validator and the capstone runner walk both trees,
so every rule above applies to the translation too, and the Chinese double dash
`——` counts as an em dash. The emphasis rule above is the one a translator hits
without noticing: dropping the space after `**` is correct Chinese typography and
silently breaks the bold, so keep the punctuation inside the emphasis and the space
after it, matching the English line for line. Code blocks are copied byte for byte (the capstones are
executed from the translation as well). Figures are not duplicated; a translated
chapter references `../../book/<slug>/assets/` directly. When you change a section
in `book/`, change the matching file in `book-zh/` in the same PR, or say in the PR
that the translation is now behind.

## Chapter structure

Chapters live in `book/<slug>/` as one file per section (`01-clarifying-requirements`
through `09-summary`, plus a `README.md` index). A chapter opens with a
Candidate/Interviewer dialogue to scope the problem, walks the design end to end, and
closes with a production section (with first-party links), an interview Q&A, and a
summary. Figures are worked matplotlib PNGs in `assets/` and mermaid diagrams. Keep
additions tight and mechanism-focused rather than padded.

A companion book covers the classic-ML half (recommenders, ranking, fraud, computer
vision, speech, NLP, forecasting) in the
[ML System Design](https://github.com/neurarch-ai/awesome-ml-system-design)
repository.
