# Templates

[CONTRIBUTING.md](../CONTRIBUTING.md) says what CI rejects. This folder says what to
write in the first place: copy a skeleton, fill it in, delete the guidance comments.

| You want to add | Copy | To |
|---|---|---|
| An interview walkthrough of a system we do not cover | [`topic.md`](topic.md) | `topics/NN-your-slug.md` |
| A full teaching chapter (10 sections, figures, capstone) | [`chapter/`](chapter/) | `book/your-slug/` |
| A paper to an existing topic's reading list | nothing | `papers.md` |
| A production system to an existing topic | nothing | that topic's `## Seen in production`, then `node tools/build.mjs` |

A topic is roughly 400 to 900 lines and can be written in one sitting. A chapter is
ten files and a capstone that has to execute. If you are unsure, write the topic:
several chapters started that way.

---

## What actually gets a change merged here

The house rules in CONTRIBUTING are mechanical and CI checks them. These four are the
editorial ones, and they are what reviews here are about:

1. **A claim carries a number or a first-party link.** "GQA saves memory" is not a
   contribution. "GQA with 8 groups cuts the cache by 8x at roughly 1 percent
   quality loss, [paper]" is. If neither a number nor a source exists, say the thing
   is not measured rather than asserting it.
2. **`## Seen in production` takes first-party engineering writeups only.** The
   company's own blog, paper, or talk. Not a summary, not a newsletter, not a
   vendor page about someone else's system. A link that requires a login is out.
3. **Say what a technique costs, in the same breath as what it buys.** Every method
   in here has a tradeoff, and an answer that lists only benefits is the answer that
   fails an interview. This is the single most common reason a draft needs another pass.
4. **Do not invent an interviewer dialogue that resolves too easily.** In section 01
   each question either removes work or changes the design. A question whose answer
   changes nothing is filler and should be cut.

## Before you open the pull request

```bash
node tools/validate-book.mjs   # required if you touched book/ or book-zh/
python3 tools/run-capstones.py # required if you touched a 10-putting-it-together.md
node tools/build.mjs           # required if you touched a Seen in production section
node tools/check-links.mjs     # optional, slow, catches a dead citation before a reviewer does
```

To read your draft the way the site will render it (sidebar, search, math, mermaid):

```bash
pip install -r tools/requirements-site.txt
node tools/build-site.mjs && mkdocs serve -f .site/mkdocs.yml
```

## Registering a new topic or chapter

Nothing scans the filesystem for content, so a new file that is not listed anywhere
is invisible. Add it in these places:

**A new topic** (`topics/NN-your-slug.md`):

- the table in the root [README.md](../README.md) under the right pipeline stage
- the list in [topics/README.md](../topics/README.md)
- at least one question in [questions.md](../questions.md) that routes to it
- `csOrder` and `tdOrder` in [tools/build.mjs](../tools/build.mjs), plus
  `tools/comparisons/NN.md` and `tools/teardowns/NN.md`, if it has case studies
- a reading list section in [papers.md](../papers.md)

**A new chapter** (`book/your-slug/`):

- the chapter table in [book/README.md](../book/README.md), under one of its `###`
  group headings. This is load-bearing beyond the README: the site's sidebar reads
  the chapter order and grouping out of that table, so a chapter missing from it is
  a chapter missing from the navigation.
- [book/reading-paths.md](../book/reading-paths.md), if it belongs to one of the paths

The Chinese edition in `book-zh/` mirrors `book/` file for file. A translation is
welcome but not required: open the English chapter, and say in the pull request that
`book-zh/your-slug/` is still missing so it can be picked up.

## Adding a paper to papers.md

One line saying what the paper changes for the reader, not what it is about. Every
topic is capped at eight, and the cap is the page's whole editorial position. If the
topic you want to add to is full, propose a swap in the pull request and say which
entry earns its place less. Check the arXiv id resolves to the title you wrote; a
wrong id is the one error nobody catches by reading.
