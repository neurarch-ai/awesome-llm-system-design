#!/usr/bin/env node
// Stage the repo as a browsable MkDocs site under .site/ (gitignored).
//
// Why this exists: the repo is ~490 markdown files and 75k lines. On GitHub that
// is a file listing, so a reader has to already know which file they want. The
// site gives it a sidebar, a search box, and stable URLs, from exactly the same
// markdown, with no second copy of the content to keep in sync.
//
// What it does:
//   1. Mirrors the repo into .site/content/ (skipping .git, .github, .site).
//   2. Derives the whole nav from the files themselves: page titles are each
//      file's H1, and the book chapter order and grouping are read out of
//      book/README.md (and book-zh/README.md) rather than restated here, so a new
//      chapter appears in the sidebar as soon as it is listed in that README.
//   3. Writes .site/mkdocs.yml.
//
// Usage:
//   node tools/build-site.mjs           # stage + write config
//   mkdocs serve -f .site/mkdocs.yml    # local preview at 127.0.0.1:8000
//   mkdocs build -f .site/mkdocs.yml    # -> .site/out
//
// Dependencies for the mkdocs step only: pip install -r tools/requirements-site.txt

import {
  cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync,
} from "node:fs";
import { basename, join } from "node:path";

const OUT = ".site";
const CONTENT = join(OUT, "content");
// tools/ is staged even though it is build machinery: every chapter summary links
// its per-topic fragment under tools/comparisons and tools/teardowns directly, so
// dropping it would break those links. The fragments stay out of the nav.
const SKIP = new Set([".git", ".github", ".site", "node_modules", ".DS_Store"]);

const SITE_NAME = "LLM System Design Interview";
const SITE_URL = "https://neurarch-ai.github.io/awesome-llm-system-design/";
const REPO_URL = "https://github.com/neurarch-ai/awesome-llm-system-design";

// ---------------------------------------------------------------- staging

function stage() {
  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(CONTENT, { recursive: true });
  for (const entry of readdirSync(".", { withFileTypes: true })) {
    if (SKIP.has(entry.name)) continue;
    cpSync(entry.name, join(CONTENT, entry.name), { recursive: true });
  }
  for (const f of walk(CONTENT)) normalize(f);
}

function walk(dir) {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else if (e.name.endsWith(".md")) out.push(p);
  }
  return out;
}

// Two rewrites, both applied to the staged copy only, so the markdown a reader sees
// on GitHub is untouched and the two renderings agree.
function normalize(file) {
  const src = readFileSync(file, "utf8");
  const out = blankLineBeforeInterruptingList(resolveFolderLinks(file, src));
  if (out !== src) writeFileSync(file, out);
}

// The book links a chapter as `[Agent Orchestration](agents/)`, which is right on
// GitHub and wrong once pretty URLs move a section page down one path segment
// (`.../reasoning-serving/09-summary/` + `../agents/` lands inside the chapter it
// started in). Point those at the folder's README instead.
function resolveFolderLinks(file, src) {
  const dir = file.slice(0, file.lastIndexOf("/"));
  return src.replace(/\]\((?!https?:|#|\/)([^)\s]+\/)\)/g, (whole, target) =>
    existsSync(join(dir, target, "README.md")) ? `](${target}README.md)` : whole,
  );
}

// CommonMark lets a list interrupt a paragraph, Python-Markdown does not, so
// `**Tricks and gotchas**` followed straight by bullets renders as a list on GitHub
// and as one run-on paragraph on the site. The repo does this in about 1750 places,
// which is most of its interview questions, tricks and mistakes lists. Insert the
// blank line those renderers disagree about.
//
// Only a list that actually interrupts a paragraph is touched: if any line since the
// last blank line was itself a list item, this is a continuation of that list, and a
// blank line there would loosen it instead of fixing anything.
function blankLineBeforeInterruptingList(src) {
  const lines = src.split("\n");
  const out = [];
  let inFence = false;
  let sawListItemInBlock = false;
  let prev = "";
  for (const line of lines) {
    const isFence = line.trimStart().startsWith("```");
    if (isFence) inFence = !inFence;
    const isItem = !inFence && /^ {0,3}([-*+]|\d{1,9}[.)]) +\S/.test(line);
    if (
      isItem &&
      !sawListItemInBlock &&
      prev.trim() !== "" &&
      !/^ {0,3}(#|>|\||```|<)/.test(prev.trimStart()) &&
      !/^\s/.test(prev)
    ) {
      out.push("");
    }
    if (line.trim() === "" || isFence) sawListItemInBlock = false;
    else if (isItem) sawListItemInBlock = true;
    out.push(line);
    prev = line;
  }
  return out.join("\n");
}

// ---------------------------------------------------------------- titles

const titleCache = new Map();

// The H1 of a page, with inline markdown stripped, used as its sidebar label.
// Falls back to the filename so a page missing an H1 still gets a readable entry
// rather than disappearing from the nav.
function titleOf(path) {
  if (titleCache.has(path)) return titleCache.get(path);
  let title = null;
  if (existsSync(path)) {
    for (const line of readFileSync(path, "utf8").split("\n")) {
      const m = line.match(/^#\s+(.+?)\s*$/);
      if (m) { title = m[1]; break; }
    }
  }
  if (!title) {
    title = basename(path, ".md").replace(/^\d+-/, "").replace(/-/g, " ");
    title = title.charAt(0).toUpperCase() + title.slice(1);
  }
  title = title
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .trim();
  titleCache.set(path, title);
  return title;
}

const mdFiles = (dir) =>
  readdirSync(dir).filter((f) => f.endsWith(".md")).sort();

// Every page in a chapter folder, index first: mkdocs-material renders a section
// whose first entry is its own index as a clickable section header.
function chapterNav(dir) {
  const files = mdFiles(dir);
  const ordered = [
    ...files.filter((f) => f === "README.md"),
    ...files.filter((f) => f !== "README.md"),
  ];
  return ordered.map((f) => ({ title: titleOf(join(dir, f)), path: join(dir, f) }));
}

// ---------------------------------------------------------------- book nav

// Read the chapter grouping straight out of the book's own README: each "### Group"
// heading starts a group, and every link to a directory that exists under it is a
// chapter, in the order the README lists them. Parsing the README instead of
// hard-coding the order here means the sidebar cannot drift from the book's own
// table of contents, and the Chinese edition needs no separate spec.
function bookNav(root) {
  const readme = join(root, "README.md");
  const lines = readFileSync(readme, "utf8").split("\n");
  const groups = [];
  let current = null;
  for (const line of lines) {
    const h3 = line.match(/^###\s+(.+?)\s*$/);
    if (h3) { current = { title: h3[1], chapters: [] }; groups.push(current); continue; }
    for (const m of line.matchAll(/\]\(([^)#:]+?)\/\)/g)) {
      const dir = join(root, m[1]);
      if (!existsSync(dir) || !statSync(dir).isDirectory()) continue;
      if (!current) continue;
      if (current.chapters.some((c) => c.dir === dir)) continue;
      current.chapters.push({ dir, title: titleOf(join(dir, "README.md")) });
    }
  }

  const nav = [{ title: titleOf(readme), path: readme }];
  // The standalone companion pages (the method, reading paths, numbers, mock
  // interview) sit next to the chapter folders and are not in any group.
  for (const f of mdFiles(root)) {
    if (f === "README.md") continue;
    nav.push({ title: titleOf(join(root, f)), path: join(root, f) });
  }
  for (const g of groups.filter((g) => g.chapters.length)) {
    nav.push({
      title: g.title,
      children: g.chapters.map((c) => ({ title: c.title, children: chapterNav(c.dir) })),
    });
  }
  return nav;
}

function topicsNav() {
  const files = mdFiles("topics");
  return [
    { title: "All topics", path: "topics/README.md" },
    ...files
      .filter((f) => f !== "README.md")
      .map((f) => ({ title: titleOf(join("topics", f)), path: join("topics", f) })),
  ];
}

// ---------------------------------------------------------------- yaml

const q = (s) => `"${String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;

function navYaml(items, depth) {
  const pad = "  ".repeat(depth);
  return items
    .map((it) =>
      it.children
        ? `${pad}- ${q(it.title)}:\n${navYaml(it.children, depth + 1)}`
        : `${pad}- ${q(it.title)}: ${it.path}`,
    )
    .join("\n");
}

const optional = (title, path) => (existsSync(path) ? [{ title, path }] : []);

function buildNav() {
  const nav = [
    { title: "Home", path: "README.md" },
    { title: "Topics", children: topicsNav() },
    { title: "The book", children: bookNav("book") },
    ...optional("Papers", "papers.md"),
    ...optional("Datasets", "datasets.md"),
    {
      title: "Case studies",
      children: [
        { title: "By topic", path: "CASE-STUDIES.md" },
        { title: "Teardowns", path: "CASE-TEARDOWNS.md" },
        { title: "By company", path: "CASE-STUDIES-BY-COMPANY.md" },
        { title: "By industry", path: "CASE-STUDIES-BY-INDUSTRY.md" },
      ],
    },
    {
      title: "Practice",
      children: [
        { title: "Question bank", path: "questions.md" },
        { title: "Deep dives", path: "deep-dives.md" },
        { title: "The answer framework", path: "framework/answer-framework.md" },
      ],
    },
    ...(existsSync("book-zh") ? [{ title: "中文版", children: bookNav("book-zh") }] : []),
    {
      title: "Contributing",
      children: [
        { title: "How to contribute", path: "CONTRIBUTING.md" },
        ...optional("Templates", "template/README.md"),
      ],
    },
  ];
  return nav;
}

// ---------------------------------------------------------------- config

function mkdocsYml(nav) {
  return `# Generated by tools/build-site.mjs. Do not edit; edit the generator.
site_name: ${q(SITE_NAME)}
site_url: ${SITE_URL}
site_description: ${q("Interview-ready walkthroughs of production LLM systems: RAG, KV cache, serving, agents, evals, guardrails.")}
repo_url: ${REPO_URL}
repo_name: neurarch-ai/awesome-llm-system-design
edit_uri: edit/main/
docs_dir: content
site_dir: out

theme:
  name: material
  language: en
  icon:
    repo: fontawesome/brands/github
  features:
    - navigation.tabs
    - navigation.indexes
    - navigation.top
    - navigation.tracking
    - toc.follow
    - search.suggest
    - search.highlight
    - search.share
    - content.code.copy
    - content.action.edit
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: deep purple
      accent: deep purple
      toggle:
        icon: material/weather-night
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: deep purple
      accent: deep purple
      toggle:
        icon: material/weather-sunny
        name: Switch to light mode

plugins:
  # jieba (see tools/requirements-site.txt) segments the Chinese edition at index
  # time and marks the word boundaries with a zero-width space, so the separator has
  # to split on it. The trailing alternative splits between two CJK characters as
  # well, which is what makes a query someone actually types ("投机解码", no spaces)
  # match text indexed as two words; without it the search box answers "no matching
  # documents" for every multi-word Chinese query.
  - search:
      separator: '[\\s\\u200b\\-,:!=\\[\\]()"/]+|(?!\\b)(?=[A-Z][a-z])|\\.(?!\\d)|&[lg]t;|(?<=[\\u4e00-\\u9fff])(?=[\\u4e00-\\u9fff])'

markdown_extensions:
  - abbr
  - admonition
  - attr_list
  - def_list
  - footnotes
  - md_in_html
  - tables
  - toc:
      permalink: true
      toc_depth: 3
  - pymdownx.details
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.tasklist:
      custom_checkbox: true
  # The book writes math as GitHub-flavoured KaTeX ($...$ and $$...$$), so the site
  # renders it with KaTeX too rather than MathJax: same engine, same output.
  - pymdownx.arithmatex:
      generic: true
  # This exact custom fence is also what makes mkdocs-material load and theme
  # mermaid itself, so the book's diagrams need no extra script.
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format

# Many pages link to files that are not nav entries (assets, scripts, the license),
# which is expected here rather than an error.
validation:
  nav:
    omitted_files: info
    not_found: warn
  links:
    absolute_links: info
    unrecognized_links: info

extra_javascript:
  - javascripts/katex.js
  - https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js
  - https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js

extra_css:
  - https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css

nav:
${navYaml(nav, 1)}
`;
}

const KATEX_JS = `// Render arithmatex's generic output with KaTeX, re-running on instant navigation.
document$.subscribe(() => {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\\\(", right: "\\\\)", display: false },
      { left: "\\\\[", right: "\\\\]", display: true },
    ],
    throwOnError: false,
  });
});
`;

// ---------------------------------------------------------------- main

const nav = buildNav();
stage();
mkdirSync(join(CONTENT, "javascripts"), { recursive: true });
writeFileSync(join(CONTENT, "javascripts", "katex.js"), KATEX_JS);
writeFileSync(join(OUT, "mkdocs.yml"), mkdocsYml(nav));

const count = (items) =>
  items.reduce((n, it) => n + (it.children ? count(it.children) : 1), 0);
console.log(`site staged: ${count(nav)} pages in nav, config at ${join(OUT, "mkdocs.yml")}`);
