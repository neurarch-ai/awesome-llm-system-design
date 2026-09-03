#!/usr/bin/env node
// Validate book/**/*.md and book-zh/**/*.md (the Chinese edition) for render-safety and house-style issues.
// Exits non-zero (with a report) if any problem is found, so it can gate CI.
//
// Checks:
//   1. Code fences balanced (even count of ``` per file).
//   2. Mermaid blocks use <br/> for line breaks, never a literal \n.
//   3. Every non-URL image reference resolves to a file on disk.
//   4. Every internal markdown link resolves (file, file.md, or directory).
//   5. No duplicate ## / ### ... headings within a file.
//   6. Math ($...$ and $$...$$) has no GitHub-KaTeX hazards:
//        a literal '*' (use \ast), a '<' before a letter (use \lt), or \operatorname (use \text).
//   7. Inline-math '$' balances once $$ blocks, code, and escaped \$ are removed
//        (an odd count means a literal money '$' is mispairing with real math; escape it as \$).
//   8. No em (—) or en (–) dashes (house style).
//   9. Emphasis (**) that CommonMark cannot pair, e.g. `**标签。**正文`, which renders
//        as literal asterisks because the closing run is not right-flanking.
//
// Usage: node tools/validate-book.mjs

import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join, dirname, resolve, extname } from "node:path";

const ROOTS = ["book", "book-zh"].filter((d) => existsSync(d));

function walk(dir) {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else if (e.name.endsWith(".md")) out.push(p);
  }
  return out;
}

// Emphasis delimiters that CommonMark will not pair, per its flanking rules.
// `**标签。**正文` renders as four literal asterisks: the closing run follows
// punctuation and precedes a letter, so it is neither preceded by whitespace nor
// followed by whitespace or punctuation, and therefore is not right-flanking.
// English prose never trips it because `**Label.** Text` puts a space after the
// closing run; a Chinese translation that drops that space silently loses the bold.
// Only runs of two or more asterisks are reported: a lone `*` in prose is usually
// notation (`Q*`, `h^*`) and cannot be told apart from a broken italic.
const PUNCT = /\p{P}/u;

function stripCodeForEmphasis(t) {
  let inFence = false;
  const lines = t.split("\n").map((l) => {
    if (l.trimStart().startsWith("```")) { inFence = !inFence; return ""; }
    return inFence ? "" : l;
  });
  const s = lines.join("\n");
  // blank out inline code spans, longest backtick runs first
  let out = "", i = 0;
  while (i < s.length) {
    if (s[i] === "`") {
      let j = i; while (j < s.length && s[j] === "`") j++;
      const tick = s.slice(i, j);
      const close = s.indexOf(tick, j);
      if (close !== -1) { out += " ".repeat(close + tick.length - i); i = close + tick.length; continue; }
    }
    out += s[i]; i++;
  }
  return out;
}

function unpairedEmphasis(par) {
  // Collect delimiter runs with their flanking properties.
  const runs = [];
  const re = /\*+/g;
  let m;
  while ((m = re.exec(par)) !== null) {
    const i = m.index, len = m[0].length;
    if (i > 0 && par[i - 1] === "\\") continue; // escaped
    const before = i > 0 ? par[i - 1] : " ";
    const after = i + len < par.length ? par[i + len] : " ";
    const bSpace = /\s/.test(before), aSpace = /\s/.test(after);
    const bPunct = PUNCT.test(before), aPunct = PUNCT.test(after);
    const canOpen = !aSpace && (!aPunct || bSpace || bPunct);
    const canClose = !bSpace && (!bPunct || aSpace || aPunct);
    runs.push({ i, len, left: len, canOpen, canClose, both: canOpen && canClose });
  }

  // CommonMark's "rule of three": when either delimiter can both open and close,
  // a pair whose lengths sum to a multiple of 3 does not match unless both
  // lengths are themselves multiples of 3. This is what stops a lone `*` from
  // closing a `**`, which is the difference between "the bold is broken" and
  // "the bold is fine and an italic sits inside it".
  const canMatch = (o, c) =>
    !((o.both || c.both) &&
      (o.len + c.len) % 3 === 0 &&
      !(o.len % 3 === 0 && c.len % 3 === 0));

  const stack = [], stray = [];
  for (const r of runs) {
    if (r.canClose) {
      for (let k = stack.length - 1; k >= 0 && r.left > 0; k--) {
        const o = stack[k];
        if (!canMatch(o, r)) continue;
        const used = Math.min(o.left, r.left);
        o.left -= used; r.left -= used;
        if (o.left === 0) stack.splice(k, 1);
      }
    }
    if (r.left > 0) {
      if (r.canOpen) stack.push(r);
      else stray.push(r);
    }
  }
  return [...stack, ...stray].filter((r) => r.left >= 2);
}

const problems = [];
const add = (file, msg) => problems.push(`${file}: ${msg}`);

const FENCE = /^```/gm;
const IMG = /!\[[^\]]*\]\(([^)]+)\)/g;
const LINK = /\[[^\]]+\]\(([^)]+)\)/g;
const HEADING = /^#{2,}\s.+$/gm;
const MATH = /\$\$[\s\S]+?\$\$|\$(?!\$)[^$\n]+?\$/g;
const IMGEXT = new Set([".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]);

const files = ROOTS.flatMap(walk);
for (const file of files) {
  const t = readFileSync(file, "utf8");
  const dir = dirname(file);

  // 1. code fences balanced
  const fences = (t.match(FENCE) || []).length;
  if (fences % 2 !== 0) add(file, `unbalanced code fences (${fences})`);

  // 2. mermaid uses <br/> not \n
  let inMermaid = false;
  for (const ln of t.split("\n")) {
    const s = ln.trim();
    if (s.startsWith("```mermaid")) { inMermaid = true; continue; }
    if (inMermaid && s.startsWith("```")) { inMermaid = false; continue; }
    if (inMermaid && ln.includes("\\n")) { add(file, "literal \\n inside a mermaid block (use <br/>)"); break; }
  }

  // 3. image references resolve
  for (const m of t.matchAll(IMG)) {
    const s = m[1].trim();
    if (s.startsWith("http") || s.startsWith("data:")) continue;
    if (!existsSync(resolve(dir, s))) add(file, `missing image: ${s}`);
  }

  // 4. internal links resolve
  for (const m of t.matchAll(LINK)) {
    let u = m[1].split("#")[0].trim();
    if (!u || u.startsWith("http") || u.startsWith("mailto") || u.startsWith("data:")) continue;
    if (IMGEXT.has(extname(u).toLowerCase())) continue;
    const tgt = resolve(dir, u);
    if (!(existsSync(tgt) || existsSync(tgt + ".md"))) add(file, `broken internal link: ${u}`);
  }

  // 5. duplicate headings
  const counts = new Map();
  for (const h of (t.match(HEADING) || [])) counts.set(h.trim(), (counts.get(h.trim()) || 0) + 1);
  for (const [h, n] of counts) if (n > 1) add(file, `duplicate heading (${n}x): ${h}`);

  // 6. KaTeX hazards inside math.
  // Strip escaped \$ first: a literal money '$' is not math, and leaving it in lets the
  // scan run from it to the next real '$', flagging ordinary prose (bold, comparisons)
  // in between. English wraps at ~80 columns so the newline usually stopped it; a
  // translated paragraph on one long line does not.
  const mathText = t.replace(/\\\$/g, "");
  for (const seg of (mathText.match(MATH) || [])) {
    const head = seg.slice(0, 48).replace(/\n/g, " ");
    if (seg.includes("*")) add(file, `literal '*' in math (use \\ast): ${head}`);
    if (/<[a-zA-Z]/.test(seg)) add(file, `'<' before a letter in math (use \\lt): ${head}`);
    if (seg.includes("\\operatorname")) add(file, `\\operatorname in math (use \\text): ${head}`);
  }

  // 7. inline-math $ parity
  let s = t.replace(/\$\$[\s\S]+?\$\$/g, "").replace(/```[\s\S]+?```/g, "").replace(/`[^`]*`/g, "").replace(/\\\$/g, "");
  const dollars = (s.match(/\$/g) || []).length;
  if (dollars % 2 !== 0) add(file, `odd inline-math '$' (${dollars}); a literal money '$' is likely mispairing with math, escape it as \\$`);

  // 8. no em/en dashes (the Chinese double dash "——" is two em dashes and fails the same way)
  if (/[–—]/.test(t)) add(file, "contains an em or en dash (use commas, periods, parentheses)");

  // 9. emphasis that cannot close, so the asterisks render literally
  const stripped = stripCodeForEmphasis(t);
  for (const par of stripped.split(/\n\s*\n/)) {
    for (const r of unpairedEmphasis(par)) {
      const ctx = par.slice(Math.max(0, r.i - 24), r.i + r.len + 20).replace(/\n/g, " ");
      add(file, `emphasis never closes, asterisks will render literally (put a space after the closing ** or move the punctuation outside it): ...${ctx}...`);
    }
  }
}

if (problems.length) {
  console.error(`\nBook validation FAILED with ${problems.length} problem(s):\n`);
  for (const p of problems) console.error("  " + p);
  process.exit(1);
}
console.log(`Book validation passed: ${files.length} files clean.`);
