#!/usr/bin/env node
// Validate book/**/*.md for render-safety and house-style issues.
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
//
// Usage: node tools/validate-book.mjs

import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join, dirname, resolve, extname } from "node:path";

const ROOT = "book";

function walk(dir) {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else if (e.name.endsWith(".md")) out.push(p);
  }
  return out;
}

const problems = [];
const add = (file, msg) => problems.push(`${file}: ${msg}`);

const FENCE = /^```/gm;
const IMG = /!\[[^\]]*\]\(([^)]+)\)/g;
const LINK = /\[[^\]]+\]\(([^)]+)\)/g;
const HEADING = /^#{2,}\s.+$/gm;
const MATH = /\$\$[\s\S]+?\$\$|\$(?!\$)[^$\n]+?\$/g;
const IMGEXT = new Set([".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]);

for (const file of walk(ROOT)) {
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

  // 6. KaTeX hazards inside math
  for (const seg of (t.match(MATH) || [])) {
    const head = seg.slice(0, 48).replace(/\n/g, " ");
    if (seg.includes("*")) add(file, `literal '*' in math (use \\ast): ${head}`);
    if (/<[a-zA-Z]/.test(seg)) add(file, `'<' before a letter in math (use \\lt): ${head}`);
    if (seg.includes("\\operatorname")) add(file, `\\operatorname in math (use \\text): ${head}`);
  }

  // 7. inline-math $ parity
  let s = t.replace(/\$\$[\s\S]+?\$\$/g, "").replace(/```[\s\S]+?```/g, "").replace(/`[^`]*`/g, "").replace(/\\\$/g, "");
  const dollars = (s.match(/\$/g) || []).length;
  if (dollars % 2 !== 0) add(file, `odd inline-math '$' (${dollars}); a literal money '$' is likely mispairing with math, escape it as \\$`);

  // 8. no em/en dashes
  if (/[–—]/.test(t)) add(file, "contains an em or en dash (use commas, periods, parentheses)");
}

if (problems.length) {
  console.error(`\nBook validation FAILED with ${problems.length} problem(s):\n`);
  for (const p of problems) console.error("  " + p);
  process.exit(1);
}
console.log(`Book validation passed: ${walk(ROOT).length} files clean.`);
