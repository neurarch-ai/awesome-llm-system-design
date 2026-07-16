#!/usr/bin/env node
// On-demand external-link checker for book/**/*.md.
// Fetches every http(s) URL and classifies it:
//   OK        2xx / 3xx
//   BLOCKED   401/403/405/406/429 (bot-block or auth wall; the page almost certainly exists)
//   DEAD      404/410, DNS failure, TLS error, or connection refused  <- the ones to fix
//
// This is deliberately NOT wired into CI: external sites rate-limit and bot-block,
// which would make a push gate flaky. Run it manually or on a schedule:
//   node tools/check-links.mjs
// Exits non-zero if any DEAD link is found.

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const ROOT = "book";
const CONCURRENCY = 12;
const TIMEOUT_MS = 20000;
const UA = "Mozilla/5.0 (compatible; neurarch-linkcheck/1.0)";

function walk(dir) {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else if (e.name.endsWith(".md")) out.push(p);
  }
  return out;
}

// url -> Set of files that reference it
const refs = new Map();
for (const f of walk(ROOT)) {
  const t = readFileSync(f, "utf8");
  for (const m of t.matchAll(/\]\((https?:\/\/[^)\s]+)\)/g)) {
    const u = m[1];
    if (!refs.has(u)) refs.set(u, new Set());
    refs.get(u).add(f);
  }
}
const urls = [...refs.keys()];
console.log(`Checking ${urls.length} unique external URLs from ${ROOT}/ ...\n`);

const BLOCKED = new Set([401, 403, 405, 406, 429]);

async function probe(u) {
  for (const method of ["HEAD", "GET"]) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
      const r = await fetch(u, { method, redirect: "follow", signal: ctrl.signal, headers: { "User-Agent": UA } });
      clearTimeout(timer);
      if (method === "HEAD" && (r.status === 405 || r.status === 501)) continue; // retry with GET
      if (r.status >= 200 && r.status < 400) return { cls: "OK", status: r.status };
      if (BLOCKED.has(r.status)) return { cls: "BLOCKED", status: r.status };
      return { cls: "DEAD", status: r.status };
    } catch (e) {
      clearTimeout(timer);
      if (method === "HEAD") continue; // some hosts refuse HEAD; try GET
      return { cls: "DEAD", status: e.cause?.code || e.name || "fetch-error" };
    }
  }
  return { cls: "DEAD", status: "no-response" };
}

const results = [];
let i = 0;
async function worker() {
  while (i < urls.length) {
    const u = urls[i++];
    results.push({ u, ...(await probe(u)) });
  }
}
await Promise.all(Array.from({ length: CONCURRENCY }, worker));

const dead = results.filter((r) => r.cls === "DEAD");
const blocked = results.filter((r) => r.cls === "BLOCKED");
console.log(`OK: ${results.length - dead.length - blocked.length}   BLOCKED (likely live): ${blocked.length}   DEAD: ${dead.length}\n`);
if (dead.length) {
  console.log("DEAD links (fix these):");
  for (const d of dead.sort((a, b) => String(a.status).localeCompare(String(b.status)))) {
    const where = [...refs.get(d.u)].map((f) => f.replace(ROOT + "/", "")).join(", ");
    console.log(`  [${d.status}] ${d.u}\n        in: ${where}`);
  }
  process.exit(1);
}
console.log("No dead links.");
