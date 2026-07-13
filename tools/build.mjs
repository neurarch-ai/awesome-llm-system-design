// Regenerates CASE-STUDIES.md, CASE-TEARDOWNS.md, CASE-STUDIES-BY-COMPANY.md,
// CASE-STUDIES-BY-INDUSTRY.md from topics/ + tools/comparisons/ + tools/teardowns/.
// Run from the repo root:  node tools/build.mjs
import fs from 'node:fs';
const TOPICS = 'topics', CMP = 'tools/comparisons', TD = 'tools/teardowns';

const csOrder = [
  ['01-rag-serving.md', 'RAG serving'],
  ['08-semantic-search-and-embeddings.md', 'Semantic search and embeddings'],
  ['02-long-context-and-kv-cache.md', 'Long-context and the KV cache'],
  ['04-inference-serving-at-scale.md', 'Inference serving at scale'],
  ['10-realtime-streaming-chat.md', 'Realtime streaming chat'],
  ['11-cost-optimization-and-model-routing.md', 'Cost optimization and model routing'],
  ['03-agent-orchestration.md', 'Agent orchestration'],
  ['09-multimodal-serving.md', 'Multimodal serving'],
  ['05-post-training-pipeline.md', 'Post-training pipeline'],
  ['13-llm-lifecycle.md', 'LLM lifecycle'],
  ['06-evaluation-system.md', 'Evaluation system'],
  ['07-safety-and-guardrails.md', 'Safety and guardrails'],
  ['12-production-monitoring-and-observability.md', 'Production monitoring and observability'],
];
function seenBullets(file) {
  const s = fs.readFileSync(`${TOPICS}/${file}`, 'utf8');
  const start = s.indexOf('## Seen in production'); if (start < 0) return [];
  const after = s.slice(start + 21); const nx = after.indexOf('\n## ');
  const region = nx < 0 ? after : after.slice(0, nx);
  return region.split('\n').filter(l => l.startsWith('- **'));
}
function buildCaseStudies() {
  let total = 0; const sections = [];
  for (const [file, name] of csOrder) {
    const bullets = seenBullets(file); if (!bullets.length) { console.error('no bullets', file); continue; }
    total += bullets.length;
    const nn = file.slice(0, 2); const cp = `${CMP}/${nn}.md`;
    const cmp = fs.existsSync(cp) ? fs.readFileSync(cp, 'utf8').trim() + '\n\n' : '';
    sections.push(`### [${name}](topics/${file}) · ${bullets.length} systems\n\n${cmp}**The systems**\n\n${bullets.join('\n')}`);
  }
  const header = [
    '# Production case studies, by topic', '',
    'The same landscape of shipped LLM systems that the broad indexes catalog',
    '(crediting the [Evidently AI ML system design database](https://www.evidentlyai.com/ml-system-design)),',
    "re-organized into this repo's own use-case taxonomy.", '',
    'Each category is not just a link list: it opens with a **similarities and',
    'differences** synthesis, a **Mermaid diagram of where the real designs diverge**,',
    'a **choices side-by-side table**, the **math that separates the approaches**, and',
    'where useful a **tradeoff quadrant plot**, all read from the underlying engineering',
    'writeups. Then the systems themselves. For the full per-case teardown of any one,',
    'see [CASE-TEARDOWNS.md](CASE-TEARDOWNS.md); browse the same systems',
    '[by company](CASE-STUDIES-BY-COMPANY.md) or [by industry](CASE-STUDIES-BY-INDUSTRY.md).',
    '', `${total} systems across the taxonomy, and growing.`, '', '---', '',
  ].join('\n');
  fs.writeFileSync('CASE-STUDIES.md', header + sections.join('\n\n---\n\n') + '\n');
  console.log(`CASE-STUDIES.md: ${total} systems, ${sections.length} categories`);
}

const tdOrder = ['01','08','02','04','10','11','03','09','05','13','06','07','12'];
function buildTeardowns() {
  const blocks = tdOrder.map(nn => fs.readFileSync(`${TD}/${nn}.md`, 'utf8').trim());
  const header = [
    '# Case teardowns', '',
    'The [case studies index](CASE-STUDIES.md) lists the shipped systems and links',
    'their engineering writeups. This document goes one level deeper: each entry is a',
    'teardown of how a real team actually built the system, read from their own',
    'writeup and presented as a diagram of their design, the interview questions that',
    'design invites, the non-obvious tricks it gets right, and the common mistakes on',
    'that kind of system with concrete fixes.', '',
    'Every teardown is faithful to a first-party engineering source. Diagrams are',
    'Mermaid and render on GitHub. Organized by the same taxonomy as the rest of the repo.',
    '', '---', '',
  ].join('\n');
  fs.writeFileSync('CASE-TEARDOWNS.md', header + blocks.join('\n\n---\n\n') + '\n');
  console.log(`CASE-TEARDOWNS.md: ${blocks.reduce((a,b)=>a+(b.match(/^### /gm)||[]).length,0)} teardowns`);
}

const norm = (c) => ({'Google Research':'Google','Google DeepMind':'Google','DeepMind':'Google','YouTube/Google':'Google','Google / ETH Zurich':'Google','Google/ETH Zurich':'Google','Meta (FAIR)':'Meta','Meta AI':'Meta','Facebook':'Meta','Microsoft Research':'Microsoft','Microsoft (MSRC)':'Microsoft','Microsoft (LLaVA)':'Microsoft','Amazon Science':'Amazon','Block (Square)':'Block','Together AI':'Together','Fireworks AI':'Fireworks','Alibaba (Qwen)':'Alibaba','Alibaba Qwen':'Alibaba','Mistral AI':'Mistral','Expedia Group':'Expedia','Grafana Labs':'Grafana','Twilio Segment':'Twilio','Red Hat (vLLM)':'Red Hat','AMD (ROCm)':'AMD','vLLM (UC Berkeley)':'vLLM','Daily (Pipecat)':'Daily','Uber Eats':'Uber','IBM Research':'IBM'}[c] || c);
const IND = {}; const put = (i, ...cs) => cs.forEach(c => IND[c] = i);
put('Big Tech and cloud','Criteo','Yahoo','Google','Meta','Microsoft','Amazon','Apple','NVIDIA','IBM','Alibaba','Snowflake','Databricks','Cloudflare','Dropbox','Salesforce','Elastic','Vespa','Red Hat','AMD','PyTorch','Hugging Face','Kuaishou');
put('AI labs and foundation models','OpenAI','Anthropic','DeepSeek','Character.AI','Cognition','Mistral','Moonshot AI','Ai2','OpenGVLab');
put('AI infra and developer tools','Anyscale','Baseten','Together','Fireworks','Modal','LangChain','vLLM','llm-d','LMSYS','LMSYS / SGLang','Replit','GitHub','GitLab','Sourcegraph','Vercel','Glean','Datadog','Honeycomb','Grafana','Slack','Discord','Twilio','Stack Overflow','Grammarly','Krisp','Vapi','Daily','LiveKit','Deepgram','AssemblyAI','ElevenLabs','Cartesia','KIVI','Feast','Tecton','MongoDB','Canva','Figma','Intercom','Segment','Algolia');
put('E-commerce and retail','Allegro','Instacart','Etsy','Wayfair','Walmart','Shopify','Mercari','Zalando','Faire','Stitch Fix','Nextdoor','Asos','Nordstrom','Ocado','Oda','Coupang','Mercado Libre','OLX','eBay','Cars24','Gousto','Picnic');
put('Media and streaming','Netflix','Spotify','Vimeo');
put('Social platforms','LinkedIn','Pinterest','Twitter','Snap','Bumble','Roblox','Glassdoor');
put('Fintech and banking','Stripe','PayPal','Nubank','Block','Ramp','Wealthsimple','Feedzai','Capital One','Coinbase','Monzo','Adyen','Goldman Sachs','Royal Bank of Canada','Lemonade');
put('Delivery and mobility','Uber','DoorDash','Grab','Gojek','Lyft','Delivery Hero','Foodpanda','Grubhub','Careem','BlaBlaCar');
put('Travel and hospitality','Airbnb','Booking','Booking.com','Expedia','GetYourGuide');
put('Professional, legal, and education','Thomson Reuters','Duolingo','Yelp','Zillow');
const isResearch = (c) => / et al\.?|University|Universit|Stanford|Berkeley|MIT|UIUC|UCSD|UC San|UT Austin|Peking|Cohere|RISELab|Kohavi|Hamilton|^"|Hidden Technical Debt|Chip Huyen|Evidently AI|Colfax|,/.test(c);
const industryOf = (c) => IND[c] || (isResearch(c) ? 'Research and academia' : 'Other');
const topicName = (f) => f.replace(/^\d\d-/, '').replace(/\.md$/, '').replace(/-/g, ' ').replace(/\b\w/g, m => m.toUpperCase()).replace(/\bLlm\b/g,'LLM').replace(/\bRag\b/g,'RAG').replace(/\bKv\b/g,'KV').replace(/\bAb\b/g,'AB');
function collect() {
  const cases = [];
  for (const f of fs.readdirSync(TOPICS).filter(x => /^\d\d.*\.md$/.test(x)).sort())
    for (const l of seenBullets(f)) { const m = l.match(/^- \*\*(.+?)\*\*\s*\[([^\]]+)\]\(([^)]+)\)/); if (m) cases.push({ company: norm(m[1].trim()), title: m[2].trim(), url: m[3].trim(), topic: topicName(f) }); }
  return cases;
}
function buildIndexes() {
  const cases = collect(); const label = 'the same LLM problem';
  const gc = new Map(); for (const c of cases) { (gc.get(c.company) || gc.set(c.company, []).get(c.company)).push(c); }
  const byCo = `# Case studies, by company\n\nThe same shipped systems as [CASE-STUDIES.md](CASE-STUDIES.md), pivoted by company so you can see how one org approaches ${label} across problems. ${cases.length} case studies, ${gc.size} companies. Deep teardowns of many of these live in [CASE-TEARDOWNS.md](CASE-TEARDOWNS.md).\n\n---\n\n` +
    [...gc.entries()].sort((a,b)=>b[1].length-a[1].length||a[0].localeCompare(b[0])).map(([co,l])=>`### ${co} (${l.length})\n\n`+l.map(c=>`- [${c.title}](${c.url}) *(${c.topic})*`).join('\n')).join('\n\n') + '\n';
  fs.writeFileSync('CASE-STUDIES-BY-COMPANY.md', byCo);
  const order = ['Big Tech and cloud','AI labs and foundation models','AI infra and developer tools','E-commerce and retail','Media and streaming','Social platforms','Fintech and banking','Delivery and mobility','Travel and hospitality','Professional, legal, and education','Research and academia','Other'];
  const gi = new Map(); for (const c of cases) { const i = industryOf(c.company); (gi.get(i) || gi.set(i, []).get(i)).push(c); }
  const byInd = `# Case studies, by industry\n\nThe same shipped systems as [CASE-STUDIES.md](CASE-STUDIES.md), pivoted by industry so you can see which ${label} patterns recur in your domain. ${cases.length} case studies across ${gi.size} industries.\n\n---\n\n` +
    order.filter(o=>gi.has(o)).map(i=>{const l=gi.get(i).sort((a,b)=>a.company.localeCompare(b.company)||a.title.localeCompare(b.title));return `## ${i} (${l.length})\n\n| Company | Case study | Topic |\n| --- | --- | --- |\n`+l.map(c=>`| ${c.company} | [${c.title}](${c.url}) | ${c.topic} |`).join('\n');}).join('\n\n') + '\n';
  fs.writeFileSync('CASE-STUDIES-BY-INDUSTRY.md', byInd);
  const unmapped = [...new Set(cases.map(c=>c.company))].filter(c=>industryOf(c)==='Other');
  console.log(`indexes: ${cases.length} cases, ${gc.size} companies${unmapped.length?', Other: '+unmapped.join(', '):''}`);
}

buildCaseStudies(); buildTeardowns(); buildIndexes();
