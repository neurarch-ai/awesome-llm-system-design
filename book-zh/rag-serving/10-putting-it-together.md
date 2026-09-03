# 10. 把它们拼起来：完整的方案

第 1 到 6 节把每个阶段的选项和取舍都讲了一遍；第 7 节展示了真实团队在哪里走上不同的路。
但它们谁都没给出一套"每个决定都已经拍板"的完整系统。这一节收官，要做三件事：
给出一套有主张的默认技术栈，好让选择困难症不至于卡住你的第一次搭建；
把本章的题目场景从头到尾走一遍，每个选择都落定并算清成本；
再展示当约束变了之后，同样这些决定会怎么翻转。
最后以一份最小可运行的 RAG 管道收尾，一个文件，不装任何依赖。

## 默认技术栈：从这里开始，有理由再偏离

本章的每个阶段都有三到六个说得过去的选项，
新手完全可能花掉一整周比较各种库，却一个 chunk 都还没检索到。
别走这条路。下面这套栈，对第一次做生产级搭建来说是个靠谱的默认值；
每一行都写清了什么时候该偏离，以及哪一节解释了原因。
工具年年在换，但每个阶段的接口（解析、分块、embedding、建索引、检索、重排、生成、评估）不会变，
所以按接口去选每个阶段，把任何具体的库都当成可替换件。

| 阶段 | 默认 | 什么时候偏离 | 为什么（章节） |
|---|---|---|---|
| 解析 | 感知版面的解析器（Docling 或 Unstructured 这一类）；表格保留为 markdown | 语料本来就是干净的 markdown：直接跳到分块 | [3](03-indexing-and-chunking.md) |
| 分块 | 递归结构化，约 400 token 封顶，10% 到 15% 重叠 | chunk 单独拿出来就失去意义：加上下文前缀 | [3](03-indexing-and-chunking.md) |
| Embedding | 一个够强的小 encoder（MiniLM / bge-small 这一类，384 维） | 召回在领域行话上到了瓶颈：换领域微调过的或者更大的模型 | [3](03-indexing-and-chunking.md) |
| 索引 | 约 10 万 chunk 以下用暴力扫描；更多就上 HNSW | 向量装不下内存：量化或者换 IVF-PQ | [3](03-indexing-and-chunking.md) |
| 检索 | 稠密 + BM25 混合，RRF 融合，top-n = 50 | 语料里压根没有精确词项查询（少见）：纯稠密 | [4](04-retrieval-and-reranking.md) |
| 重排 | Cross-encoder 收到 top-m = 8 | 首 token 预算低于约 800ms：缩小 n 或者干脆跳过 | [4](04-retrieval-and-reranking.md)、[6](06-serving-and-scaling.md) |
| 生成 | 中档 instruct 模型；引用来源 ID；检索弱就拒答 | 答案需要多跳推理：沿着范式阶梯往上爬 | [5](05-generation-and-grounding.md)、[2](02-frame-the-system.md) |
| 评估 | 100 条查询的黄金集，标注好相关 chunk，在动任何参数之前就建好 | 没有例外。先把黄金集建起来 | [3](03-indexing-and-chunking.md) |

最后一行是新手最容易跳过、事后最后悔的：没有黄金集，
每个分块和 embedding 的决定都只是凭感觉，你根本判断不出一次改动到底有没有帮上忙。
花一个下午做标注，第一次换组件的时候就回本了。

## 完整的方案

回到[第 1 节](01-clarifying-requirements.md)的场景：
5,000 万篇内部文档，10,000 名员工，峰值 20 QPS，p99 首 token 低于 1.5 秒，
新鲜度在一小时以内，答案要带引用并支持拒答，还要按用户做 ACL。
下面是整个系统，每个选择都已拍板，以及它胜出的理由。

| 决策 | 选择 | 为什么它胜出 |
|---|---|---|
| 解析 | 感知版面，保留表格，剥掉模板文字 | 5,000 万篇文档里有多栏 PDF；解析一旦搞砸，下游无从补救 |
| 分块 | 递归结构化，400 token 封顶，50 token 重叠 | wiki 和设计文档有现成的标题可用；重叠让骑跨边界的答案保持完整 |
| Embedding | 384 维 encoder，写路径和读路径分开部署 | 维度决定索引内存（见下）；写路径靠批量，读路径受延迟约束 |
| 索引 | HNSW，分片，int8 量化，每个 chunk 带 ACL 元数据 | 4,500 万 chunk 下要兼顾召回和延迟；ACL 必须在搜索内部过滤 |
| 检索 | 稠密 + BM25 混合加 RRF，top-n = 50 | 工单 ID 和产品代号都是精确词项查询，稠密检索会把它们糊掉 |
| 重排 | Cross-encoder，top-m = 8 | 相比直接塞 top-50，prefill token 少了约 5 倍；精度上去了，成本下来了 |
| 生成 | 中档模型，流式输出，引用校验，拒答规则 | "有依据，否则闭嘴"就是明确写下来的质量标准 |
| 新鲜度 | 变更驱动的 upsert，按文档 ID 打墓碑标记 | 一小时的要求排除了每晚重建的做法 |
| 评估 | 离线用黄金集算 recall@k；线上抽样跑 LLM judge 并盯引用失败率 | 检索召回是质量上限，所以它是第一个要盯的数字 |

**索引容量估算。** 5,000 万篇文档，每篇约 300 token，按 400 token 封顶、50 token 重叠切分，
大约得到 4,500 万个 chunk（见[第 3 节](03-indexing-and-chunking.md)）。
384 维、float32 的话就是 4500 万 x 384 x 4 字节，原始向量约 69 GB；
int8 量化能把它压到 17 GB 左右，再加上 HNSW 的图开销，
分到两三个带副本的分片上很宽裕。
同样的语料换成 768 维，上面每个数字都要翻倍，
这也是为什么 embedding 维度是和索引一起定的，而不是只看榜单排名。

**延迟。** [第 6 节](06-serving-and-scaling.md)那份组件预算，
带重排时 p99 落在约 1040ms，在 1.5 秒的预算里还留有余量：
查询 embedding 约 20ms，带 ACL 过滤的 ANN 约 40ms，
50 个候选上的 cross-encoder 约 80ms，8 个 chunk 的 prefill 约 250ms，
到第一个解码 token 约 600ms。

**每次查询的成本。** 拼好的 prompt 大致是 300 token 的 system prompt，
加上 8 个 chunk x 400 token，再加上查询本身，接近 3,600 个输入 token，
答案约 300 token：

$$\text{cost/query} \approx T_{\text{in}} \cdot p_{\text{in}} + T_{\text{out}} \cdot p_{\text{out}} + c_{\text{rerank}} + c_{\text{embed}}$$

按中档模型的示意价格（每百万输入和输出 token 分别是 $0.25 和 $1.25），
LLM 这块大约是 $0.0009 加 $0.0004；查询 embedding 可以忽略不计，
自托管的 cross-encoder 摊下来也只占 LLM 成本的一小部分。
就算它 $0.0015 一次查询，17 万次查询下来大约 $250 一天。
真正值得记住的数字是那个反事实：如果没有重排器，
把 top-50 个 chunk 全塞进去，prompt 就变成约 20,000 token，
LLM 成本和 prefill 延迟都要乘以 5。
重排器不是质量上的奢侈品，它是那个能把自己成本挣回来的组件。

**头一个月会坏在哪儿。** 早期运维里有三种失败模式占了大头，所以上线前就要把它们的信号接出来：
新鲜度滞后（索引 upsert 队列的堆积深度对照那个一小时的承诺）、
引用校验失败（[第 5 节](05-generation-and-grounding.md)那个亚毫秒级检查；
失败率往上走，说明生成器正在偏离它的上下文），
以及分人群的召回（ACL 可见范围很窄的用户最先撞上候选集过小的问题，
他们会来报"什么都搜不到"，而全局指标看着一切正常）。

## 同样的技术，换一组约束

实践中真正重要的复盘问题，不是"哪个分块器最好"，而是"在我的约束下哪个分块器最好"。
下面是同一条管道搭了三遍。只有企业那一列是上面那套方案；
另外两列保持完全相同的阶段接口，几乎每一个实现选择都换掉了。

| | 创业公司文档机器人 | 企业知识库（本章） | 批量合规问答 |
|---|---|---|---|
| 语料 / 流量 | 2,000 篇文档，约 3 万 chunk；0.2 QPS | 5,000 万篇文档，4,500 万 chunk；20 QPS | 500 万篇文档；每晚 10 万次查询，无交互 |
| 延迟预算 | 几秒也没关系 | p99 首 token < 1.5s | 没有；只看吞吐和成本 |
| 索引 | 进程内暴力扫描（3 万 chunk 下几毫秒）；完全不用调 ANN | 分片的 int8 HNSW，ACL 在搜索内部 | IVF-PQ，省内存；按批次节奏重建 |
| 检索 / 重排 | 纯稠密 top-10，在黄金集显示精度吃紧之前不上重排器 | 混合 RRF top-50，cross-encoder 收到 top-8 | 混合，狠重排到 top-4：prefill 成本占了账单大头 |
| 生成 | 强力的 API 模型；0.2 QPS 下模型价格无关紧要 | 中档，流式，引用校验 | 小模型，超大批次，前缀缓存，抢占式算力 |
| 新鲜度 | 每次部署全量重新 embedding；整个语料一批就跑完了 | 变更驱动的 upsert，一小时 SLA | 每晚重建是特性，不是妥协 |
| 评估 | 50 条手工维护的黄金查询 | 黄金集 + 线上 judge 抽样 + 引用失败率 | 对每个批次抽一片做 LLM judge |
| 什么算过度设计 | ANN 索引、重排器、各种缓存、任何 agent 化的东西 | 为单个事实型查询上 GraphRAG | 流式、语义缓存、低延迟服务栈 |

由此有两条教训。
第一，创业公司那一列基本上都是在做减法：3 万 chunk 下暴力扫描既精确又快，
还省掉了一整片需要调参的面；0.2 QPS 下每一层缓存都是累赘。
如果整个语料能塞进一个上下文窗口、查询又很稀疏，
[第 2 节](02-frame-the-system.md)那份硬塞对比会告诉你，你现在可能压根还不需要检索。
第二，批量那一列展示了延迟和成本调换了位置：
既然没有首 token 预算，重排就可以更狠（m = 4），模型可以更小，
所有东西都按硬件吃得下的最大批次去跑。

## 每个约束决定什么

浓缩版的决策指南。从需求里读出左边那一列，右边几列会告诉你它先动哪个杠杆，
在你去比较任何工具之前。

| 你的约束 | 它动的杠杆 | 经验法则 |
|---|---|---|
| chunk 数量 | 索引家族 | 约 10 万以下：暴力扫描。到约 1,000 万：内存里跑 HNSW。再往上：分片、量化，或者 IVF-PQ |
| 首 token 预算 | 重排深度 n、保留的 chunk 数 m | 低于约 800ms：n 降到约 20，或者跳过 cross-encoder；低于约 2s：n = 50 全开，m = 8 |
| 每次查询的成本 | 先动 m，再动模型档位 | prefill 随 m 增长；把 m 减半，在还没碰模型之前就大致把 LLM 成本砍了一半 |
| 新鲜度 | upsert 策略、范式上限 | 分钟级到小时级：增量 upsert；同时也排除了在热路径上做昂贵的逐 chunk LLM 富化 |
| 查询重复度 | 缓存层 | 重复很多的内部流量：embedding 缓存加前缀缓存立刻见效；长尾的公开流量：不见效 |
| 查询里有精确标识符 | 混合检索 | 工单 ID、SKU、行话：BM25 和稠密并行是必选项，一步融合换 3 到 5 个点的召回 |
| 多跳或者通读全语料的问题 | 范式的台阶 | 只在失败模式确实要求时才往[阶梯](02-frame-the-system.md)上爬：先改写，再 CRAG，再图谱或者 agent 化 |
| 按用户的权限 | 索引选型 | ACL 必须在 ANN 搜索内部过滤；挑一个原生支持元数据过滤的索引 |
| 回答质量的底线 | 拒答 + 校验 | 引用校验几乎不要钱；检索置信度低于阈值就拒答，别去猜 |

## 最小可运行的 RAG

对每一份框架教程的评价都一样：读者装好了五个库，还是没看清这条管道长什么样。
所以这里把完整的读写路径放进一个文件，零安装。
每个生产组件都换成了接口相同的最小实现：
encoder 变成词袋余弦，ANN 索引变成列表扫描，
cross-encoder 变成精确词项重合度，LLM 变成拼好的 prompt 本身。
形状本身就是这一课；本章的每一节，升级的都是这个文件里的某一个函数。

```python
"""The whole read-and-write path in one file, runnable with no installs."""
import math, re
from collections import Counter

# --- write path -------------------------------------------------------------

def chunk(doc_id, text, size=40, overlap=8):
    """Fixed-size sliding window over words; production: structural chunking."""
    words, step, out = text.split(), size - overlap, []
    for i in range(0, len(words), step):
        out.append({"id": f"{doc_id}#{len(out)}", "text": " ".join(words[i:i + size])})
    return out

STOP = {"the", "a", "an", "is", "are", "for", "of", "to", "and", "what", "when", "does"}

def embed(text):
    """Bag-of-words vector; production: a transformer encoder (e.g. MiniLM).
    Stopwords are stripped for the same reason parsers strip boilerplate:
    high-frequency filler dominates the vector and poisons the match."""
    return Counter(t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP)

def cosine(a, b):
    dot = sum(a[t] * b[t] for t in a if t in b)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0

INDEX = []                                   # production: HNSW / IVF-PQ with ACL metadata

def upsert(doc_id, text, acl):
    global INDEX
    INDEX = [e for e in INDEX if e["doc"] != doc_id]          # tombstone old chunks
    for c in chunk(doc_id, text):
        INDEX.append({"doc": doc_id, "id": c["id"], "text": c["text"],
                      "vec": embed(c["text"]), "acl": acl})

# --- read path --------------------------------------------------------------

def retrieve(query, user, n=4):
    """ACL filter runs inside the search, not after it."""
    visible = [e for e in INDEX if user in e["acl"]]
    return sorted(visible, key=lambda e: cosine(embed(query), e["vec"]), reverse=True)[:n]

def rerank(query, candidates, m=2):
    """Exact-term overlap; production: a cross-encoder over (query, chunk) pairs."""
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    return sorted(candidates, key=lambda e: len(terms & set(e["text"].lower().split())), reverse=True)[:m]

def answer(query, user):
    chunks = rerank(query, retrieve(query, user))
    if not chunks or cosine(embed(query), chunks[0]["vec"]) < 0.05:
        return "I could not find a grounded answer."           # abstain, don't guess
    prompt = "Answer ONLY from these sources and cite their ids:\n"
    prompt += "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    prompt += f"\nQuestion: {query}"
    return prompt                # production: LLM call + verify cited ids exist in prompt

# --- demo -------------------------------------------------------------------

upsert("wiki/oncall", "The on-call rotation for the payments team changes every "
       "Monday at 09:00 UTC. Escalations page the secondary after 15 minutes.", acl={"alice", "bob"})
upsert("design/refunds", "Refunds above 500 dollars require manager approval and "
       "are processed by the billing service within two business days.", acl={"alice"})

print(answer("when does the on-call rotation change?", user="bob"))
print("---")
print(answer("what is the refund approval threshold?", user="bob"))   # ACL: bob can't see it
```

跑一下，这两条查询用大约六十行代码演示了本章两条没得商量的原则：
bob 那条 on-call 问题拿回了一个有依据的答案，附带可引用的 chunk ID；
而他那条退款问题返回的是拒答消息，
因为唯一能回答它的文档在他的 ACL 之外，而且过滤是在检索内部跑的，
所以系统压根没看见它，没泄露它，也没围着它编。
把 `embed` 换成真的 encoder，`INDEX` 换成 ANN 索引，`rerank` 换成 cross-encoder，
最后那条 prompt 接进一次 LLM 调用，你就把本章重建出来了。
