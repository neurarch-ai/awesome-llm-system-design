# 10. 把它们拼起来：完整的方案

第 1 到第 6 节把每个阶段的选项和权衡都讲了一遍，第 7 节展示了真实团队在哪里分道扬镳。但它们都没给出一个每个决定都已经拍板的完整系统。这一节收官做三件事：给出一套有立场的默认技术栈，让选择困难症不至于卡住第一版；把本章的场景从头到尾走一遍，每个选择都定下来并且算出规模；再展示同样这些决定在约束变了之后会怎么翻转。最后以一个最小可运行的 ANN 索引收尾，一个文件，不用装任何东西。

## 默认技术栈：从这里开始，有理由再偏离

本章每个阶段都有三到六个说得过去的选项，一个第一次搭的人可能一个星期都在比较索引库，一条查询都还没服务出去。跳过这一步。下面这套栈是第一版生产系统的合理默认值；每一行都写了什么时候该偏离，以及哪一节解释了原因。库年年都在换，但每个阶段的接口（embed、索引、检索、融合、重排、评估）不会变，所以按接口逐阶段来选，把任何具体的库都当成可替换的。

| 阶段 | 默认 | 什么时候偏离 | 为什么（章节） |
|---|---|---|---|
| Embedding 模型 | 小的 384 维 bi-encoder（MiniLM-L6 那一档），批量写一套部署，延迟敏感的查询另一套 | 召回够不到线而内存还有富余：BGE-large / E5-large；多语言语料：multilingual-e5 | [3](03-the-embedding-service.md) |
| 索引 | 语料装得进内存就用 HNSW；增量插入白送 | 有内存预算限制的十亿级：IVF-PQ；一台普通机器：DiskANN；双塔点积检索：ScaNN 那种各向异性 PQ | [4](04-vector-index.md) |
| 量化 | 8-bit 向量，加上对短名单的全精度重打分 | 全精度下语料也能宽裕地装进内存：跳过；极端规模：PQ 码（每条向量 24 字节，压缩 64 倍） | [4](04-vector-index.md) |
| 混合融合 | 稠密 + BM25 并行，用 RRF 融合 | 查询是纯自然语言、完全没有精确 token（少见）：只用稠密；已经有 Elasticsearch 而且问题出在词项不匹配：SPLADE | [5](05-hybrid-and-reranking.md) |
| 重排 | 结果给人看的时候，在融合后的 top-100 上跑 cross-encoder | 短名单反正要喂给一个会重新打分的下游排序器：跳过 | [5](05-hybrid-and-reranking.md) |
| 过滤 | 推进索引里做，绝不放在后面 | 过滤放过的文档不到约 1%：按分区拆子索引直接路由 | [6](06-serving-and-scaling.md) |
| 新鲜度 | 持久队列，GPU 批处理 embedding worker，两个索引都做增量 upsert | 商品目录稳定、能接受一天的陈旧：定时全量重建更简单 | [6](06-serving-and-scaling.md)、[3](03-the-embedding-service.md) |
| 评估 | 在往下游传的那个 k 上量 recall@k，在基于时间的划分上对着全量扫描的上界来量 | 永远不要偏离。先把标注查询集建起来 | [4](04-vector-index.md)、[8](08-interview-qa.md) |

最后一行是新手最容易跳过、之后最后悔的一条：没有标注查询集和一个暴力扫描出来的召回上界，每一个索引旋钮都只是凭感觉，而一个设得太低的 `ef` 会悄无声息地把召回封顶，编码器上花再多钱也没用。花一下午做标注，第一次调 `nprobe` 时就回本了。

## 完整的方案

回到[第 1 节](01-clarifying-requirements.md)的场景：一亿篇文本文档，每季度增长 10%，top-k 检索要在 p99 50ms 以内，召回要高，带属性过滤，查询构成是自然语言加精确编码，插入和删除的新鲜度要在分钟级。下面是整个系统，每个选择都已敲定，并附上它为什么胜出。

| 决定 | 选择 | 为什么它胜出 |
|---|---|---|
| 编码器 | MiniLM-L6 那一档，384 维，读写分开部署 | CPU 上约 5ms 的查询推理，塞得进 50ms 预算里 embed 这一格；1024 维会让索引内存几乎翻三倍 |
| 查询 embedding | 带缓存；只有未命中才跑编码器推理 | 重复查询很常见；一次缓存命中把 4-8ms 变成 1ms 以内 |
| 索引 | HNSW，分片，增量插入 | 量化之后语料装得进内存；分钟级的新鲜度 SLA 排除了只能重建的结构 |
| 压缩 | 8-bit 向量（Voyager 的 E4M3 那一档），对短名单做全精度重打分 | 内存砍掉四分之三，召回损失由重打分补回来；永远不要把压缩后的分数当成最终结果 |
| 检索 | 稠密 ANN + BM25 并行，RRF 融合 | 面试官钉住了精确编码类的查询；只用稠密会漏掉它们，而 RRF 不需要分数校准 |
| 重排 | 在融合后的 top-100 上跑 cross-encoder，按界面分别开关 | 面向人的界面需要前 3 条的精度；喂给下游排序器的界面跳过它，省下 10-30ms |
| 过滤 | 推进索引里做；高选择性的取值单独拆子索引 | 一个丢掉 99% 候选的后过滤会把整个 ANN 预算浪费掉 |
| 新鲜度 | 队列，GPU 批量 embedding，向量索引和词法索引一起 upsert | 分钟级 SLA；两路通道必须保持同步，否则混合融合的效果会退化 |
| 评估 | 在下游用的那个 k 上量 recall@k 并对比全量扫描上界，基于时间的划分，用在线 A/B 把关 | 第 1 节钉住了检索漏召会直接伤害产品，所以召回是第一个要盯的数字 |

**索引内存。** 一亿条 384 维 float32 向量，原始向量大约 153 GB（[第 3 节](03-the-embedding-service.md)）。HNSW 每条向量还要加 `M * 8` 字节的图边：M = 32 时又是约 26 GB，落在[第 4 节](04-vector-index.md)给出的约 178 GB 全精度数字附近。把向量量化到 8 bit，向量 payload 降到约 38 GB，于是整个索引大约 64 GB：四个分片，每片约 16 GB，再按查询吞吐做副本，全精度向量则放在热路径之外，专门用来给短名单重打分。同一份语料如果是 1024 维，原始就要从 409 GB 起步，这也是为什么编码器维度是跟索引一起定的，而不是只看榜单排名。

**延迟。** [第 6 节](06-serving-and-scaling.md)的分项预算在常见路径上落在 50ms 以内还有余量：缓存命中的查询 embedding 约 1ms（未命中 4-8ms），2-15ms 的 ANN 检索与 2-8ms 的 BM25 并行，所以只算较慢的那一路，RRF 融合约 1ms，cross-encoder 重排 10-30ms，网络 2-5ms。取中间值，开着重排器时 p99 加起来接近 40ms。长尾一飙，第一个被截短或跳过的就是重排器；其余环节要么有缓存，要么并行，要么快到可以忽略。

**成本。** 线上账单是内存租金，不是按查询算的 token：一个约 64 GB 的分片索引意味着几台内存吃紧的副本，加上写路径上的 GPU embedding worker，在加副本之前，成本不随查询量变化。举个例子：四个分片乘两份副本就是八个索引节点。本章的校准参照是 Vespa 公开的十亿级方案，十亿条 int8 向量在 50ms 内做到 90% 的 recall@10，每月约 \$6K（[第 7 节](07-how-teams-do-it-in-production.md)）；一个一亿文档的服务远在这之下。真正值得记住的数字是反事实：不做量化，每个分片的内存几乎要翻三倍；选 1024 维的编码器，整个集群规模都要成倍放大，换来的召回提升[第 3 节](03-the-embedding-service.md)已经表明是趋平的。

**第一个月会出什么问题。** 早期运维里有三种失效模式最常见，所以上线之前就要把它们的信号接好：墓碑堆积（HNSW 的删除会把死节点仍然连在图里，于是在高频变更下召回慢慢下滑、延迟慢慢爬升，全程不报错；盯住墓碑占比，并按[第 4 节](04-vector-index.md)留出重建的预算）；按过滤类别拆开看的召回（带高选择性过滤的查询最先撞上[第 8 节](08-interview-qa.md)那套低通过率的算术，这批用户会来说"搜索坏了"，而全局召回看起来还很正常）；以及 embedding 缓存未命中（新出现的长尾查询在 p99 上要背上完整的 4-8ms 编码器开销；缓存命中率下滑就是延迟预算即将漏掉的早期预警）。

## 同样的技术在不同约束下

实践中真正重要的复盘问题不是"哪个索引最好"，而是"在我的约束下哪个索引最好"。下面是同一条流水线搭了三遍。只有中间那一列是上面那套方案；另外两列保持完全相同的阶段接口，却几乎换掉了每一个实现选择。两侧列里的语料规模仅为示意。

| | 内部 wiki 搜索 | 商品目录搜索（本章） | 十亿商品的市场检索 |
|---|---|---|---|
| 语料 / 消费方 | 50 万篇文档；结果给人看 | 一亿篇文档；人和下游模型都要用，按更难的那种情况设计 | 十亿以上商品；只喂给一个下游的学习式排序器 |
| 延迟预算 | 几百毫秒也没关系 | 整次搜索调用 p99 < 50ms | 检索喂给一个自带预算的排序器；吞吐和内存才是主导 |
| 编码器 | 小的 384 维 bi-encoder；整个语料在一个批处理任务里重新 embed | 384 维，查询侧带缓存，写路径用 GPU 批处理 | Matryoshka：ANN 用短的前缀维度，排序器用完整向量，一次训练搞定 |
| 索引 | 默认的 HNSW，单进程，不分片，全精度 | 分片的 8-bit HNSW，增量 upsert | IVF-PQ（如果是双塔 MIPS 就用 ScaNN 那种各向异性 PQ），加全精度重打分 |
| 混合 / 重排 | BM25 + RRF 仍然是必须的（工单号、黑话）；宽松的预算装得下 cross-encoder | 混合 RRF；cross-encoder 按界面开关 | 词法通道视查询构成而定；不用 cross-encoder，下游排序器会重新打分 |
| 新鲜度 | 发布时重新 embed；整个语料一个批次就够 | 队列 + 增量 upsert，分钟级 SLA | 每周全量批量构建加每日增量，Instacart / LinkedIn 那种做法 |
| 评估 | 100 条标注查询，对着整个语料的全量扫描来量 | recall@k 对比全量扫描上界，按时间划分，A/B 把关 | 在排序器实际消费的那个 k 上量 recall@500，外加长尾商品的覆盖率 |
| 什么算过度设计 | 分片、量化、重建调度器 | DiskANN、PQ 码 | cross-encoder、按查询类别开关重排、把延迟往 50ms 以下调 |

有两个教训。第一，wiki 那一列基本上都是做减法：50 万向量上，单个全精度 HNSW 进程已经足够精确，哪儿都放得下，还顺手把分片、量化、重建调度从需要调的东西里划掉了；但词法通道在每一轮删减里都活了下来，因为精确 token 类的查询是用户的属性，不是规模的属性。第二，市场那一列展示了当最终排序由下游排序器负责时会变什么：cross-encoder 消失了，召回改在排序器消费的那个大 k 上量，选索引家族的是内存预算而不是延迟，而这正是 PQ 压缩和 Matryoshka 维度值回它们那份复杂度的场景。

## 每个约束各自决定什么

压缩版的决策指南。从需求里读出左边那一列，右边几列告诉你在比较任何工具之前，它先动的是哪个杠杆。

| 你的约束 | 它动的杠杆 | 经验法则 |
|---|---|---|
| 语料规模 vs 内存 | 索引家族 | 装得进内存：HNSW。有预算限制的十亿级：IVF-PQ。一台机器上十亿：DiskANN |
| 延迟预算 | `ef` / `nprobe`，重排开关 | 把宽度旋钮往上调到召回过线，剩下的预算花在重排上；cross-encoder 的价码是 10-30ms |
| 查询构成里有没有精确 token | 混合检索 | SKU、错误码、人名：稠密之外并行跑 BM25 再用 RRF 融合是必须的，不是一项优化 |
| 结果的消费方 | 重排 | 人要看前 3 条：上 cross-encoder。下游排序器会重新打分：跳过 |
| 新鲜度 SLA | upsert 策略 | 分钟级：增量插入（HNSW、FreshDiskANN）。天级：定时重建更简单，还能顺带把图修好 |
| 过滤的选择性 | 过滤放在哪里 | 放过大部分文档：索引内过滤就行。放过不到约 1%：分区并路由到子索引 |
| 相似度目标 | 量化器 | 双塔点积检索：各向异性 PQ 或者做 L2 归一化；按欧氏距离调的量化器会悄无声息地损失 MIPS 召回 |
| 变更频率 | 重建预算 | 删除很多：跟踪墓碑占比并安排重建；增量插入不可能永远免费 |
| 召回底线 | 编码器维度、重打分 | 选能过线的最小模型；压缩过的短名单一律用全精度重打分 |

## 最小可运行的 ANN 索引

所有向量数据库教程的读后感都一样：读者把服务跑起来了，还是不明白 `nprobe` 为什么存在。所以这里把本章最核心的那个权衡，召回与工作量的关系，压进一个文件里，零安装。每个生产组件都被换成了接口相同、体量最小的东西：训练好的粗量化器变成几轮 Lloyd 迭代的简易 k-means，FAISS 的倒排表变成一个 dict，真实 embedding 变成带种子的高斯簇，而全量暴力扫描扮演的正是它在生产里的角色：那个所有 ANN 数字都要对着量的 ground-truth 召回上界。

```python
"""A toy IVF index in one file: build, probe, and measure recall vs work."""
import random

random.seed(7)
DIM, N_CLUSTERS, N_VECS, N_QUERIES, K = 16, 24, 3000, 60, 10

# --- synthetic corpus: clustered vectors, like real embeddings ---------------

def rand_unit():
    return [random.gauss(0, 1) for _ in range(DIM)]

true_centers = [rand_unit() for _ in range(N_CLUSTERS)]
corpus = []
for _ in range(N_VECS):
    c = random.choice(true_centers)
    corpus.append([x + random.gauss(0, 0.35) for x in c])

def dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))

# --- index build: k-means-lite centroids + inverted lists --------------------

NLIST = 32
centroids = random.sample(corpus, NLIST)
for _ in range(8):                                   # a few Lloyd iterations
    sums = [[0.0] * DIM for _ in range(NLIST)]
    counts = [0] * NLIST
    for v in corpus:
        j = min(range(NLIST), key=lambda i: dist2(v, centroids[i]))
        counts[j] += 1
        for d in range(DIM):
            sums[j][d] += v[d]
    centroids = [[s / c for s in sums[i]] if (c := counts[i]) else centroids[i]
                 for i in range(NLIST)]

inverted = {i: [] for i in range(NLIST)}             # centroid id -> vector ids
for vid, v in enumerate(corpus):
    inverted[min(range(NLIST), key=lambda i: dist2(v, centroids[i]))].append(vid)

# --- search: exact scan vs probing the n_probe nearest lists -----------------

def exact_topk(q):
    return sorted(range(N_VECS), key=lambda vid: dist2(q, corpus[vid]))[:K]

def ivf_topk(q, n_probe):
    lists = sorted(range(NLIST), key=lambda i: dist2(q, centroids[i]))[:n_probe]
    cands = [vid for i in lists for vid in inverted[i]]
    return sorted(cands, key=lambda vid: dist2(q, corpus[vid]))[:K], len(cands)

queries = [[x + random.gauss(0, 0.35) for x in random.choice(true_centers)]
           for _ in range(N_QUERIES)]
truth = [set(exact_topk(q)) for q in queries]

print(f"{N_VECS} vectors, {NLIST} lists, recall@{K} over {N_QUERIES} queries")
print("n_probe  recall@10  corpus scanned")
for n_probe in (1, 2, 4, 8, 16):
    hits = scanned = 0
    for q, t in zip(queries, truth):
        found, n_cands = ivf_topk(q, n_probe)
        hits += len(t & set(found))
        scanned += n_cands
    recall = hits / (K * N_QUERIES)
    frac = scanned / (N_VECS * N_QUERIES)
    print(f"{n_probe:>7}  {recall:>9.3f}  {frac:>13.1%}")
```

跑一下，那张表就是[第 4 节](04-vector-index.md)全部论点的五行版：`n_probe = 1` 时索引只扫了语料的 4.0%，就已经拿到 0.803 的 recall@10；2 个 probe 时扫 7.3% 换来 0.985；到 4 个 probe，碰了 14.2% 的向量就达到 1.000 的召回。召回随 `n_probe` 上升，而扫描比例始终远低于全量扫描要付的 100%，这两列之间的差距就是 ANN 检索全部的经济学理由。它也说明了这个旋钮为什么会饱和：一旦探测的单元已经覆盖了查询真正的邻域，再多的 probe 买到的只有延迟，这正是[第 6 节](06-serving-and-scaling.md)把 `nprobe` 和 `ef` 当成要对着召回底线来调的预算、而不是要拉满的旋钮的原因。把高斯簇换成真实 embedding，把简易 k-means 循环换成训练好的粗量化器，把 dict 换成带 PQ 码的 FAISS 倒排表，把精确扫描换成一个离线评估任务，你就把本章的索引层重建出来了。
