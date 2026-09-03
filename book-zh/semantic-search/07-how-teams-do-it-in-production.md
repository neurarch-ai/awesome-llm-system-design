# 7. 真实团队在生产环境里怎么做

每一个生产级的语义搜索服务，骨架都是同一套：离线把语料编码，建 ANN 索引，请求进来时 embed 查询，检索近似邻居，跟一路词法通道融合，可选地再重排。各家的差别在于内存预算花在哪儿、向量压得有多狠，以及检索是一段还是两段。

## 真实设计在哪里分道扬镳

| 系统 | ANN 索引 | 压缩 | 混合词法通道 | 重排 | 关键设计选择 |
|---|---|---|---|---|---|
| Spotify（Voyager） | HNSW（hnswlib） | E4M3 8-bit 浮点（相比 Annoy 省 4 倍） | 无 | 无 | 无状态的内存态 K8s pod；部署时加载索引文件；Python 和 Java binding 共用同一份距离计算 |
| Meta（Faiss） | GPU 上的 IVF-PQ | OPQ + IMI + 20 字节 PQ 码 | 未说明 | 短名单用全精度精修 | 十亿级的 GPU 加速；k-selection kernel 全在寄存器里；Deep1B 上 2ms 达到约 40% 的 recall@1 |
| Google（ScaNN） | 分区 + 各向异性 PQ | 各向异性的学习式量化 | 未说明 | 全精度重打分 | 针对 MIPS 惩罚平行方向的误差；在 glove-100-angular 上同等召回下 QPS 翻倍 |
| Microsoft（DiskANN） | SSD 上的 Vamana 图 | 遍历用的 PQ 码放在 DRAM | 未说明 | 从 SSD 取全精度 | 一台普通机器装十亿向量；从 SSD 读，约 5ms 达到 95% 召回 |
| Vespa | HNSW-IF 混合 | int8 向量 | 有（稠密 + 倒排文件） | 全精度重打分（深度 4000） | 十亿条 100 维 int8 向量；50ms 内 recall@10 达 90%，每月约 \$6K；对 CRUD 友好 |
| LinkedIn | IVFPQ（Galene） | Matryoshka：ANN 用 2048 维，排序器用 4096 维 | 未说明 | DCNv2 学习式排序器（L2 阶段） | 十亿以上的 profile；每周全量批处理 + 每日 CDC 增量；一个模型同时训出两种维度 |
| Instacart | FAISS ANN | 未说明 | 无（关键词和类目作为 EBR 的补充） | 下游排序器 | 每天重建 FAISS 索引；查询 embedding 缓存命中率 95%；A/B 里加购率 +4.1% |
| Etsy | HNSW | 4-bit PQ | 有（词项 + 神经） | 未说明 | 紧凑的 4-bit PQ 码；购买率 +5.58%；用难负样本训练 |
| Dropbox | 未说明 | 8-bit 自定义缩放，保留完整维度 | 无 | 未说明 | 用 MTEB 跑了 11 个模型来选型，最后选了 multilingual-e5-large（MRR 0.5044，第二名 0.3299）；每篇文档 4KB 的元数据上限决定了维度选择 |
| Walmart | 倒排索引 + 神经 | 未说明 | 有（倒排 + 神经） | 未说明 | 面向长尾商品查询的混合系统，这类查询里词法精确性很关键；代价是要维护两路通道 |
| Faire | Elasticsearch 上的 SPLADE | 稀疏神经 | 有（稀疏神经就是那路词法风格的通道） | 未说明 | 可解释的稀疏神经检索；在已有的 ES 基础设施上做词项扩展 |

表里的分界线很清楚。把语料整个放在内存里的系统会选 HNSW，因为它在给定延迟下召回最好（Spotify、Vespa 的中心点那一层、Etsy）。预算有限又要做到十亿级的系统会压成 PQ 码，必要时再落到 SSD 上（Meta 的 Faiss、DiskANN）。查询类型混杂的系统一定会在稠密之外并行跑一路词法（Vespa、Etsy、Walmart）。需要一个模型同时给出两档质量的系统会用 Matryoshka embedding（LinkedIn）。

**Dropbox 这个案例被低估了。** 在碰任何索引之前，他们先认真做了一轮模型选型：把 MTEB benchmark 适配到自己的多语言场景，测了 11 个模型，然后发现第一名和第二名之间的 MRR 相对差距超过 50%。教训是：如果模型选型没做扎实，换个更好的索引类型基本补不上这个差距。

## 这些系统（第一手技术文章）

- **Spotify** [Introducing Voyager: Spotify's new nearest-neighbor search library](https://engineering.atspotify.com/2023/10/introducing-voyager-spotifys-new-nearest-neighbor-search-library)：用 hnswlib 包的 HNSW，E4M3 8-bit 量化，无状态的内存态 Kubernetes 部署。

- **Vespa** [Billion-scale vector search using hybrid HNSW-IF](https://blog.vespa.ai/vespa-hybrid-billion-scale-vector-search/)：中心点层用内存里的 HNSW，加上落盘的倒排文件，撑起十亿条 int8 向量；50ms 内 recall@10 达 90%；支持 CRUD。

- **Meta** [Faiss: a library for efficient similarity search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)：GPU 加速的 IVF-PQ，用 OPQ、IMI 和 PQ 组合做十亿级相似度检索。

- **Google Research** [Announcing ScaNN: efficient vector similarity search](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/)：针对 MIPS 调过的各向异性学习式量化；在 ann-benchmarks 上召回与 QPS 的曲线领先。

- **Microsoft Research** [DiskANN](https://www.microsoft.com/en-us/research/project/project-akupara-approximate-nearest-neighbor-search-for-large-scale-semantic-search/)：SSD 承载的 Vamana 图；一台机器上十亿向量，约 5ms 达到 95% 召回。

- **LinkedIn** [Semantic Search for AI Agents at Scale](https://www.linkedin.com/blog/engineering/ai/semantic-search-for-ai-agents-at-scale-retrieval-and-ranking-for-linkedins-hiring-assistant)：在十亿以上的 profile 上，用 Matryoshka embedding 做两段式 IVFPQ 检索加 DCNv2 排序。

- **Pinterest** [Advancements in Embedding-Based Retrieval at Pinterest Homefeed](https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e)：双塔 ANN 检索，多路 embedding 扇出加兴趣过滤。

- **Instacart** [How Instacart uses embeddings to improve search relevance](https://company.instacart.com/how-its-made/how-instacart-uses-embeddings-to-improve-search-relevance)：ITEMS bi-encoder 通过 FAISS 提供服务，每天重建索引，查询缓存命中率 95%。

- **Etsy** [Unified Embedding Based Personalized Retrieval in Etsy Search](https://arxiv.org/abs/2306.04833)：HNSW 配 4-bit PQ，词项加神经的混合检索；购买率 +5.58%。

- **Dropbox** [Selecting a model for semantic search at Dropbox scale](https://dropbox.tech/machine-learning/selecting-model-semantic-search-dropbox-ai)：用适配过的 MTEB 对 11 个模型做选型，multilingual-e5-large 胜出；8-bit 量化加自定义缩放。

- **Walmart** [Semantic Retrieval at Walmart](https://arxiv.org/abs/2412.04637)：面向长尾商品查询的倒排索引加神经检索混合方案。

- **Faire** [Beyond BM25 and dense embeddings](https://craft.faire.com/beyond-bm25-and-dense-embeddings-841a7b18ce27)：在 Elasticsearch 上跑 SPLADE 稀疏神经检索，语义可解释。

- **GitHub** [Inside Copilot's new code embedding model](https://github.blog/news-insights/product-news/copilot-new-embedding-model-vs-code/)：自研的代码 embedding 模型，在更低延迟下把 Copilot 的检索质量提升了 37.6%。

- **Uber Eats** [Scaling Multilingual Semantic Search in Uber Eats](https://arxiv.org/abs/2603.06586)：在六个市场上跨店铺、菜品、生鲜的多语言检索。

- **Amazon** [Semantic Product Search](https://arxiv.org/abs/1907.00937)：双塔模型在预计算的商品目录 embedding 上做 kNN 检索；双塔商品搜索的 KDD 2019 经典参考。

完整对比和更多案例研究见 [topics/08-semantic-search-and-embeddings.md](../../topics/08-semantic-search-and-embeddings.md)。
