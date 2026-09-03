# 10. 把它们拼起来：完整的方案

第 1 到 6 节把每个阶段的选项和取舍都讲了一遍，第 7 节展示了真实团队在哪里分道扬镳。
它们都没有给出的，是一个每个决定都已经落定的完整系统。这一节做三件事：
给一套有主见的默认技术栈，免得选择困难卡住第一版；把本章的场景从头到尾走一遍，每个选择都定下来并算清成本；
再展示同样这些决定在约束变了之后怎么翻转。最后收在一个最小的可运行图像 token 预算计算器上，一个文件，不用装任何东西。

## 默认技术栈：从这里开始，有理由再偏离

本章每个阶段都有三到六个说得过去的选项，第一次动手的人可能花一周比较连接器，一张图都还没服务出去。
别这样。下面这套栈是第一版生产部署的合理默认；每一行都写清了什么时候该偏离，以及哪一节解释了原因。
模型每年都在换，但每个阶段的接口（编码、投影、交错、解码、评估）不会变，
所以按接口逐阶段选型，把任何具体模型都当成可替换的。

| 阶段 | 默认选择 | 什么时候偏离 | 为什么（对应小节） |
|---|---|---|---|
| 融合策略 | 晚期融合：预训练编码器加预训练 LLM，用一个训练出来的 projector 粘起来 | 产品必须生成图像而不只是读图：改早期融合，并接受预训练那笔账 | [4](04-model-choices.md) |
| 视觉编码器 | 冻结的 CLIP ViT-L/14 这一档（有 SigLIP 就用 SigLIP） | 目标领域离编码器的预训练分布很远（文档、医疗、卫星）：解冻或者加 adapter | [4](04-model-choices.md) |
| 连接器 / projector | MLP projector，每个 patch 一个解码器 token | 每个请求的成本必须被严格框住、不管分辨率如何：换 Perceiver / Q-Former resampler | [3](03-the-projector-and-tokens.md) |
| 分辨率与分块 | 固定 336px，不分块；在网关降采样 | 任务需要 OCR、图表或密集文字：1024px 加分块和 tile 标签，按请求开，不要全局开 | [3](03-the-projector-and-tokens.md)、[1](01-clarifying-requirements.md) |
| 图像 token 预算 | 上线前先把 token 公式算一遍；限制每个请求的 token 数 | 永远别跳过这个公式。"一张图就是一个 token"正是本章要终结的那个错误 | [3](03-the-projector-and-tokens.md)、[8](08-interview-qa.md) |
| 服务布局 | 两级：可批处理的 DP 编码器，加带连续批处理的 TP 解码器；纯文本请求绕行 | 流量几乎 100% 带图而且量不大：单 server 更简单，也够用 | [6](06-serving-and-scaling.md) |
| 缓存 | 按图像内容 hash 索引的编码器 embedding；图像 hash 折进前缀缓存的 key | 流量是长尾的、几乎不重复的图：缓存赚不到什么，钱花在别处 | [6](06-serving-and-scaling.md) |
| 评估 | VQAv2 软准确率加 POPE 对抗切分 F1，并按分辨率档位跟踪 TTFT 和单请求成本 | 没有例外。只看准确率不看服务成本，上线的会是一个正确但用不起的东西 | [5](05-evaluation.md) |

最后一行是新手最爱跳过、后来最后悔的一行：一个把图像 token 预算翻倍换来的离线准确率提升，
会让 prefill 计算变成四倍，而 benchmark 那套框架不会告诉你任何事。从第一天起就在服务分辨率下跟踪 TTFT。

## 完整的方案

回到 [第 1 节](01-clarifying-requirements.md) 那个场景：一个视觉问答服务，每个请求一张图、最大 1024x1024，
流式输出文本答案，首 token 延迟低于 2 秒，混合负载里 30% 的请求带图、70% 是纯文本。
任务是通用视觉理解；密集文字和图表是以后的事。下面是整个系统，每个选择都已落定，并给出它为什么胜出。

| 决定 | 选择 | 为什么它胜出 |
|---|---|---|
| 融合 | 晚期融合 | 只读的 VQA；复用预训练编码器和 LLM，用零头的训练成本就能拿到这个能力 |
| 视觉编码器 | 冻结的 CLIP ViT-L/14，336px | 任务是自然图像问答；在这个预算下，一个强的预训练骨干胜过从零开始训 |
| Projector | MLP，接在编码器倒数第二层 | 细节随成本增长，而 576 个 token 还负担得起；CLIP 最后一层会丢掉解码器需要的局部细节 |
| 分辨率策略 | 在网关把每次上传都降到 336px | 通用问答不需要原生 1024px；成熟的做法是按任务定分辨率，而不是默认拉满 |
| Token 预算 | 每请求 576 个图像 token，在进编码器之前就强制卡死 | 一次超大上传不能把编码器打爆显存，也不能把整个 batch 饿死 |
| 服务布局 | 编码器层（数据并行）和解码层（张量并行加连续批处理）分开 | 编码器是一趟无状态、可批处理的计算，只占参数量的百分之几；对它做 TP 白费同步，又省不下计算 |
| 路由 | 纯文本请求完全绕开视觉层 | 七成流量一分钱图像基础设施的钱都不用付；这是结构性的赢，不是一项优化 |
| 缓存 | 按内容 hash 缓存编码器 embedding；hash 折进前缀缓存的 key | 重复的图跳过编码；KV key 里的 hash 让两张占位符相同的图不会撞在一起 |
| 评估 | VQAv2 软准确率、POPE 对抗切分 F1，以及每个分辨率档位的 TTFT 和成本 | VQA 奖励自信的回答；POPE 抓那些自信但编出来的；TTFT 让 2 秒这个承诺保持诚实 |

**图像 token 数。** 在定下来的 336px、14 像素的 patch 下，patch 网格是 24x24 = 576 个 token
（[第 3 节](03-the-projector-and-tokens.md)）。谁都不该默认走的那个替代方案：
原生 1024px、patch 16 是 64x64 = 4096 个 token，多 7 倍，而这个任务根本不需要读小字。
选 576 这个 token 数，是让下游每一个数字都成立的那一个决定。

**每张图的 prefill 成本。** 一个带图请求大约是 576 个图像 token 加一个 64 token 左右的问题，
接近 640 个 token 的 prefill，是一个 30 token 纯文本请求序列长度的 21 倍左右。
要是我们按原生 1024px 服务，同一个请求就是 4096 个 token 起步，也就是
[第 1 节](01-clarifying-requirements.md) 里那个 130 倍的爆炸；而 prefill 的计算量随序列长度平方增长，
所以延迟的差距比 token 比例难看得多。

**每个会话的 KV cache。** 在 [第 3 节](03-the-projector-and-tokens.md) 那个作为参照的 32 层 GQA 解码器上
（8 个 KV head、head 维度 128、fp16），576 个图像 token 每请求大约多占 72 MB 的 KV cache，
而 4096 个 token 时大约是 512 MB。省下的这 7 倍，正是让解码层每张 GPU 能多扛好几倍并发带图会话的原因。

**延迟。** 以下是示意值，和 [第 6 节](06-serving-and-scaling.md) 的构成图一致：
网关上校验和降采样几十毫秒，编码器那一趟几十毫秒（缓存命中就是零），然后是约 640 个 token 的 prefill 和第一个解码出来的 token。
336px 时 prefill 和 decode 大致持平，TTFT 稳稳落在一秒以内，
2 秒预算还剩一半，留给排队、冷缓存和多轮对话的长前缀。

**单请求成本。** 示意值，按每百万输入 token \$0.25、每百万输出 token \$1.25 算：
约 640 个输入 token 加一个 150 token 左右的回答，一个带图请求大约 \$0.0004，
而占七成的纯文本请求只花其十分之一。真正值得内化的是那个反事实：
按原生 1024px 服务，图像那一侧的输入账单会翻 6 倍以上，prefill 延迟涨得更多，
而在通用 VQA 上换不来任何可测量的质量提升。分辨率策略不是在质量上让步，它是那个能把自己赚回来的组件。

**上线第一个月会坏在哪。** 早期运维里有三种故障模式占主导，所以在上线前就把它们的信号接好：
细节幻觉（尽管产品定位是通用问答，用户还是会传收据和图表，而 336px 下模型读不出来，于是它开始猜；
带文字图片上的标记率上升，就是要盯的那个 [POPE 式](05-evaluation.md) 信号）、
token 预算爆炸（第一个"就为一个客户"把分辨率上限调高的同事，会让 prefill 变成四倍；
告警要按分辨率档位设在 TTFT p99 上，不要设全局的），
以及编码器和解码器的耦合（如果两层共用 GPU 或者共用一个队列，大的带图请求会队头阻塞那七成文本流量，
纯文本 p99 随带图流量一起上涨，就是[服务拆分](06-serving-and-scaling.md)已经悄悄退化的信号）。

## 同样的技术，换一组约束

实践中真正要紧的复盘问题不是"哪个连接器最好"，而是"在我的约束下哪个连接器最好"。
下面是同一条三段式流水线搭了三遍。只有中间那一列是上面那套方案，另外两列保留了完全相同的
编码器-projector-解码器接口，几乎把每一个实现选择都换掉了。

| | 消费级照片问答（本章） | 发票与文档助手 | 商品库批量生成描述 |
|---|---|---|---|
| 任务 / 流量 | 通用 VQA；30% 带图、70% 纯文本；交互式 | OCR 级别地读密集页面；几乎每个请求都带一份文档 | 每晚给上百万张商品图生成描述；没有人在等 |
| 延迟预算 | TTFT < 2 秒 | 几秒可以接受；标准是小字要读对 | 没有；只看吞吐和每千张图的成本 |
| 编码器 + 分辨率 | 冻结的 CLIP 这一档，固定降到 336px | 原生或分块的 1024px 加 tile 标签；为文档解冻或适配编码器 | 固定低分辨率；描述要的是大意，不是字形 |
| 连接器 / token 预算 | MLP，576 个 token | MLP 加分块；每页几千个 token，认了 | resampler（Q-Former / Perceiver 这一档），固定 32 到 64 个 token |
| 服务 | 两级，纯文本绕行，embedding 缓存 | 单队列就够；赢在限制每页的块数 | 在 spot 算力上跑超大 batch；embedding 缓存收益极高，因为商品图会重复 |
| 评估 | VQAv2 + POPE F1，按档位看 TTFT | DocVQA（ANLS）+ ChartQA（宽松）+ TextVQA；精确匹配会惩罚 OCR 失误 | 抽样人工或 LLM 评审看描述质量；每千张图的成本 |
| 什么算过度设计 | 分块、原生分辨率、图像生成 | 用 resampler：那个固定上限删掉的恰好是产品赖以卖钱的细节 | 两级服务、流式输出、TTFT 看板、高分辨率 |

从中掉出来两条教训。第一，文档那一列把本章的头号经济学整个反过来了：
照片问答那套把力气花在压 token 上，文档那套却刻意花 token（分块、tile 标签、原生分辨率），
因为小于一个词的细节就是产品本身，它的评估也从软投票的 VQA 家族换成了容忍 OCR 失误的那些指标。
第二，批处理那一列展示了延迟和成本互换了约束地位：没有人在等，
resampler 那固定的 32 到 64 个 token 让每张图的成本都是同一个很小的数，
batch 想开多大就开到硬件的极限，而 embedding 缓存在这里发挥得最好，因为商品库每天都在重发同样的图。

## 每条约束各自决定什么

压缩过的决策指南。从需求里读出左列，右边两列告诉你在比较任何模型之前，它先动的是哪个杠杆。

| 你的约束 | 它动的杠杆 | 经验法则 |
|---|---|---|
| 任务必须还原多少细节 | 连接器和分辨率一起 | 只要大意：resampler 或低分辨率 MLP。丰富理解：336px 的 MLP。OCR 和图表：约 1024px 的分块加 tile 标签 |
| 首 token 预算 | 图像 token 数 | token 数以平方推高 prefill；先砍分辨率或给连接器封顶，再去动解码器 |
| 图文混合流量 | 路由与分层 | 只要纯文本占比不小：让它绕开视觉层；队列分开，图片永远堵不住文本 |
| 图像重复度 | 编码器 embedding 缓存 | 商品库、围绕一张图的多轮对话：按内容 hash 缓存；长尾上传：别做 |
| 围绕一张图的多轮对话 | 前缀缓存的 key | 把图像 hash 折进 KV 前缀 key，否则两张占位符相同的图会撞在一起 |
| 多图请求 | 单请求 token 上限 | k 张图线性叠加；限制每请求的图片数，或者用 resampler 压掉多出来的那些 |
| 单请求成本 | 先看 token 预算，再看模型档位 | 把图像 token 砍一半，图像那侧的账单大致就减半，用不着先去找更便宜的模型 |
| 编码器占参数的比例 | 编码器的并行方式 | 只占百分之几时，DP 副本胜过 TP 切分；一次 all-gather 取代逐层 all-reduce |
| 长宽比的多样性 | 分辨率策略 | 长宽比乱七八糟的截图和页面：用动态原生分辨率；尺寸统一的照片：固定裁剪更便宜 |
| 必须生成图像 | 融合策略 | 只有早期融合能吐出视觉 token；只读的产品不该为它的预训练买单 |

## 最小的可运行 token 预算

每一份 VLM 事故复盘读起来都一样：有人没算 token 就上了一个分辨率或分块的改动，
结果是账单或者 TTFT 曲线先发现了它。所以这里把整个成本模型放进一个文件，不用装任何东西。
每个生产组件都被换成接口相同的最小实现：分块策略变成一次向上取整的除法，
projector 变成一个整数池化因子，服务账单变成一个按示意价格算的每页 token 数。
形状才是重点：分辨率以平方推高 token 数，分块把成本量化成阶梯，而连接器再把它抠回来。

```python
"""Image-token budget calculator: resolution -> tiles -> tokens -> prefill bill."""
import math

PATCH = 16            # pixels per patch side (Qwen2-VL / Pixtral class)
TILE = 512            # tiling policy: images above this side are split into tiles
POOL = 4              # projector pooling factor (2x2 patch merge); 1 = plain MLP
TEXT_TOKENS = 60      # text prompt tokens accompanying the images
IMAGES_PER_PAGE = 4   # images in one document page / request
PRICE_IN = 0.25       # illustrative $ per million input tokens

def tiles_for(H, W, tile=TILE):
    """How many tile crops the tiling policy produces; 1 if the image fits."""
    return math.ceil(H / tile) * math.ceil(W / tile)

def tokens_per_tile(H, W, patch=PATCH, tile=TILE):
    """Patch grid of one crop: the full image if it fits, else a full tile."""
    side = min(max(H, W), tile)
    return (side // patch) ** 2

def image_tokens(H, W, patch=PATCH, tile=TILE, pool=1):
    """Raw decoder tokens for one image under the tiling policy, then pooled."""
    raw = tiles_for(H, W, tile) * tokens_per_tile(H, W, patch, tile)
    return raw // pool

def page_bill(H, W, pool):
    """Prefill tokens and cost for a page of IMAGES_PER_PAGE images plus text."""
    toks = IMAGES_PER_PAGE * image_tokens(H, W, pool=pool) + TEXT_TOKENS
    return toks, toks * PRICE_IN / 1e6

def main():
    resolutions = [336, 512, 672, 1024]
    base = image_tokens(resolutions[0], resolutions[0])
    hdr = (f"{'res':>5} {'tiles':>5} {'raw tok':>8} {'pooled':>7} "
           f"{'vs 336':>7} {'page tok':>9} {'page $':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in resolutions:
        raw = image_tokens(r, r)
        pooled = image_tokens(r, r, pool=POOL)
        toks, cost = page_bill(r, r, pool=1)
        print(f"{r:>5} {tiles_for(r, r):>5} {raw:>8} {pooled:>7} "
              f"{raw / base:>6.1f}x {toks:>9} {cost:>9.4f}")
    print()
    toks_raw, cost_raw = page_bill(1024, 1024, pool=1)
    toks_p, cost_p = page_bill(1024, 1024, pool=POOL)
    print(f"1024px page, plain MLP projector : {toks_raw:>6} prefill tokens  ${cost_raw:.4f}")
    print(f"1024px page, {POOL}x pooled projector : {toks_p:>6} prefill tokens  ${cost_p:.4f}")
    print(f"resolution 336 -> 1024 is {1024/336:.1f}x the side, "
          f"{image_tokens(1024,1024)/base:.1f}x the tokens (quadratic); "
          f"pooling claws back {POOL}x")

if __name__ == "__main__":
    main()
```

跑一遍，这张表用大约六十行就把本章那三条成本主张变成了具体的数。
patch 16 下 336px 的图是 441 个 token，1024px 是 4096 个，
和 [第 3 节](03-the-projector-and-tokens.md) 为 Pixtral 这一档网格推出来的是同一个数：
边长 3 倍，token 9.3 倍，平方关系正在生效。
tiles 那一列展示了分块策略怎么把成本量化成阶梯：一张 672px 的上传会落进和整页 1024px 一样的四个补齐块里，
付一模一样的 4096 token 账单，这就是为什么分辨率上限该定在块的边界上。
pooled 那一列则展示了连接器怎么把成本抠回来：一个 4 倍 patch-merge 的 projector
把一页四张图的 1024px 请求从 16,444 个 prefill token（约 \$0.0041，示意值）降到 4,156 个（约 \$0.0010），
resampler 用固定上限拉的是同一个杠杆，只是拉得更狠。
把 POOL 依次改成 1、4、16，你就是在走[连接器的取舍曲线](03-the-projector-and-tokens.md)，
从 MLP 一路走向 Q-Former；本章的每一节，都是这个文件里某一个常量的取值策略。
