# 4. 污染与题目效度

有两种情况会让一个 benchmark 数字毫无意义，而 harness 跑得一点毛病没有：模型已经见过答案，或者题目本来就是错的。这两个都是测量问题，不是模型问题，而且在面试里都经常被一句"我们对训练数据做了去重"糊弄过去。

## 五种泄漏

大家谈污染的时候，好像它只有一种。其实有五种，需要的防御各不相同。

| 种类 | 怎么发生的 | 为什么去重抓不到 |
|---|---|---|
| 逐字的题目泄漏 | Benchmark 文件本身就在爬取的语料里 | 只有能搜索训练语料才抓得到，而这只有训练方能做 |
| 近似重复泄漏 | 改写、翻译、换了格式的镜像、带答案的论坛帖子 | 精确匹配的去重会把它们直接放过去 |
| 格式泄漏 | 模型学的是 benchmark 的答题风格，而不是它的题目 | 根本没有任何重复；模型学会的是考试的形状 |
| 蒸馏泄漏 | 在一个本身被污染的模型的输出上训练，或者在从 benchmark 题目生成的合成数据上训练 | 被污染的文本从来没出现在你的语料里 |
| 选择泄漏 | 什么都没漏进训练；是你看着测试集来挑 checkpoint、超参或者 prompt | 这是经由实验者发生的过拟合，而且悄悄累积 |

选择泄漏是资深候选人会点名、初级候选人会漏掉的那一种。扫一遍训练配方的时候把 benchmark 跑 200 次，测试集就变成了验证集。修法是流程上的：留一个封存的切片，带**查询预算**，每次查看都记录下来，并且把一个根据封存切片的比较挑出来的 checkpoint，视为消耗了一次查看额度。

## 检测：真正跑得动的方法

| 方法 | 需要什么 | 能告诉你什么 | 局限 |
|---|---|---|---|
| 题目和训练语料之间的 n-gram 或子串重叠 | 能访问训练语料 | 逐字和轻度编辑过的泄漏的直接证据 | 只有训练模型的人能用；对 n、大小写、空白归一化敏感 |
| Embedding 近似重复搜索 | 训练语料加一个 embedder | 能抓住改写和翻译 | 阈值是主观判断；在网页规模下很贵 |
| Canary 字符串 | Benchmark 发布时带了一个 canary GUID | 模型能复现 canary，说明文件进了训练 | 只能证明文件在，不在证明不了任何事 |
| 基于 token 概率的成员推断（Min-K% Prob、Min-K%++） | 模型的 token 对数概率 | 某段文本在预训练里出现过的统计证据 | 需要 log-prob；对训练很充分或被改写过的文本效果弱（[Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2310.16789)、[Min-K%++](https://arxiv.org/abs/2404.02936)） |
| 时间切分对比 | 题目发布日期加模型的训练截止日期 | 最干净的黑盒信号：截止日期之后的题目和之前难度匹配的题目分数对比 | 需要 benchmark 给题目打时间戳，而且难度必须真的匹配 |
| 函数式孪生题 | 按同一规格重新生成的、题目全新的 benchmark | 在孪生集上大幅下降，是对原集过拟合的直接证据 | 昂贵；孪生集必须难度匹配，这是难点 |
| 扰动和置换测试 | 只需要 API 访问 | 对选项重排或表面改写敏感，暗示记住了表面形式 | 和一般的脆弱性混在一起，所以只是提示，不是定论 |

函数式孪生题的那个结果值得当作具体锚点记住：按同一规格从头重建一个小学数学 benchmark，在同样的模型上重跑，发现某些模型家族有系统性的差距，另一些则没有，这正是污染会留下的印记（[A Careful Examination of Large Language Model Performance on Grade School Arithmetic](https://arxiv.org/abs/2405.00332)）。一篇关于从静态评估走向动态评估的综述收集了工具箱里剩下的部分（[Recent Advances in LLM Benchmarks against Data Contamination](https://arxiv.org/abs/2502.17521)）。

```python
def min_k_percent(token_logprobs, k=0.2):     # k = fraction of least-likely tokens
    n = max(1, int(len(token_logprobs) * k))  # how many tokens to keep
    worst = sorted(token_logprobs)[:n]        # the least likely tokens in the text
    return sum(worst) / n                     # higher (closer to 0) => more likely memorized
# Compare the score for benchmark items against a held-out reference distribution of
# same-domain text the model provably could not have seen; a shifted distribution is
# the signal, a single number means nothing.
```

"和参考分布比较"这一点是大家最容易弄错的。一个 Min-K% 的值单独看没有任何解释力，只有对照同一统计量在训练截止日期之后、同一体裁文本上的分布，它才有意义。

## 预防：让泄漏在构造上就有界的设计

- **按时间闸门的 live benchmark。**从每个模型截止日期之后的来源抽题，并持续刷新。[LiveBench](https://arxiv.org/abs/2406.19314) 从近期的竞赛、论文和新闻里构造问题，按计划轮换题库的一部分；[LiveCodeBench](https://arxiv.org/abs/2403.07974) 给每道题打上发布日期，这样就能在严格晚于模型截止日期的窗口上评估。代价是题库在脚下不停变，所以跨时间的比较需要钉住窗口。
- **私有内部集。**从自己的工作负载里构建，从不发布，未经数据保留审查从不发给第三方 API。这是唯一能*证明*、而不是争辩无泄漏故事的集合。
- **在构造上就 held-out。**程序化生成、带验证器的题目（模板化的符号任务、带属性测试的生成代码）提供无限的新题，代价是构念更窄。
- **Canary 和许可证卫生。**发布任何 benchmark 都附上 canary GUID 和禁止训练的许可证，同时清楚这两者都是行业规范而不是强制手段。
- **报告日期。**每个 benchmark 结果都应带上模型的训练截止日期和题库的日期范围。读者据此就能自行判断污染风险，而不必信任你的去污染声明。

## 另一半：题目效度

即使一个 benchmark 毫无泄漏，题目坏了它照样撒谎。

- **标注错误**封住了可达到的分数上限，并且恰好在模型拉开差距的地方加噪声。对 MMLU 的重新标注发现了个位数百分比的错误率，分布在解析问题、多个正确选项和无法回答的问题上（[Are We Done with MMLU?](https://arxiv.org/abs/2406.04127)）；经过对抗过滤的常识 benchmark 也表现出类似的效度问题（[What the HellaSwag?](https://arxiv.org/abs/2504.07825)）。两个候选落在标注错误带之内的时候，benchmark 就是排不了序，句号。
- **弱的结果验证**是 agent 场景里的对应物：环境依据薄弱得撑不住的证据就宣布成功。那篇关于严谨 agent benchmark 的审计发现，有些任务设置的测试套件会接受错误的解法，成功标准会把什么都不做算成通过，并且把修法打包成一份覆盖任务规格、结果验证和报告的检查清单（[Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825)）。
- **格式伪影。**如果不看题目也能答到高于随机水平，那 benchmark 有一部分测的是选项模式的区分（[Answer Matching Outperforms Multiple Choice](https://arxiv.org/abs/2507.02856)）。便宜的审计方法：把题干去掉跑一遍 benchmark，报告无题干基线。任何明显高于随机的结果都是红旗，应该和分数一起发布。

## 诊断一个可疑的结果

| 症状 | 第一假设 | 能区分的测试 |
|---|---|---|
| 一个模型只在某个老 benchmark 上远超同行 | 那个 benchmark 上的污染 | 在同一构念的截止日期之后的题目上打分，或者用函数式孪生集 |
| 选项重排或题目改写之后分数崩掉 | 记住了表面形式，或者脆弱 | 扰动集加无题干基线 |
| 你的微调在目标 benchmark 上涨了 5 分，其他地方涨了 0 分 | 格式过拟合或选择泄漏 | 同一构念的自由生成版本；查一下开发期间这个集合被查询了多少次 |
| 所有人都在 90 分以上，而且运行之间排序会翻转 | 饱和加标注噪声 | 算出区间；把这个 benchmark 从排序用途里退役 |
| 一个 agent 通过了它明显没完成的任务 | 弱的结果验证 | 人工审计 20 条通过的轨迹；发布 agent 数字之前这项审计不是可选的 |

整节的主线：**一个好得可疑的数字，在证明不是之前，就是污染、坏题或者协议 bug。**面试奖励那个先把这句话说出来、然后指出会去测三者之中哪一个的候选人。
