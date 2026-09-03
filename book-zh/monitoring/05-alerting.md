# 5. 告警

## 核心规则：对比率和变化量告警，不对单个事件告警

不管什么规模，一个每出现一条被标记的答案就叫 on-call 的系统，只会训练出一支忽略告警的团队。单独一条无依据的论断、单独一次 judge 不同意、单独一个慢请求，都是噪声。信号是**变更之后的比率偏移**，合适的工具是相对最近基线计算的 z-score（当前比率离基线有几个标准差）。

对无依据率来说：

$$z_t = \frac{r_t - r_{\text{ref}}}{\sqrt{\,r_{\text{ref}}(1 - r_{\text{ref}}) / n_t\,}}$$

其中 $r_t$ 是当前窗口（含 $n_t$ 条打过分的 trace）里的无依据率，$r_{\text{ref}}$ 是参考窗口的基线比率。当 $z_t$ 超过阈值（通常取 $z \geq 3$）时叫人。

```python
import numpy as np
def rate_zscore(r_t, r_ref, n_t):
    # z-score for a rate shift vs a baseline proportion, using the binomial standard error
    se = np.sqrt(r_ref * (1 - r_ref) / n_t)                # standard error of the baseline rate
    return float((r_t - r_ref) / se)
# rate_zscore(0.08, 0.05, 500) -> 3.0779 (above the z>=3 page threshold)
```

这样一来，真正的幻觉飙升（把比率推高好几个百分点的那种）会很快触发，而日常的噪声不会。

注意分母里 $n_t$ 的作用：窗口更小（采样更重）会收窄置信区间，更容易发现小的比率偏移，但同时也抬高了最小可检测的效应量。采样率和窗口大小要一起调，不能各调各的。

## on-call 分级：告警速度要和答错的代价匹配

不是每个质量信号都配得上凌晨三点的电话。告警的紧急程度要和一个坏答案的代价对得上：

| 信号 | 检测延迟 | 响应级别 |
|---|---|---|
| 护栏拦截率飙升（安全回退） | 分钟级 | 立即叫人（PagerDuty 或同类） |
| 无依据率 z-score 飙升 | 分钟到小时 | 一小时内叫人 |
| judge 分数滚动平均低于阈值 | 小时级 | 开工单，下个工作日处理 |
| 输入分布漂移（embedding 距离上升） | 小时到天 | 每周看板复盘，主动排查 |
| 定时冻结评估集回放出现回退 | 取决于调度周期（小时到天） | 拦住待发布的版本，不叫人 |
| 单请求成本 p95 飙升 | 分钟级 | 超过成本预算阈值就叫人 |

关键的区分在于**拦（block）**和**叫（page）**：前者阻止一次发布触达用户，后者把人叫醒。冻结评估集上的回退是拦；judge 分数的缓慢漂移是开一张工单。

## 每次变更都回放

每次换模型、每次改 prompt，都是质量可能悄悄变化的时刻。把冻结评估集回放接成在每个这类事件上自动触发，赶在流量切到新配置之前。这就是上线前评估里那道部署关卡的持续版本：关卡现在永远在跑，而不只跑一次。

一次变更的两步走：

1. 用冻结评估集跑一遍候选版本。分数回退到可接受区间以下，就拦住发布。
2. 冻结评估通过之后，把一片金丝雀流量路由到候选版本，监控线上的代理分数、反馈和延迟，至少覆盖一个完整的流量周期（二十四小时能覆盖一个昼夜模式），再扩大范围。

## 避开尾部采样的陷阱

均匀随机采样会把人工审核和 judge 的预算花在常见的、简单的请求上，很少碰到那种正在造成损害的罕见失败模式。采样要分层：

- 显式反馈低或负面的请求（丢弃、点踩）多抽。
- 修改率或重试率高的请求多抽。
- 检索分数低的请求多抽（模型是在薄弱甚至空的上下文上作答的）。
- 护栏（自动安全过滤器）触发或者差点触发的请求多抽。
- 保留一片均匀抽样的基线，这样整体质量还有一个无偏估计。

这种分层设计，才能让人工审核队列始终对准失败，而不是把预算烧在那些本来就会得高分的例子上。具体做法是按上面的准则给每条 trace 一个权重，然后按权重比例抽样，而不是均匀抽：

```python
import numpy as np
def stratified_sample(n, weights, k):
    # draw k of n trace indices with probability proportional to per-trace weights;
    # weight up the suspicious tail (low feedback, high retry, low retrieval score)
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()                                        # normalize weights into a probability distribution
    return np.random.choice(n, size=k, replace=False, p=p)
# stratified_sample(1000, weights, 50) -> 50 indices biased toward high-weight (suspicious) traces
```
