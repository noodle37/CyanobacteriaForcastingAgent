# 在 GIFT-Eval 上评测本项目的 Agent

本文说明如何把 tsfm-agent 接入 GIFT-Eval 基准、在基准的任务切分与指标体系下评测智能体,
以及当前实现与官方协议之间的已知偏差。

撰写日期:2026-11-06

---

## 一、接入目的与定位

GIFT-Eval(Salesforce,arXiv:2410.10393)是通用时序预测评测基准,包含 23 个数据集、
28 个频率目录、97 个任务配置,覆盖 7 个领域。其官方协议把被评测对象定义为
「一个把上下文序列映射到预测分布的函数」,通过 GluonTS 的 `Predictor` 接口接入,
由 `gluonts.model.evaluate_model` 按固定参数计算 11 项指标。

本项目接入 GIFT-Eval 的目的是**评测智能体**:在本项目原有的 HTTP 服务、工具集合、
选模与报告流程完全不变的前提下,把它放到一个公开、可复现的基准环境里,得到与其
他智能体系统可比的数字,并能测量智能体带来的精度变化与推理开销。

接入过程不涉及训练或微调任何时序基础模型。

---

## 二、环境准备

### 2.1 评测环境

评测依赖与 Agent 服务依赖完全隔离,避免互相影响:

| 环境 | 用途 | 关键依赖 |
| --- | --- | --- |
| `.venv-agent` | Agent 服务(含 ReAct 与固定流水线两种模式) | langchain、langgraph、pandas、numpy |
| `.venv-tsfm` | 时序基础模型服务(Chronos-2、TimesFM-3) | torch、chronos、timesfm |
| `.venv-gift` | 基准评测 | gluonts 0.15.1、datasets 2.17.1、numpy 1.26 |

`.venv-gift` 建立在 conda 环境之上并继承其中的 torch,以避免重复下载:

```bash
cd /home/jxy/tsfm-agent
/home/jxy/miniconda3/envs/chronos-cyano/bin/python -m venv --system-site-packages .venv-gift
.venv-gift/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "gluonts~=0.15.1" "datasets~=2.17.1" "numpy~=1.26.0" "pandas>=2.0.0" \
    python-dotenv toolz
```

gluonts 版本与基准的 `pyproject.toml` 锁定一致(0.15.1)。`evaluate_model` 在不同
gluonts 版本上的返回结构与 `TestData` 迭代语义有差异,版本必须对齐。

### 2.2 基准源码

基准源码置于 `third_party/GIFT-Eval/`,`run_eval.py` 会自动把其中的 `src` 加入
模块搜索路径,无需安装为包。

若需重新获取(本机 GitHub 直连不可达,须走代理):

```bash
curl -sL -o /tmp/gift.tar.gz \
  "https://gh-proxy.com/https://github.com/SalesforceAIResearch/GIFT-Eval/archive/refs/heads/main.tar.gz"
tar xzf /tmp/gift.tar.gz -C third_party --strip-components=1
```

### 2.3 数据集

数据托管在 HuggingFace 的 `Salesforce/GiftEval`,本机 huggingface.co 直连不可达,
使用镜像:

```bash
HF_ENDPOINT=https://hf-mirror.com .venv-gift/bin/python - <<'PY'
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Salesforce/GiftEval",
    repo_type="dataset",
    local_dir="/home/jxy/data/gift_eval",
    allow_patterns=["covid_deaths/*"],   # 按需列出需要的子集
)
PY
```

下载后的目录结构可直接被 `datasets.load_from_disk` 读取:每个数据集目录下是
`data-*.arrow`、`state.json` 与 `dataset_info.json`,无需格式转换。

然后把该目录写入环境变量:

```bash
export GIFT_EVAL=/home/jxy/data/gift_eval
```

---

## 三、启动 Agent

评测通过 HTTP 调用 Agent,两种运行模式都可用,接口一致,只需更换地址。

**固定流水线模式**(不配置语言模型,流程为 ingest → select_model → forecast →
baseline → report):单次预测约 0.05 秒,适合批量评测与回归验证。

```bash
./scripts/run_agent_fixed.sh 8001      # 监听 127.0.0.1:8001
```

注意 `agent/settings.py` 用 `load_dotenv` 载入仓库根目录的 `.env`,而 `load_dotenv`
默认不覆盖已存在的变量。因此脚本里必须把三个语言模型变量**置空**而不是 `unset`,
否则键不存在时会被 `.env` 重新填回,Agent 仍会走 ReAct 路径。

**ReAct 模式**(语言模型参与选模与分析):单次运行约 158 秒,适合评测智能体本身。

```bash
./scripts/run_agent.sh                  # 监听 127.0.0.1:8000
```

---

## 四、运行评测

```bash
GIFT_EVAL=/home/jxy/data/gift_eval \
.venv-gift/bin/python -m evaluation.gift_eval.run_eval \
    --dataset covid_deaths \
    --dataset hierarchical_sales/D \
    --term short \
    --limit-series 10 \
    --agent-url http://127.0.0.1:8001 \
    --out evaluation/results/gift_eval
```

常用参数:

| 参数 | 说明 |
| --- | --- |
| `--dataset` | 数据集名,可重复。含频率目录时写成 `hierarchical_sales/D` |
| `--term` | 预测长度档位:`short`、`medium`、`long` |
| `--limit-series` | 每个任务最多评测多少条序列 |
| `--sample-strategy` | 抽样策略:`stride`(默认)、`head`、`random` |
| `--agent-url` | Agent 地址,决定评测固定流水线还是 ReAct |
| `--max-context` | 提交给 Agent 的最大上下文点数,默认 2048 |
| `--max-horizon` | Agent 策略允许的最大预测步数,默认 64 |
| `--model-name` | 结果中记录的名字 |

输出为 `all_results.csv`,前 15 列与官方提交格式一致(列名、顺序完全相同),
其后附加本次评测的信息列:序列数、上下文截断数、平均上下文长度、平均耗时、
Agent 选中的模型、预测长度、季节周期与备注。

---

## 五、适配层做了什么

`evaluation/gift_eval/agent_predictor.py` 把 Agent 的 HTTP 接口包装成 GluonTS 预测器:

1. 对测试集中的每个实例取出历史序列,用 `LastValueImputation` 补齐缺测;
2. 按 `max_context` 截断,并把截断情况记入轨迹;
3. 向 `{agent_url}/v1/runs` 提交请求。**请求体不含 `actual`**——真值只存在于
   GluonTS 的标签侧,Agent 在评分前看不到未来值;
4. 把返回的点预测与 10%、90% 分位扩展为 0.1 至 0.9 共九档;
5. 组装成 `QuantileForecast`,预测原点取 `forecast_start(entry)`。

`evaluation/gift_eval/run_eval.py` 负责其余部分:按官方方式构造 `Dataset`
(多变量数据集按官方做法 `to_univariate=True` 拆成一维)、以
`get_seasonality(freq)` 取季节周期、按官方参数调用 `evaluate_model`
(`axis=None`、`mask_invalid_label=True`、`allow_nan_forecast=False`)、
汇总为结果文件。

---

## 六、与官方协议之间的已知偏差

这些偏差会直接影响数字的可比性,记录在案:

1. **分位数为插值近似。** Agent 的对外契约只有点预测与 10%、90% 两个分位,
   基准的加权分位数损失需要九档。0.2 至 0.4 与 0.6 至 0.8 各档由分段线性插值
   得到,并非模型真实输出的分位。这会使 `mean_weighted_sum_quantile_loss` 与
   `MSIS` 系统性偏离官方数值。彻底消除需要让模型适配器直接输出九档分位。

2. **上下文长度受限。** Agent 策略上限为 2048 点,`context_tail` 为 512 点。
   基准中不少数据集的序列长度远超此值,长历史序列会被截断,直接影响 MASE 与
   分位数损失。结果文件中的 `n_context_truncated` 与 `mean_used_len` 列记录了
   截断情况;若与官方条目比较,应先确认该行未被截断。

3. **预测步数上限 64。** Agent 策略 `horizon_max` 为 64。基准的 medium 与 long
   档预测长度分别为基准值的 10 倍与 15 倍(日频为 300 与 450 步),全部超出该
   上限,因此**当前只有 short 档任务可评测**。评测程序会跳过超出上限的任务并
   给出提示。要覆盖中长档,需先放宽 `configs/permissions.yaml` 中的
   `horizon_max`,并确认模型在该步数下的推理开销。

4. **`MSE[mean]` 与 `MSE[0.5]` 相等。** 预测对象只提供分位键,没有 `mean` 键,
   GluonTS 会以中位数代替均值。官方提交同样如此(榜单中各条目的这两个指标相等),
   因此这一口径与官方一致,无需修改。

5. **抽样结果不覆盖全部序列。** 指标由抽样得到,统计意义取决于抽样规模与策略。
   取前若干条序列(`--sample-strategy head`)可能系统性地取到退化样本:例如
   `car_parts_with_missing` 的前若干条序列标签窗口整体缺测,指标会算成 nan。
   默认的等距抽样可避免这一点。

6. **ReAct 模式下,Agent 不保证按请求的步数返回预测。** ReAct 的 `forecast` 工具允许
   自行指定 `horizon`,模型可能选择与请求不同的步数。GluonTS 的
   `BatchForecast.__getitem__` 会把同一批次内所有预测数组堆叠,长度一旦不一致即抛出
   `all input arrays must have the same shape`,整批失败。适配层已加校验:返回步数少于
   任务要求时报明确错误,多于时按任务长度截取。评测程序的固定流水线模式不存在此问题,
   因为它总是按请求的步数预测。

7. **单任务失败不中断整批评测。** 任一任务抛出异常时,评测程序把该任务记为失败行
   (`notes` 字段写明原因),继续处理其余任务,便于在一批任务中定位问题。

---

## 七、当前状态与阻塞项

已验证通过的部分:

- 评测环境、基准源码引入、数据集下载与加载;
- 固定流水线模式下的端到端评测:日频、周频、月频三类任务均产出完整的 11 项指标,
  单条序列约 0.05 秒;
- 缺测数据的插值处理与等距抽样策略。

尚未打通的部分:

- **ReAct 模式评测受语言模型服务商的额度限制阻塞。** 实测请求返回
  `PermissionDeniedError: 403 - Free quota exhausted`,Agent 以 `partial` 状态结束且
  不返回预测。这是账户侧的配额问题,需要充值或关闭「仅使用免费额度」模式后才能继续。
  适配层与失败处理已经就位,额度恢复后可直接重跑。
---

## 八、下一步

以下事项尚未开展,待评测规模确定后推进:

- 让模型适配器直接输出九档分位,消除偏差 1;
- 放宽策略上限后覆盖 medium 与 long 档任务;
- 在完整 97 个任务配置上跑通,与官方条目逐行对照,验证本地部署与官方复现一致;
- 统计 ReAct 模式的推理开销,与固定流水线模式对照,量化智能体带来的精度变化与成本。
