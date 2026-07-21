# 实验结果记录

## 当前状态

已完成：

- 阅读远端 commit `a4259b9e2eed65b49ed2d42c1dc1ddf1c8c89e3b`。
- 确认该 commit 修复 disk prefix KV cache restore，关键变更在 `CPUKVCacheManager` 和 `kvcache_demo`。
- 确认 `kvcache_demo` 支持：
  - `--dump`
  - `--load`
  - `--split-step`
  - `--load-prefix-step`
- 手机端已创建独立实验目录并部署二进制：
  - `/data/data/com.termux/files/home/csm/mobiinfer/experiments/kv_cache_restore_prefill/bin/kvcache_demo`
  - `/data/data/com.termux/files/home/csm/mobiinfer/experiments/kv_cache_restore_prefill/bin/libMNN.so`

尚未执行正式手机 benchmark。

## 服务器快速测试

已执行。

使用：

```text
/home/ma-user/workspace/csm/mobiinfer/build/kvcache_demo
/home/ma-user/workspace/csm/mobiinfer/transformers/llm/export/model/config.json
```

### 修复前问题

`build/kvcache_demo` 设置了：

```cpp
llm->set_config(R"({"async":false,"kvcache_mmap":true,"tmp_path":"tmp"})");
```

但在 `llm->load()` 初始化 runtime hint 时没有创建 `tmp` 目录，导致 CPU KV cache 文件创建失败：

```text
Failed to create the file: tmp/<addr>.k
Failed to create the file: tmp/<addr>.v
Failed to resize the kvcache files!
Failed to memory-map the kvcache!
```

原始代码随后仍把 `mKVCacheInDisk` 设为 `true`，`ProcessKey<float>` 写入无效 mmap 地址，触发 segfault。gdb backtrace 崩溃点：

```text
MNN::CPUKVCacheManager::ProcessKey<float>
MNN::CPUKVCacheManager::onUpdateKV
MNN::CPUAttention::onExecute
MNN::Transformer::Llm::generate
```

### 修复内容

补丁文件：

```text
experiments/kv_cache_restore_prefill/kv_cache_restore_tmpdir_fix.patch
```

包含两处修改：

- `Llm::setRuntimeHint`: 当 `kvcache_mmap=true` 时，先创建 `tmp_path`。
- `CPUKVCacheManager::onAlloc`: disk KV 文件或 mmap 失败时，不再继续写无效地址，改为 fallback 到 memory KV。

### 服务器验证结果

输入：

- `prefix tokens`: 416
- `variable tokens`: 72
- `decode tokens`: 1

结果：

| case | prefill time s | decode time s |
| --- | ---: | ---: |
| split-step baseline | 4.15 | 0.16 |
| dump prefix one-time cost | 3.24 | N/A |
| load-prefix-step restore | 1.09 | 0.15 |

服务器端单 case prefill 加速：

```text
speedup = 4.15 / 1.09 = 3.81x
reduction = 1 - 1.09 / 4.15 = 73.7%
```

## 手机正式 case

待服务器快速测试通过后执行。

计划使用：

```text
/data/data/com.termux/files/home/csm/mobiinfer/bench20/prompts/prompt_00.txt
/data/data/com.termux/files/home/csm/mobiinfer/bench20/configs/config_baseline_w8a8_ar.json
```

结果表待补充：

| case | prefix tokens | variable tokens | baseline prefill s | restore prefill s | speedup | reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| server prompt_00 smoke | 416 | 72 | 4.15 | 1.09 | 3.81x | 73.7% |
| phone prompt_00 | TBD | TBD | TBD | TBD | TBD | TBD |

## 2026-07-21 图片 continuation 修复

### 现象

greedy 下，文本 `split-step` 与 `load-prefix-step` 输出一致；图片 continuation 在 restore 后从
第一个生成 token 就分叉并很快出现乱码。图片输入结构为：纯文本 prefix 409 token，后续文本和
图片合计 1288 token；缓存中仅有 409-token 文本 prefix KV。

### 隔离实验

| 实验 | split / restore 结果 | 结论 |
| --- | --- | --- |
| 纯文本 continuation 72 token | 一致 | 基础 prefix restore 可用 |
| 长纯文本 continuation 1734 token | 首 token 字节一致 | 排除长度和大扩容 |
| 图片 continuation，deepstack 置零 | `Deep` / `on`，不一致 | 排除 deepstack 本身 |
| 图片 continuation，在 PendingRead 前完成视觉编码 | `markdown` / `markdown` | 确认调用顺序是根因 |

### 根因与修复

旧 `load-prefix-step` 顺序为：

1. 设置 prefix KV `PendingRead`。
2. 对 `prefix + variable` 做多模态编码。
3. prefill variable。

多模态 tokenization 会实际运行 vision encoder。`Omni` 的 processor/vision runtime 能看到 KVMeta；
因此图片预处理发生在语言模型 prefix cache 的 `PendingRead` 状态下，污染了 image continuation。
纯文本 tokenization 不运行 vision encoder，所以不受影响。

修复后先编码完整多模态 prompt并切出 variable token，同时建立 image embedding、deepstack 和
T/H/W position ids；随后才调用 `loadPromptKVCachePrefixOnly` 安装 `PendingRead`，最后 prefill
variable。图片本身仍不进入 prefix cache。

### 最终一致性与性能

配置：`config.greedy.json`，生成 32 token。

| case | prefix tokens | variable tokens | prefill s | decode s |
| --- | ---: | ---: | ---: | ---: |
| image split-step | 409 | 1288 | 15.57 | 6.29 |
| image load-prefix-step | 409 | 1288 | 12.25 | 6.96 |

清洗后的两份生成结果逐字节一致，SHA-256 均为：

```text
9e18999bb9bdb00d5691b87027a64d2abfb365c427aed68438fe8e89a733e7ad
```

prefill 加速比：`15.57 / 12.25 = 1.27x`；耗时下降：`21.3%`。

最终日志：

- `logs/final_img_split_greedy32.log`
- `logs/final_img_load_greedy32.log`

## A8W8 Lookahead d2 与 Prefix Restore 组合

输入是 lookahead benchmark 的 `prompt_00.txt`，显式加入与 `use_template:true` 等价的模板边界。
模型和 lookahead 参数来自远端 `artifacts/lookahead_bench/README.md`：A8W8 CPU、greedy、d2、
FCFS、预构建 `assistant_min20_top1_n1_4.mnnngram`。运行时成功加载 981,584 条 ngram。

缓存内容为 412-token 模板化纯文本 prefix，共 28 层；continuation 为 1293 token，包含后续文本、
9 条 Action History、图片和 assistant 模板尾部。完整 prompt 为 1705 token，与原 README 的
`llm_demo` prompt_00 日志一致。

### 单次结果

| strategy | prefix | decode tokens | prefill s | decode s | decode tok/s | prefill+decode s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AR | split | 114 | 12.54 | 10.50 | 10.86 | 23.04 |
| AR | restore | 114 | 8.68 | 8.85 | 12.88 | 17.53 |
| Lookahead d2 | split | 112 | 12.77 | 6.33 | 17.69 | 19.10 |
| Lookahead d2 | restore | 112 | 8.71 | 4.73 | 23.68 | 13.44 |

正确性：

- AR split/restore 清洗后输出逐字节一致，SHA-256：
  `b2e8ac41b40b2cadbe8e1839def7b0292aa039600ab96901d4915ca7f96cac30`。
- Lookahead split/restore 清洗后输出逐字节一致，SHA-256：
  `e7b0725643cc97b33b18f251f20527bbb4d5e2b3d4d9ab027613040c6edb4b11`。
- AR 与 lookahead 的输出不同，分别生成 114 和 112 token，因此跨策略 decode 使用 token/s 比较。

单次观察：

- Lookahead 路径的 prefix restore prefill：`12.77 / 8.71 = 1.47x`，下降 31.8%。
- Lookahead split 相对 AR split 的 decode 吞吐：`17.69 / 10.86 = 1.63x`。
- Lookahead restore 相对 AR split 的 `prefill+decode`：`23.04 / 13.44 = 1.71x`。

这是一个 case 的单次顺序运行。restore 本身不应改善 decode，但日志中的后跑任务 decode 更快，说明存在
page cache、CPU 频率或系统负载影响；因此 `1.71x` 只能作为当前 smoke result，正式性能结论需要交错顺序
并重复多轮取中位数。以上总时间不包含约 1.2 秒的 vision encoder 时间和模型加载时间。

日志目录：

```text
/home/ma-user/workspace/csm/mobiinfer/experiments/kv_cache_restore_prefill/lookahead_cpu/logs/
```

## Pruned A8W8 Lookahead D3 + KV Restore，20 Cases

实验使用 `prompt_00` 到 `prompt_19`、vocab-pruned A8W8 模型、greedy、lookahead d3 v3、
FCFS 和 hash lf1 `.mnnngram3` 表。运行时成功 mmap 981,584 条 ngram。20 个 prompt 有两种
公共 prefix，因此分别 dump：

- `pruned_d3_prefix_a`: 413 token，16 个 case。
- `pruned_d3_prefix_b`: 384 token，4 个 case (`08/14/17/19`)。

两份缓存均为 28 层。split 与 restore 各使用 `-P4`，并为每个进程分配独立 tmp 目录。

### 正确性

- 20/20 case 的 split/restore 生成内容逐字节一致。
- 20/20 case 的 restore prefill 都快于 split。
- 没有 `INTERNAL_ERROR`、崩溃或 prefix load 失败。

### 性能

| 指标 | Split | Restore |
| --- | ---: | ---: |
| 平均 prefix token | 407.20 | 407.20 cached |
| 平均 continuation token | 768.05 | 768.05 |
| 平均 decode token | 96.40 | 96.40 |
| Prefill 总时间 | 166.62s | 110.57s |
| 平均 prefill | 8.331s | 5.529s |
| Decode 总时间 | 65.59s | 63.59s |
| 聚合 decode 速度 | 29.39 tok/s | 30.32 tok/s |
| Prefill + decode 总时间 | 232.21s | 174.16s |

聚合 prefill 加速为 `1.507x`，耗时下降 33.64%；逐 case prefill 加速范围
`1.151x - 2.485x`，中位数 `1.603x`。`prefill+decode` 加速为 `1.333x`，总耗时下降 25.0%。
decode 总时间只相差约 3%，说明主要收益来自省略公共 prefix prefill。

### 质量

仓库 `evaluate_outputs.py` 对 restore 日志的结果：

- JSON OK: 100%
- Action match: 80%
- 可比坐标 case: 14
- 平均 coordinate overlap: 0.549

质量结果描述的是 pruned d3 模型本身；由于 split/restore 20/20 输出完全一致，KV restore 没有引入质量变化。

但相对 raw full 路径，两段 prefill 尚未完全等价：

- 后来的有效旧基线 `v3cmp_w8g128_d3_v3_hash_20_eval.json`：JSON 100%、Action 85%、坐标 0.575。
- 当前二进制重新运行 raw full：JSON 100%、Action 85%、坐标 0.575，与有效旧基线一致。
- 当前 split/restore：JSON 100%、Action 80%、坐标 0.549。
- raw full 与 split 全文逐字节一致 16/20，action 一致 19/20。
- case 02 从 raw full 的 `swipe` 变成 split/restore 的 `click`。

因此 5 个百分点的 action 下降不是磁盘 KV restore 额外引入的，但确实来自两段 prefill 相对完整
prefill 的行为差异。这是后续需要修复的正确性问题，不能仅以 split/restore 一致宣称与 raw 等价。

结构化结果：

- `pruned_d3_kv20/results.json`
- `pruned_d3_kv20/results.csv`
- `pruned_d3_kv20/quality_eval.json`
- `pruned_d3_kv20/quality_eval_raw_current.json`
- `pruned_d3_kv20/raw_split_comparison.json`
- `pruned_d3_kv20/summary.txt`
