# 数据格式说明

本实验用于测量 KV cache restore 对 prefill 速度的影响。

## 输入文件

实验 prompt 拆成两段：

- `prefix.txt`: 可复用的公共前缀，通常包含 system prompt、Action Space、Response Format、Constraints 等固定内容。
- `variable.txt`: 每个任务变化的部分，通常从 `### Current Task`、Action History、截图占位信息和最终用户请求开始。
- `prefix_img.txt`: 图片 case 的可复用纯文本前缀，不包含 `<img>`。
- `variable_img.txt`: 图片 case 的动态 continuation，包含后续文本和一个真实的
  `<img>image_path<hw>height,width</hw></img>` 标签。

合并语义等价于原始完整 prompt：

```text
<prefix.txt 内容><variable.txt 内容>
```

图片 case 的缓存仍然只包含 `prefix_img.txt` 的文本 KV。加载后重新计算
`variable_img.txt` 的文本 token、image embedding、deepstack 和 T/H/W position ids。

## Lookahead 组合输入

`lookahead_cpu/` 使用 README 中的 A8W8 prompt_00，并显式加入 chat template 边界：

- `prefix_template.txt`: `<|im_start|>user\n` + `prefix_img.txt`。
- `variable_template.txt`: `variable_img.txt` +
  `<|im_end|>\n<|im_start|>assistant\n`。
- `config_ar.json`: A8W8 greedy AR 配置。
- `config_lookahead.json`: A8W8 greedy lookahead d2 配置，使用绝对 `.mnnngram` 路径。

两段合计编码为 1705 token，其中缓存 prefix 412 token，动态 continuation 1293 token；
这与原 `llm_demo + use_template:true` 的 prompt_00 token 数一致。

## Pruned D3 20-Case 输入

`pruned_d3_kv20/` 使用 `prompt_00.txt` 到 `prompt_19.txt`。20 个 case 在
`### Current Task` 前存在两种固定 prefix：

- A 类：16 个 case，模板化后 413 token。
- B 类：`08/14/17/19`，少一个空行和一条 Constraint，模板化后 384 token。

每个 case 的 `runs/NN/prefix.txt` 包含 `<|im_start|>user\n` 和对应固定 prefix；
`runs/NN/variable.txt` 从 `### Current Task` 开始，并以
`<|im_end|>\n<|im_start|>assistant\n` 结束。两类 prefix 分别使用独立的 28 层 KV cache。

## 模型配置

手机端现有 bench 配置：

```text
/data/data/com.termux/files/home/csm/mobiinfer/bench20/configs/config_baseline_w8a8_ar.json
```

该配置指向：

```text
/data/data/com.termux/files/home/csm/mobiinfer/models/baseline_w8a8/
```

服务器端快速测试使用：

```text
/home/ma-user/workspace/csm/mobiinfer/transformers/llm/export/model/config.json
```

## 输出日志

`kvcache_demo` 关键输出字段：

- `split prefix tokens num`: 未 restore 路径中 prefix token 数。
- `cached prefix tokens num`: restore 路径中从磁盘恢复的 prefix token 数。
- `variable tokens num`: variable token 数。
- `prefill time`: 当前运行记录的 prefill 总耗时，单位秒。
- `decode time`: decode 耗时，单位秒。

本实验比较：

- baseline: `--split-step prefix.txt variable.txt 1`
- restore: 先 `--dump <cache_name> prefix.txt`，再 `--load-prefix-step <cache_name> prefix.txt variable.txt 1`

一致性比较会先从日志中去掉设备信息、`Loaded prefix` 诊断行和统计区，再对模型输出执行
`cmp` 与 `sha256sum`。使用 greedy 时，split 与 restore 必须逐字节一致。

速度提升计算：

```text
speedup = baseline_prefill_time_s / restore_prefill_time_s
reduction = 1 - restore_prefill_time_s / baseline_prefill_time_s
```
