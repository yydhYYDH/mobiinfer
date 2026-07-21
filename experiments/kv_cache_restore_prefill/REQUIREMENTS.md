# 需求文档

目标：测出加上 KV cache restore 之后，prefill 速度的提升。当前只需要先测一个 case。

## 实验范围

- 先在服务器 `sai-a100` 上用 x86 `build/kvcache_demo` 做快速功能测试。
- 功能测试通过后，再在 `oneplus13` 手机上跑一个正式 case。
- 手机实验使用独立目录，不覆盖已有 bench 根目录二进制和库。

## 对比方法

同一组 `prefix.txt + variable.txt`，对比两条路径：

1. 不使用 restore：
   `kvcache_demo config.json --split-step prefix.txt variable.txt 1`
2. 使用 restore：
   先 `kvcache_demo config.json --dump cache_name prefix.txt`，再
   `kvcache_demo config.json --load-prefix-step cache_name prefix.txt variable.txt 1`

## 指标

核心指标：

- baseline prefill time
- restore prefill time
- speedup
- reduction

辅助指标：

- prefix token 数
- variable token 数
- dump prefix prefill time
- decode time

## 正确性要求

- 使用 greedy 采样。
- `prefix + continuation` 的 token 边界必须与完整 prompt 编码一致。
- 图片 continuation 在 split 和 restore 路径上的 `input_ids`、image embedding、deepstack、
  T/H/W position ids、attention mask 必须一致。
- restore 的 prefix KV 层序和长度必须与 dump 一致。
- 先比较首个生成 token，再比较 32-token 输出；最终输出必须逐字节一致。
- prefix dump 不包含图片。图片及其附加输入在每次 continuation 中重新计算。

## Lookahead 组合要求

- 使用 README 的 A8W8 greedy lookahead d2 参数和预构建 ngram 表。
- 对同一个模板化 prompt_00 比较四种组合：AR split、AR restore、lookahead split、lookahead restore。
- AR 的 split/restore 输出必须逐字节一致。
- lookahead 的 split/restore 输出必须逐字节一致。
- AR 与 lookahead 可能产生不同长度或内容，跨策略性能比较需要同时报告 token/s，不能只比较 decode 总时间。
- `prefill + decode` 不包含 vision encoder 时间；组合端到端结论必须注明该边界。

## Pruned D3 20-Case 要求

- 使用 vocab-pruned A8W8 模型和 `config_cpu_retest_w8g128_vocab_pruned_d3_v3_192_template.json`。
- 使用 greedy、lookahead d3、FCFS 和 hash lf1 `.mnnngram3` 表。
- 识别并分别缓存两种公共 prefix，禁止把 A 类缓存用于 B 类 case。
- 每个 case 使用独立 tmp 目录；split 和 restore 都按 README 的 `-P4` 运行。
- 20 个 case 必须逐一比较 split/restore 输出，不能只比较平均速度。
- 同时比较当前二进制的 raw full 与 split；split/restore 一致不能替代 raw 等价性验证。
- 同时输出逐 case CSV、聚合 JSON 和仓库标准质量评测 JSON。

## 注意事项

- `--dump` 的耗时是生成可复用 KV cache 的一次性成本，不计入单次 restore 推理的 prefill 加速比。
- 手机端正式实验前必须先在服务器上验证 demo 流程可跑通。
- 如果手机端需要部署新二进制，放到 `experiments/kv_cache_restore_prefill/bin`，不要覆盖根目录已有 bench 产物。
