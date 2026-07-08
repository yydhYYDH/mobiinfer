# Vocab Pruning Notes

## 2026-06-24 taobao.md tokenization mismatch

When testing `/home/ma-user/workspace/csm/mobi-autoround/data/prompt_template/longde-no-reasoning/taobao.md` with image `/home/ma-user/workspace/csm/mobi-autoround/data/taobao/taobao_1_4.jpg`, the original tokenizer produced 602 input ids while the first pruned tokenizer produced 607.

The mismatch came from four BPE tokens that had count 0 in `mobimind_e2e_train.jsonl` but appear in this markdown template due to formatting/newlines:

| old token id | token repr | decoded text | pruned fallback |
|---:|---|---|---|
| 515 | `{Ċ` | `{"\\n"}` | `{` + `\\n` |
| 15752 | `)",Ċ` | `)",\\n` | `)",` + `\\n` |
| 1837 | `"ĊĊ` | `"\\n\\n` | `"` + `\\n\\n` |
| 47989 | `)ĊĊĊĊĊĊ` | `)\\n\\n\\n\\n\\n\\n` | `)\\n\\n` + `\\n\\n` + `\\n\\n` |

Action: force keep these token ids in the pruning script so this prompt keeps the same segmentation as the original tokenizer.
