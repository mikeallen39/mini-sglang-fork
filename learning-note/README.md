# Mini-SGLang Learning Notes

这套笔记不是项目文档的重复整理，而是给你自己用来真正吃透代码的学习工作区。

建议使用方式：

1. 先读 [00-study-method.md](./00-study-method.md)，明确怎么看代码、怎么记笔记。
2. 再按 [01-14day-plan.md](./01-14day-plan.md) 的顺序推进，不要跳着看。
3. 每读完一个模块，就复制 `templates/` 下的模板新建一份自己的笔记。
4. 每做一次实验，就单独记录，不要把实验观察混在“模块说明”里。

推荐产出物：

- `notes/系统总览.md`
- `notes/请求主路径.md`
- `notes/core-req-batch-context.md`
- `notes/scheduler-main-loop.md`
- `notes/engine-init-and-forward.md`
- `notes/kvcache.md`
- `notes/attention-backends.md`
- `notes/model-loading.md`
- `notes/distributed.md`
- `notes/适配改动复盘.md`

你真正要逼自己写下来的，不是“这个文件做了什么”，而是：

- 这个模块的输入/输出是什么
- 它维护了什么状态
- 谁创建这些状态，谁销毁这些状态
- 它依赖哪些不变量
- 如果它坏了，系统会出现什么具体症状

如果你能把这些写清楚，这个项目就开始变成你的知识，而不是你暂时看过的代码。
