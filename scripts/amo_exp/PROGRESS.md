# HVPO LLM 实验 - 进度与关键决策记录

## 环境修复 (已完成)
- **根因**: 2026-07-08 安装 vllm 0.11 时, 其依赖 `transformers>=4.55.2`(无上限) 把 transformers
  从项目要求的 <5.0 自动升到 5.13.0, 突破了 setup.py/requirements.txt 的 `<5.0` 约束。
- **表现**: transformers 5.x 删除了 `AutoModelForVision2Seq` (verl 用) + 改了 tokenization_utils
  导出 + 新增 `all_tied_weights_keys` (align_anything 奖励模型类缺失) → verl 与 align_anything 均崩。
- **解法 (已执行)**: `pip install transformers==4.57.3` (满足 vllm>=4.55.2 且 <5.0, 项目原始目标版本)。
  连带 huggingface_hub 1.22 → 0.36.2。verl 与 align_anything **均无需改代码**, 之前对 align_anything
  的 25 处 patch 已用 `git checkout align_anything/` 全部回退。
- **保留的唯一运行期设置**: 起奖励服务需 `XFORMERS_IGNORE_FLASH_VERSION_CHECK=1`
  (flash-attn 2.8.3 vs xformers 允许 <=2.8.2, 与 transformers 无关; 官方跳过开关)。

## 已下载资产
- 基座: /data/Qwen/Qwen2.5-{1.5B,3B}-Instruct (完整, 可加载)
- safe 奖励模型: playground/reward_model/checkpoints/Qwen2.5-7B-SafeRLHF-{RM,CM} (各 27G/6shard, 完整)
  - 注: hf CLI 对 hf-mirror metadata 不兼容, 小文件用 curl 补齐 (scripts/amo_exp/curl_hf_download.py)
- UniEval (news 用): ~/.cache/huggingface/hub/models--MingZhong--unieval-{sum,fact,dialog}

## 数据 (已核验)
- data/PKU-SafeRLHF/{train:73907, test:8211}.parquet, extra_info.question ✓ (safe 主实验, 目标强冲突)
- data/CNN_DailyMail/{train:287113, test:11490}.parquet, extra_info.article ✓ (news 对照, 目标不冲突)

## 脚本 (scripts/amo_exp/, 未被 git 跟踪, 已备份 /tmp/amo_exp_backup)
- run_exp.sh <safe|news> <1.5b|3b> <grpo|hvpo> [steps] [overrides] — 锁 GPU2,3, 用 amo env python
- serve_rewards.sh <safe|news|stop> — 奖励服务锁 GPU0,1, 带 xformers flag
- curl_hf_download.py — 绕过 hf CLI 的下载器

## 冒烟测试 (已通过)
safe 1.5b hvpo 2步小 batch: 全链路跑通, 零真错误。日志见 /tmp/amo_reward_logs/smoke_hvpo.log
关键指标正常: [hv_contribution] 35-43, safe_helpfulness/harmless 两目标独立打分, HVPO 优势正常。
时间: ~45s/rollout step (batch=32)。

## 受控对比设计
- GRPO 基线: reward_manager=amo_vanilla (加权和标量) + adv=grpo
- HVPO:      reward_manager=amo_hvpo   (超体积贡献 ΔHV) + adv=hvpo (同 GRPO 形式)
- 唯一变量 = 奖励信号。评测 verl.trainer.amo_eval 内置 HV 计算 (ref=[0,0]) + 各维均值。

## 待办
- [ ] 确定正式 run 规模 (steps/epoch, batch)
- [ ] 跑 safe: 1.5b/3b × grpo/hvpo (4 run)
- [ ] 跑 news: 1.5b/3b × grpo/hvpo (4 run)
- [ ] 评测 + HV 对比表 + Pareto 图 + 分析
