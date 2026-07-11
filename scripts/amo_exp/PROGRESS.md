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

## 稳定训练配置 (2026-07 调通)
- GPU: 训练与奖励模型同卡 GPU 0,1 (TRAIN_GPUS=0,1, run_exp.sh 默认)
- 关键内存安全设置 (避免 OOM, 与 vLLM 共存):
  - actor_rollout_ref.rollout.gpu_memory_utilization=0.35
  - ppo/log_prob/ref micro_batch_size_per_gpu=4
  - 不要用 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (与 vLLM mem pool 冲突)
- 快速验证: data.train_batch_size=64, ppo_mini_batch_size=32, 50 步 ~= 1.5-2h
  - 单步时间随 batch 缩放; 跑满 1 epoch 与 batch 无关(总量固定, ~40h@2卡)
- trainer.val_before_train=False (跳过昂贵的 8211 全量预验证)
- data.val_files=test_small.parquet (200 子集, 周期验证快)
- SAVE_FREQ=25 -> checkpoint 存 checkpoints/amo_pku-saferlhf/qwen2.5-1.5b_grpo/global_step_N
  - resume_mode=auto 可自动续训到完整 epoch (checkpoint 可复用)
- 公平对比: GRPO 与 HVPO 用完全相同的步数/batch, 仅 adv_estimator + reward_manager 不同

## GRPO run (进行中)
log: /tmp/amo_reward_logs/run_safe_1.5b_grpo.log

## GRPO run 完成 (2026-07-10)
- 50 步, batch 64, GPU 0,1。checkpoint: global_step_25, global_step_50 (可复用续训)
- critic/score/mean: 3.0 (start) -> 4.79 (step50), +60% (GRPO 优化加权和奖励, 符合预期)
- 用时约 1h15m (~87s/step compute + reward scoring overhead)
- log: /tmp/amo_reward_logs/run_safe_1.5b_grpo.log

## HVPO run 启动 (相同配置, 仅 adv_estimator=hvpo + reward_manager=amo_hvpo)
- log: /tmp/amo_reward_logs/run_safe_1.5b_hvpo.log

## 评测流水线 (scripts/amo_exp/eval_pipeline.sh)
- merge LoRA -> generate(GPU2,3) -> amo_eval(HV, ref=origin) + 各维均值
- GRPO step50 已 merge 成功: checkpoints/.../qwen2.5-1.5b_grpo/global_step_50/actor/merge
- 注意: 不要在训练(vLLM占卡)时跑 generation(另一个vLLM), 会 shm 冲突挂起!
  评测必须等 HVPO 训练完成、释放 GPU 后再做。
- 注意: safe 奖励分不在[0,1](约-3~+6), amo_eval 用 ref=origin 算 HV 会 clamp 负值;
  分析时需对两方法用相同参考点(取全体最小值以下)重算 HV 才公平。
