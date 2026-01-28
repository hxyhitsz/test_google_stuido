
import { CategoryType, Question } from './types';

export const INTERVIEW_CONTENT: Question[] = [
  {
    id: "transformer-encoder-decoder",
    category: CategoryType.TRANSFORMER,
    title: "Encoder vs Decoder Blocks",
    description: "Fundamental architectural differences between understanding (BERT) and generating (GPT).",
    content: [
      {
        type: 'markdown',
        content: `### 核心设计差异
1. **Encoder (双向)**: 允许当前 Token 看到前后的所有信息。常用于特征提取、分类。
2. **Decoder (自回归)**: 使用 **Causal Mask**，确保计算第 $i$ 个 Token 时只能看到 $1$ 到 $i-1$。
3. **Cross-Attention**: 在 Seq2Seq 结构中，Decoder 通过此模块“查阅” Encoder 的输出。`
      },
      {
        type: 'code',
        content: `import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, cross_kv=None):
        # Self-Attention
        # is_decoder=True 时需传入 attn_mask (通常为下三角矩阵)
        mask = self.get_causal_mask(x.size(1)) if self.is_decoder else None
        out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + out)

        # Cross-Attention (如有)
        if cross_kv is not None:
            out, _ = self.attn(x, cross_kv, cross_kv)
            x = self.ln1(x + out)

        # FFN
        x = self.ln2(x + self.ffn(x))
        return x`
      }
    ]
  },
  {
    id: "ppo-algorithm",
    category: CategoryType.RL_ALGO,
    title: "PPO: Proximal Policy Optimization",
    description: "The gold standard for RLHF alignment using clipping for stability.",
    content: [
      {
        type: 'markdown',
        content: `### PPO 损失函数逻辑
PPO 通过 **Clipping** 限制策略更新幅度。它计算 $r_t(\theta) = \frac{\pi_\theta}{\pi_{old}}$，并取：
$L = \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$`
      },
      {
        type: 'code',
        content: `def ppo_update(old_logps, new_logps, advantages, eps=0.2):
    # ratio = exp(log(new) - log(old))
    ratio = (new_logps - old_logps).exp()
    
    # 未剪裁目标
    surr1 = ratio * advantages
    # 剪裁后的目标 (限制更新在 [1-eps, 1+eps])
    surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
    
    # 取两者最小值以实现保守更新
    loss = -torch.min(surr1, surr2).mean()
    return loss`
      }
    ]
  },
  {
    id: "grpo-deepseek",
    category: CategoryType.RL_ALGO,
    title: "GRPO (DeepSeek-R1)",
    description: "Group Relative Policy Optimization - Scaling RL without a Critic network.",
    content: [
      {
        type: 'markdown',
        content: `### GRPO (Group Relative)
DeepSeek-R1 的核心创新。不需要额外的 Critic 网络，而是对一个 Prompt 生成 $G$ 个回复，通过组内回复的相对 Reward 计算 Advantage。`
      },
      {
        type: 'code',
        content: `def compute_grpo_advantage(rewards):
    # rewards shape: [GroupSize]
    # 对同一 Prompt 的一组采样计算均值和标准差
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    
    # 相对优势 = (当前奖励 - 组内平均) / 标准差
    advantages = (rewards - mean) / std
    return advantages

# 优点：显著降低显存占用，支持超大规模 RL 训练`
      }
    ]
  },
  {
    id: "qlora-implementation",
    category: CategoryType.LLM_ENG,
    title: "QLoRA: 4-bit Quantized LoRA",
    description: "Enabling fine-tuning of 70B models on a single 48GB GPU.",
    content: [
      {
        type: 'markdown',
        content: `### QLoRA 三大支柱
1. **NF4 (NormalFloat 4)**: 针对正态分布权重的最优量化数据类型。
2. **Double Quantization**: 对量化常数再量化，节省约 0.37 bits/param。
3. **Paged Optimizers**: 利用 CPU 内存管理显存尖峰。`
      },
      {
        type: 'code',
        content: `class QLoRALayer(nn.Module):
    def __init__(self, in_f, out_f, r=8):
        super().__init__()
        # 假设 base_weight 已量化为 NF4
        self.register_buffer('base_weight', torch.randn(out_f, in_f, dtype=torch.uint8))
        self.register_buffer('scales', torch.randn(out_f, 1))
        
        # LoRA 适配器 (保持高精度 FP32/BF16)
        self.lora_A = nn.Parameter(torch.zeros(in_f, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f))
        
    def forward(self, x):
        # 1. 在 CUDA 核内进行反量化计算 (W * x)
        # 2. 计算 LoRA 增量 (x * A * B)
        lora_out = (x @ self.lora_A @ self.lora_B)
        # 3. 相加
        return self.dequantize_compute(x) + lora_out`
      }
    ]
  },
  {
    id: "mha-gqa-mqa",
    category: CategoryType.ATTENTION,
    title: "MHA / GQA / MQA",
    description: "Comparing attention heads mechanisms for efficiency.",
    content: [
      {
        type: 'markdown',
        content: `### KV Cache 优化
1. **MHA**: 每个 Q 有独立的 K, V。
2. **MQA**: 所有 Q 共享一对 K, V (节省显存极多，但质量略降)。
3. **GQA**: 每组 Q 共享一对 K, V (折中方案，LLaMA-3 标配)。`
      },
      {
        type: 'code',
        content: `def gqa_logic(q, k, v, num_groups):
    # q: [B, L, n_q, d]
    # k, v: [B, L, n_kv, d]
    group_size = q.size(2) // k.size(2)
    
    # 扩展 k, v 以匹配 q 的头数
    k = k.repeat_interleave(group_size, dim=2)
    v = v.repeat_interleave(group_size, dim=2)
    
    # 之后进行标准 MHA 计算
    return compute_attention(q, k, v)`
      }
    ]
  }
];
