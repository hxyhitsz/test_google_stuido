
import { CategoryType, Question } from './types';

export const INTERVIEW_CONTENT: Question[] = [
  {
    id: "mha-gqa-mqa",
    category: CategoryType.ATTENTION,
    title: "MHA / GQA / MQA Implementation",
    description: "Multi-Head, Grouped-Query, and Multi-Query Attention architectures compared.",
    content: [
      {
        type: 'markdown',
        content: `### 核心逻辑对比
1. **MHA (Multi-Head Attention)**: 每个 Query 都有对应的 Key 和 Value 头。
2. **MQA (Multi-Query Attention)**: 所有 Query 共享一组 Key 和 Value 头 (极大地节省显存，KV Cache 友好)。
3. **GQA (Grouped Query Attention)**: 将 Query 分组，每组共享一组 Key 和 Value。它是 MHA 和 MQA 的折中方案 (LLaMA-2/3 常用)。`
      },
      {
        type: 'code',
        content: `import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralAttention(nn.Module):
    def __init__(self, dim, num_q_heads, num_kv_heads):
        super().__init__()
        self.dim = dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_q_heads
        
        # 组大小 (多少个 Q 共享一个 KV)
        self.group_size = num_q_heads // num_kv_heads
        
        self.q_proj = nn.Linear(dim, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        # [B, L, n_q, d]
        q = self.q_proj(x).view(B, L, self.num_q_heads, self.head_dim)
        # [B, L, n_kv, d]
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim)

        # 核心：将 k, v 扩展到和 q 相同的头数
        k = k.repeat_interleave(self.group_size, dim=2) 
        v = v.repeat_interleave(self.group_size, dim=2) 

        # 计算 Attention
        q = q.transpose(1, 2) # [B, n_q, L, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)`
      }
    ]
  },
  {
    id: "sliding-window-attention",
    category: CategoryType.ATTENTION,
    title: "Sliding Window Attention",
    description: "Mistral-style attention for handling long sequences with fixed complexity.",
    content: [
      {
        type: 'markdown',
        content: `### 滑动窗口注意力 (SWA)
通过只在当前 Token 附近的窗口内计算注意力，将复杂度从 $O(L^2)$ 降低到 $O(L \times W)$。
虽然单层感受野有限，但随着层数加深，高层能够覆盖整个输入序列。`
      },
      {
        type: 'code',
        content: `def sliding_window_mask(seq_len, window_size):
    # 生成因果掩码 (Lower Triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # 结合窗口限制
    # 只允许关注 [i - window_size, i] 范围内的 token
    window_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=-window_size+1)
    return mask * window_mask

# 在 Attention Forward 中使用:
# scores = scores.masked_fill(mask == 0, float("-inf"))`
      }
    ]
  },
  {
    id: "transformer-encoder-decoder",
    category: CategoryType.TRANSFORMER,
    title: "Encoder vs Decoder Architecture",
    description: "Fundamental blocks: Bidirectional Encoder and Autoregressive Decoder.",
    content: [
      {
        type: 'markdown',
        content: `### 核心差异
1. **Encoder**: 允许双向关注 (Full Mask)，用于理解上下文 (如 BERT)。
2. **Decoder**: 使用 Masked Self-Attention (Causal Mask)，保证生成的自回归性 (如 GPT)。
3. **Cross-Attention**: Decoder 关注 Encoder 输出的桥梁。`
      },
      {
        type: 'code',
        content: `class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, cross_kv=None):
        # 1. Self-Attention
        # 如果是 Decoder, 需要传入因果掩码 (is_causal=True)
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_output)
        
        # 2. Cross-Attention (仅在特定 Decoder 结构中)
        if cross_kv is not None:
            # Query 来自 Decoder, Key/Value 来自 Encoder
            ca_output, _ = self.attn(x, cross_kv, cross_kv)
            x = self.ln1(x + ca_output)

        # 3. Feed Forward
        x = self.ln2(x + self.ffn(x))
        return x`
      }
    ]
  },
  {
    id: "ppo-rlhf",
    category: CategoryType.RL_ALGO,
    title: "PPO (Proximal Policy Optimization)",
    description: "The classic RLHF algorithm for alignment with reward models.",
    content: [
      {
        type: 'markdown',
        content: `### PPO 核心公式
PPO 通过裁剪 (Clip) 目标函数来防止策略更新过大，保证训练稳定性。
$L^{CLIP} = \mathbb{E} [ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) ]$`
      },
      {
        type: 'code',
        content: `def ppo_loss(old_logps, new_logps, advantages, epsilon=0.2):
    # 计算概率比率 ratio = pi_new / pi_old
    ratio = torch.exp(new_logps - old_logps)
    
    # 未裁剪的目标
    surr1 = ratio * advantages
    # 裁剪后的目标
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    
    # 取两者最小值，实现保守更新
    loss = -torch.min(surr1, surr2).mean()
    return loss`
      }
    ]
  },
  {
    id: "grpo-deepseek",
    category: CategoryType.RL_ALGO,
    title: "GRPO (Group Relative Policy Optimization)",
    description: "DeepSeek-R1's innovation: Removing the Reward Model via group-based rewards.",
    content: [
      {
        type: 'markdown',
        content: `### GRPO 原理 (DeepSeek-R1)
GRPO 抛弃了传统 PPO 的价值模型 (Critic/Reward Model)，改为对同一提示生成一组样本。
通过组内分数的 **均值** 和 **标准差** 进行归一化，作为相对优势 (Advantage)。
这大大节省了推理显存和训练开销。`
      },
      {
        type: 'code',
        content: `def compute_grpo_advantages(group_rewards):
    # group_rewards: [GroupSize]
    # 对同一 Prompt 生成的多个 Output 的 Reward 进行归一化
    mean = group_rewards.mean()
    std = group_rewards.std() + 1e-8
    
    # 核心：组内相对得分即为优势函数
    advantages = (group_rewards - mean) / std
    return advantages

# GRPO Loss 与 PPO 类似，但 Advantage 是计算出来的，而非模型预测
def grpo_step(policy, ref_model, prompts, beta=0.1):
    # 1. 采样组 (Group Sampling)
    # 2. 计算奖励 (Rule-based or Outcome-based)
    # 3. 计算优势 (Advantages)
    # 4. KL 散度约束 (约束策略不要偏离基座模型太远)`
      }
    ]
  },
  {
    id: "qlora-implementation",
    category: CategoryType.LLM_ENG,
    title: "QLoRA (Quantized LoRA)",
    description: "Pushing LoRA efficiency with 4-bit NormalFloat and double quantization.",
    content: [
      {
        type: 'markdown',
        content: `### QLoRA 核心技术
1. **4-bit NormalFloat (NF4)**: 针对正态分布初始化的权重设计的量化数据类型。
2. **Double Quantization**: 对量化常数本身再进行量化，进一步节省空间。
3. **Paged Optimizers**: 利用 CPU 内存管理显存峰值。`
      },
      {
        type: 'code',
        content: `def mock_nf4_quantization(weight):
    # 模拟 NF4 量化逻辑
    # 1. 获取缩放因子 (AbsMax)
    absmax = weight.abs().max()
    # 2. 映射到 NF4 查找表范围
    # 实际库中使用 bitsandbytes 处理高效位操作
    quantized = torch.round(weight / absmax * 7) # 简化的 3-bit 示意
    return quantized, absmax

# QLoRA 结构
class QLoRALinear(nn.Module):
    def __init__(self, base_layer, r=8):
        super().__init__()
        # base_layer 是量化后的权重 (不可导)
        self.base_layer = base_layer 
        self.base_layer.weight.requires_grad = False
        
        # LoRA 分支 (全精度 FP32 或 BF16)
        self.lora_A = nn.Parameter(torch.zeros(base_layer.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, base_layer.out_features))
        
    def forward(self, x):
        # 1. 反量化前向 (De-quantize on the fly)
        # 实际实现中，FP16 激活与 NF4 权重在 CUDA 核内进行计算
        out = self.base_layer(x)
        # 2. LoRA 修正
        out += (x @ self.lora_A @ self.lora_B)
        return out`
      }
    ]
  },
  {
    id: "rope-implementation",
    category: CategoryType.ATTENTION,
    title: "RoPE (Rotary Position Embedding)",
    description: "The core position encoding for modern LLMs like LLaMA and Mistral.",
    content: [
      {
        type: 'markdown',
        content: `### 旋转位置编码 (RoPE)
RoPE 通过旋转矩阵将位置信息注入。它具有外推性，且在计算注意力时只依赖相对位置。`
      },
      {
        type: 'code',
        content: `def precompute_rope_freqs(dim, max_len, theta=10000.0):
    # 计算旋转角度: theta_i = theta ^ (-2i/d)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    # 外积得到所有位置的角度矩阵 [max_len, dim/2]
    freqs = torch.outer(t, freqs)
    # 转为极坐标形式 (复数)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(x, freqs_complex):
    # x: [B, L, H, D] -> 转换为复数表示
    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    # 广播相乘实现旋转: x * e^(i*m*theta)
    x_rotated = x_complex * freqs_complex
    # 转回实数
    return torch.view_as_real(x_rotated).flatten(-2)`
      }
    ]
  },
  {
    id: "adamw-optimizer",
    category: CategoryType.DL_BASICS,
    title: "AdamW Optimizer",
    description: "Adam with decoupled weight decay, standard for modern LLM training.",
    content: [
      {
        type: 'markdown',
        content: `### AdamW vs Adam
Adam 在计算梯度更新时会将 Weight Decay 混入一阶/二阶动量，这会导致 L2 正则化的效果打折扣。
AdamW 将权重衰减直接应用在参数更新步，实现了真正的解耦。`
      },
      {
        type: 'code',
        content: `def step_adamw(params, grads, exp_avg, exp_avg_sq, lr, beta1, beta2, wd, step):
    # params, grads: list of tensors
    # exp_avg, exp_avg_sq: buffers
    for p, g, m, v in zip(params, grads, exp_avg, exp_avg_sq):
        # 1. 权重衰减 (Decoupled Weight Decay)
        p.data.mul_(1 - lr * wd)
        
        # 2. 更新动量
        m.mul_(beta1).add_(g, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        
        # 3. 偏差修正
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        
        # 4. 参数更新
        p.data.addcdiv_(m_hat, v_hat.sqrt().add(1e-8), value=-lr)`
      }
    ]
  },
  {
    id: "focal-loss",
    category: CategoryType.DL_BASICS,
    title: "Focal Loss",
    description: "Handles class imbalance by down-weighting easy examples.",
    content: [
      {
        type: 'markdown',
        content: `### Focal Loss 公式
$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$
- $\gamma$ (focusing parameter): 增大对难样本的关注。
- $\alpha$ (balancing factor): 调节正负样本比例。`
      },
      {
        type: 'code',
        content: `def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)
    
    # 核心公式: (1-p_t)^gamma * CE
    loss = alpha * (1 - p_t)**gamma * ce_loss
    return loss.mean()`
      }
    ]
  },
  {
    id: "iou-ciou",
    category: CategoryType.CV_CORE,
    title: "IoU & CIoU Implementation",
    description: "Intersection over Union and Complete IoU for Bounding Box regression.",
    content: [
      {
        type: 'markdown',
        content: `### CIoU (Complete IoU)
解决 IoU 缺陷：
1. **Overlap Area**: 基础 IoU。
2. **Central Distance**: 惩罚中心点距离。
3. **Aspect Ratio**: 惩罚长宽比不一致。`
      },
      {
        type: 'code',
        content: `def calculate_ciou(box1, box2):
    # box: [x1, y1, x2, y2]
    # 1. 计算 IoU
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])
    
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    iou = inter_area / union
    
    # 2. 中心点距离
    c1 = torch.tensor([(box1[0]+box1[2])/2, (box1[1]+box1[3])/2])
    c2 = torch.tensor([(box2[0]+box2[2])/2, (box2[1]+box2[3])/2])
    rho2 = torch.sum((c1 - c2)**2)
    
    # 3. 最小闭合框对角线
    outer_x1 = torch.min(box1[0], box2[0])
    outer_y1 = torch.min(box1[1], box2[1])
    outer_x2 = torch.max(box1[2], box2[2])
    outer_y2 = torch.max(box1[3], box2[3])
    diag2 = (outer_x2 - outer_x1)**2 + (outer_y2 - outer_y1)**2
    
    # 4. 长宽比惩罚
    w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
    w2, h2 = box2[2]-box2[0], box2[3]-box2[1]
    import math
    v = (4 / math.pi**2) * (torch.atan(w1/h1) - torch.atan(w2/h2))**2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)
        
    return iou - (rho2 / diag2 + alpha * v)`
      }
    ]
  },
  {
    id: "dpo-loss",
    category: CategoryType.RL_ALGO,
    title: "DPO (Direct Preference Optimization)",
    description: "Fine-tuning LLMs with human preferences without an explicit Reward Model.",
    content: [
      {
        type: 'markdown',
        content: `### DPO Loss 逻辑
DPO 核心公式利用 Bradley-Terry 模型，通过策略模型 $ \pi_\theta $ 和参考模型 $ \pi_{ref} $ 的 Log 概率差来替代 Reward 计算。`
      },
      {
        type: 'code',
        content: `def dpo_loss(policy_chosen_logps, policy_rejected_logps, 
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # 计算当前模型与参考模型在 chosen/rejected 上的偏置
    chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratios = policy_rejected_logps - ref_rejected_logps
    
    # DPO 核心: sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    logits = beta * (chosen_log_ratios - rejected_log_ratios)
    
    # 我们希望 chosen 的概率远大于 rejected
    loss = -F.logsigmoid(logits).mean()
    
    # 额外指标: chosen vs rejected 奖励准确率
    reward_acc = (logits > 0).float().mean()
    return loss, reward_acc`
      }
    ]
  },
  {
    id: "lora-implementation",
    category: CategoryType.LLM_ENG,
    title: "LoRA (Low-Rank Adaptation)",
    description: "Efficiently fine-tuning LLMs by injecting low-rank matrices.",
    content: [
      {
        type: 'markdown',
        content: `### LoRA 原理
$W_{updated} = W + \Delta W = W + B \times A$
- $W$: 冻结的预训练权重。
- $A$: 低秩矩阵 (Gaussian 初始化)。
- $B$: 低秩矩阵 (零初始化)。
- $r$: 秩 (Rank)，通常为 8, 16 等。`
      },
      {
        type: 'code',
        content: `class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features) 
        self.linear.weight.requires_grad = False
        
        self.r = r
        self.scaling = lora_alpha / r
        
        # 低秩适配矩阵
        self.lora_A = nn.Parameter(torch.randn(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        # 初始化: A 为高斯分布, B 为 0 (保证初始输出为 0)
        import math
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_output = self.linear(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_output + lora_output`
      }
    ]
  }
];
