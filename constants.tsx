
import { Question } from './types';

export const INTERVIEW_CONTENT: Question[] = [
  // --- 一、大模型核心注意力 / 编码模块 ---
  {
    id: "gqa-mqa-mha",
    category: "Attention & Encoding",
    title: "MHA / GQA / MQA Implementation",
    description: "KV Cache 显存优化技术演进：从多头到分组头再到单头。",
    content: [
      { type: 'markdown', content: "### 核心对比\n- **MHA**: $H_q = H_k = H_v$，每个 Query 都有自己的 KV。\n- **GQA**: $H_q = n \times H_k$，每组 Query 共享一对 KV（Llama-3 标配）。\n- **MQA**: $H_k = H_v = 1$，所有 Query 共享一对 KV。" },
      { type: 'code', content: "def grouped_query_attention(q, k, v, num_groups):\n    # q: [B, H_q, L, D], k/v: [B, H_kv, L, D]\n    # H_q 必须能被 H_kv 整除，num_groups = H_q // H_kv\n    B, H_q, L, D = q.shape\n    H_kv = k.shape[1]\n    \n    # 1. 扩展 k, v 匹配 q 的头数\n    # repeat_interleave 会将 [B, 1, L, D] 变为 [B, G, L, D]\n    k = k.repeat_interleave(H_q // H_kv, dim=1)\n    v = v.repeat_interleave(H_q // H_kv, dim=1)\n    \n    # 2. 标准 Scaled Dot-Product Attention\n    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)\n    attn = torch.softmax(scores, dim=-1)\n    return torch.matmul(attn, v)" }
    ]
  },
  {
    id: "rope-embedding",
    category: "Attention & Encoding",
    title: "RoPE (Rotary Positional Embedding)",
    description: "旋转位置编码：通过复数空间旋转实现相对位置敏感。",
    content: [
      { type: 'markdown', content: "### 为什么用 RoPE?\n1. **相对位置敏感**: 两个 Token 的内积只取决于它们的相对距离。\n2. **外推性**: 理论上支持比训练长度更长的序列（配合插值）。" },
      { type: 'code', content: "def apply_rope(q, k, cos, sin):\n    # q, k: [B, L, H, D]\n    # cos, sin: [L, D/2]\n    def rotate_half(x):\n        # 将最后维度 D 拆分，前一半和后一半交换并取负\n        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]\n        return torch.cat((-x2, x1), dim=-1)\n\n    # 核心公式: x * cos + rotate_half(x) * sin\n    q_embed = (q * cos) + (rotate_half(q) * sin)\n    k_embed = (k * cos) + (rotate_half(k) * sin)\n    return q_embed, k_embed" }
    ]
  },

  // --- 二、深度学习基础：优化器 / 损失函数 ---
  {
    id: "adamw-optimizer",
    category: "DL foundations",
    title: "AdamW: Decoupled Weight Decay",
    description: "解耦权重衰减：修正了 Adam 在 L2 正则化时的权重更新逻辑。",
    content: [
      { type: 'markdown', content: "### Adam vs AdamW\nAdam 在计算梯度时加入 L2 项，导致惩罚项被梯度自适应调整。**AdamW** 直接在权重更新步减去 $\lambda \theta$，解耦了衰减与梯度比例。" },
      { type: 'code', content: "def adamw_step(w, grad, m, v, lr, wd, beta1=0.9, beta2=0.999, eps=1e-8):\n    # 1. 解耦权重衰减 (Decoupled Weight Decay)\n    w = w - lr * wd * w\n    \n    # 2. 更新一阶/二阶动量\n    m = beta1 * m + (1 - beta1) * grad\n    v = beta2 * v + (1 - beta2) * (grad ** 2)\n    \n    # 3. 偏差修正\n    m_hat = m / (1 - beta1)\n    v_hat = v / (1 - beta2)\n    \n    # 4. 参数更新\n    return w - lr * m_hat / (torch.sqrt(v_hat) + eps)" }
    ]
  },
  {
    id: "focal-loss",
    category: "DL foundations",
    title: "Focal Loss for Imbalance",
    description: "解决样本不平衡：让模型更关注难分类的样本。",
    content: [
      { type: 'markdown', content: "### 公式核心\n$FL(p_t) = -(1-p_t)^\gamma \log(p_t)$\n- 当 $p_t$ 接近 1 (易分样本)，权重因子 $(1-p_t)^\gamma$ 趋近 0，降低损失贡献。" },
      { type: 'code', content: "def focal_loss(preds, targets, alpha=0.25, gamma=2.0):\n    # preds: sigmoid 后的概率\n    # targets: 0 或 1\n    ce_loss = F.binary_cross_entropy(preds, targets, reduction='none')\n    p_t = preds * targets + (1 - preds) * (1 - targets)\n    \n    # 难易样本加权: (1 - p_t)^gamma\n    loss = alpha * ((1 - p_t) ** gamma) * ce_loss\n    return loss.mean()" }
    ]
  },

  // --- 三、计算机视觉核心 ---
  {
    id: "ciou-implementation",
    category: "Computer Vision",
    title: "CIoU (Complete IoU)",
    description: "最全的框回归损失：考虑了重叠面积、中心点距离和长宽比。",
    content: [
      { type: 'markdown', content: "### CIoU 三要素\n1. **IoU**: 重叠区域。\n2. **Distance**: 中心点欧式距离比外接框对角线。\n3. **Aspect Ratio**: 形状相似度惩罚 $\nu$。" },
      { type: 'code', content: "def compute_ciou(box1, box2):\n    # box: [x1, y1, x2, y2]\n    inter, union, iou = get_iou(box1, box2)\n    \n    # 1. 中心距离惩罚项\n    d2 = dist(center(box1), center(box2))**2\n    c2 = diag(outer_box(box1, box2))**2\n    distance_term = d2 / c2\n    \n    # 2. 长宽比惩罚项\n    v = (4 / pi**2) * (atan(w1/h1) - atan(w2/h2))**2\n    alpha = v / ((1 - iou) + v + 1e-6)\n    \n    return iou - (distance_term + alpha * v)" }
    ]
  },
  {
    id: "soft-nms",
    category: "Computer Vision",
    title: "Soft NMS",
    description: "非极大值抑制改进：对重叠框不直接删除，而是降低其置信度。",
    content: [
      { type: 'markdown', content: "### 核心价值\n在密集场景下，防止将真实的遮挡目标由于 IoU 过大而误删。" },
      { type: 'code', content: "def soft_nms(boxes, scores, iou_thresh=0.5, sigma=0.5):\n    # boxes: [N, 4], scores: [N]\n    indices = scores.sort(descending=True).indices\n    for i in range(len(indices)):\n        idx = indices[i]\n        for j in range(i + 1, len(indices)):\n            jdx = indices[j]\n            iou = compute_iou(boxes[idx], boxes[jdx])\n            # 高斯加权衰减: score = score * exp(-iou^2 / sigma)\n            if iou > iou_thresh:\n                scores[jdx] *= math.exp(-(iou**2) / sigma)\n    return boxes[scores > 0.01]" }
    ]
  },

  // --- 四、强化学习算法 ---
  {
    id: "dpo-loss",
    category: "RL & Alignment",
    title: "DPO (Direct Preference Optimization)",
    description: "直接偏好优化：绕过奖励模型，直接在偏好数据上微调策略。",
    content: [
      { type: 'markdown', content: "### DPO 核心思想\n利用 Bradley-Terry 模型，将 RLHF 的 KL 散度约束下的奖励最大化问题转化为一个简单的二分类交叉熵损失。" },
      { type: 'code', content: "def dpo_loss(policy_logps, ref_logps, beta=0.1):\n    # policy_logps: [chosen, rejected] 当前模型的 log 概率\n    # ref_logps: [chosen, rejected] 参考模型的 log 概率\n    \n    # 计算当前模型与参考模型的 log 概率差 (Implicit Reward)\n    pi_logratios = policy_logps_chosen - policy_logps_rejected\n    ref_logratios = ref_logps_chosen - ref_logps_rejected\n    \n    logits = pi_logratios - ref_logratios\n    # 核心公式: -log(sigmoid(beta * (logits)))\n    return -F.logsigmoid(beta * logits).mean()" }
    ]
  },

  // --- 六、大模型长文本 / 高效训练 ---
  {
    id: "lora-module",
    category: "LLM Engineering",
    title: "LoRA (Low-Rank Adaptation)",
    description: "低秩适配：保持原权重冻结，通过旁路矩阵 A 和 B 进行高效训练。",
    content: [
      { type: 'markdown', content: "### 参数计算\n训练参数量为 $2 \times r \times d$，其中 $r \ll d$。推理时可将 $W + AB$ 合并，实现零额外延迟。" },
      { type: 'code', content: "class LoRALinear(nn.Module):\n    def __init__(self, in_dim, out_dim, r=8, lora_alpha=16):\n        self.base_layer = nn.Linear(in_dim, out_dim) # 冻结\n        self.lora_A = nn.Parameter(torch.randn(in_dim, r)) # 降维\n        self.lora_B = nn.Parameter(torch.zeros(r, out_dim)) # 升维，初始化为 0\n        self.scaling = lora_alpha / r\n\n    def forward(self, x):\n        # 结果 = 基础输出 + (x @ A @ B) * 缩放因子\n        return self.base_layer(x) + (x @ self.lora_A @ self.lora_B) * self.scaling" }
    ]
  },
  {
    id: "top-p-sampling",
    category: "LLM Engineering",
    title: "Top-P (Nucleus) Sampling",
    description: "核采样：选择累积概率达到阈值 P 的最小 Token 集合。",
    content: [
      { type: 'markdown', content: "### 相比 Top-K 的优势\nTop-K 截断点固定，而 Top-P 的截断窗口随概率分布的平坦程度动态变化，生成更灵活。" },
      { type: 'code', content: "def top_p_sampling(logits, p=0.9):\n    sorted_logits, indices = torch.sort(logits, descending=True)\n    probs = torch.softmax(sorted_logits, dim=-1)\n    \n    # 计算累积概率\n    cum_probs = torch.cumsum(probs, dim=-1)\n    \n    # 移除累积概率超过 p 的 token (保留第一个超过 p 的)\n    mask = cum_probs > p\n    mask[..., 1:] = mask[..., :-1].clone()\n    mask[..., 0] = 0\n    \n    sorted_logits[mask] = -float('Inf')\n    return torch.softmax(sorted_logits, dim=-1)" }
    ]
  }
];
