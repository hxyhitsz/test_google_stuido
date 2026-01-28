
import React, { useState, useMemo, useEffect } from 'react';
import ReactDOM from 'react-dom/client';

// --- Types ---
enum CategoryType {
  ATTENTION = "Attention & Encoding",
  TRANSFORMER = "Transformer Arch",
  RL_ALGO = "RL & Alignment",
  LLM_ENG = "LLM Engineering",
  DL_BASICS = "DL Basics",
  CV_CORE = "CV Core"
}

interface Question {
  id: string;
  title: string;
  category: CategoryType;
  description: string;
  content: { type: 'markdown' | 'code'; content: string }[];
}

// --- Constants ---
const INTERVIEW_CONTENT: Question[] = [
  {
    id: "transformer-encoder-decoder",
    category: CategoryType.TRANSFORMER,
    title: "Encoder vs Decoder Architecture",
    description: "Fundamental blocks of Transformer: Bidirectional vs Autoregressive.",
    content: [
      { type: 'markdown', content: "### 核心设计差异\n1. **Encoder**: 双向关注，适合理解上下文。\n2. **Decoder**: 单向掩码(Causal Mask)，适合自回归生成。" },
      { type: 'code', content: "class TransformerBlock(nn.Module):\n    def __init__(self, dim, heads, is_decoder=False):\n        super().__init__()\n        self.is_decoder = is_decoder\n        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)\n        # ... 实现代码" }
    ]
  },
  {
    id: "grpo-deepseek",
    category: CategoryType.RL_ALGO,
    title: "GRPO (Group Relative Policy Optimization)",
    description: "DeepSeek-R1's innovation for reinforcement learning scaling.",
    content: [
      { type: 'markdown', content: "### GRPO 原理\n通过组内采样奖励的均值和标准差来计算 Advantage，无需 Critic 网络。" },
      { type: 'code', content: "def compute_grpo_advantage(rewards):\n    mean = rewards.mean()\n    std = rewards.std() + 1e-8\n    return (rewards - mean) / std" }
    ]
  },
  {
    id: "ppo-algorithm",
    category: CategoryType.RL_ALGO,
    title: "PPO: Clipping for Stability",
    description: "Policy optimization with ratio clipping to prevent drastic changes.",
    content: [
      { type: 'markdown', content: "### PPO 剪切机制\n$L = \\min(r_t A_t, \\text{clip}(r_t, 1-\\epsilon, 1+\\epsilon) A_t)$" },
      { type: 'code', content: "def ppo_loss(ratio, adv, eps=0.2):\n    surr1 = ratio * adv\n    surr2 = torch.clamp(ratio, 1-eps, 1+eps) * adv\n    return -torch.min(surr1, surr2).mean()" }
    ]
  },
  {
    id: "qlora-nf4",
    category: CategoryType.LLM_ENG,
    title: "QLoRA: NF4 & Double Quant",
    description: "Efficient LLM fine-tuning via 4-bit quantization and LoRA adapters.",
    content: [
      { type: 'markdown', content: "### QLoRA 技术要点\n使用 NF4 量化权重，并在反向传播时实时反量化以保持精度。" },
      { type: 'code', content: "class QLoRALayer(nn.Module):\n    def forward(self, x):\n        base_out = quant_matmul(x, self.nf4_weight)\n        adapter_out = x @ self.lora_A @ self.lora_B\n        return base_out + adapter_out" }
    ]
  },
  {
    id: "mha-gqa-mqa",
    category: CategoryType.ATTENTION,
    title: "MHA / GQA / MQA",
    description: "Head grouping techniques for KV-cache memory optimization.",
    content: [
      { type: 'markdown', content: "### KV Cache 演进\n- **MHA**: 每个 Q 独立 KV\n- **MQA**: 所有 Q 共享一对 KV\n- **GQA**: 每组 Q 共享一对 KV" },
      { type: 'code', content: "def gqa_logic(q, k, v, num_groups):\n    k = k.repeat_interleave(q.size(2)//k.size(2), dim=2)\n    v = v.repeat_interleave(q.size(2)//v.size(2), dim=2)\n    return standard_attn(q, k, v)" }
    ]
  }
];

// --- Internal Components ---
const CodeBlock: React.FC<{ code: string }> = ({ code }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="rounded-lg overflow-hidden border border-gray-200 bg-gray-900 shadow-sm my-4">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Implementation</span>
        <button onClick={handleCopy} className="text-gray-400 hover:text-white">
          {copied ? <i className="fas fa-check text-green-400"></i> : <i className="far fa-copy"></i>}
        </button>
      </div>
      <div className="p-4 overflow-x-auto">
        <pre className="text-sm code-font leading-relaxed text-indigo-100"><code>{code}</code></pre>
      </div>
    </div>
  );
};

// --- App Root ---
const App: React.FC = () => {
  const [selectedId, setSelectedId] = useState(INTERVIEW_CONTENT[0].id);
  const [searchTerm, setSearchTerm] = useState("");
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => { setIsLoaded(true); }, []);

  const filteredItems = useMemo(() => 
    INTERVIEW_CONTENT.filter(q => 
      q.title.toLowerCase().includes(searchTerm.toLowerCase()) || 
      q.category.toLowerCase().includes(searchTerm.toLowerCase())
    )
  , [searchTerm]);

  const current = useMemo(() => INTERVIEW_CONTENT.find(q => q.id === selectedId), [selectedId]);

  const categories = Object.values(CategoryType);

  const formatText = (text: string) => {
    return text
      .replace(/^### (.*$)/gim, '<h3 class="text-lg font-bold text-gray-900 mb-2 mt-4">$1</h3>')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-gray-900 font-bold">$1</strong>')
      .replace(/\$(.*?)\$/g, '<code class="bg-indigo-50 text-indigo-700 px-1 rounded">$1</code>')
      .replace(/\n/g, '<br/>');
  };

  if (!isLoaded) return null;

  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <aside className="w-72 border-r flex flex-col shrink-0 bg-gray-50/50">
        <div className="p-6 border-b bg-white">
          <h1 className="text-lg font-bold text-indigo-600 flex items-center gap-2 mb-4">
            <i className="fas fa-microchip"></i> AI Interview
          </h1>
          <input 
            type="text" 
            placeholder="Search topics..." 
            className="w-full px-3 py-1.5 text-sm border rounded-lg focus:ring-1 focus:ring-indigo-400 outline-none"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <nav className="flex-1 overflow-y-auto custom-scrollbar p-3">
          {categories.map(cat => {
            const items = filteredItems.filter(i => i.category === cat);
            if (items.length === 0) return null;
            return (
              <div key={cat} className="mb-4">
                <h3 className="px-3 py-1 text-[10px] font-bold text-gray-400 uppercase tracking-widest">{cat}</h3>
                {items.map(item => (
                  <button
                    key={item.id}
                    onClick={() => setSelectedId(item.id)}
                    className={`w-full text-left px-3 py-2 text-xs rounded-md transition-all mb-0.5
                      ${selectedId === item.id ? 'bg-indigo-600 text-white shadow-md' : 'text-gray-600 hover:bg-gray-100'}`}
                  >
                    {item.title}
                  </button>
                ))}
              </div>
            );
          })}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-10 bg-white">
        {current ? (
          <div className="max-w-3xl mx-auto opacity-100 transition-opacity duration-300">
            <header className="mb-8 pb-6 border-b">
              <span className="text-[10px] font-bold text-indigo-500 uppercase tracking-widest block mb-2">{current.category}</span>
              <h2 className="text-3xl font-extrabold text-gray-900 mb-3">{current.title}</h2>
              <p className="text-gray-500 text-sm">{current.description}</p>
            </header>
            
            <div className="space-y-8">
              {current.content.map((block, i) => (
                <div key={i}>
                  {block.type === 'markdown' ? (
                    <div 
                      className="text-sm text-gray-700 leading-relaxed"
                      dangerouslySetInnerHTML={{ __html: formatText(block.content) }} 
                    />
                  ) : (
                    <CodeBlock code={block.content} />
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-gray-300">Select a topic</div>
        )}
      </main>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);
