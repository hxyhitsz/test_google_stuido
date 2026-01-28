
import React, { useState, useMemo, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";

// --- Import Constants ---
import { INTERVIEW_CONTENT } from './constants';

const CodeBlock = ({ code }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="rounded-xl overflow-hidden border border-gray-700 bg-[#0d1117] shadow-xl my-6">
      <div className="flex items-center justify-between px-4 py-2 bg-[#161b22] border-b border-gray-800">
        <div className="flex gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
          <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[10px] font-mono text-gray-500 uppercase tracking-widest">Python 3.10+</span>
          <button onClick={handleCopy} className="text-gray-400 hover:text-indigo-400 transition-colors">
            {copied ? <i className="fas fa-check text-green-400"></i> : <i className="far fa-copy"></i>}
          </button>
        </div>
      </div>
      <div className="p-5 overflow-x-auto custom-scrollbar">
        <pre className="text-sm font-mono leading-relaxed text-indigo-100/90">
          <code className="whitespace-pre">{code}</code>
        </pre>
      </div>
    </div>
  );
};

const App = () => {
  const [selectedId, setSelectedId] = useState(INTERVIEW_CONTENT[0].id);
  const [searchTerm, setSearchTerm] = useState("");
  const [aiAnalysis, setAiAnalysis] = useState(null);
  const [isAiLoading, setIsAiLoading] = useState(false);

  const current = useMemo(() => INTERVIEW_CONTENT.find(q => q.id === selectedId), [selectedId]);
  
  const categories = useMemo(() => [...new Set(INTERVIEW_CONTENT.map(item => item.category))], []);

  const filteredItems = useMemo(() => 
    INTERVIEW_CONTENT.filter(q => 
      q.title.toLowerCase().includes(searchTerm.toLowerCase()) || 
      q.category.toLowerCase().includes(searchTerm.toLowerCase())
    )
  , [searchTerm]);

  const handleConsultAI = async () => {
    if (!current) return;
    setIsAiLoading(true);
    setAiAnalysis(null);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: 'gemini-3-pro-preview',
        contents: `You are a Senior Machine Learning Interviewer at Google/OpenAI. Analyze the following concept: "${current.title}". 
        1. Explain the deep technical intuition.
        2. Mention 2-3 "Trap Questions" or common candidate mistakes when explaining this.
        3. Discuss production trade-offs (e.g., memory vs speed).
        Format the output clearly using Markdown.`,
      });
      setAiAnalysis(response.text);
    } catch (err) {
      setAiAnalysis("Analysis failed. Please try again later.");
    } finally {
      setIsAiLoading(false);
    }
  };

  const formatMD = (text) => {
    return text
      .replace(/^### (.*$)/gim, '<h3 class="text-xl font-bold text-gray-900 mb-4 mt-8 border-l-4 border-indigo-500 pl-3">$1</h3>')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-indigo-900 font-bold">$1</strong>')
      .replace(/\$(.*?)\$/g, '<code class="bg-indigo-50 text-indigo-700 px-1.5 py-0.5 rounded font-mono text-sm">$1</code>')
      .replace(/\n/g, '<br/>');
  };

  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <aside className="w-80 border-r flex flex-col shrink-0 bg-gray-50/80 backdrop-blur">
        <div className="p-6 border-b bg-white/50">
          <h1 className="text-xl font-black text-indigo-600 flex items-center gap-3 mb-5 italic">
            <i className="fas fa-microchip"></i> AI INTERVIEW PRO
          </h1>
          <div className="relative group">
            <input 
              type="text" 
              placeholder="Search concepts..." 
              className="w-full pl-10 pr-4 py-2.5 text-sm border-0 ring-1 ring-gray-200 rounded-xl focus:ring-2 focus:ring-indigo-500 transition-all bg-white"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <i className="fas fa-search absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400 group-focus-within:text-indigo-500 transition-colors"></i>
          </div>
        </div>
        
        <nav className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-6">
          {categories.map(cat => {
            const items = filteredItems.filter(i => i.category === cat);
            if (items.length === 0) return null;
            return (
              <div key={cat}>
                <h3 className="px-2 py-1 text-[11px] font-bold text-indigo-400 uppercase tracking-[0.2em] mb-2">{cat}</h3>
                <div className="space-y-1">
                  {items.map(item => (
                    <button
                      key={item.id}
                      onClick={() => {
                        setSelectedId(item.id);
                        setAiAnalysis(null);
                      }}
                      className={`w-full text-left px-3 py-2.5 text-sm rounded-lg transition-all duration-200 flex items-center gap-2
                        ${selectedId === item.id 
                          ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-100 translate-x-1' 
                          : 'text-gray-600 hover:bg-white hover:shadow-sm'}`}
                    >
                      <i className={`fas fa-chevron-right text-[10px] ${selectedId === item.id ? 'opacity-100' : 'opacity-0'}`}></i>
                      <span className="truncate font-medium">{item.title}</span>
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </nav>
      </aside>

      {/* Content Area */}
      <main className="flex-1 overflow-y-auto bg-white relative">
        {current ? (
          <div className="max-w-4xl mx-auto px-12 py-16 animate-in fade-in duration-500">
            <header className="mb-12">
              <div className="flex items-center gap-2 text-indigo-500 font-bold text-xs uppercase tracking-widest mb-4">
                <span className="w-6 h-[2px] bg-indigo-500"></span>
                {current.category}
              </div>
              <h2 className="text-4xl font-black text-gray-900 mb-6 tracking-tight leading-tight">{current.title}</h2>
              <p className="text-lg text-gray-500 leading-relaxed font-medium">{current.description}</p>
            </header>

            <div className="mb-12 flex gap-4">
              <button 
                onClick={handleConsultAI}
                disabled={isAiLoading}
                className="px-6 py-3 bg-gray-900 text-white rounded-xl font-bold flex items-center gap-3 hover:bg-black transition-all shadow-xl hover:-translate-y-1 active:translate-y-0 disabled:opacity-50 disabled:translate-y-0"
              >
                {isAiLoading ? <i className="fas fa-spinner fa-spin"></i> : <i className="fas fa-brain-circuit"></i>}
                {isAiLoading ? "Analyzing..." : "Expert Deep Dive"}
              </button>
            </div>

            {aiAnalysis && (
              <div className="mb-12 p-8 bg-indigo-50/50 rounded-2xl border border-indigo-100 shadow-sm animate-in zoom-in-95 duration-300">
                <div className="flex items-center gap-2 text-indigo-600 font-black text-xs uppercase tracking-widest mb-6 border-b border-indigo-100 pb-4">
                  <i className="fas fa-user-tie text-lg"></i>
                  Gemini Interviewer Insight
                </div>
                <div className="prose prose-indigo max-w-none text-gray-800 whitespace-pre-wrap leading-relaxed">
                  {aiAnalysis}
                </div>
              </div>
            )}
            
            <div className="space-y-12">
              {current.content.map((block, i) => (
                <div key={i} className="content-fade">
                  {block.type === 'markdown' ? (
                    <div 
                      className="text-gray-700 text-lg leading-relaxed"
                      dangerouslySetInnerHTML={{ __html: formatMD(block.content) }} 
                    />
                  ) : (
                    <CodeBlock code={block.content} />
                  )}
                </div>
              ))}
            </div>

            <footer className="mt-32 pt-12 border-t border-gray-100 flex justify-between items-center text-xs font-bold text-gray-400 uppercase tracking-widest">
              <span>© 2025 AI Engineer Prep • Pro Series</span>
              <div className="flex gap-8">
                <a href="#" className="hover:text-indigo-600">Github Repo</a>
                <a href="#" className="hover:text-indigo-600">Roadmap</a>
              </div>
            </footer>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-gray-300 gap-6 opacity-40">
            <i className="fas fa-layer-group text-8xl"></i>
            <p className="text-2xl font-black uppercase tracking-widest">Select Concept to Begin</p>
          </div>
        )}
      </main>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
