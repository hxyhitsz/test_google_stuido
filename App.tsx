
import React, { useState, useMemo } from 'react';
import { INTERVIEW_CONTENT } from './constants';
import { CategoryType, Question } from './types';

// --- Components ---

const CodeBlock: React.FC<{ code: string }> = ({ code }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group rounded-xl overflow-hidden border border-gray-200 shadow-sm bg-gray-900">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Implementation</span>
        <button onClick={handleCopy} className="text-gray-400 hover:text-white transition-colors">
          {copied ? <i className="fas fa-check text-green-400"></i> : <i className="far fa-copy"></i>}
        </button>
      </div>
      <div className="p-5 overflow-x-auto">
        <pre className="text-sm code-font leading-relaxed text-blue-100">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
};

const Sidebar: React.FC<{ 
  selectedId: string, 
  onSelect: (id: string) => void,
  searchTerm: string,
  onSearch: (s: string) => void 
}> = ({ selectedId, onSelect, searchTerm, onSearch }) => {
  const categories = Object.values(CategoryType);

  return (
    <aside className="w-80 h-full border-r bg-white flex flex-col shrink-0">
      <div className="p-6 border-b">
        <h1 className="text-xl font-bold text-indigo-600 flex items-center gap-2 mb-4">
          <i className="fas fa-microchip"></i> AI Interview Pro
        </h1>
        <div className="relative">
          <input 
            type="text" 
            placeholder="Search knowledge..." 
            className="w-full pl-9 pr-4 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
            value={searchTerm}
            onChange={(e) => onSearch(e.target.value)}
          />
          <i className="fas fa-search absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 text-sm"></i>
        </div>
      </div>
      
      <nav className="flex-1 overflow-y-auto custom-scrollbar p-2">
        {categories.map(cat => {
          const items = INTERVIEW_CONTENT.filter(q => 
            q.category === cat && 
            q.title.toLowerCase().includes(searchTerm.toLowerCase())
          );
          if (items.length === 0) return null;
          return (
            <div key={cat} className="mb-4">
              <h3 className="px-4 py-2 text-[10px] font-bold text-gray-400 uppercase tracking-widest">{cat}</h3>
              <div className="space-y-1">
                {items.map(item => (
                  <button
                    key={item.id}
                    onClick={() => onSelect(item.id)}
                    className={`w-full text-left px-4 py-2.5 text-sm rounded-lg transition-all
                      ${selectedId === item.id 
                        ? 'bg-indigo-50 text-indigo-700 font-semibold shadow-sm' 
                        : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'}`}
                  >
                    {item.title}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </nav>
    </aside>
  );
};

// --- Main App ---

const App: React.FC = () => {
  const [selectedId, setSelectedId] = useState(INTERVIEW_CONTENT[0].id);
  const [searchTerm, setSearchTerm] = useState("");

  const current = useMemo(() => INTERVIEW_CONTENT.find(q => q.id === selectedId), [selectedId]);

  const renderMarkdown = (text: string) => {
    return text
      .replace(/^### (.*$)/gim, '<h3 class="text-xl font-bold text-gray-900 mb-4 mt-6">$1</h3>')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-gray-900 font-bold">$1</strong>')
      .replace(/\$(.*?)\$/g, '<code class="bg-indigo-50 text-indigo-700 px-1 py-0.5 rounded text-[0.9em]">$1</code>')
      .replace(/\n/g, '<br/>');
  };

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      <Sidebar 
        selectedId={selectedId} 
        onSelect={setSelectedId} 
        searchTerm={searchTerm} 
        onSearch={setSearchTerm} 
      />
      
      <main className="flex-1 overflow-y-auto bg-white">
        {current ? (
          <div className="max-w-4xl mx-auto px-8 py-12 lg:px-16 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <header className="mb-12 border-b border-gray-100 pb-8">
              <span className="inline-block px-2.5 py-1 rounded-md bg-indigo-50 text-indigo-600 text-[10px] font-bold uppercase tracking-wider mb-4">
                {current.category}
              </span>
              <h1 className="text-4xl font-extrabold text-gray-900 tracking-tight mb-4">{current.title}</h1>
              <p className="text-lg text-gray-500 leading-relaxed max-w-2xl">{current.description}</p>
            </header>

            <div className="space-y-12">
              {current.content.map((cell, idx) => (
                <div key={idx} className="relative">
                  {cell.type === 'markdown' ? (
                    <div 
                      className="prose prose-slate max-w-none text-gray-700 leading-7"
                      dangerouslySetInnerHTML={{ __html: renderMarkdown(cell.content) }}
                    />
                  ) : (
                    <CodeBlock code={cell.content} />
                  )}
                </div>
              ))}
            </div>

            <footer className="mt-20 pt-10 border-t border-gray-100 flex items-center justify-between text-gray-400 text-xs">
              <p>© 2025 AI Interview Mastery Guide</p>
              <div className="flex gap-4">
                <a href="#" className="hover:text-indigo-600">Documentation</a>
                <a href="#" className="hover:text-indigo-600">GitHub Repository</a>
              </div>
            </footer>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-gray-300">
            <p>Select a topic to start learning</p>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
