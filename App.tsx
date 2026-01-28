import React, { useState, useMemo } from 'react';
import Sidebar from './components/Sidebar';
import CodeBlock from './components/CodeBlock';
import { INTERVIEW_CONTENT } from './constants';
import { GoogleGenAI } from "@google/genai";

// Main Application component for the AI Interview Pro platform
const App: React.FC = () => {
  const [selectedId, setSelectedId] = useState<string>(INTERVIEW_CONTENT[0].id);
  const [searchTerm, setSearchTerm] = useState("");
  const [aiResponse, setAiResponse] = useState<string | null>(null);
  const [isAiLoading, setIsAiLoading] = useState(false);

  const current = useMemo(() => 
    INTERVIEW_CONTENT.find(q => q.id === selectedId)
  , [selectedId]);
  
  const categories = useMemo(() => 
    [...new Set(INTERVIEW_CONTENT.map(item => item.category))]
  , []);

  // Use Gemini API to provide expert-level interview analysis for the selected topic
  const handleAskAI = async () => {
    if (!current) return;
    setIsAiLoading(true);
    setAiResponse(null);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      // Using gemini-3-pro-preview for high-quality technical reasoning and interview prep
      const response = await ai.models.generateContent({
        model: 'gemini-3-pro-preview',
        contents: `You are an expert interviewer for top-tier AI companies. Explain the following concept in depth for a senior role interview: ${current.title}. Description: ${current.description}. Focus on technical nuances, implementation trade-offs, and common pitfalls candidates face when explaining this topic.`,
      });
      setAiResponse(response.text || "No response received from AI.");
    } catch (error) {
      console.error("AI deep dive analysis failed:", error);
      setAiResponse("Unable to fetch AI explanation. Please check your connection and try again.");
    } finally {
      setIsAiLoading(false);
    }
  };

  const formatMD = (text: string) => {
    return text
      .replace(/^### (.*$)/gim, '<h3 class="text-xl font-bold text-gray-900 mb-3 mt-6">$1</h3>')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-indigo-900 font-bold">$1</strong>')
      .replace(/\$(.*?)\$/g, '<code class="bg-indigo-50 text-indigo-700 px-1.5 py-0.5 rounded font-mono text-sm">$1</code>')
      .replace(/\n/g, '<br/>');
  };

  return (
    <div className="flex h-screen bg-white font-sans text-gray-900 overflow-hidden">
      <Sidebar 
        categories={categories}
        questions={INTERVIEW_CONTENT}
        selectedId={selectedId}
        onSelect={(id) => {
          setSelectedId(id);
          setAiResponse(null); // Clear previous AI analysis when topic changes
        }}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
      />

      <main className="flex-1 overflow-y-auto bg-white custom-scrollbar">
        {current ? (
          <div className="max-w-4xl mx-auto p-8 lg:p-16">
            <header className="mb-12 pb-8 border-b border-gray-100">
              <div className="text-xs font-bold text-indigo-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
                {current.category}
              </div>
              <h2 className="text-4xl font-extrabold text-gray-900 mb-4 tracking-tight leading-tight">{current.title}</h2>
              <p className="text-lg text-gray-600 leading-relaxed max-w-2xl">{current.description}</p>
            </header>
            
            <div className="space-y-12">
              <div className="flex justify-start">
                <button 
                  onClick={handleAskAI}
                  disabled={isAiLoading}
                  className="group flex items-center gap-3 px-6 py-3 bg-indigo-600 text-white rounded-xl font-semibold shadow-lg shadow-indigo-200 hover:bg-indigo-700 hover:-translate-y-0.5 active:translate-y-0 transition-all disabled:opacity-50 disabled:translate-y-0 disabled:shadow-none"
                >
                  {isAiLoading ? (
                    <i className="fas fa-circle-notch fa-spin text-lg"></i>
                  ) : (
                    <i className="fas fa-sparkles text-lg group-hover:rotate-12 transition-transform"></i>
                  )}
                  {isAiLoading ? "Consulting Gemini..." : "Ask Gemini for Expert Insight"}
                </button>
              </div>

              {aiResponse && (
                <div className="p-8 bg-indigo-50/50 rounded-2xl border border-indigo-100 shadow-sm transition-all animate-in fade-in slide-in-from-top-4 duration-500">
                  <div className="flex items-center gap-2 text-indigo-700 font-bold text-sm uppercase tracking-wider mb-4">
                    <i className="fas fa-robot text-lg"></i>
                    Gemini AI Interviewer Response
                  </div>
                  <div className="text-gray-800 leading-relaxed whitespace-pre-wrap max-w-none text-base">
                    {aiResponse}
                  </div>
                </div>
              )}

              {current.content.map((block, i) => (
                <div key={i} className="animate-in fade-in slide-in-from-bottom-4 duration-500 delay-150">
                  {block.type === 'markdown' ? (
                    <div 
                      className="text-gray-700 leading-loose text-lg"
                      dangerouslySetInnerHTML={{ __html: formatMD(block.content) }} 
                    />
                  ) : (
                    <CodeBlock code={block.content} />
                  )}
                </div>
              ))}
            </div>

            <footer className="mt-24 pt-10 border-t border-gray-100 text-sm text-gray-400 flex flex-col md:flex-row justify-between items-center gap-4">
              <div className="flex items-center gap-2">
                <i className="fas fa-brain text-indigo-300"></i>
                <span>© 2025 AI Interview Pro Mastery • Senior Engineering Prep</span>
              </div>
              <div className="flex gap-6">
                <a href="#" className="hover:text-indigo-600 transition-colors">Curriculum</a>
                <a href="#" className="hover:text-indigo-600 transition-colors">Mock Prep</a>
                <a href="#" className="hover:text-indigo-600 transition-colors">Resources</a>
              </div>
            </footer>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-gray-300 gap-4">
            <i className="fas fa-book-open text-6xl opacity-20"></i>
            <p className="text-xl font-medium opacity-40">Select a core concept to explore technical details</p>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
