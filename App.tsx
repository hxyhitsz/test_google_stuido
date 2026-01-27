
import React, { useState, useMemo } from 'react';
import Sidebar from './components/Sidebar';
import CodeBlock from './components/CodeBlock';
import { INTERVIEW_CONTENT } from './constants';
import { CategoryType, Question } from './types';

const App: React.FC = () => {
  const [selectedId, setSelectedId] = useState(INTERVIEW_CONTENT[0].id);
  const [searchTerm, setSearchTerm] = useState("");

  const selectedQuestion = useMemo(() => 
    INTERVIEW_CONTENT.find(q => q.id === selectedId), 
  [selectedId]);

  const categories = Object.values(CategoryType);

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Sidebar */}
      <Sidebar 
        categories={categories}
        questions={INTERVIEW_CONTENT}
        selectedId={selectedId}
        onSelect={setSelectedId}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
      />

      {/* Main Content (Notebook Style) */}
      <main className="flex-1 overflow-y-auto bg-white p-8 md:p-12">
        {selectedQuestion ? (
          <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500">
            {/* Header */}
            <header className="border-b pb-6">
              <div className="flex items-center gap-2 text-indigo-600 text-sm font-semibold uppercase tracking-wide mb-2">
                <i className="fas fa-tag"></i> {selectedQuestion.category}
              </div>
              <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight">
                {selectedQuestion.title}
              </h1>
              <p className="mt-3 text-lg text-gray-600 leading-relaxed">
                {selectedQuestion.description}
              </p>
            </header>

            {/* Notebook Cells */}
            <div className="space-y-10">
              {selectedQuestion.content.map((section, idx) => (
                <div key={idx} className="space-y-4">
                  {section.type === 'markdown' ? (
                    <div className="prose prose-indigo max-w-none text-gray-700 leading-relaxed">
                      {/* Simple naive markdown rendering for this demo */}
                      <div dangerouslySetInnerHTML={{ 
                        __html: section.content
                          .replace(/^### (.*$)/gim, '<h3 class="text-xl font-bold text-gray-800 mb-4 mt-8">$1</h3>')
                          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                          .replace(/\$(.*?)\$/g, '<code class="bg-indigo-50 px-1 py-0.5 rounded text-indigo-700">$1</code>')
                          .replace(/\n/g, '<br/>')
                      }} />
                    </div>
                  ) : (
                    <CodeBlock code={section.content} />
                  )}
                </div>
              ))}
            </div>

            {/* Footer / Tip */}
            <footer className="mt-16 pt-8 border-t">
              <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded-r-lg">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <i className="fas fa-lightbulb text-amber-400"></i>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm text-amber-700 font-medium">
                      Interview Tip:
                    </p>
                    <p className="text-sm text-amber-700 mt-1">
                      Candidates are often asked to derive these formulas from scratch. Focus on understanding why the change was made (e.g., why AdamW decouples weight decay) rather than just memorizing the code.
                    </p>
                  </div>
                </div>
              </div>
            </footer>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-gray-400">
            <i className="fas fa-search text-6xl mb-4"></i>
            <p className="text-xl">Select a topic or search to get started.</p>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
