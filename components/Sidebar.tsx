
import React from 'react';
import { CategoryType, Question } from '../types';

interface SidebarProps {
  categories: CategoryType[];
  questions: Question[];
  selectedId: string;
  onSelect: (id: string) => void;
  searchTerm: string;
  onSearchChange: (val: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ 
  categories, 
  questions, 
  selectedId, 
  onSelect,
  searchTerm,
  onSearchChange
}) => {
  return (
    <div className="w-80 h-full border-r bg-white flex flex-col">
      <div className="p-4 border-b">
        <h1 className="text-xl font-bold text-indigo-600 mb-4 flex items-center gap-2">
           <i className="fas fa-brain"></i> AI Interview Pro
        </h1>
        <div className="relative">
          <input 
            type="text" 
            placeholder="Search topics..." 
            className="w-full pl-9 pr-4 py-2 bg-gray-100 border-none rounded-lg focus:ring-2 focus:ring-indigo-500 text-sm"
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
          />
          <i className="fas fa-search absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 text-sm"></i>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {categories.map(category => {
          const categoryQuestions = questions.filter(q => 
            q.category === category && 
            (q.title.toLowerCase().includes(searchTerm.toLowerCase()) || 
             q.description.toLowerCase().includes(searchTerm.toLowerCase()))
          );

          if (categoryQuestions.length === 0) return null;

          return (
            <div key={category} className="py-2">
              <h2 className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider bg-gray-50">
                {category}
              </h2>
              <div className="mt-1 space-y-px">
                {categoryQuestions.map(q => (
                  <button
                    key={q.id}
                    onClick={() => onSelect(q.id)}
                    className={`w-full text-left px-4 py-2.5 text-sm transition-colors flex flex-col gap-0.5
                      ${selectedId === q.id 
                        ? 'bg-indigo-50 text-indigo-700 border-r-4 border-indigo-600' 
                        : 'text-gray-700 hover:bg-gray-100'}`}
                  >
                    <span className="font-medium truncate">{q.title}</span>
                    <span className="text-xs text-gray-500 truncate">{q.description}</span>
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Sidebar;
