
import React, { useState } from 'react';

interface CodeBlockProps {
  code: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group rounded-xl overflow-hidden border border-gray-200 shadow-sm">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Python Implementation</span>
        <button 
          onClick={handleCopy}
          className="text-gray-400 hover:text-indigo-600 transition-colors"
        >
          {copied ? <i className="fas fa-check text-green-500"></i> : <i className="far fa-copy"></i>}
        </button>
      </div>
      <div className="p-4 bg-gray-900 overflow-x-auto">
        <pre className="text-sm code-font leading-relaxed text-blue-100">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
};

export default CodeBlock;
