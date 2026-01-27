
export enum CategoryType {
  ATTENTION = "Attention & Encoding",
  DL_BASICS = "DL Basics",
  CV_CORE = "CV Core",
  RL_ALGO = "RL & Alignment",
  TRANSFORMER = "Transformer Arch",
  LLM_ENG = "LLM Engineering"
}

export interface CodeSnippet {
  title: string;
  description: string;
  code: string;
  category: CategoryType;
  tags: string[];
}

export interface Question {
  id: string;
  title: string;
  category: CategoryType;
  description: string;
  content: {
    type: 'markdown' | 'code';
    content: string;
  }[];
}
