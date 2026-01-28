
export enum CategoryType {
  ATTENTION = "Attention & Encoding",
  TRANSFORMER = "Transformer Arch",
  RL_ALGO = "RL & Alignment",
  LLM_ENG = "LLM Engineering",
  DL_BASICS = "DL Basics",
  CV_CORE = "CV Core"
}

export interface ContentSection {
  type: 'markdown' | 'code';
  content: string;
}

export interface Question {
  id: string;
  title: string;
  category: CategoryType;
  description: string;
  content: ContentSection[];
}
