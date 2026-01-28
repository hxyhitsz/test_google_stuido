// Define types for the Interview application to ensure type safety and resolve module errors
export type CategoryType = string;

export interface ContentBlock {
  type: 'markdown' | 'code';
  content: string;
}

export interface Question {
  id: string;
  category: CategoryType;
  title: string;
  description: string;
  content: ContentBlock[];
}
