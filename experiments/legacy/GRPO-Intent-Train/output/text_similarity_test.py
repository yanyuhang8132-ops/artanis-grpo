#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本相似度测试脚本
支持多种相似度计算方法：
- BLEU Score
- ROUGE Score
- Cosine Similarity (TF-IDF)
- Semantic Similarity (Sentence Transformers)
- Jaccard Similarity
- Edit Distance
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import argparse
import os

# 文本处理
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import jieba

# 相似度计算
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# 语义相似度 - 已禁用
# try:
#     from sentence_transformers import SentenceTransformer
#     SENTENCE_TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     SENTENCE_TRANSFORMERS_AVAILABLE = False
#     print("Warning: sentence-transformers not installed. Semantic similarity will be skipped.")


class TextSimilarityCalculator:
    """文本相似度计算器"""
    
    def __init__(self, language='zh', use_semantic=False):
        """
        初始化相似度计算器
        
        Args:
            language: 语言类型，'zh'为中文，'en'为英文
            use_semantic: 是否使用语义相似度（已禁用）
        """
        self.language = language
        self.use_semantic = False  # 语义相似度已禁用
        
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def tokenize_text(self, text: str) -> List[str]:
        """文本分词"""
        if self.language == 'zh':
            return list(jieba.cut(text.strip()))
        else:
            return word_tokenize(text.lower().strip())
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """计算BLEU分数"""
        ref_tokens = self.tokenize_text(reference)
        cand_tokens = self.tokenize_text(candidate)
        
        # 使用平滑函数避免0分
        smoothing = SmoothingFunction().method1
        
        try:
            score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
            return score
        except:
            return 0.0
    
    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """计算ROUGE分数 - 已禁用"""
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_cosine_similarity(self, reference: str, candidate: str) -> float:
        """计算TF-IDF余弦相似度"""
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([reference, candidate])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def calculate_semantic_similarity(self, reference: str, candidate: str) -> float:
        """计算语义相似度 - 已禁用"""
        return 0.0
    
    def calculate_jaccard_similarity(self, reference: str, candidate: str) -> float:
        """计算Jaccard相似度"""
        ref_tokens = set(self.tokenize_text(reference))
        cand_tokens = set(self.tokenize_text(candidate))
        
        intersection = len(ref_tokens.intersection(cand_tokens))
        union = len(ref_tokens.union(cand_tokens))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_edit_distance_similarity(self, reference: str, candidate: str) -> float:
        """计算编辑距离相似度"""
        try:
            max_len = max(len(reference), len(candidate))
            if max_len == 0:
                return 1.0
            
            edit_distance = len(list(difflib.unified_diff(reference, candidate)))
            similarity = 1 - (edit_distance / (2 * max_len))
            return max(0.0, similarity)
        except:
            return 0.0
    
    def calculate_all_similarities(self, reference: str, candidate: str) -> Dict[str, float]:
        """计算所有相似度指标"""
        similarities = {}
        
        # BLEU Score
        similarities['bleu_score'] = self.calculate_bleu_score(reference, candidate)
        
        # Cosine Similarity (TF-IDF)
        similarities['cosine_similarity'] = self.calculate_cosine_similarity(reference, candidate)
        
        # Semantic Similarity
        similarities['semantic_similarity'] = self.calculate_semantic_similarity(reference, candidate)
        
        # Jaccard Similarity
        similarities['jaccard_similarity'] = self.calculate_jaccard_similarity(reference, candidate)
        
        # Edit Distance Similarity
        similarities['edit_distance_similarity'] = self.calculate_edit_distance_similarity(reference, candidate)
        
        # 计算平均相似度（不包括语义相似度）
        valid_scores = []
        for key, score in similarities.items():
            if key != 'semantic_similarity':
                valid_scores.append(score)
        
        similarities['average_similarity'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        return similarities


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """加载数据文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
    else:
        raise ValueError("Unsupported file format. Please use .json, .jsonl, or .csv")
    
    return data


def test_similarity(data_file: str, reference_field: str = 'reference', 
                   model_output_field: str = 'model_output', 
                   language: str = 'zh', output_file: str = None):
    """
    测试文本相似度
    
    Args:
        data_file: 数据文件路径
        reference_field: 参考文本字段名
        model_output_field: 模型输出字段名
        language: 语言类型
        output_file: 输出文件路径
    """
    print(f"Loading data from {data_file}...")
    data = load_data(data_file)
    print(f"Loaded {len(data)} records")
    
    # 初始化相似度计算器
    calculator = TextSimilarityCalculator(language=language)
    
    results = []
    
    print("Calculating similarities...")
    for i, item in enumerate(data):
        if i % 100 == 0:
            print(f"Processing {i}/{len(data)}")
        
        reference = str(item.get(reference_field, ''))
        model_output = str(item.get(model_output_field, ''))
        
        if not reference or not model_output:
            print(f"Warning: Empty text found at index {i}")
            continue
        
        # 计算所有相似度
        similarities = calculator.calculate_all_similarities(reference, model_output)
        
        # 添加原始数据
        result = {
            'index': i,
            'reference': reference,
            'model_output': model_output,
            **similarities
        }
        
        results.append(result)
    
    # 计算统计信息
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("🎯 最终结果")
    print("="*50)
    avg_similarity = df_results['average_similarity'].mean()
    print(f"📊 平均文本相似度: {avg_similarity:.4f}")
    
    # 简化显示，只显示平均值
    print(f"\n📈 各指标平均分数:")
    similarity_columns = ['bleu_score', 'cosine_similarity', 'jaccard_similarity', 'edit_distance_similarity']
    
    for col in similarity_columns:
        if col in df_results.columns:
            mean_score = df_results[col].mean()
            print(f"   {col:20}: {mean_score:.4f}")
    
    print("="*50)
    
    # 保存结果
    if output_file:
        if output_file.endswith('.csv'):
            df_results.to_csv(output_file, index=False, encoding='utf-8')
        elif output_file.endswith('.json'):
            df_results.to_json(output_file, orient='records', ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return df_results


def main():
    parser = argparse.ArgumentParser(description='Text Similarity Testing Script')
    parser.add_argument('--data_file', type=str, required=True, help='Path to data file')
    parser.add_argument('--reference_field', type=str, default='reference', help='Reference text field name')
    parser.add_argument('--model_output_field', type=str, default='model_output', help='Model output field name')
    parser.add_argument('--language', type=str, default='zh', choices=['zh', 'en'], help='Language type')
    parser.add_argument('--output_file', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    test_similarity(
        data_file=args.data_file,
        reference_field=args.reference_field,
        model_output_field=args.model_output_field,
        language=args.language,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()