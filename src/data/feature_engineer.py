"""
Feature engineering for programming problem difficulty prediction
"""
import re
import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class ProgrammingFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features specific to programming problems"""
    
    def __init__(self, keyword_dict: Dict[str, List[str]] = None):
        """
        Initialize feature extractor
        
        Args:
            keyword_dict: Dictionary of keyword categories and their keywords
        """
        self.keyword_dict = keyword_dict or self._get_default_keywords()
        self.feature_names = []
    
    def _get_default_keywords(self) -> Dict[str, List[str]]:
        """Get default programming keywords"""
        return {
            'graph': ['graph', 'tree', 'node', 'edge', 'vertex', 'dfs', 'bfs', 
                     'dijkstra', 'spanning', 'shortest', 'path', 'connected', 'cycle'],
            'dp': ['dynamic', 'programming', 'dp', 'memoization', 'recursion', 
                   'subproblem', 'optimal', 'state'],
            'greedy': ['greedy', 'optimization', 'minimum', 'maximum', 'optimal'],
            'math': ['prime', 'gcd', 'lcm', 'modulo', 'factorial', 'fibonacci',
                    'combinatorics', 'probability', 'number theory'],
            'data_structures': ['stack', 'queue', 'heap', 'priority', 'set', 
                               'map', 'hash', 'array', 'list'],
            'string': ['string', 'pattern', 'substring', 'palindrome', 'anagram',
                      'match', 'parse'],
            'sorting': ['sort', 'merge', 'quick', 'insertion', 'order'],
            'search': ['binary search', 'linear search', 'search', 'find'],
            'complexity': ['time complexity', 'space complexity', 'O(n)', 'O(1)',
                          'logarithmic', 'polynomial', 'exponential'],
            'advanced': ['segment tree', 'fenwick', 'suffix', 'trie', 
                        'convex hull', 'flow', 'bipartite']
        }
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)"""
        return self
    
    def transform(self, X):
        """
        Transform text to features
        
        Args:
            X: List of text strings
        
        Returns:
            Feature matrix
        """
        features = []
        
        for text in X:
            text_lower = text.lower() if isinstance(text, str) else ""
            feature_vector = self._extract_features(text_lower, text)
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_features(self, text_lower: str, text_original: str) -> List[float]:
        """Extract all features from text"""
        features = []
        
        # 1. Text statistics
        features.extend(self._text_statistics(text_original))
        
        # 2. Keyword features
        features.extend(self._keyword_features(text_lower))
        
        # 3. Complexity indicators
        features.extend(self._complexity_indicators(text_lower))
        
        # 4. Mathematical indicators
        features.extend(self._math_indicators(text_original))
        
        # 5. Structure indicators
        features.extend(self._structure_indicators(text_lower))
        
        return features
    
    def _text_statistics(self, text: str) -> List[float]:
        """Extract basic text statistics"""
        if not text:
            return [0] * 10
        
        words = text.split()
        sentences = text.split('.')
        
        features = [
            len(text),  # Character count
            len(words),  # Word count
            len(sentences),  # Sentence count
            len(text) / max(len(words), 1),  # Avg word length
            len(words) / max(len(sentences), 1),  # Avg sentence length
            len(set(words)) / max(len(words), 1),  # Lexical diversity
            sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Uppercase ratio
            sum(1 for c in text if c.isdigit()) / max(len(text), 1),  # Digit ratio
            sum(1 for c in text if c in '.,;:!?') / max(len(text), 1),  # Punctuation ratio
            text.count('\n') / max(len(text), 1),  # Newline ratio
        ]
        
        return features
    
    def _keyword_features(self, text: str) -> List[float]:
        """Extract keyword-based features"""
        features = []
        
        for category, keywords in self.keyword_dict.items():
            # Count occurrences of keywords in this category
            count = sum(text.count(keyword) for keyword in keywords)
            # Normalize by text length
            normalized_count = count / max(len(text.split()), 1)
            features.append(normalized_count)
            
            # Binary feature: category present or not
            features.append(1.0 if count > 0 else 0.0)
        
        return features
    
    def _complexity_indicators(self, text: str) -> List[float]:
        """Extract algorithmic complexity indicators"""
        features = []
        
        # Big O notation patterns
        big_o_patterns = [r'O\(1\)', r'O\(n\)', r'O\(n\^2\)', r'O\(log', r'O\(n log']
        for pattern in big_o_patterns:
            features.append(1.0 if re.search(pattern, text, re.IGNORECASE) else 0.0)
        
        # Time/space constraints
        features.append(1.0 if 'time limit' in text else 0.0)
        features.append(1.0 if 'memory limit' in text else 0.0)
        
        # Large number indicators
        features.append(1.0 if re.search(r'10\^[5-9]', text) else 0.0)
        features.append(1.0 if re.search(r'10\^[1-4]', text) else 0.0)
        
        return features
    
    def _math_indicators(self, text: str) -> List[float]:
        """Extract mathematical notation indicators"""
        features = []
        
        # LaTeX math expressions
        features.append(text.count('$') / max(len(text), 1))
        
        # Mathematical symbols
        math_symbols = ['≤', '≥', '≠', '∈', '∑', '∏', '∫', '√', '±']
        features.append(sum(text.count(sym) for sym in math_symbols) / max(len(text), 1))
        
        # Formulas (simple heuristic)
        formula_patterns = [r'\w+\s*[+\-*/=]\s*\w+', r'\w+\^\w+']
        formula_count = sum(len(re.findall(pat, text)) for pat in formula_patterns)
        features.append(formula_count / max(len(text.split()), 1))
        
        # Numbers
        numbers = re.findall(r'\b\d+\b', text)
        features.append(len(numbers) / max(len(text.split()), 1))
        
        # Large numbers (complexity indicators)
        large_numbers = [n for n in numbers if len(n) >= 6]
        features.append(len(large_numbers) / max(len(numbers), 1) if numbers else 0)
        
        return features
    
    def _structure_indicators(self, text: str) -> List[float]:
        """Extract problem structure indicators"""
        features = []
        
        # Input/output specification
        features.append(1.0 if 'input' in text else 0.0)
        features.append(1.0 if 'output' in text else 0.0)
        features.append(1.0 if 'example' in text else 0.0)
        features.append(1.0 if 'constraint' in text else 0.0)
        
        # Multi-test case
        features.append(1.0 if 'test case' in text else 0.0)
        
        # Array/matrix indicators
        features.append(1.0 if re.search(r'array|matrix|grid', text) else 0.0)
        
        # Query/operation indicators
        features.append(1.0 if re.search(r'query|operation|update', text) else 0.0)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        names = []
        
        # Text statistics
        names.extend(['char_count', 'word_count', 'sentence_count', 'avg_word_len',
                     'avg_sentence_len', 'lexical_diversity', 'uppercase_ratio',
                     'digit_ratio', 'punct_ratio', 'newline_ratio'])
        
        # Keyword features
        for category in self.keyword_dict.keys():
            names.append(f'{category}_count')
            names.append(f'{category}_present')
        
        # Complexity indicators
        names.extend(['has_O1', 'has_On', 'has_On2', 'has_Olog', 'has_Onlog',
                     'has_time_limit', 'has_memory_limit', 'large_constraints', 
                     'small_constraints'])
        
        # Math indicators
        names.extend(['latex_density', 'math_symbol_density', 'formula_density',
                     'number_density', 'large_number_ratio'])
        
        # Structure indicators
        names.extend(['has_input', 'has_output', 'has_example', 'has_constraint',
                     'multi_test', 'has_array', 'has_query'])
        
        return names


class CombinedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Combine TF-IDF with custom features"""
    
    def __init__(self, 
                 max_features=5000,
                 ngram_range=(1, 2),
                 keyword_dict=None):
        """
        Initialize combined feature extractor
        
        Args:
            max_features: Maximum TF-IDF features
            ngram_range: N-gram range for TF-IDF
            keyword_dict: Keyword dictionary for custom features
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents='unicode',
            lowercase=True,
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words=None  # We handle this in preprocessing
        )
        
        self.custom_extractor = ProgrammingFeatureExtractor(keyword_dict)
        self.fitted = False
    
    def fit(self, X, y=None):
        """Fit both extractors"""
        self.tfidf_vectorizer.fit(X)
        self.custom_extractor.fit(X)
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform text to combined features"""
        if not self.fitted:
            raise ValueError("Extractor must be fitted before transform")
        
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(X).toarray()
        
        # Custom features
        custom_features = self.custom_extractor.transform(X)
        
        # Combine
        combined_features = np.hstack([tfidf_features, custom_features])
        
        return combined_features
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names"""
        tfidf_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
        custom_names = self.custom_extractor.get_feature_names()
        return tfidf_names + custom_names
