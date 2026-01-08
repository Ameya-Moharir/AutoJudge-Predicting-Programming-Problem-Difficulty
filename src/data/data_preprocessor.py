"""
Text preprocessing for programming problem descriptions
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Optional
import unicodedata

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Advanced text preprocessing for programming problems"""
    
    def __init__(self, 
                 lowercase=True,
                 remove_special_chars=False,
                 remove_stopwords=True,
                 lemmatization=True,
                 min_word_length=2):
        """
        Initialize preprocessor
        
        Args:
            lowercase: Convert text to lowercase
            remove_special_chars: Remove special characters
            remove_stopwords: Remove stop words
            lemmatization: Apply lemmatization
            min_word_length: Minimum word length to keep
        """
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_stopwords = remove_stopwords
        self.lemmatization = lemmatization
        self.min_word_length = min_word_length
        
        # Initialize tools
        self.lemmatizer = WordNetLemmatizer() if lemmatization else None
        
        # Custom stop words (exclude programming-specific terms)
        self.stop_words = set(stopwords.words('english'))
        
        # Remove these from stop words as they're important in programming
        programming_terms = {
            'only', 'no', 'not', 'all', 'each', 'every', 'same', 'different',
            'first', 'second', 'last', 'more', 'less', 'most', 'least',
            'before', 'after', 'between', 'above', 'below', 'up', 'down'
        }
        self.stop_words -= programming_terms
    
    def preprocess(self, text: str, preserve_code=True) -> str:
        """
        Preprocess text
        
        Args:
            text: Input text
            preserve_code: Whether to preserve code snippets
        
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Extract and preserve code snippets if requested
        code_snippets = []
        if preserve_code:
            text, code_snippets = self._extract_code(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Handle mathematical expressions (preserve important symbols)
        text = self._preserve_math_notation(text)
        
        # Remove special characters but preserve important programming symbols
        if self.remove_special_chars:
            text = self._clean_special_chars(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words
        if self.remove_stopwords:
    # Keep programming-related stop words
            important_stopwords = {
                'not', 'no', 'all', 'any', 'each', 'every', 'most', 'least',
                'first', 'last', 'next', 'previous', 'before', 'after',
                'more', 'less', 'same', 'different', 'both', 'either',
                'may', 'must', 'can', 'should', 'will'
            }
            tokens = [
                w for w in tokens 
                if w not in self.stop_words or w in important_stopwords
            ]
        
        # Lemmatization
        if self.lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        
        # Join tokens
        processed_text = ' '.join(tokens)
        
        # Add back code indicators
        if code_snippets:
            processed_text += ' CODE_PRESENT'
        
        return processed_text.strip()
    
    def _extract_code(self, text: str) -> tuple:
        """Extract code snippets from text"""
        code_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'`[^`]+`',  # Inline code
        ]
        
        code_snippets = []
        for pattern in code_patterns:
            snippets = re.findall(pattern, text)
            code_snippets.extend(snippets)
            text = re.sub(pattern, '', text)
        
        return text, code_snippets
    
    def _preserve_math_notation(self, text: str) -> str:
        """Preserve mathematical notation"""
        # Replace common math symbols with words
        replacements = {
            r'\$.*?\$': ' MATH_EXPR ',  # LaTeX math
            r'O\([^\)]+\)': ' COMPLEXITY ',  # Big O notation
            r'≤': ' leq ',
            r'≥': ' geq ',
            r'≠': ' neq ',
            r'∈': ' in ',
            r'∑': ' sum ',
            r'∏': ' product ',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _clean_special_chars(self, text: str) -> str:
        """Clean special characters while preserving important ones"""
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\-_+*/<>=]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple texts
        
        Args:
            texts: List of texts
        
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
    
    def combine_fields(self, 
                       title: str = "",
                       description: str = "",
                       input_desc: str = "",
                       output_desc: str = "") -> str:
        """
        Combine multiple text fields with appropriate weighting
        
        Args:
            title: Problem title
            description: Problem description
            input_desc: Input description
            output_desc: Output description
        
        Returns:
            Combined and weighted text
        """
        # Weight different fields (title and description are more important)
        combined = []
        
        if title:
            # Repeat title to give it more weight
            combined.extend([title] * 3)
        
        if description:
            # Description is most important
            combined.extend([description] * 5)
        
        if input_desc:
            combined.append(input_desc)
        
        if output_desc:
            combined.append(output_desc)
        
        return ' '.join(combined)
