"""
Data loading utilities for AutoJudge
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm
from ..utils.logger import logger


class DataLoader:
    """Load and manage programming problem datasets"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing raw data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_jsonl(self, file_path: str) -> pd.DataFrame:
        """
        Load JSONL file
        
        Args:
            file_path: Path to JSONL file
        
        Returns:
            DataFrame with loaded data
        """
        data = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} samples")
        
        return df
    
    def download_dataset(self, url: str, output_path: str = None) -> str:
        """
        Download dataset from URL
        
        Args:
            url: Dataset URL
            output_path: Output file path
        
        Returns:
            Path to downloaded file
        """
        if output_path is None:
            output_path = self.data_dir / url.split('/')[-1]
        else:
            output_path = Path(output_path)
        
        if output_path.exists():
            logger.info(f"Dataset already exists: {output_path}")
            return str(output_path)
        
        logger.info(f"Downloading dataset from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded dataset to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def load_task_complexity_dataset(self) -> pd.DataFrame:
        """
        Load TaskComplexity dataset
        
        Returns:
            DataFrame with TaskComplexity data
        """
        url = "https://raw.githubusercontent.com/AREEG94FAHAD/TaskComplexityEval-24/main/problems_data.jsonl"
        
        try:
            file_path = self.download_dataset(url)
            df = self.load_jsonl(file_path)
            return self._standardize_columns(df)
        except Exception as e:
            logger.warning(f"Could not load TaskComplexity dataset: {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and format"""
        # Map common column name variations
        column_mapping = {
            'title': 'title',
            'problem_title': 'title',
            'description': 'description',
            'problem_description': 'description',
            'input_description': 'input_description',
            'input_desc': 'input_description',
            'output_description': 'output_description',
            'output_desc': 'output_description',
            'problem_class': 'difficulty',
            'class': 'difficulty',
            'difficulty': 'difficulty',
            'problem_score': 'score',
            'difficulty_score': 'score',
            'score': 'score'
        }
        
        # Rename columns
        df.columns = [column_mapping.get(col.lower(), col.lower()) for col in df.columns]
        
        # Standardize difficulty labels
        if 'difficulty' in df.columns:
            df['difficulty'] = df['difficulty'].str.lower().str.strip()
            df['difficulty'] = df['difficulty'].map({
                'easy': 'easy',
                'medium': 'medium',
                'med': 'medium',
                'hard': 'hard',
                'difficult': 'hard'
            }).fillna(df['difficulty'])
        
        return df
    
    def create_sample_dataset(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Create a small sample dataset for testing
        
        Args:
            n_samples: Number of samples to create
        
        Returns:
            Sample DataFrame
        """
        import numpy as np
        
        templates = {
            'easy': [
                {
                    'title': 'Simple Addition',
                    'description': 'Given two numbers A and B, output their sum.',
                    'input_description': 'Two integers A and B (1 ≤ A, B ≤ 100)',
                    'output_description': 'Output A + B',
                    'difficulty': 'easy',
                    'score': np.random.uniform(1, 3)
                },
                {
                    'title': 'Count Characters',
                    'description': 'Given a string, count the number of characters.',
                    'input_description': 'A single string S (1 ≤ |S| ≤ 100)',
                    'output_description': 'Output the length of S',
                    'difficulty': 'easy',
                    'score': np.random.uniform(1, 3)
                }
            ],
            'medium': [
                {
                    'title': 'Binary Search',
                    'description': 'Given a sorted array and a target value, find the index using binary search.',
                    'input_description': 'An array of n sorted integers and a target value (1 ≤ n ≤ 10^5)',
                    'output_description': 'Output the index or -1 if not found',
                    'difficulty': 'medium',
                    'score': np.random.uniform(4, 6)
                },
                {
                    'title': 'Longest Common Subsequence',
                    'description': 'Find the longest common subsequence between two strings using dynamic programming.',
                    'input_description': 'Two strings S1 and S2 (1 ≤ |S1|, |S2| ≤ 1000)',
                    'output_description': 'Output the length of LCS',
                    'difficulty': 'medium',
                    'score': np.random.uniform(4, 6)
                }
            ],
            'hard': [
                {
                    'title': 'Shortest Path in Graph',
                    'description': 'Find shortest path in weighted graph using Dijkstra algorithm with heap optimization.',
                    'input_description': 'Graph with n nodes and m edges (1 ≤ n ≤ 10^5, 1 ≤ m ≤ 10^6)',
                    'output_description': 'Output shortest distance or -1 if unreachable',
                    'difficulty': 'hard',
                    'score': np.random.uniform(7, 10)
                },
                {
                    'title': 'Segment Tree Range Queries',
                    'description': 'Implement segment tree to handle range minimum queries and point updates efficiently.',
                    'input_description': 'Array of n integers and q queries (1 ≤ n, q ≤ 10^5)',
                    'output_description': 'Output result for each query',
                    'difficulty': 'hard',
                    'score': np.random.uniform(7, 10)
                }
            ]
        }
        
        data = []
        samples_per_class = n_samples // 3
        
        for difficulty in ['easy', 'medium', 'hard']:
            for _ in range(samples_per_class):
                template = np.random.choice(templates[difficulty])
                sample = template.copy()
                # Add some variation
                sample['score'] += np.random.uniform(-0.5, 0.5)
                data.append(sample)
        
        df = pd.DataFrame(data)
        logger.info(f"Created sample dataset with {len(df)} samples")
        
        return df
    
    def combine_datasets(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple datasets
        
        Args:
            dataframes: List of DataFrames to combine
        
        Returns:
            Combined DataFrame
        """
        if not dataframes:
            return pd.DataFrame()
        
        # Filter non-empty dataframes
        dataframes = [df for df in dataframes if not df.empty]
        
        if not dataframes:
            return pd.DataFrame()
        
        combined = pd.concat(dataframes, ignore_index=True)
        
        # Remove duplicates based on title
        if 'title' in combined.columns:
            combined = combined.drop_duplicates(subset=['title'], keep='first')
        
        logger.info(f"Combined dataset has {len(combined)} samples")
        
        return combined
    
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate dataset format
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        required_columns = ['title', 'description', 'difficulty', 'score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for empty values
        if 'description' in df.columns and df['description'].isna().any():
            errors.append("Some descriptions are missing")
        
        # Check difficulty values
        if 'difficulty' in df.columns:
            valid_difficulties = {'easy', 'medium', 'hard'}
            invalid = df[~df['difficulty'].isin(valid_difficulties)]
            if not invalid.empty:
                errors.append(f"Invalid difficulty values found: {invalid['difficulty'].unique()}")
        
        # Check score range
        if 'score' in df.columns:
            if df['score'].isna().any():
                errors.append("Some scores are missing")
            elif (df['score'] < 0).any() or (df['score'] > 10).any():
                errors.append("Scores should be between 0 and 10")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Dataset validation passed")
        else:
            logger.error(f"Dataset validation failed: {errors}")
        
        return is_valid, errors
