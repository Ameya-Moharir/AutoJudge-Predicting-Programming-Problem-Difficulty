#!/usr/bin/env python
"""
Train AutoJudge models
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import TextPreprocessor
from src.data.feature_engineer import CombinedFeatureExtractor
from src.models.classifier import DifficultyClassifier
from src.models.regressor import DifficultyScoreRegressor
from src.utils.logger import logger
from src.utils.config import config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train AutoJudge models')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with sample data')
    parser.add_argument('--full', action='store_true',
                       help='Full training with downloaded datasets')
    parser.add_argument('--dataset', type=str,
                       help='Path to custom dataset')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models')
    return parser.parse_args()


def load_and_prepare_data(quick=False, custom_dataset=None):
    """
    Load and prepare data
    
    Args:
        quick: Use sample data
        custom_dataset: Path to custom dataset
    
    Returns:
        DataFrame with prepared data
    """
    loader = DataLoader()
    
    if custom_dataset:
        logger.info(f"Loading custom dataset: {custom_dataset}")
        df = loader.load_jsonl(custom_dataset)
    elif quick:
        logger.info("Creating sample dataset for quick training...")
        df = loader.create_sample_dataset(n_samples=300)
    else:
        logger.info("Loading full datasets...")
        datasets = []
        
        # Load TaskComplexity dataset
        try:
            tc_df = loader.load_task_complexity_dataset()
            if not tc_df.empty:
                datasets.append(tc_df)
        except Exception as e:
            logger.warning(f"Could not load TaskComplexity dataset: {e}")
        
        # If no datasets loaded, use sample
        if not datasets:
            logger.warning("No datasets loaded, using sample data")
            df = loader.create_sample_dataset(n_samples=500)
        else:
            df = loader.combine_datasets(datasets)
    
    # Validate dataset
    is_valid, errors = loader.validate_dataset(df)
    if not is_valid:
        logger.error(f"Dataset validation failed: {errors}")
        logger.info("Attempting to fix dataset...")
        df = fix_dataset(df)
    
    return df


def fix_dataset(df):
    """Fix common dataset issues"""
    # Fill missing descriptions
    if 'description' in df.columns:
        df['description'] = df['description'].fillna('')
    
    # Fill missing input/output descriptions
    if 'input_description' in df.columns:
        df['input_description'] = df['input_description'].fillna('')
    if 'output_description' in df.columns:
        df['output_description'] = df['output_description'].fillna('')
    
    # Ensure difficulty column exists
    if 'difficulty' not in df.columns and 'problem_class' in df.columns:
        df['difficulty'] = df['problem_class']
    
    # Standardize difficulty labels
    if 'difficulty' in df.columns:
        df['difficulty'] = df['difficulty'].str.lower().str.strip()
        df = df[df['difficulty'].isin(['easy', 'medium', 'hard'])]
    
    # Ensure score column exists
    if 'score' not in df.columns and 'problem_score' in df.columns:
        df['score'] = df['problem_score']
    
    # Generate scores if missing
    if 'score' not in df.columns or df['score'].isna().all():
        score_map = {'easy': (1, 3), 'medium': (4, 6), 'hard': (7, 10)}
        df['score'] = df['difficulty'].apply(
            lambda x: np.random.uniform(*score_map.get(x, (5, 5)))
        )
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['description', 'difficulty', 'score'])
    
    return df


def preprocess_texts(df):
    """
    Preprocess all text fields
    
    Args:
        df: DataFrame with raw text
    
    Returns:
        Array of preprocessed combined texts
    """
    logger.info("Preprocessing text data...")
    
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_special_chars=False,
        remove_stopwords=True,
        lemmatization=True,
        min_word_length=2
    )
    
    # Combine text fields
    combined_texts = []
    for _, row in df.iterrows():
        combined = preprocessor.combine_fields(
            title=row.get('title', ''),
            description=row.get('description', ''),
            input_desc=row.get('input_description', ''),
            output_desc=row.get('output_description', '')
        )
        combined_texts.append(combined)
    
    # Preprocess
    processed_texts = preprocessor.batch_preprocess(combined_texts)
    
    logger.info(f"Preprocessed {len(processed_texts)} texts")
    
    return processed_texts


def extract_features(texts, fit=True, extractor=None):
    """
    Extract features from texts
    
    Args:
        texts: List of preprocessed texts
        fit: Whether to fit the extractor
        extractor: Pre-fitted extractor (if fit=False)
    
    Returns:
        Features array and fitted extractor
    """
    logger.info("Extracting features...")
    
    if extractor is None:
        extractor = CombinedFeatureExtractor(
            max_features=config.get('preprocessing.max_features', 5000),
            ngram_range=tuple(config.get('preprocessing.ngram_range', [1, 2]))
        )
    
    if fit:
        features = extractor.fit_transform(texts)
        logger.info(f"Extracted {features.shape[1]} features")
    else:
        features = extractor.transform(texts)
    
    return features, extractor


def train_models(X_train, y_train_class, y_train_score, X_test, y_test_class, y_test_score):
    """
    Train classification and regression models
    
    Args:
        X_train: Training features
        y_train_class: Training difficulty labels
        y_train_score: Training difficulty scores
        X_test: Test features
        y_test_class: Test difficulty labels
        y_test_score: Test difficulty scores
    
    Returns:
        Trained classifier and regressor
    """
    # Scale features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    logger.info("\n" + "="*80)
    logger.info("TRAINING CLASSIFICATION MODEL")
    logger.info("="*80)
    
    classifier = DifficultyClassifier()
    classifier.fit(X_train_scaled, y_train_class)
    
    # Evaluate classifier
    class_results = classifier.evaluate(X_test_scaled, y_test_class)
    
    # Train regressor
    logger.info("\n" + "="*80)
    logger.info("TRAINING REGRESSION MODEL")
    logger.info("="*80)
    
    regressor = DifficultyScoreRegressor()
    regressor.fit(X_train_scaled, y_train_score)
    
    # Evaluate regressor
    reg_results = regressor.evaluate(X_test_scaled, y_test_score)
    
    return classifier, regressor, scaler, class_results, reg_results


def save_models(classifier, regressor, extractor, scaler, output_dir):
    """
    Save all models and artifacts
    
    Args:
        classifier: Trained classifier
        regressor: Trained regressor
        extractor: Feature extractor
        scaler: Feature scaler
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving models to {output_dir}...")
    
    classifier.save(output_dir / 'classifier.pkl')
    regressor.save(output_dir / 'regressor.pkl')
    joblib.dump(extractor, output_dir / 'feature_extractor.pkl')
    joblib.dump(scaler, output_dir / 'scaler.pkl')
    
    logger.info("All models saved successfully!")


def print_summary(class_results, reg_results, n_train, n_test):
    """Print training summary"""
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Training samples: {n_train}")
    logger.info(f"Test samples: {n_test}")
    logger.info("")
    logger.info("CLASSIFICATION RESULTS:")
    logger.info(f"  Accuracy: {class_results['accuracy']:.4f}")
    logger.info(f"  Weighted Precision: {class_results['classification_report']['weighted avg']['precision']:.4f}")
    logger.info(f"  Weighted Recall: {class_results['classification_report']['weighted avg']['recall']:.4f}")
    logger.info(f"  Weighted F1: {class_results['classification_report']['weighted avg']['f1-score']:.4f}")
    logger.info("")
    logger.info("REGRESSION RESULTS:")
    logger.info(f"  RMSE: {reg_results['rmse']:.4f}")
    logger.info(f"  MAE: {reg_results['mae']:.4f}")
    logger.info(f"  R² Score: {reg_results['r2']:.4f}")
    logger.info(f"  MAPE: {reg_results['mape']:.2f}%")
    logger.info("="*80)


def main():
    """Main training function"""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("AutoJudge Model Training")
    logger.info("="*80)
    
    # Load data
    df = load_and_prepare_data(
        quick=args.quick,
        custom_dataset=args.dataset
    )
    
    logger.info(f"\nDataset statistics:")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Difficulty distribution:")
    logger.info(df['difficulty'].value_counts())
    logger.info(f"\nScore statistics:")
    logger.info(df['score'].describe())
    
    # Preprocess texts
    texts = preprocess_texts(df)
    
    # Extract features
    features, extractor = extract_features(texts)
    
    # Prepare labels
    y_class = df['difficulty'].values
    y_score = df['score'].values
    
    # Split data
    logger.info(f"\nSplitting data (test_size={args.test_size})...")
    X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = train_test_split(
        features, y_class, y_score,
        test_size=args.test_size,
        random_state=42,
        stratify=y_class
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train models
    classifier, regressor, scaler, class_results, reg_results = train_models(
        X_train, y_train_class, y_train_score,
        X_test, y_test_class, y_test_score
    )
    
    # Save models
    save_models(classifier, regressor, extractor, scaler, args.output_dir)
    
    # Print summary
    print_summary(class_results, reg_results, len(X_train), len(X_test))
    
    logger.info("\nTraining completed successfully!")
    logger.info(f"Models saved to: {args.output_dir}")
    logger.info("\nTo run the web application, execute:")
    logger.info("  python web/app.py")


if __name__ == '__main__':
    main()
