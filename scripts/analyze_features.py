#!/usr/bin/env python
"""
Feature Engineering Analysis Script
Demonstrates feature extraction and analyzes feature importance
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import TextPreprocessor
from src.data.feature_engineer import CombinedFeatureExtractor
from src.utils.logger import logger


def analyze_features():
    """Analyze feature extraction and importance"""
    
    logger.info("="*80)
    logger.info("FEATURE ENGINEERING ANALYSIS")
    logger.info("="*80)
    
    # Load data
    loader = DataLoader()
    df = loader.load_task_complexity_dataset()
    
    if df.empty:
        logger.info("Using sample data...")
        df = loader.create_sample_dataset(n_samples=300)
    
    logger.info(f"\nLoaded {len(df)} samples")
    
    # Preprocessing
    logger.info("\n" + "="*80)
    logger.info("TEXT PREPROCESSING")
    logger.info("="*80)
    
    preprocessor = TextPreprocessor()
    
    # Show before/after example
    sample_text = df.iloc[0]['description'][:200]
    logger.info(f"\nOriginal text (first 200 chars):")
    logger.info(f"  {sample_text}")
    
    preprocessed = preprocessor.preprocess(sample_text)
    logger.info(f"\nPreprocessed text:")
    logger.info(f"  {preprocessed[:200]}")
    
    # Preprocess all texts
    logger.info("\nPreprocessing all texts...")
    df['combined_text'] = df.apply(
        lambda x: preprocessor.combine_fields(
            x['title'], x['description'],
            x.get('input_description', ''),
            x.get('output_description', '')
        ), axis=1
    )
    
    logger.info(f"✓ Preprocessed {len(df)} texts")
    
    # Feature extraction
    logger.info("\n" + "="*80)
    logger.info("FEATURE EXTRACTION")
    logger.info("="*80)
    
    extractor = CombinedFeatureExtractor()
    
    logger.info("Fitting feature extractor...")
    X = extractor.fit_transform(df['combined_text'].values)
    
    logger.info(f"\n✓ Extracted {X.shape[1]} features from {X.shape[0]} samples")
    
    # Feature breakdown
    logger.info("\n" + "="*80)
    logger.info("FEATURE BREAKDOWN")
    logger.info("="*80)
    
    tfidf_features = extractor.tfidf.get_feature_names_out()
    custom_features = extractor.programming_extractor.feature_names_
    
    logger.info(f"\nTF-IDF Features: {len(tfidf_features)}")
    logger.info(f"Custom Features: {len(custom_features)}")
    logger.info(f"Total Features: {len(tfidf_features) + len(custom_features)}")
    
    # Show custom features
    logger.info("\n" + "="*80)
    logger.info("CUSTOM PROGRAMMING FEATURES")
    logger.info("="*80)
    
    logger.info("\nFeature categories:")
    categories = {
        'Text Statistics': ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 'lexical_diversity'],
        'Algorithmic Keywords': [f for f in custom_features if 'keyword' in f],
        'Complexity Indicators': [f for f in custom_features if 'complexity' in f or 'constraint' in f],
        'Mathematical Features': [f for f in custom_features if 'math' in f or 'formula' in f],
        'Structure Indicators': [f for f in custom_features if 'input' in f or 'output' in f or 'test' in f]
    }
    
    for category, features in categories.items():
        matching = [f for f in custom_features if any(kw in f for kw in features)]
        logger.info(f"\n{category}: {len(matching)} features")
        for feat in matching[:5]:  # Show first 5
            logger.info(f"  - {feat}")
        if len(matching) > 5:
            logger.info(f"  ... and {len(matching) - 5} more")
    
    # Sample feature values
    logger.info("\n" + "="*80)
    logger.info("SAMPLE FEATURE VALUES")
    logger.info("="*80)
    
    # Get custom feature values for a sample
    sample_idx = 0
    sample_custom_start = len(tfidf_features)
    sample_custom_values = X[sample_idx, sample_custom_start:].toarray().flatten()
    
    logger.info(f"\nCustom features for sample problem:")
    logger.info(f"Title: {df.iloc[sample_idx]['title'][:50]}...")
    logger.info(f"Difficulty: {df.iloc[sample_idx]['difficulty']}")
    logger.info("\nFeature values:")
    
    for i, (name, value) in enumerate(zip(custom_features, sample_custom_values)):
        if i < 10:  # Show first 10
            logger.info(f"  {name:<30} = {value:.4f}")
    
    logger.info(f"  ... and {len(custom_features) - 10} more features")
    
    # Top TF-IDF terms by difficulty
    logger.info("\n" + "="*80)
    logger.info("TOP TF-IDF TERMS BY DIFFICULTY")
    logger.info("="*80)
    
    for difficulty in ['easy', 'medium', 'hard']:
        diff_mask = df['difficulty'] == difficulty
        if diff_mask.sum() > 0:
            diff_X = X[diff_mask, :len(tfidf_features)]
            mean_tfidf = diff_X.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-10:][::-1]
            top_terms = [tfidf_features[i] for i in top_indices]
            
            logger.info(f"\n{difficulty.upper()} - Top 10 terms:")
            for i, term in enumerate(top_terms, 1):
                logger.info(f"  {i:2d}. {term}")
    
    # Feature statistics
    logger.info("\n" + "="*80)
    logger.info("FEATURE STATISTICS")
    logger.info("="*80)
    
    feature_stats = pd.DataFrame({
        'mean': X.mean(axis=0).A1,
        'std': np.sqrt(X.power(2).mean(axis=0).A1 - np.power(X.mean(axis=0).A1, 2)),
        'max': X.max(axis=0).toarray().flatten()
    })
    
    logger.info(f"\nOverall feature statistics:")
    logger.info(f"  Mean of means: {feature_stats['mean'].mean():.6f}")
    logger.info(f"  Mean of stds:  {feature_stats['std'].mean():.6f}")
    logger.info(f"  Sparsity:      {(X == 0).sum() / X.size:.2%}")
    
    # Save feature names
    logger.info("\n" + "="*80)
    logger.info("SAVING FEATURE INFORMATION")
    logger.info("="*80)
    
    output_dir = Path(__file__).parent.parent / 'logs'
    output_dir.mkdir(exist_ok=True)
    
    # Save all feature names
    all_features = list(tfidf_features) + list(custom_features)
    feature_file = output_dir / 'feature_names.txt'
    with open(feature_file, 'w') as f:
        f.write("# All Features\n")
        f.write(f"# Total: {len(all_features)}\n\n")
        f.write("## TF-IDF Features\n")
        for feat in tfidf_features[:100]:
            f.write(f"{feat}\n")
        f.write(f"... and {len(tfidf_features) - 100} more\n\n")
        f.write("## Custom Programming Features\n")
        for feat in custom_features:
            f.write(f"{feat}\n")
    
    logger.info(f"✓ Feature names saved to: {feature_file}")
    
    # Save feature statistics
    stats_file = output_dir / 'feature_statistics.csv'
    feature_stats['feature_name'] = all_features
    feature_stats.to_csv(stats_file, index=False)
    logger.info(f"✓ Feature statistics saved to: {stats_file}")
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    
    return X, df, extractor


def main():
    """Main function"""
    try:
        X, df, extractor = analyze_features()
        
        logger.info("\n✓ Feature engineering analysis complete!")
        logger.info(f"  - {X.shape[1]} features extracted")
        logger.info(f"  - {X.shape[0]} samples processed")
        logger.info("\nNext steps:")
        logger.info("  1. Review feature breakdown above")
        logger.info("  2. Check logs/ for saved feature information")
        logger.info("  3. Train models: python scripts/train_models.py --full")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
