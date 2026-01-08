#!/usr/bin/env python
"""
Data Exploration Script
Analyzes the dataset and displays statistics, distributions, and insights
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.utils.logger import logger

def explore_dataset():
    """Explore and analyze the dataset"""
    
    logger.info("="*80)
    logger.info("DATA EXPLORATION")
    logger.info("="*80)
    
    # Load data
    loader = DataLoader()
    
    # Try to load TaskComplexity dataset
    df = loader.load_task_complexity_dataset()
    
    if df.empty:
        logger.info("No dataset found. Using sample data for exploration...")
        df = loader.create_sample_dataset(n_samples=500)
    
    logger.info(f"\nLoaded {len(df)} samples")
    
    # Basic info
    logger.info("\n" + "="*80)
    logger.info("DATASET OVERVIEW")
    logger.info("="*80)
    logger.info(f"\nShape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Missing values
    logger.info("\n" + "="*80)
    logger.info("MISSING VALUES")
    logger.info("="*80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    logger.info(f"\n{missing_df[missing_df['Missing Count'] > 0]}")
    
    if missing_df['Missing Count'].sum() == 0:
        logger.info("✓ No missing values found!")
    
    # Difficulty distribution
    logger.info("\n" + "="*80)
    logger.info("DIFFICULTY CLASS DISTRIBUTION")
    logger.info("="*80)
    class_dist = df['difficulty'].value_counts()
    class_pct = (class_dist / len(df)) * 100
    
    for cls in ['easy', 'medium', 'hard']:
        if cls in class_dist.index:
            count = class_dist[cls]
            pct = class_pct[cls]
            logger.info(f"  {cls.capitalize():<10} : {count:>5} ({pct:>5.1f}%)")
    
    # Score statistics
    logger.info("\n" + "="*80)
    logger.info("DIFFICULTY SCORE STATISTICS")
    logger.info("="*80)
    score_stats = df['score'].describe()
    logger.info(f"\n{score_stats}")
    
    # Text length analysis
    logger.info("\n" + "="*80)
    logger.info("TEXT LENGTH ANALYSIS")
    logger.info("="*80)
    
    df['title_length'] = df['title'].str.len()
    df['desc_length'] = df['description'].str.len()
    
    logger.info("\nTitle lengths:")
    logger.info(f"  Mean: {df['title_length'].mean():.1f} characters")
    logger.info(f"  Min:  {df['title_length'].min()}")
    logger.info(f"  Max:  {df['title_length'].max()}")
    
    logger.info("\nDescription lengths:")
    logger.info(f"  Mean: {df['desc_length'].mean():.1f} characters")
    logger.info(f"  Min:  {df['desc_length'].min()}")
    logger.info(f"  Max:  {df['desc_length'].max()}")
    
    # Score by difficulty
    logger.info("\n" + "="*80)
    logger.info("AVERAGE SCORE BY DIFFICULTY")
    logger.info("="*80)
    score_by_diff = df.groupby('difficulty')['score'].agg(['mean', 'std', 'min', 'max'])
    logger.info(f"\n{score_by_diff}")
    
    # Sample problems
    logger.info("\n" + "="*80)
    logger.info("SAMPLE PROBLEMS")
    logger.info("="*80)
    
    for difficulty in ['easy', 'medium', 'hard']:
        sample = df[df['difficulty'] == difficulty].iloc[0] if difficulty in df['difficulty'].values else None
        if sample is not None:
            logger.info(f"\n{difficulty.upper()} Example:")
            logger.info(f"  Title: {sample['title'][:50]}...")
            logger.info(f"  Description: {sample['description'][:100]}...")
            logger.info(f"  Score: {sample['score']}")
    
    # Visualizations (if matplotlib available)
    try:
        logger.info("\n" + "="*80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*80)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Difficulty distribution
        class_dist.plot(kind='bar', ax=axes[0, 0], color=['green', 'orange', 'red'])
        axes[0, 0].set_title('Difficulty Class Distribution')
        axes[0, 0].set_xlabel('Difficulty')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=0)
        
        # 2. Score distribution
        axes[0, 1].hist(df['score'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Difficulty Score Distribution')
        axes[0, 1].set_xlabel('Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(df['score'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 1].legend()
        
        # 3. Score by difficulty (boxplot)
        df.boxplot(column='score', by='difficulty', ax=axes[1, 0])
        axes[1, 0].set_title('Score Distribution by Difficulty')
        axes[1, 0].set_xlabel('Difficulty')
        axes[1, 0].set_ylabel('Score')
        plt.sca(axes[1, 0])
        plt.xticks(rotation=0)
        
        # 4. Description length by difficulty
        df.boxplot(column='desc_length', by='difficulty', ax=axes[1, 1])
        axes[1, 1].set_title('Description Length by Difficulty')
        axes[1, 1].set_xlabel('Difficulty')
        axes[1, 1].set_ylabel('Length (characters)')
        plt.sca(axes[1, 1])
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path(__file__).parent.parent / 'logs' / 'data_exploration.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Visualizations saved to: {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")
    
    # Validation
    logger.info("\n" + "="*80)
    logger.info("DATA VALIDATION")
    logger.info("="*80)
    
    is_valid, errors = loader.validate_dataset(df)
    if is_valid:
        logger.info("✓ Dataset is valid!")
    else:
        logger.error("✗ Dataset has validation errors:")
        for error in errors:
            logger.error(f"  - {error}")
    
    logger.info("\n" + "="*80)
    logger.info("EXPLORATION COMPLETE")
    logger.info("="*80)
    
    return df


def main():
    """Main function"""
    try:
        df = explore_dataset()
        logger.info(f"\n✓ Successfully explored {len(df)} problems")
        logger.info("\nNext steps:")
        logger.info("  1. Review the statistics above")
        logger.info("  2. Check logs/data_exploration.png for visualizations")
        logger.info("  3. Run training: python scripts/train_models.py --full")
        
    except Exception as e:
        logger.error(f"Error during exploration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
