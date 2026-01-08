#!/usr/bin/env python
"""
Model Evaluation Analysis Script
Loads trained models and performs comprehensive evaluation
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import TextPreprocessor
from src.utils.logger import logger


def load_models():
    """Load trained models"""
    models_dir = Path(__file__).parent.parent / 'models'
    
    try:
        classifier = joblib.load(models_dir / 'classifier.pkl')
        regressor = joblib.load(models_dir / 'regressor.pkl')
        feature_extractor = joblib.load(models_dir / 'feature_extractor.pkl')
        scaler = joblib.load(models_dir / 'scaler.pkl')
        
        logger.info("✓ Models loaded successfully")
        return classifier, regressor, feature_extractor, scaler
    
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        logger.error("Please train models first: python scripts/train_models.py --full")
        sys.exit(1)


def evaluate_models():
    """Evaluate trained models"""
    
    logger.info("="*80)
    logger.info("MODEL EVALUATION ANALYSIS")
    logger.info("="*80)
    
    # Load models
    classifier, regressor, feature_extractor, scaler = load_models()
    
    # Load data
    loader = DataLoader()
    df = loader.load_task_complexity_dataset()
    
    if df.empty:
        logger.warning("No dataset found. Cannot evaluate without data.")
        return
    
    # Split data (same as training)
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['difficulty']
    )
    
    logger.info(f"\nDataset split:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Test:  {len(test_df)} samples")
    
    # Preprocess test data
    preprocessor = TextPreprocessor()
    
    test_df['combined_text'] = test_df.apply(
        lambda x: preprocessor.combine_fields(
            x['title'], x['description'],
            x.get('input_description', ''),
            x.get('output_description', '')
        ), axis=1
    )
    
    # Extract features
    X_test = feature_extractor.transform(test_df['combined_text'].values)
    X_test_scaled = scaler.transform(X_test)
    
    y_test_class = test_df['difficulty'].values
    y_test_score = test_df['score'].values
    
    # Classification evaluation
    logger.info("\n" + "="*80)
    logger.info("CLASSIFICATION EVALUATION")
    logger.info("="*80)
    
    y_pred_class = classifier.predict(X_test_scaled)
    y_pred_proba = classifier.predict_proba(X_test_scaled)
    
    # Accuracy
    accuracy = (y_pred_class == y_test_class).mean()
    logger.info(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test_class, y_pred_class))
    
    # Confusion matrix
    logger.info("Confusion Matrix:")
    cm = confusion_matrix(y_test_class, y_pred_class, labels=['easy', 'medium', 'hard'])
    cm_df = pd.DataFrame(
        cm,
        index=['True Easy', 'True Medium', 'True Hard'],
        columns=['Pred Easy', 'Pred Medium', 'Pred Hard']
    )
    logger.info(f"\n{cm_df}")
    
    # Per-class accuracy
    logger.info("\nPer-class Accuracy:")
    for i, label in enumerate(['easy', 'medium', 'hard']):
        class_acc = cm[i, i] / cm[i].sum()
        logger.info(f"  {label.capitalize():<10}: {class_acc:.4f} ({class_acc*100:.1f}%)")
    
    # Regression evaluation
    logger.info("\n" + "="*80)
    logger.info("REGRESSION EVALUATION")
    logger.info("="*80)
    
    y_pred_score = regressor.predict(X_test_scaled)
    
    # Metrics
    mae = mean_absolute_error(y_test_score, y_pred_score)
    rmse = np.sqrt(mean_squared_error(y_test_score, y_pred_score))
    r2 = r2_score(y_test_score, y_pred_score)
    mape = np.mean(np.abs((y_test_score - y_pred_score) / y_test_score)) * 100
    
    logger.info(f"\nRegression Metrics:")
    logger.info(f"  MAE:   {mae:.4f}")
    logger.info(f"  RMSE:  {rmse:.4f}")
    logger.info(f"  R²:    {r2:.4f}")
    logger.info(f"  MAPE:  {mape:.2f}%")
    
    # Error analysis
    errors = y_pred_score - y_test_score
    logger.info(f"\nError Statistics:")
    logger.info(f"  Mean Error:     {errors.mean():.4f}")
    logger.info(f"  Std Error:      {errors.std():.4f}")
    logger.info(f"  Min Error:      {errors.min():.4f}")
    logger.info(f"  Max Error:      {errors.max():.4f}")
    logger.info(f"  Median Error:   {np.median(errors):.4f}")
    
    # Predictions within tolerance
    within_05 = (np.abs(errors) <= 0.5).sum() / len(errors)
    within_10 = (np.abs(errors) <= 1.0).sum() / len(errors)
    within_15 = (np.abs(errors) <= 1.5).sum() / len(errors)
    
    logger.info(f"\nPrediction Accuracy:")
    logger.info(f"  Within ±0.5: {within_05:.2%}")
    logger.info(f"  Within ±1.0: {within_10:.2%}")
    logger.info(f"  Within ±1.5: {within_15:.2%}")
    
    # Sample predictions
    logger.info("\n" + "="*80)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("="*80)
    
    sample_indices = [0, len(test_df)//3, 2*len(test_df)//3]
    
    for idx in sample_indices:
        sample = test_df.iloc[idx]
        pred_class = y_pred_class[idx]
        true_class = y_test_class[idx]
        pred_score = y_pred_score[idx]
        true_score = y_test_score[idx]
        proba = y_pred_proba[idx]
        
        logger.info(f"\nSample {idx + 1}:")
        logger.info(f"  Title: {sample['title'][:60]}...")
        logger.info(f"  True:  {true_class.upper()} (score: {true_score:.2f})")
        logger.info(f"  Pred:  {pred_class.upper()} (score: {pred_score:.2f})")
        logger.info(f"  Confidence: Easy={proba[0]:.2%}, Medium={proba[1]:.2%}, Hard={proba[2]:.2%}")
        logger.info(f"  Score Error: {abs(pred_score - true_score):.2f}")
    
    # Visualizations
    logger.info("\n" + "="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Easy', 'Medium', 'Hard'],
                   yticklabels=['Easy', 'Medium', 'Hard'])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Predicted vs Actual scores
        axes[0, 1].scatter(y_test_score, y_pred_score, alpha=0.5)
        axes[0, 1].plot([0, 10], [0, 10], 'r--', label='Perfect Prediction')
        axes[0, 1].set_title('Predicted vs Actual Scores')
        axes[0, 1].set_xlabel('Actual Score')
        axes[0, 1].set_ylabel('Predicted Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prediction errors distribution
        axes[1, 0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(0, color='red', linestyle='--', label='Zero Error')
        axes[1, 0].set_title('Prediction Error Distribution')
        axes[1, 0].set_xlabel('Prediction Error (Predicted - Actual)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. Error by difficulty class
        error_by_class = pd.DataFrame({
            'error': np.abs(errors),
            'difficulty': y_test_class
        })
        error_by_class.boxplot(column='error', by='difficulty', ax=axes[1, 1])
        axes[1, 1].set_title('Absolute Error by Difficulty')
        axes[1, 1].set_xlabel('Difficulty')
        axes[1, 1].set_ylabel('Absolute Error')
        plt.sca(axes[1, 1])
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path(__file__).parent.parent / 'logs' / 'evaluation_results.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Visualizations saved to: {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")
    
    # Save detailed results
    results_file = Path(__file__).parent.parent / 'logs' / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("CLASSIFICATION METRICS:\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm_df) + "\n\n")
        
        f.write("REGRESSION METRICS:\n")
        f.write(f"  MAE:  {mae:.4f}\n")
        f.write(f"  RMSE: {rmse:.4f}\n")
        f.write(f"  R²:   {r2:.4f}\n")
        f.write(f"  MAPE: {mape:.2f}%\n\n")
        
        f.write("PREDICTION ACCURACY:\n")
        f.write(f"  Within ±0.5: {within_05:.2%}\n")
        f.write(f"  Within ±1.0: {within_10:.2%}\n")
        f.write(f"  Within ±1.5: {within_15:.2%}\n")
    
    logger.info(f"✓ Results saved to: {results_file}")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    
    return {
        'accuracy': accuracy,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def main():
    """Main function"""
    try:
        results = evaluate_models()
        
        if results:
            logger.info("\n✓ Model evaluation complete!")
            logger.info(f"  Classification Accuracy: {results['accuracy']:.2%}")
            logger.info(f"  Regression RMSE: {results['rmse']:.4f}")
            logger.info(f"  Regression R²: {results['r2']:.4f}")
            logger.info("\nCheck logs/ for detailed results and visualizations")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
