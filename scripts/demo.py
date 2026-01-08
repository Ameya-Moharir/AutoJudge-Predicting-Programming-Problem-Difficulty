#!/usr/bin/env python
"""
AutoJudge Demo Script
Demonstrates the complete workflow from data loading to prediction
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import TextPreprocessor
from src.data.feature_engineer import CombinedFeatureExtractor
from src.models.classifier import DifficultyClassifier
from src.models.regressor import DifficultyScoreRegressor
import joblib


def demo_preprocessing():
    """Demonstrate text preprocessing"""
    print("=" * 80)
    print("DEMONSTRATION: Text Preprocessing")
    print("=" * 80)
    
    preprocessor = TextPreprocessor()
    
    # Example problem
    title = "Binary Search Tree"
    description = """
    Given a binary search tree, implement the following operations:
    1. Insert a new node
    2. Delete a node
    3. Find the kth smallest element
    The tree may have up to 10^5 nodes.
    """
    input_desc = "Number of operations n, followed by n operations"
    output_desc = "Result of each query operation"
    
    # Combine fields
    combined = preprocessor.combine_fields(
        title=title,
        description=description,
        input_desc=input_desc,
        output_desc=output_desc
    )
    
    print("\nOriginal combined text (first 200 chars):")
    print(combined[:200] + "...")
    
    # Preprocess
    processed = preprocessor.preprocess(combined)
    
    print("\nProcessed text (first 200 chars):")
    print(processed[:200] + "...")
    
    print("\n✓ Preprocessing complete")


def demo_feature_extraction():
    """Demonstrate feature extraction"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Feature Extraction")
    print("=" * 80)
    
    preprocessor = TextPreprocessor()
    
    texts = [
        "Simple addition problem: Given two numbers, output their sum",
        "Find shortest path in weighted graph using Dijkstra algorithm with binary heap optimization",
        "Implement segment tree to handle range minimum queries and point updates efficiently"
    ]
    
    processed_texts = preprocessor.batch_preprocess(texts)
    
    # Extract features
    extractor = CombinedFeatureExtractor(max_features=100, ngram_range=(1, 2))
    features = extractor.fit_transform(processed_texts)
    
    print(f"\nExtracted {features.shape[1]} features from {features.shape[0]} texts")
    print(f"Feature matrix shape: {features.shape}")
    
    # Show some feature names
    feature_names = extractor.get_feature_names()
    print(f"\nSample features (first 10):")
    for i, name in enumerate(feature_names[:10]):
        print(f"  {i+1}. {name}")
    
    print("\n✓ Feature extraction complete")


def demo_prediction():
    """Demonstrate prediction with trained models"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Prediction (requires trained models)")
    print("=" * 80)
    
    model_dir = Path(__file__).parent.parent / 'models'
    
    # Check if models exist
    required_files = ['classifier.pkl', 'regressor.pkl', 'feature_extractor.pkl', 'scaler.pkl']
    missing = [f for f in required_files if not (model_dir / f).exists()]
    
    if missing:
        print("\n⚠️  Models not found. Please train models first:")
        print("    python scripts/train_models.py --quick")
        return
    
    # Load models
    print("\nLoading models...")
    classifier = joblib.load(model_dir / 'classifier.pkl')
    regressor = joblib.load(model_dir / 'regressor.pkl')
    extractor = joblib.load(model_dir / 'feature_extractor.pkl')
    scaler = joblib.load(model_dir / 'scaler.pkl')
    
    preprocessor = TextPreprocessor()
    
    # Test problems
    test_problems = [
        {
            'title': 'Simple Sum',
            'description': 'Given two integers A and B, output their sum.',
            'expected': 'easy'
        },
        {
            'title': 'Longest Common Subsequence',
            'description': 'Find the longest common subsequence of two strings using dynamic programming. The strings can have up to 1000 characters.',
            'expected': 'medium'
        },
        {
            'title': 'Maximum Flow',
            'description': 'Implement maximum flow algorithm using Dinic\'s algorithm with BFS. The graph can have up to 10^5 nodes and 10^6 edges.',
            'expected': 'hard'
        }
    ]
    
    print("\nMaking predictions...\n")
    
    for i, problem in enumerate(test_problems, 1):
        print(f"Problem {i}: {problem['title']}")
        print(f"Expected: {problem['expected'].upper()}")
        
        # Preprocess
        combined = preprocessor.combine_fields(
            title=problem['title'],
            description=problem['description']
        )
        processed = preprocessor.preprocess(combined)
        
        # Extract features
        features = extractor.transform([processed])
        features_scaled = scaler.transform(features)
        
        # Predict
        pred_class = classifier.predict(features_scaled)[0]
        pred_proba = classifier.predict_proba(features_scaled)[0]
        pred_score = regressor.predict(features_scaled)[0]
        
        print(f"Predicted: {pred_class.upper()}")
        print(f"Score: {pred_score:.2f}/10")
        print(f"Confidence: {max(pred_proba)*100:.1f}%")
        print(f"  Easy: {pred_proba[0]*100:.1f}%")
        print(f"  Medium: {pred_proba[1]*100:.1f}%")
        print(f"  Hard: {pred_proba[2]*100:.1f}%")
        
        # Check if correct
        correct = "✓" if pred_class == problem['expected'] else "✗"
        print(f"Result: {correct}")
        print()
    
    print("✓ Predictions complete")


def demo_dataset_loading():
    """Demonstrate dataset loading"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Dataset Loading")
    print("=" * 80)
    
    loader = DataLoader()
    
    # Create sample dataset
    print("\nCreating sample dataset...")
    df = loader.create_sample_dataset(n_samples=30)
    
    print(f"\nDataset info:")
    print(f"  Total samples: {len(df)}")
    print(f"\n  Difficulty distribution:")
    print(df['difficulty'].value_counts().to_string())
    
    print(f"\n  Score statistics:")
    print(df['score'].describe().to_string())
    
    # Validate
    is_valid, errors = loader.validate_dataset(df)
    print(f"\n  Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if not is_valid:
        for error in errors:
            print(f"    - {error}")
    
    print("\n✓ Dataset loading complete")


def main():
    """Run all demonstrations"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "AutoJudge Demo Script" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        demo_dataset_loading()
        demo_preprocessing()
        demo_feature_extraction()
        demo_prediction()
        
        print("\n" + "=" * 80)
        print("All demonstrations completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Train models: python scripts/train_models.py --quick")
        print("  2. Start web app: python web/app.py")
        print("  3. Open browser: http://localhost:5000")
        print()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
