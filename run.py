#!/usr/bin/env python3
"""
SMS Spam Filter - Complete Training Pipeline
============================================

This script handles the complete training pipeline for the SMS spam filter:
1. Data loading and validation
2. Text preprocessing
3. Model training and comparison
4. Evaluation and visualization
5. Model saving for deployment

Usage:
    python run.py [--data-path path/to/data.csv] [--model-type naive_bayes]

Author: Your Name
Date: 2024
"""

import os
import sys
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
try:
    from data_loader import DataLoader
    from preprocessor import TextPreprocessor
    from model_trainer import SpamClassifierTrainer
    from predictor import SpamPredictor
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure all modules are in the 'src/' directory")
    sys.exit(1)

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    SMS SPAM FILTER TRAINER                   ║
    ║                                                              ║
    ║  🤖 AI-Powered SMS Spam Detection System                    ║
    ║  📊 Complete Training & Evaluation Pipeline                 ║
    ║  🚀 Production-Ready Model Generation                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'results', 'logs', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created/verified directory: {directory}")

def load_and_explore_data(data_path, logger):
    """Load data and perform initial exploration"""
    logger.info(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        print(f"\n❌ Data file not found: {data_path}")
        print("Please ensure your data file exists with columns: 'label', 'message'")
        print("Example data format:")
        print("label,message")
        print("ham,Hey are you free for lunch?")
        print("spam,Congratulations! You won $1000!")
        return None, None
    
    try:
        # Load data
        loader = DataLoader(data_path)
        data = loader.load_data()
        data_info = loader.get_data_info()
        
        # Print data info
        print(f"\n📊 DATASET OVERVIEW")
        print("=" * 50)
        print(f"Total samples: {data_info['total_samples']:,}")
        print(f"Spam messages: {data_info['spam_count']:,} ({data_info['spam_percentage']:.1f}%)")
        print(f"Ham messages: {data_info['ham_count']:,} ({100-data_info['spam_percentage']:.1f}%)")
        
        # Check for class imbalance
        if data_info['spam_percentage'] < 10 or data_info['spam_percentage'] > 90:
            print("⚠️  Warning: Significant class imbalance detected!")
            
        # Show sample messages
        print(f"\n📝 SAMPLE MESSAGES")
        print("=" * 50)
        
        spam_sample = data[data['label'] == 'spam'].sample(min(2, data_info['spam_count']))
        ham_sample = data[data['label'] == 'ham'].sample(min(2, data_info['ham_count']))
        
        print("🚨 SPAM Examples:")
        for idx, row in spam_sample.iterrows():
            print(f"   '{row['message'][:70]}...'" if len(row['message']) > 70 else f"   '{row['message']}'")
            
        print("\n✅ HAM Examples:")
        for idx, row in ham_sample.iterrows():
            print(f"   '{row['message'][:70]}...'" if len(row['message']) > 70 else f"   '{row['message']}'")
        
        return loader, data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        print(f"❌ Error loading data: {e}")
        return None, None

def analyze_text_characteristics(data, logger):
    """Analyze text characteristics of the dataset"""
    print(f"\n📈 TEXT ANALYSIS")
    print("=" * 50)
    
    # Message length analysis
    data['message_length'] = data['message'].str.len()
    data['word_count'] = data['message'].str.split().str.len()
    
    # Group by label
    stats_by_label = data.groupby('label').agg({
        'message_length': ['mean', 'std', 'min', 'max'],
        'word_count': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    print("Message Statistics by Type:")
    print(stats_by_label)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Message length distribution
    data[data['label'] == 'ham']['message_length'].hist(bins=50, alpha=0.7, label='Ham', ax=axes[0,0], color='green')
    data[data['label'] == 'spam']['message_length'].hist(bins=50, alpha=0.7, label='Spam', ax=axes[0,0], color='red')
    axes[0,0].set_xlabel('Message Length (characters)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Message Length Distribution')
    axes[0,0].legend()
    
    # Word count distribution
    data[data['label'] == 'ham']['word_count'].hist(bins=30, alpha=0.7, label='Ham', ax=axes[0,1], color='green')
    data[data['label'] == 'spam']['word_count'].hist(bins=30, alpha=0.7, label='Spam', ax=axes[0,1], color='red')
    axes[0,1].set_xlabel('Word Count')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Word Count Distribution')
    axes[0,1].legend()
    
    # Box plots
    data.boxplot(column='message_length', by='label', ax=axes[1,0])
    axes[1,0].set_title('Message Length by Type')
    axes[1,0].set_xlabel('Message Type')
    axes[1,0].set_ylabel('Message Length')
    
    data.boxplot(column='word_count', by='label', ax=axes[1,1])
    axes[1,1].set_title('Word Count by Type')
    axes[1,1].set_xlabel('Message Type')
    axes[1,1].set_ylabel('Word Count')
    
    plt.tight_layout()
    plt.savefig('results/text_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Text analysis completed and saved")

def generate_wordclouds(data, logger):
    """Generate word clouds for spam and ham messages"""
    print(f"\n🎨 GENERATING WORD CLOUDS")
    print("=" * 50)
    
    try:
        # Separate messages
        spam_text = ' '.join(data[data['label'] == 'spam']['message'])
        ham_text = ' '.join(data[data['label'] == 'ham']['message'])
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Spam word cloud
        spam_wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Reds',
            max_words=100,
            relative_scaling=0.5,
            stopwords=set(['will', 'now', 'get', 'go', 'call', 'text', 'mobile'])
        ).generate(spam_text)
        
        axes[0].imshow(spam_wordcloud, interpolation='bilinear')
        axes[0].set_title('SPAM Messages - Most Common Words', fontsize=16, fontweight='bold', color='darkred')
        axes[0].axis('off')
        
        # Ham word cloud
        ham_wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Greens',
            max_words=100,
            relative_scaling=0.5,
            stopwords=set(['will', 'now', 'get', 'go', 'call', 'text', 'mobile'])
        ).generate(ham_text)
        
        axes[1].imshow(ham_wordcloud, interpolation='bilinear')
        axes[1].set_title('HAM Messages - Most Common Words', fontsize=16, fontweight='bold', color='darkgreen')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Word clouds generated and saved")
        logger.info("Word clouds generated successfully")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not generate word clouds: {e}")
        logger.warning(f"Word cloud generation failed: {e}")

def train_and_compare_models(X_train, X_test, y_train, y_test, preprocessor, logger):
    """Train multiple models and compare their performance"""
    print(f"\n🤖 MODEL TRAINING & COMPARISON")
    print("=" * 50)
    
    # Models to compare
    models_config = {
        'naive_bayes': {
            'name': 'Naive Bayes',
            'description': 'Probabilistic classifier based on Bayes theorem'
        },
        'logistic_regression': {
            'name': 'Logistic Regression',
            'description': 'Linear classifier with logistic function'
        },
        'random_forest': {
            'name': 'Random Forest',
            'description': 'Ensemble of decision trees'
        }
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    # Train each model
    for model_type, config in models_config.items():
        print(f"\n🔧 Training {config['name']}...")
        print(f"   Description: {config['description']}")
        
        try:
            # Initialize trainer
            trainer = SpamClassifierTrainer(model_type=model_type)
            trainer.create_pipeline(preprocessor, max_features=5000)
            
            # Train model
            trainer.train(X_train, y_train)
            
            # Evaluate model
            metrics = trainer.evaluate(X_test, y_test, verbose=False)
            results[model_type] = {
                'trainer': trainer,
                'metrics': metrics,
                'config': config
            }
            
            print(f"   ✅ {config['name']} Results:")
            print(f"      Accuracy:  {metrics['accuracy']:.4f}")
            print(f"      Precision: {metrics['precision']:.4f}")
            print(f"      Recall:    {metrics['recall']:.4f}")
            print(f"      F1-Score:  {metrics['f1_score']:.4f}")
            
            # Generate visualizations
            trainer.plot_confusion_matrix(
                metrics['y_true'],
                metrics['y_pred'],
                save_path=f"results/{model_type}_confusion_matrix.png"
            )
            
            trainer.plot_feature_importance(
                top_n=15,
                save_path=f"results/{model_type}_feature_importance.png"
            )
            
            # Track best model (using F1-score as primary metric)
            if metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                best_model = trainer
                best_model_type = model_type
                
        except Exception as e:
            print(f"   ❌ Error training {config['name']}: {e}")
            logger.error(f"Error training {model_type}: {e}")
            continue
    
    # Display comparison
    print(f"\n📊 MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<10} {'Status'}")
    print("-" * 80)
    
    for model_type, result in results.items():
        metrics = result['metrics']
        status = "🏆 BEST" if model_type == best_model_type else ""
        print(f"{result['config']['name']:<20} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<11.4f} "
              f"{metrics['recall']:<9.4f} "
              f"{metrics['f1_score']:<10.4f} "
              f"{status}")
    
    return results, best_model, best_model_type

def create_comparison_visualizations(results, logger):
    """Create comparison visualizations for all models"""
    print(f"\n📈 CREATING COMPARISON VISUALIZATIONS")
    print("=" * 50)
    
    try:
        # Prepare data for visualization
        model_names = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for model_type, result in results.items():
            metrics = result['metrics']
            model_names.append(result['config']['name'])
            accuracy_scores.append(metrics['accuracy'])
            precision_scores.append(metrics['precision'])
            recall_scores.append(metrics['recall'])
            f1_scores.append(metrics['f1_score'])
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bar plot of all metrics
        x = np.arange(len(model_names))
        width = 0.2
        
        axes[0,0].bar(x - 1.5*width, accuracy_scores, width, label='Accuracy', color='skyblue')
        axes[0,0].bar(x - 0.5*width, precision_scores, width, label='Precision', color='lightcoral')
        axes[0,0].bar(x + 0.5*width, recall_scores, width, label='Recall', color='lightgreen')
        axes[0,0].bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='gold')
        
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_title('Model Performance Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(model_names)
        axes[0,0].legend()
        axes[0,0].set_ylim(0, 1.1)
        
        # F1-Score comparison (most important)
        colors = ['gold' if score == max(f1_scores) else 'lightgray' for score in f1_scores]
        bars = axes[0,1].bar(model_names, f1_scores, color=colors)
        axes[0,1].set_ylabel('F1-Score')
        axes[0,1].set_title('F1-Score Comparison (Higher is Better)')
        axes[0,1].set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Precision vs Recall scatter
        axes[1,0].scatter(recall_scores, precision_scores, s=200, alpha=0.7, c=['red', 'blue', 'green'])
        for i, name in enumerate(model_names):
            axes[1,0].annotate(name, (recall_scores[i], precision_scores[i]), 
                             xytext=(5, 5), textcoords='offset points')
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision vs Recall')
        axes[1,0].grid(True, alpha=0.3)
        
        # Performance radar chart (if more than 2 models)
        if len(model_names) >= 2:
            from math import pi
            
            categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            N = len(categories)
            
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            axes[1,1].set_theta_offset(pi / 2)
            axes[1,1].set_theta_direction(-1)
            axes[1,1].set_thetagrids(np.degrees(angles[:-1]), categories)
            
            for i, (model_type, result) in enumerate(results.items()):
                metrics = result['metrics']
                values = [metrics['accuracy'], metrics['precision'], 
                         metrics['recall'], metrics['f1_score']]
                values += values[:1]
                
                color = ['red', 'blue', 'green'][i % 3]
                axes[1,1].plot(angles, values, 'o-', linewidth=2, 
                              label=result['config']['name'], color=color)
                axes[1,1].fill(angles, values, alpha=0.25, color=color)
            
            axes[1,1].set_ylim(0, 1)
            axes[1,1].set_title('Model Performance Radar Chart')
            axes[1,1].legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        else:
            axes[1,1].text(0.5, 0.5, 'Radar chart needs\n≥2 models', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comparison visualizations created and saved")
        logger.info("Model comparison visualizations generated")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not create comparison visualizations: {e}")
        logger.warning(f"Comparison visualization failed: {e}")

def test_model_predictions(model_path, logger):
    """Test the saved model with example predictions"""
    print(f"\n🧪 TESTING SAVED MODEL")
    print("=" * 50)
    
    try:
        # Load the saved model
        predictor = SpamPredictor(model_path)
        
        # Test messages
        test_messages = [
            ("Free money! Click here now! Limited time offer!", "spam"),
            ("Hey, are you free for lunch tomorrow?", "ham"),
            ("Congratulations! You've won $1000! Claim now!", "spam"),
            ("Can you pick up milk on your way home?", "ham"),
            ("URGENT: Your account will be suspended! Call now!", "spam"),
            ("Thanks for helping me move today", "ham"),
            ("Win big at casino! No deposit required!", "spam"),
            ("Meeting moved to 3pm in conference room B", "ham")
        ]
        
        print("🔍 Testing model with example messages:")
        print("-" * 60)
        
        correct_predictions = 0
        total_predictions = len(test_messages)
        
        for message, expected in test_messages:
            prediction = predictor.predict(message)
            probabilities = predictor.predict_proba(message)
            confidence = max(probabilities.values())
            
            # Check if prediction matches expected
            is_correct = prediction == expected
            correct_predictions += is_correct
            
            status = "✅" if is_correct else "❌"
            
            print(f"{status} Message: '{message[:50]}{'...' if len(message) > 50 else ''}'")
            print(f"   Expected: {expected.upper()}, Predicted: {prediction.upper()}, "
                  f"Confidence: {confidence:.3f}")
            print()
        
        accuracy = correct_predictions / total_predictions
        print(f"📊 Test Results: {correct_predictions}/{total_predictions} correct "
              f"(Accuracy: {accuracy:.1%})")
        
        logger.info(f"Model testing completed. Accuracy on test examples: {accuracy:.3f}")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        logger.error(f"Model testing failed: {e}")

def save_training_report(results, best_model_type, data_info, logger):
    """Save a comprehensive training report"""
    print(f"\n📋 GENERATING TRAINING REPORT")
    print("=" * 50)
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = f"results/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# SMS Spam Filter - Training Report\n\n")
            f.write(f"**Generated:** {timestamp}\n\n")
            
            f.write(f"## Dataset Overview\n")
            f.write(f"- Total samples: {data_info['total_samples']:,}\n")
            f.write(f"- Spam messages: {data_info['spam_count']:,} ({data_info['spam_percentage']:.1f}%)\n")
            f.write(f"- Ham messages: {data_info['ham_count']:,} ({100-data_info['spam_percentage']:.1f}%)\n\n")
            
            f.write(f"## Model Performance\n\n")
            f.write(f"| Model | Accuracy | Precision | Recall | F1-Score |\n")
            f.write(f"|-------|----------|-----------|--------|----------|\n")
            
            for model_type, result in results.items():
                metrics = result['metrics']
                marker = " 🏆" if model_type == best_model_type else ""
                f.write(f"| {result['config']['name']}{marker} | "
                       f"{metrics['accuracy']:.4f} | "
                       f"{metrics['precision']:.4f} | "
                       f"{metrics['recall']:.4f} | "
                       f"{metrics['f1_score']:.4f} |\n")
            
            f.write(f"\n## Best Model\n")
            f.write(f"**Selected Model:** {results[best_model_type]['config']['name']}\n")
            f.write(f"**F1-Score:** {results[best_model_type]['metrics']['f1_score']:.4f}\n\n")
            
            f.write(f"## Files Generated\n")
            f.write(f"- Model: `models/spam_classifier.pkl`\n")
            f.write(f"- Visualizations: `results/` directory\n")
            f.write(f"- Logs: `logs/` directory\n\n")
            
            f.write(f"## Next Steps\n")
            f.write(f"1. Deploy model using Flask app: `python app/app.py`\n")
            f.write(f"2. Test API endpoints with example messages\n")
            f.write(f"3. Monitor model performance in production\n")
            f.write(f"4. Retrain with new data as needed\n")
        
        print(f"✅ Training report saved: {report_path}")
        logger.info(f"Training report generated: {report_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save training report: {e}")
        logger.warning(f"Training report generation failed: {e}")

def print_deployment_instructions(model_path, best_model_type, logger):
    """Print deployment instructions"""
    print(f"\n🚀 DEPLOYMENT READY!")
    print("=" * 60)
    print(f"✅ Model training completed successfully!")
    print(f"📁 Model saved at: {model_path}")
    print(f"🏆 Best performing model: {best_model_type}")
    
    print(f"\n📋 NEXT STEPS:")
    print("1. 🌐 Deploy using Flask web app:")
    print("   python app/app.py")
    print()
    print("2. 📱 Test the API:")
    print("   curl -X POST http://localhost:5000/predict \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"message\": \"Free money! Click now!\"}'")
    print()
    print("3. 🔍 Monitor and evaluate:")
    print("   - Check model performance regularly")
    print("   - Collect feedback on predictions")
    print("   - Retrain with new data when needed")
    print()
    print("4. 📊 Review training artifacts:")
    print("   - Confusion matrices: results/*_confusion_matrix.png")
    print("   - Feature importance: results/*_feature_importance.png")
    print("   - Training logs: logs/training_*.log")
    print("   - Full report: results/training_report_*.md")

def main(args):
    """Main training pipeline"""
    # Setup
    print_banner()
    logger = setup_logging()
    create_directories()
    
    logger.info("Starting SMS Spam Filter training pipeline")
    logger.info(f"Configuration: data_path={args.data_path}, model_type={args.model_type}")
    
    try:
        # Step 1: Load and explore data
        print(f"\n🔄 STEP 1: DATA LOADING & EXPLORATION")
        print("=" * 60)
        
        loader, data = load_and_explore_data(args.data_path, logger)
        if loader is None:
            return 1
        
        data_info = loader.get_data_info()
        
        # Step 2: Text analysis (optional)
        if not args.skip_analysis:
            analyze_text_characteristics(data, logger)
            generate_wordclouds(data, logger)
        
        # Step 3: Prepare data
        print(f"\n🔄 STEP 2: DATA PREPARATION")
        print("=" * 60)
        
        X_train, X_test, y_train, y_test = loader.get_train_test_split(
            test_size=args.test_size, random_state=42
        )
        
        print(f"✅ Data split completed:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Testing samples: {len(X_test):,}")
        print(f"   Test size ratio: {args.test_size:.1%}")
        
        # Step 4: Initialize preprocessor
        print(f"\n🔄 STEP 3: TEXT PREPROCESSING SETUP")
        print("=" * 60)
        
        preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
        print("✅ Text preprocessor initialized with:")
        print("   - Lowercasing")
        print("   - URL/Email removal")
        print("   - Tokenization") 
        print("   - Stopword removal")
        print("   - Stemming")
        
        # Step 5: Model training
        print(f"\n🔄 STEP 4: MODEL TRAINING & EVALUATION")
        print("=" * 60)
        
        if args.model_type == 'auto':
            # Train and compare all models
            results, best_model, best_model_type = train_and_compare_models(
                X_train, X_test, y_train, y_test, preprocessor, logger
            )
            
            if not results:
                print("❌ No models were trained successfully")
                return 1
                
            # Create comparison visualizations
            create_comparison_visualizations(results, logger)
            
        else:
            # Train specific model
            print(f"Training single model: {args.model_type}")
            trainer = SpamClassifierTrainer(model_type=args.model_type)
            trainer.create_pipeline(preprocessor, max_features=args.max_features)
            trainer.train(X_train, y_train)
            
            metrics = trainer.evaluate(X_test, y_test, verbose=True)
            
            # Generate visualizations for single model
            trainer.plot_confusion_matrix(
                metrics['y_true'], metrics['y_pred'],
                save_path=f"results/{args.model_type}_confusion_matrix.png"
            )
            trainer.plot_feature_importance(
                top_n=15, save_path=f"results/{args.model_type}_feature_importance.png"
            )
            
            best_model = trainer
            best_model_type = args.model_type
            results = {
                args.model_type: {
                    'trainer': trainer, 
                    'metrics': metrics,
                    'config': {'name': args.model_type.replace('_', ' ').title()}
                }
            }
        
        # Step 6: Save best model
        print(f"\n🔄 STEP 5: MODEL SAVING")
        print("=" * 60)
        
        model_path = "models/spam_classifier.pkl"
        best_model.save_model(model_path)
        
        print(f"✅ Best model saved successfully!")
        print(f"   Model type: {best_model_type}")
        print(f"   Model path: {model_path}")
        print(f"   F1-Score: {results[best_model_type]['metrics']['f1_score']:.4f}")
        
        # Step 7: Test saved model
        test_model_predictions(model_path, logger)
        
        # Step 8: Generate report
        save_training_report(results, best_model_type, data_info, logger)
        
        # Step 9: Print deployment instructions
        print_deployment_instructions(model_path, best_model_type, logger)
        
        # Final success message
        print(f"\n🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"⏱️  Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏆 Best model: {results[best_model_type]['config']['name']}")
        print(f"📊 Final F1-Score: {results[best_model_type]['metrics']['f1_score']:.4f}")
        print(f"📁 All outputs saved in: models/, results/, and logs/ directories")
        
        logger.info("SMS Spam Filter training pipeline completed successfully")
        logger.info(f"Best model: {best_model_type} with F1-score: {results[best_model_type]['metrics']['f1_score']:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user")
        logger.warning("Training pipeline interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        logger.error(f"Critical error in training pipeline: {e}", exc_info=True)
        print("\n🔍 Please check:")
        print("1. Data file exists and has correct format")
        print("2. All required modules are installed")
        print("3. All custom modules are in the 'src/' directory")
        print("4. Sufficient disk space for model and results")
        print("5. Check the log file for detailed error information")
        return 1

def validate_environment():
    """Validate the environment and dependencies"""
    print(f"\n🔍 ENVIRONMENT VALIDATION")
    print("=" * 50)
    
    # Package mapping: display_name -> actual_import_name
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'scikit-learn': 'sklearn',  # scikit-learn is imported as sklearn
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'wordcloud': 'wordcloud',
        'nltk': 'nltk'
    }
    
    missing_packages = []
    
    for display_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - MISSING")
            missing_packages.append(display_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages detected!")
        print(f"Please install missing packages:")
        for pkg in missing_packages:
            if pkg == 'scikit-learn':
                print(f"   pip install scikit-learn")
            else:
                print(f"   pip install {pkg}")
        print(f"\nOr install all at once:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    # Check custom modules (with better error handling)
    print(f"\n🔍 Checking custom modules...")
    custom_modules = ['data_loader', 'preprocessor', 'model_trainer', 'predictor']
    missing_modules = []
    
    for module in custom_modules:
        try:
            # Try to import from src directory
            import importlib.util
            module_path = os.path.join('src', f'{module}.py')
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location(module, module_path)
                if spec is not None:
                    print(f"✅ {module}")
                    continue
            
            # Try regular import as fallback
            __import__(module)
            print(f"✅ {module}")
        except (ImportError, FileNotFoundError, AttributeError):
            print(f"❌ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Missing custom modules!")
        print(f"Please ensure these files exist in 'src/' directory:")
        for module in missing_modules:
            print(f"   - src/{module}.py")
        print(f"\nAlternatively, you can run with --skip-validation to bypass this check")
        return False
    
    print(f"\n✅ Environment validation passed!")
    return True

def show_help():
    """Show detailed help and usage examples"""
    help_text = """
SMS Spam Filter Training Pipeline
================================

This script provides a complete training pipeline for SMS spam classification.

USAGE EXAMPLES:
  # Train with automatic model selection (recommended)
  python run.py --data-path data/spam.csv

  # Train specific model
  python run.py --model-type naive_bayes

  # Custom configuration
  python run.py --data-path my_data.csv --test-size 0.3 --max-features 10000

  # Skip analysis for faster training
  python run.py --skip-analysis

REQUIRED DATA FORMAT:
  Your CSV file should have two columns:
  - 'label': 'spam' or 'ham'
  - 'message': The SMS text content

  Example:
  label,message
  ham,Hey are you free for lunch?
  spam,Congratulations! You won $1000!

OUTPUT FILES:
  models/
    └── spam_classifier.pkl        # Trained model
  results/
    ├── *_confusion_matrix.png     # Confusion matrices
    ├── *_feature_importance.png   # Feature importance plots
    ├── model_comparison.png       # Model comparison charts
    ├── text_analysis.png          # Data analysis charts
    ├── wordclouds.png            # Word clouds
    └── training_report_*.md       # Comprehensive report
  logs/
    └── training_*.log            # Training logs

DEPLOYMENT:
  After training, deploy using:
  python app/app.py

For more information, visit: https://github.com/yourproject/sms-spam-filter
    """
    print(help_text)

if __name__ == "__main__":
    # Check for help request
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        sys.exit(0)
    
    # Parse arguments first
    parser = argparse.ArgumentParser(description='SMS Spam Filter Training Pipeline')
    parser.add_argument('--data-path', default='data/spam.csv', 
                       help='Path to the SMS dataset CSV file')
    parser.add_argument('--model-type', default='auto', 
                       choices=['auto', 'naive_bayes', 'logistic_regression', 'random_forest'],
                       help='Model type to train (auto selects best)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--max-features', type=int, default=5000,
                       help='Maximum TF-IDF features (default: 5000)')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip data analysis and visualization steps')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip environment validation (use with caution)')
    
    args = parser.parse_args()
    
    # Validate environment (unless skipped)
    if not args.skip_validation and not validate_environment():
        print("\n❌ Environment validation failed. Please fix the issues above.")
        print("Or use --skip-validation to bypass this check (not recommended)")
        sys.exit(1)
    
    # Run main pipeline
    exit_code = main(args)
    sys.exit(exit_code)