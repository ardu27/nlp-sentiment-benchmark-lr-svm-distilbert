import json
import logging
import os
import time

import config
from src.data import DataHandler
from src.preprocessing import TextCleaner
from src.models import LREngine, SVMEngine, TransformerEngine
from src.evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_pipeline():
    handler = DataHandler(config.DATASET_PATH)
    X_train_raw, X_test_raw, y_train, y_test = handler.load_and_prepare()

    cleaner = TextCleaner()
    logging.info("Cleaning training texts...")
    X_train = cleaner.full_clean(X_train_raw.tolist())
    logging.info("Cleaning testing texts...")
    X_test = cleaner.full_clean(X_test_raw.tolist())

    # 1. LR Pipeline
    lr_engine = LREngine()
    lr_engine.train_with_optimization(X_train, y_train)
    predictions_lr = lr_engine.predict(X_test)
    acc_lr, f1_lr = ModelEvaluator.evaluate_and_plot(y_test, predictions_lr, save_path="cm_lr.png")
    saved_path_lr = lr_engine.save_model()

    # 2. SVM Pipeline
    svm_engine = SVMEngine()
    svm_engine.train_with_optimization(X_train, y_train)
    predictions_svm = svm_engine.predict(X_test)
    acc_svm, f1_svm = ModelEvaluator.evaluate_and_plot(y_test, predictions_svm, save_path="cm_svm.png")
    saved_path_svm = svm_engine.save_model()
    
    # 3. Transformer Pipeline
    bert_engine = TransformerEngine()
    transformer_dir = config.TRANSFORMER_SAVE_PATH
    if os.path.exists(transformer_dir) and os.listdir(transformer_dir):
        logging.info("Found existing DistilBERT model. Skipping training and loading directly.")
        bert_engine.load_model(transformer_dir)
        saved_path_bert = transformer_dir
    else:
        bert_engine.train(X_train_raw, y_train)
        saved_path_bert = bert_engine.save_model()
    
    predictions_bert = bert_engine.predict(X_test_raw)
    acc_bert, f1_bert = ModelEvaluator.evaluate_and_plot(y_test, predictions_bert, save_path="cm_bert.png")

    # Tracking
    experiment_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path_lr": saved_path_lr,
        "model_path_svm": saved_path_svm,
        "transformer_path": saved_path_bert,
        "metrics_lr": {
            "accuracy": acc_lr,
            "f1_score": f1_lr
        },
        "metrics_svm": {
            "accuracy": acc_svm,
            "f1_score": f1_svm
        },
        "metrics_bert": {
            "accuracy": acc_bert,
            "f1_score": f1_bert
        }
    }
    
    tracking_file = config.EXPERIMENT_TRACKING_FILE
    experiments = []
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, "r") as f:
                experiments = json.load(f)
        except Exception:
            pass
            
    experiments.append(experiment_data)
    with open(tracking_file, "w") as f:
        json.dump(experiments, f, indent=4)

    logging.info("Pipeline executed successfully and experiment tracking updated!")

    print("\n")
    print(f"{'Model':<15} | {'Accuracy':<8} | {'F1 Score':<8}")
    print("-" * 35)
    print(f"{'LR (baseline)':<15} | {acc_lr:^8.4f} | {f1_lr:^8.4f}")
    print(f"{'SVM':<15} | {acc_svm:^8.4f} | {f1_svm:^8.4f}")
    print(f"{'DistilBERT':<15} | {acc_bert:^8.4f} | {f1_bert:^8.4f}")
    print("\n")

if __name__ == "__main__":
    run_pipeline()