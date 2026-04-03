DATASET_PATH = "data/review.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

GRID_PARAMS_LR = {
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'lr__C': [0.1, 1.0, 10.0]
}

GRID_PARAMS_SVM = {
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'svm__C': [0.1, 1.0, 10.0]
}

MODELS_DIR_LR = "models/lr"
MODELS_DIR_SVM = "models/svm"
EXPERIMENT_TRACKING_FILE = "experiment_tracking.json"

TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_SAVE_PATH = "models/transformer"
TRANSFORMER_EPOCHS = 3
TRANSFORMER_BATCH_SIZE = 16