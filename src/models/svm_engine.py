import logging
import joblib
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import config

logger = logging.getLogger(__name__)

class SVMEngine:
    def __init__(self):
        self.model = None
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")),
            ('svm', LinearSVC(class_weight='balanced', max_iter=1000, dual=False))
        ])

    def train_with_optimization(self, X_train, y_train):
        logger.info("Starting automatic optimization (GridSearchCV) for SVM...")

        self.model = GridSearchCV(
            self.pipeline,
            param_grid=config.GRID_PARAMS_SVM,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        self.model.fit(X_train, y_train)

        logger.info(f"Best parameters found for SVM: {self.model.best_params_}")
        logger.info("SVM Training completed.")

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self):
        if not os.path.exists(config.MODELS_DIR_SVM):
            os.makedirs(config.MODELS_DIR_SVM)
            
        timestamp = time.strftime("%Y%m%d_%H%M")
        model_filename = f"model_svm_{timestamp}.pkl"
        save_path = os.path.join(config.MODELS_DIR_SVM, model_filename)
        
        best_estimator = self.model.best_estimator_ if hasattr(self.model, 'best_estimator_') else self.model
        joblib.dump(best_estimator, save_path)
        
        logger.info(f"SVM Model saved at: {save_path}")
        return save_path
