import pandas as pd
from sklearn.model_selection import train_test_split
import config
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_prepare(self):
        logger.info(f"Loading local data from: {self.file_path}")
        df = pd.read_csv(self.file_path)

        if 'text' not in df.columns or 'stars' not in df.columns:
            raise ValueError("CSV file missing required columns: 'text' and 'stars'")

        # Normalizare (0 si 1) pentru LogisticRegression / SVC / Transformer
        df['Sentiment'] = df['stars'].apply(lambda x: 1 if float(x) > 3 else 0)

        X = df['text']
        y = df['Sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )

        logger.info(f"Prepared {len(X_train)} train texts and {len(X_test)} test texts.")
        return X_train, X_test, y_train, y_test
