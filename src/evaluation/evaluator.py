import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    @staticmethod
    def evaluate_and_plot(y_true, y_pred, save_path='confusion_matrix.png'):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        print("\n--- PERFORMANCE REPORT ---")
        print(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Confusion matrix saved as '{save_path}'")

        return acc, f1
