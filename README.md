# sentiment-analysis-ml-comparison

I built this project as a 2nd year Computer Science student to explore how different Machine Learning models handle the same natural language processing (NLP) task. The goal is to classify text reviews as either **Positive** or **Negative** and compare how "traditional" mathematical models perform against a modern "deep learning" Transformer model.

## What I Learned
The most important thing I learned from this project is that **the newest and most complex model isn't always the winner**. I started this project thinking that DistilBERT (a powerful neural network) would easily beat everything else, but I discovered that for smaller datasets, a well-tuned classical model like SVM can actually be more accurate and much faster. I also learned how crucial "text cleaning" (removing noise, fixing negations) is for traditional models, whereas deep learning models prefer the raw, messy text.

## Models Compared

*   **Logistic Regression (LR):** This is my baseline. It's a simple but effective model that looks for linear relationships between word frequencies and the sentiment. It's very fast and easy to understand.
*   **Support Vector Machine (SVM):** This model tries to find the "best" possible line (hyperplane) that separates the two classes with the widest margin. It's a classic power-player in text classification.
*   **DistilBERT (Fine-Tuned):** A "distilled" version of BERT, one of the most famous deep learning models. Unlike the others, it understands the context and order of words, not just their frequency. I used "Transfer Learning" by taking a pre-trained model and fine-tuning it on my specific reviews.

## Results

| Model           | Accuracy | F1 Score |
|-----------------|----------|----------|
| LR (baseline)   |  0.8520  |  0.8517  |
| **SVM**         |  **0.8625**  |  **0.8594**  |
| DistilBERT      |  0.8245  |  0.8284  |

### Why did SVM win?
In plain English: For the size of the dataset I used (around 10,000 reviews), the SVM was better at spotting the clear patterns in the words I cleaned. DistilBERT is like a "supercomputer" that needs a massive amount of data to truly shine. With a smaller dataset, the SVM found the right answers more consistently, while DistilBERT was arguably "overthinking" the simple patterns or just didn't have enough examples to learn the complex ones.

## Project Structure

```text
ai_pr1/
├── data/               # Contains the raw CSV dataset (review.csv)
├── models/             # Where the trained .pkl and transformer files are saved
├── src/                # All the source code, organized into modules:
│   ├── data/           # Loading and preparing the CSV data
│   ├── models/         # The "engines" for LR, SVM, and DistilBERT
│   ├── preprocessing/  # Text cleaning logic (removing punctuation, etc.)
│   └── evaluation/     # Logic for calculating accuracy and drawing matrices
├── config.py           # Central place for all settings and hyperparameters
├── main.py             # Script to train all 3 models and compare them
├── app.py              # The Streamlit web interface to try the models
└── requirements.txt    # List of libraries you need to install
```

## How to Run

### 1. Prerequisites
- Python 3.8 or newer
- A virtual environment (recommended)

### 2. Setup
Clone the repo and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Training
To train all three models from scratch and see the comparison table:
```bash
python main.py
```
> **Note:** DistilBERT training is very heavy. On a standard laptop (CPU), it can take **1-2 hours**. If you have a GPU, it will be much faster. If you just want to test the app, you can skip this if you already have the models saved.

### 4. Running the Demo
To open the interactive web interface:
```bash
streamlit run app.py
```
This will open a tab in your browser where you can type your own reviews and see how each model classifies them in real-time.

## Tech Stack
- **scikit-learn**: Used for the LR and SVM models and the TF-IDF feature extraction.
- **transformers**: Provided the pre-trained DistilBERT model and tokenizer.
- **torch (PyTorch)**: The deep learning engine that runs DistilBERT.
- **streamlit**: Allowed me to build a beautiful web UI with just a few lines of Python.
- **eli5**: Used for "Explainable AI" to show exactly which words influenced the models' decisions.
- **nltk**: Used for tokenizing text and lemmatizing words.
- **pandas**: Used for handling the CSV data tables.

## Future Improvements
1.  **Bigger Dataset**: I'd like to try this on 100,000+ reviews to see at exactly what point DistilBERT starts overtaking the SVM.
2.  **GPU Training**: I want to learn how to use CUDA properly to speed up the Transformer training.
3.  **Experimental Hyperparameters**: I'd love to try different "learning rates" and "batch sizes" for BERT to see if I can squeeze out more accuracy.
