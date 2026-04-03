import re
import string
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


class TextCleaner:
    def __init__(self, language='english'):
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('punkt')
            nltk.download('punkt_tab')

        all_stopwords = set(stopwords.words(language))
        negations = {'no', 'not', 'nor', 'didnt', 'doesnt', 'isnt', 'wasnt', 'wouldnt', 'cant', 'couldnt', 'wont',
                     'arent', 'hasnt', 'hadnt', 'havent', 'dont'}

        self.stop_words = all_stopwords - negations

        self.lemmatizer = WordNetLemmatizer()

    def remove_noise(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+|#', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        return text

    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 1
        ]
        return " ".join(cleaned_tokens)

    def full_clean(self, text_list):
        cleaned_data = []
        for text in text_list:
            noise_free = self.remove_noise(text)
            final_text = self.tokenize_and_lemmatize(noise_free)
            cleaned_data.append(final_text)
        return cleaned_data
