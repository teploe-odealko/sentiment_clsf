import pandas as pd
from nltk.tokenize import RegexpTokenizer
import pymorphy2


def lemmatize_str(text):
    tokenizer = RegexpTokenizer("[A-Za-zА-Яа-я]+")
    tokens = tokenizer.tokenize(text)

    # Lowercase and lemmatize
    morph = pymorphy2.MorphAnalyzer()
    tokens_norm = [morph.parse(t.lower())[0].normal_form for t in tokens]
    # tokens_clean = [t for t in tokens_norm if t not in stop_word_list]
    return ' '.join(tokens_norm)

def preprocess_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the reviews (lemmatize with spacy)

        Args:
            reviews: Source data.
        Returns:
            Preprocessed (lemmatized) data.
    """

    reviews.text = reviews.text.apply(lemmatize_str)
    return reviews
