import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

regular_punct = list(string.punctuation)
stop_words = set(stopwords.words('english'))


def transform_text(text):
    tokens = nltk.word_tokenize(text)  # Tokenize text into words
    tokens = remove_stopwords(tokens, stop_words)  # Remove stopwords
    tokens = remove_punctuation(tokens, regular_punct)  # Remove punctuation
    tokens = lowcase(tokens)  # Convert to lowercase
    tokens = lemmatize_tokens(tokens)  # Lemmatize words
    print(tokens)


def remove_stopwords(tokens, stop_words):
    return [word for word in tokens if word.lower() not in stop_words]


def remove_punctuation(text, punct_list):
    if isinstance(text, list):  # Checking if text is a list of tokens
        return [word for word in text if word not in punct_list]
    else:
        for punc in punct_list:
            if punc in text:
                text = text.replace(punc, ' ')
        return text.strip()


def lowcase(text):
    if isinstance(text, list):  # Checking if text is a list of tokens
        return [word.lower() for word in text]
    else:
        return text.lower()


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]


# main
transform_text("sooo sad i will miss you here in san diego!!!")
