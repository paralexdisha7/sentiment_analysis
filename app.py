import streamlit as st 
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


# preprocess


regular_punct = list(string.punctuation)
stop_words = set(stopwords.words('english'))

def transform_text(text):
    tokens = nltk.word_tokenize(text)  # Tokenize text into words
    tokens = remove_stopwords(tokens, stop_words)  # Remove stopwords
    tokens = remove_punctuation(tokens, regular_punct)  # Remove punctuation
    tokens = lowcase(tokens)  # Convert to lowercase
    tokens = lemmatize_tokens(tokens)  # Lemmatize words
    # print(tokens)
    return tokens


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




#__main__

st.title("Sentiment Analysis...")
input_tweet = st.text_area("enter the tweet...")


if st.button('Analyze'):
    # preprocess
    transformed_text = transform_text(input_tweet)
    # vectorize
   
    # Join tokens into a single string
    transformed_text_str = ' '.join(transformed_text)
    # vectorize
    vector_tweet = tfidf.transform([transformed_text_str])
    # predict
    result = model.predict(vector_tweet)
    print(result)
    
    st.header(result[0])

