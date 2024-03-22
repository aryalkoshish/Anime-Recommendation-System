import re
import pandas as pd
import string
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from contractions import CONTRACTION_MAP # Custom dictionary for handling contractions
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# Download necessary resources for NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

review_df = pd.read_csv('/Users/shweta/Downloads/Anime_3014/reviews.csv')
review_df_subset = review_df.iloc[:5000]

# Custom function to handle contractions
contraction_map = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}
def expand_contractions(text, contraction_mapping=contraction_map):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        if expanded_contraction:
            expanded_contraction = first_char + expanded_contraction[1:]
        else:
            expanded_contraction = match  # Return original match if expansion not found
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# Custom function to handle negations (e.g., not happy -> not_happy)
def handle_negations(text):
    negations = ['no', 'not', 'none', 'neither', 'never', 'nobody', 'nothing']
    words = text.split()
    for i in range(len(words)):
        if words[i] in negations:
            if i < len(words) - 1:
                words[i + 1] = 'not_' + words[i + 1]
    return ' '.join(words)


# Custom function to preprocess text
def preprocess_text(text):
    # Handle contractions
    text = expand_contractions(text)

    # Handle negations
    text = handle_negations(text)

    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the words to their base form
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


# Apply preprocessing to the text column in your DataFrame
review_df_subset['preprocessed_text'] = review_df_subset['text'].apply(preprocess_text)

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to assign sentiment scores to each review
def get_sentiment_score(text):
    # Analyze sentiment
    sentiment_scores = sid.polarity_scores(text)

    # Determine sentiment label based on compound score
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


# Apply sentiment analysis to each review in the DataFrame
review_df_subset['sentiment'] = review_df_subset['preprocessed_text'].apply(get_sentiment_score)

review_df_subset.to_csv('/Users/shweta/Downloads/review_df_subset.csv', index=False)
