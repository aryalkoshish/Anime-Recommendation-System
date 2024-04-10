import re
import numpy as np
import pandas as pd
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
from textblob import TextBlob
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

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
abbreviation_mapping_df = pd.read_csv('/Users/shweta/Downloads/slangs.csv')
review_df_subset = review_df.iloc[:5000]

# Function to find abbreviations using regex
def find_abbreviations(s):
    return re.findall(r'\b[A-Z]{2,}\b', s)  # Ensure the regular expression pattern is balanced

# Apply the function to extract abbreviations
review_df_subset['abbreviations'] = review_df_subset['text'].apply(find_abbreviations)
num_data_with_abbreviations = review_df_subset[review_df_subset['abbreviations'].apply(len) > 0].shape[0]

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
def calculate_sentiment_score(text):
    # Create a TextBlob object
    blob = TextBlob(text)

    # Get the sentiment polarity
    sentiment_score = blob.sentiment.polarity

    # Assign sentiment label based on polarity
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'


# Apply sentiment analysis to each review in the DataFrame
review_df_subset['sentiment'] = review_df_subset['preprocessed_text'].apply(calculate_sentiment_score)

tokenized_sentences = [text.split() for text in review_df_subset['preprocessed_text']]

# Train Word2Vec model
word2vec_model = Word2Vec(tokenized_sentences, min_count=1)

def count_words(text):
    return len(text.split())

# Function to calculate the number of sentences in each review
def count_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)

# Function to perform sentiment analysis and get sentiment scores
def get_sentiment_scores(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity
def average_word_embedding(text):
    tokens = text.split()
    embeddings = []
    for token in tokens:
        if token in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[token])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)  # Return zero vector if no embeddings found

# Apply the function to each review
review_df_subset['review_length'] = review_df_subset['preprocessed_text'].apply(count_words)
review_df_subset['num_sentences'] = review_df_subset['preprocessed_text'].apply(count_sentences)
review_df_subset['polarity'], review_df_subset['subjectivity'] = zip(*review_df_subset['preprocessed_text'].apply(get_sentiment_scores))
review_df_subset['word_embeddings'] = review_df_subset['preprocessed_text'].apply(average_word_embedding)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(review_df_subset['preprocessed_text'])


# Step 1: Split the dataset into training and testing sets
X = review_df_subset['preprocessed_text']  # Text data
y = review_df_subset['sentiment']  # Sentiment labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Feature Extraction using TF-IDF
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 3: Choose appropriate algorithm (Naive Bayes)
model = MultinomialNB()

# Step 4: Train the model
model.fit(X_train_tfidf, y_train)

# Step 5: Evaluate the performance of the model
y_pred = model.predict(X_test_tfidf)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted',zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Step 3: Choose Support Vector Machines (SVM) algorithm
svm_model = SVC(kernel='linear')

# Step 4: Train the SVM model
svm_model.fit(X_train_tfidf, y_train)

# Step 5: Evaluate the performance of the model
y_pred_svm = svm_model.predict(X_test_tfidf)

# Calculate evaluation metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted',zero_division=1)
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

# Print the evaluation metrics for SVM
print("SVM Accuracy:", accuracy_svm)
print("SVM Precision:", precision_svm)
print("SVM Recall:", recall_svm)
print("SVM F1-score:", f1_svm)

# Topic modeling with LDA
lda_model = LatentDirichletAllocation(n_components=6, random_state=42)
lda_topics = lda_model.fit_transform(tfidf_matrix)

# Visualize topics with word clouds
def visualize_topics(lda_model, feature_names, n_words=20):
    for idx, topic in enumerate(lda_model.components_):
        # Get top words for each topic
        top_words_idx = topic.argsort()[:-n_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]

        # Create word cloud for each topic
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_words))

        # Plot word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {idx + 1}')
        plt.axis('off')
        plt.show()

# Visualize topics using word clouds
visualize_topics(lda_model, tfidf_vectorizer.get_feature_names_out())

# # Calculate average sentiment score for each topic
topic_sentiment = []
for topic_idx, topic in enumerate(lda_topics):
    top_reviews_idx = topic.argsort()[-10:]  # Example: Top 10 reviews for each topic
    topic_reviews = review_df_subset.iloc[top_reviews_idx]
    avg_sentiment = topic_reviews['polarity'].mean()
    topic_sentiment.append(avg_sentiment)

# Visualize sentiment scores for topics
plt.figure(figsize=(8, 5))
plt.bar(range(len(topic_sentiment)), topic_sentiment, color='skyblue')
plt.xlabel('Topic')
plt.ylabel('Average Sentiment Score')
plt.title('Sentiment Scores for Topics')
plt.xticks(range(len(topic_sentiment)), [f'Topic {i+1}' for i in range(len(topic_sentiment))])
plt.show()

def calculate_similarity(lda_model, feature_names, words_or_phrases):
    similarities = []
    # Iterate over each word or phrase
    for word_or_phrase in words_or_phrases:
        # Calculate cosine similarity with each topic
        similarity_with_topics = []
        for topic in lda_model.components_:
            # Get top words for the topic
            top_words_idx = topic.argsort()[:-len(topic) - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            # Calculate cosine similarity between word_or_phrase and top words of the topic
            similarity = cosine_similarity([word2vec_model.wv[word_or_phrase]], [word2vec_model.wv[word] for word in top_words])
            similarity_with_topics.append(similarity[0][0])  # Extracting the scalar value from the array
        similarities.append(similarity_with_topics)
    return np.array(similarities)

# Example words or phrases for semantic relationship exploration
words_or_phrases = ["action", "romance", "comedy", "drama"]

# Calculate similarities
similarities = calculate_similarity(lda_model, tfidf_vectorizer.get_feature_names_out(), words_or_phrases)
n_topics = 6
# Visualize the semantic relationship using a heatmap

plt.figure(figsize=(10, 6))
sns.heatmap(similarities, annot=True, xticklabels=[f"Topic {i+1}" for i in range(n_topics)], yticklabels=words_or_phrases, cmap="YlGnBu")
plt.title("Semantic Relationship between Words/Phrases and LDA Topics")
plt.xlabel("LDA Topics")
plt.ylabel("Words/Phrases")
plt.show()

#  Model Selection and Hyperparameter Tuning (SVM)
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)
best_params = grid_search.best_params_

# Train the model with best hyperparameters
svm_model = SVC(**best_params)
svm_model.fit(X_train_tfidf, y_train)
print("Best parameters:", grid_search.best_params_)
# Evaluate the performance of the model
y_pred = svm_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Define the pipeline with TF-IDF vectorizer and Multinomial Naive Bayes classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Define parameter grid for grid search
param_grid = {
    'tfidf__max_features': [1000, 5000, 10000],  # Maximum number of features for TF-IDF vectorizer
    'tfidf__ngram_range': [(1, 1), (1, 2)],       # N-gram range for TF-IDF vectorizer
    'clf__alpha': [0.1, 0.5, 1.0],                # Alpha parameter for Multinomial Naive Bayes
}

# Initialize GridSearchCV with the pipeline, parameter grid, and cross-validation strategy
grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='accuracy', verbose=1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters found by the grid search
print("Best parameters:", grid_search.best_params_)

# Evaluate the best model on the test data
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print("Accuracy on test set:", accuracy)

# LIME Explanation
pipeline = make_pipeline(tfidf_vectorizer, model)
class_names = ['negative', 'positive','neutral']
explainer = LimeTextExplainer(class_names=class_names)
print(X_test.index)
#index = [1501, 2586, 2653, 1055,  705,  106,  589, 2468, 2413, 1600]
index=1000
text = X_test.iloc[index]
print(text)
exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=6)
with open(f"data_{index}.html", "w") as file:
    file.write(exp.as_html())

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# Perform t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(tfidf_matrix.toarray())

# Define a color map for sentiments
color_map = {'positive': 'Green', 'negative': 'red', 'neutral': 'Yellow'}

# Plot PCA result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=review_df_subset['sentiment'].map(color_map))
plt.title('PCA')

# Plot t-SNE result
plt.subplot(1, 2, 2)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=review_df_subset['sentiment'].map(color_map))
plt.title('t-SNE')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(review_df_subset['sentiment'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

sns.set_style("whitegrid")

# Plot histogram of sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(review_df_subset['score'], bins=20, color='skyblue', kde=True)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Plot box plot of sentiment scores
plt.figure(figsize=(8, 6))
sns.boxplot(y=review_df_subset['score'], color='lightblue')
plt.title('Box Plot of Sentiment Scores')
plt.ylabel('Sentiment Score')
plt.show()


