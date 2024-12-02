import pandas as pd
import numpy as np
import glob
import os
import logging
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import umap.umap_ as umap
import hdbscan

# Define NLTK data directory explicitly
nltk_data_path = 'c:/python39/nltk_data'
nltk.data.path = [nltk_data_path]  # Ensure NLTK only searches in the specified directory

# Download or confirm availability of NLTK resources
try:
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('wordnet', download_dir=nltk_data_path)
except Exception as e:
    print("Error downloading NLTK resources:", e)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define the data directory
data_dir = 'c:/python39/femalechat'

# Get list of all CSV files in the directory
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
logger.info(f'Found {len(csv_files)} CSV files in directory {data_dir}.')

# Initialize an empty list to store messages
all_messages = []

# Loop over CSV files and collect messages
for csv_file in csv_files:
    logger.info(f'Reading file {csv_file}')
    data = pd.read_csv(csv_file)
    messages = data['messageText'].dropna().tolist()
    all_messages.extend(messages)
    logger.info(f'Collected {len(messages)} messages from {csv_file}. Total messages collected: {len(all_messages)}.')

logger.info(f'Finished collecting messages. Total messages: {len(all_messages)}')

# Text Preprocessing Function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from text
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize with preserve_line=True to avoid sent_tokenize
    tokens = nltk.word_tokenize(text, preserve_line=True)
    # Remove stop words and lemmatize
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back to string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Preprocess all messages
logger.info('Preprocessing messages...')
preprocessed_messages = [preprocess_text(message) for message in tqdm(all_messages)]
logger.info('Preprocessing completed.')

# Load the Sentence Transformer model
logger.info('Loading Sentence Transformer model...')
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info('Model loaded successfully.')

# Encode messages into embeddings
logger.info('Encoding messages into embeddings...')
embeddings = model.encode(preprocessed_messages, batch_size=64, show_progress_bar=True)
logger.info('Encoding completed.')

# Optional: Save embeddings for future use
embeddings_path = os.path.join(data_dir, 'female_chat_embeddings_preprocessed.npy')
np.save(embeddings_path, embeddings)
logger.info(f'Embeddings saved to {embeddings_path}.')

# Reduce dimensionality with UMAP for visualization and clustering
logger.info('Reducing dimensionality with UMAP...')
reducer = umap.UMAP(n_components=5, random_state=42)
embeddings_reduced = reducer.fit_transform(embeddings)
logger.info('Dimensionality reduction completed.')

# Perform clustering with HDBSCAN
logger.info('Performing clustering with HDBSCAN...')
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean', cluster_selection_epsilon=0.01)
cluster_labels = clusterer.fit_predict(embeddings_reduced)
logger.info('Clustering completed.')

# Create a dataframe with messages and cluster labels
cluster_data = pd.DataFrame({
    'originalMessage': all_messages,
    'preprocessedMessage': preprocessed_messages,
    'topic_label': cluster_labels
})

# Replace -1 labels (noise) with NaN
cluster_data['topic_label'] = cluster_data['topic_label'].replace(-1, np.nan)

# Count the number of messages in each cluster
cluster_counts = cluster_data['topic_label'].value_counts().sort_values(ascending=False)
logger.info('Number of messages per topic:')
for topic_label, count in cluster_counts.items():
    logger.info(f'Topic {int(topic_label)}: {count} messages')

# Sort the dataframe by topic_label based on the number of messages per topic
cluster_data['topic_label'] = cluster_data['topic_label'].astype('category')
cluster_data['topic_label'] = cluster_data['topic_label'].cat.reorder_categories(cluster_counts.index.tolist())
cluster_data.sort_values('topic_label', inplace=True)

# Save the messages with topic labels to a new CSV file
output_csv_path = os.path.join(data_dir, 'messages_with_topics.csv')
cluster_data.to_csv(output_csv_path, index=False)
logger.info(f'Messages with topic labels saved to {output_csv_path}.')

logger.info('Script completed successfully.')
