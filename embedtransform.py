import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import umap.umap_ as umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import glob
import os
import numpy as np
import logging

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

# Load the Sentence Transformer model
logger.info('Loading Sentence Transformer model...')
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info('Model loaded successfully.')

# Encode messages into embeddings
logger.info('Encoding messages into embeddings...')
embeddings = model.encode(all_messages, batch_size=64, show_progress_bar=True)
logger.info('Encoding completed.')

# Optional: Save embeddings for future use
embeddings_path = 'female_chat_embeddings.npy'
np.save(embeddings_path, embeddings)
logger.info(f'Embeddings saved to {embeddings_path}.')

# Reduce dimensionality with UMAP for visualization and clustering
logger.info('Reducing dimensionality with UMAP...')
reducer = umap.UMAP(n_components=50, random_state=42)
embeddings_reduced = reducer.fit_transform(embeddings)
logger.info('Dimensionality reduction completed.')

# Perform clustering with HDBSCAN
logger.info('Performing clustering with HDBSCAN...')
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean')
cluster_labels = clusterer.fit_predict(embeddings_reduced)
logger.info('Clustering completed.')

# Create a dataframe with messages and cluster labels
cluster_data = pd.DataFrame({'messageText': all_messages, 'cluster_label': cluster_labels})
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
logger.info(f'Number of clusters found: {num_clusters}')

# Print sample messages from each cluster
for cluster in set(cluster_labels):
    if cluster == -1:
        logger.info('Skipping noise cluster.')
        continue
    logger.info(f'\nCluster {cluster}:')
    cluster_messages = cluster_data[cluster_data['cluster_label'] == cluster]['messageText']
    sample_size = min(5, len(cluster_messages))
    logger.info(f'Sample messages from Cluster {cluster}: {cluster_messages.sample(n=sample_size, random_state=42).tolist()}')

# Alternatively, perform topic modeling with LDA
logger.info('Performing topic modeling with LDA...')
# Vectorize messages using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.9, min_df=10)
tfidf_matrix = vectorizer.fit_transform(cluster_data['messageText'])
logger.info('TF-IDF vectorization completed.')

# Perform LDA
n_topics = 10  # Adjust the number of topics as needed
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_topics = lda_model.fit_transform(tfidf_matrix)
logger.info('LDA topic modeling completed.')

# Display the topics
def display_topics(model, feature_names, no_top_words):
    for idx, topic in enumerate(model.components_):
        topic_words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        logger.info(f"\nTopic {idx+1}: {topic_words}")

tf_feature_names = vectorizer.get_feature_names_out()
display_topics(lda_model, tf_feature_names, no_top_words=10)

logger.info('Script completed successfully.')
