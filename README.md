# Topic-Modeling

![GitHub Repo Size](https://img.shields.io/github/repo-size/ChrisRoyse/Topic-Modeling)
![GitHub last commit](https://img.shields.io/github/last-commit/ChrisRoyse/Topic-Modeling)
![License](https://img.shields.io/github/license/ChrisRoyse/Topic-Modeling)

## Twitch Chat Analysis and Topic Modeling

This repository contains four Python scripts designed to process and analyze large volumes of Twitch chat data. The scripts embed, vectorize, and cluster chat messages from Twitch streams, enabling contextual topic modeling and classification using advanced Natural Language Processing (NLP) techniques.

---

## Table of Contents

- [Overview](#overview)
- [Processing Pipeline](#processing-pipeline)
- [Scripts](#scripts)
  - [1. preprocessing_and_clustering.py](#1-preprocessing_and_clusteringpy)
  - [2. embedding_and_clustering.py](#2-embedding_and_clusteringpy)
  - [3. berttopic_theme_generation.py](#3-berttopic_theme_generationpy)
  - [4. merge_topics_with_themes.py](#4-merge_topics_with_themespy)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Directory Structure](#directory-structure)
  - [Running the Scripts](#running-the-scripts)
- [Output](#output)
- [Features and Functionality](#features-and-functionality)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository contains scripts developed to process chat logs from 64 female Twitch streamers, totaling approximately 309,000 messages. The objective is to analyze chat data to identify prevalent topics and themes within these communities using advanced Natural Language Processing (NLP) techniques.

---

## Processing Pipeline

1. **Data Collection and Preprocessing:** Aggregate and clean chat messages from CSV files.
2. **Embedding:** Transform messages into numerical representations using Sentence Transformers.
3. **Dimensionality Reduction:** Reduce embedding dimensions with UMAP for efficient clustering.
4. **Clustering:** Group similar messages using HDBSCAN.
5. **Topic Modeling:** Identify and name topics within clusters using BERTopic.
6. **Merging and Classification:** Associate messages with thematic labels for analysis.

---

## Scripts

### 1. preprocessing_and_clustering.py

**Purpose:** Performs extensive preprocessing, embedding, dimensionality reduction, and clustering.

**Features:**

- Preprocesses text (lowercasing, URL removal, tokenization, stopword removal, lemmatization).
- Generates embeddings using SentenceTransformer.
- Reduces dimensions with UMAP.
- Clusters messages with HDBSCAN using refined parameters.
- Saves messages with topic labels to a CSV file.

**Dependencies:**

- pandas
- numpy
- nltk
- re
- sentence_transformers
- umap-learn
- hdbscan
- tqdm
- logging
- os

```python
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
```

---

### 2. embedding_and_clustering.py

**Purpose:** Collects chat messages, generates embeddings, reduces dimensionality, and performs clustering.

**Features:**

- Reads and aggregates messages from multiple CSV files.
- Uses SentenceTransformer for message embeddings.
- Applies UMAP for dimensionality reduction.
- Clusters messages with HDBSCAN.
- Optionally performs topic modeling with Latent Dirichlet Allocation (LDA).

**Dependencies:**

- pandas
- sentence_transformers
- umap-learn
- hdbscan
- scikit-learn
- numpy
- logging

```python
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
```

---

### 3. berttopic_theme_generation.py

**Purpose:** Generates themes for each topic cluster using BERTopic.

**Features:**

- Reads messages with topic labels.
- Applies BERTopic to extract representative themes per cluster.
- Handles outlier topics where no prominent theme is detected.
- Saves generated themes to a CSV file.

**Dependencies:**

- pandas
- bertopic
- tqdm
- os

```python
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm
import os

# Path to your CSV file
csv_path = 'c:/python39/messages_with_topics.csv'

# Read the CSV file
df = pd.read_csv(csv_path)

# Drop rows with NaN in 'topic_label' or 'originalMessage'
df = df.dropna(subset=['topic_label', 'originalMessage'])

# Convert 'topic_label' to string to ensure consistent grouping
df['topic_label'] = df['topic_label'].astype(str)

# Group messages by 'topic_label'
grouped = df.groupby('topic_label')

# Initialize a dictionary to store themes
topic_themes = {}

# Initialize BERTopic model
topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")

# Iterate over each group and generate themes
print("Generating themes for each topic...")
for topic_label, group in tqdm(grouped):
    # Get all messages in the group
    messages = group['originalMessage'].tolist()
    
    # Fit BERTopic model on the messages
    topics, probs = topic_model.fit_transform(messages)
    
    # Extract topic info
    topic_info = topic_model.get_topic_info()
    
    # Check if there are any non-outlier topics
    if (topic_info.Topic != -1).any():
        # Get the most frequent non-outlier topic
        main_topic = topic_info.loc[topic_info.Topic != -1].iloc[0]['Topic']
        # Get the representative words for the main topic
        representative_words = topic_model.get_topic(main_topic)
        # Format the theme
        theme = ', '.join([word for word, _ in representative_words])
    else:
        # Assign a placeholder theme if all topics are outliers
        theme = "No prominent theme detected (outliers only)"
    
    # Store the theme
    topic_themes[topic_label] = theme

# Create a DataFrame for the themes
themes_df = pd.DataFrame(list(topic_themes.items()), columns=['topic_label', 'theme'])

# Save the themes to a CSV file
output_path = 'c:/python39/topic_themes.csv'
themes_df.to_csv(output_path, index=False)
print(f"Themes saved to {output_path}")

# Optional: Print the themes
for topic_label, theme in topic_themes.items():
    print(f"\nTopic {topic_label} Theme:\n{theme}\n")
```

---

### 4. merge_topics_with_themes.py

**Purpose:** Merges clustered messages with thematic labels.

**Features:**

- Reads messages with topic labels and themes from CSV files.
- Cleans and synchronizes `topic_label` data types.
- Merges dataframes to associate messages with themes.
- Handles missing themes gracefully.

**Dependencies:**

- pandas

```python
import pandas as pd

# Paths to your CSV files
messages_csv_path = 'c:/python39/sorted_messages_with_topics.csv'
themes_csv_path = 'c:/python39/topic_themes.csv'

# Read the CSV files
df_messages = pd.read_csv(messages_csv_path)
df_themes = pd.read_csv(themes_csv_path)

# Ensure 'topic_label' columns are of the same type and format
def clean_topic_label(label):
    if pd.isna(label):
        return None
    else:
        # Convert to string and strip whitespace
        label_str = str(label).strip()
        # Remove any decimal points if label is a float represented as a string
        if label_str.endswith('.0'):
            label_str = label_str[:-2]
        return label_str

df_messages['topic_label'] = df_messages['topic_label'].apply(clean_topic_label)
df_themes['topic_label'] = df_themes['topic_label'].apply(clean_topic_label)

# Identify topic_labels present in messages but not in themes
messages_labels = set(df_messages['topic_label'].dropna())
themes_labels = set(df_themes['topic_label'].dropna())

missing_labels = messages_labels - themes_labels

if missing_labels:
    print("These topic_labels are in messages but not in themes:")
    print(missing_labels)
    print("\nRows with missing themes will have 'No theme available' as the theme.")
else:
    print("All topic_labels in messages are present in themes.")

# Merge the DataFrames on 'topic_label', adding the 'theme' column to df_messages
df_merged = df_messages.merge(df_themes, on='topic_label', how='left')

# Fill missing themes with a placeholder
df_merged['theme'] = df_merged['theme'].fillna('No theme available')

# Save the merged DataFrame to a new CSV file
output_csv_path = 'c:/python39/sorted_messages_with_themes.csv'
df_merged.to_csv(output_csv_path, index=False)

print(f"Merged file saved to {output_csv_path}")
```

---

## Getting Started

### Prerequisites

- **Python 3.x** is required to run the scripts.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ChrisRoyse/Topic-Modeling.git
   cd Topic-Modeling
   ```

2. **Install the required libraries:**

   ```bash
   pip install pandas numpy nltk sentence-transformers umap-learn hdbscan bertopic scikit-learn tqdm
   ```

3. **Download NLTK data:**

   The scripts are configured to download necessary NLTK data. Ensure that the `nltk_data` directory is correctly set in the scripts or adjust the path as needed.

### Directory Structure

```
Topic-Modeling/
│
├── preprocessing_and_clustering.py
├── embedding_and_clustering.py
├── berttopic_theme_generation.py
├── merge_topics_with_themes.py
├── requirements.txt
├── README.md
└── data/
    ├── femalechat/
    │   ├── streamer1.csv
    │   ├── streamer2.csv
    │   └── ... 
    ├── messages_with_topics.csv
    ├── topic_themes.csv
    └── sorted_messages_with_themes.csv
```

### Running the Scripts

1. **Preprocessing and Clustering:**

   ```bash
   python preprocessing_and_clustering.py
   ```

2. **Embedding and Clustering:**

   ```bash
   python embedding_and_clustering.py
   ```

3. **BERTopic Theme Generation:**

   ```bash
   python berttopic_theme_generation.py
   ```

4. **Merging Topics with Themes:**

   ```bash
   python merge_topics_with_themes.py
   ```

Ensure that the paths specified in each script match the locations of your data files.

---

## Output

- **messages_with_topics.csv:** Contains original and preprocessed messages with assigned topic labels.
- **topic_themes.csv:** Maps each topic label to its corresponding theme.
- **sorted_messages_with_themes.csv:** Merges messages with their respective themes for comprehensive analysis.

---

## Features and Functionality

- **Data Aggregation:** Efficiently reads and aggregates chat messages from multiple CSV files.
- **Text Preprocessing:** Cleans and preprocesses text data for optimal analysis.
- **Embeddings:** Utilizes Sentence Transformers to convert text data into numerical embeddings.
- **Dimensionality Reduction:** Employs UMAP to reduce the dimensionality of embeddings for better clustering performance.
- **Clustering:** Uses HDBSCAN to identify clusters of similar messages.
- **Topic Modeling:** Applies BERTopic and LDA to extract meaningful themes and topics from clustered data.
- **Data Merging:** Integrates topic labels with thematic descriptions for enhanced interpretability.

---

## Future Enhancements

- **Visualization:** Incorporate visualizations of topic distributions and cluster embeddings.
- **Real-time Analysis:** Adapt scripts for real-time chat analysis and topic modeling.
- **Multi-language Support:** Extend preprocessing and modeling to support multiple languages.
- **Improved Preprocessing:** Implement more advanced text preprocessing techniques to handle slang, emojis, and other chat-specific nuances.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository.**
2. **Create a new branch:**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit your changes:**

   ```bash
   git commit -m 'Add some feature'
   ```

4. **Push to the branch:**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a pull request.**

Please ensure that your code adheres to the existing style and that you include appropriate documentation and comments.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any questions or suggestions, please contact [Chris Royse](mailto:chrisroyse@example.com).

---

© 2024 GitHub, Inc.

---

**Note:** Ensure that you update the paths in the scripts (`c:/python39/...`) to match your local directory structure. Additionally, consider adding a `requirements.txt` file to simplify the installation of dependencies:

```txt
pandas
numpy
nltk
sentence-transformers
umap-learn
hdbscan
bertopic
scikit-learn
tqdm
```

You can create this file in your repository and instruct users to install dependencies using:

```bash
pip install -r requirements.txt
```

This completes the setup of your GitHub repository with a detailed `README.md`. If you have any further customization or additional information you'd like to include, feel free to modify the content accordingly.
