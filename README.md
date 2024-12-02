# Twitch Chat Analysis and Topic Modeling

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A repository containing four Python scripts designed to process and analyze large volumes of Twitch chat data. The scripts embed, vectorize, and cluster chat messages from Twitch streams, enabling contextual topic modeling and classification using advanced NLP techniques.

## Table of Contents

- [Overview](#overview)
- [Processing Pipeline](#processing-pipeline)
- [Scripts](#scripts)
  - [1. `embedding_and_clustering.py`](#1-embedding_and_clusteringpy)
  - [2. `merge_topics_with_themes.py`](#2-merge_topics_with_themespy)
  - [3. `bertopic_theme_generation.py`](#3-bertopic_theme_generationpy)
  - [4. `preprocessing_and_clustering.py`](#4-preprocessing_and_clusteringpy)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Directory Structure](#directory-structure)
  - [Running the Scripts](#running-the-scripts)
- [Output](#output)
- [Features and Functionality](#features-and-functionality)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This repository contains scripts developed to process chat logs from 64 female Twitch streamers, totaling approximately 309,000 messages. The objective is to analyze chat data to identify prevalent topics and themes within these communities using advanced Natural Language Processing (NLP) techniques.

### Processing Pipeline

1. **Data Collection and Preprocessing**: Aggregate and clean chat messages from CSV files.
2. **Embedding**: Transform messages into numerical representations using Sentence Transformers.
3. **Dimensionality Reduction**: Reduce embedding dimensions with UMAP for efficient clustering.
4. **Clustering**: Group similar messages using HDBSCAN.
5. **Topic Modeling**: Identify and name topics within clusters using BERTopic.
6. **Merging and Classification**: Associate messages with thematic labels for analysis.

## Scripts

### 1. `embedding_and_clustering.py`

**Purpose**: Collects chat messages, generates embeddings, reduces dimensionality, and performs clustering.

**Features**:
- Reads and aggregates messages from multiple CSV files.
- Uses `SentenceTransformer` for message embeddings.
- Applies UMAP for dimensionality reduction.
- Clusters messages with HDBSCAN.
- Optionally performs topic modeling with Latent Dirichlet Allocation (LDA).

**Dependencies**:
- `pandas`
- `sentence_transformers`
- `umap-learn`
- `hdbscan`
- `scikit-learn`
- `numpy`
- `logging`

### 2. `merge_topics_with_themes.py`

**Purpose**: Merges clustered messages with thematic labels.

**Features**:
- Reads messages with topic labels and themes from CSV files.
- Cleans and synchronizes `topic_label` data types.
- Merges dataframes to associate messages with themes.
- Handles missing themes gracefully.

**Dependencies**:
- `pandas`

### 3. `bertopic_theme_generation.py`

**Purpose**: Generates themes for each topic cluster using BERTopic.

**Features**:
- Reads messages with topic labels.
- Applies BERTopic to extract representative themes per cluster.
- Handles outlier topics where no prominent theme is detected.
- Saves generated themes to a CSV file.

**Dependencies**:
- `pandas`
- `bertopic`
- `tqdm`
- `os`

### 4. `preprocessing_and_clustering.py`

**Purpose**: Performs extensive preprocessing, embedding, dimensionality reduction, and clustering.

**Features**:
- Preprocesses text (lowercasing, URL removal, tokenization, stopword removal, lemmatization).
- Generates embeddings using `SentenceTransformer`.
- Reduces dimensions with UMAP.
- Clusters messages with HDBSCAN using refined parameters.
- Saves messages with topic labels to a CSV file.

**Dependencies**:
- `pandas`
- `numpy`
- `nltk`
- `re`
- `sentence_transformers`
- `umap-learn`
- `hdbscan`
- `tqdm`
- `logging`
- `os`

## Getting Started

### Prerequisites

- **Python 3.x**
- Install required libraries:

  ```bash
  pip install pandas numpy nltk sentence-transformers umap-learn hdbscan bertopic scikit-learn tqdm
