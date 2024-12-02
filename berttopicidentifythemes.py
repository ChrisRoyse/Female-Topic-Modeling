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
