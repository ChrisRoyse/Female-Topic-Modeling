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
