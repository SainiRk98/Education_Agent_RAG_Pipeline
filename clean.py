# import pandas as pd
# import re
# from sentence_transformers import SentenceTransformer
# import hdbscan

# def clean_data(input_file="questions.csv", output_file="clustered_questions.csv"):

# # ========= Step 1: Data Preparation =========
#     print("📂 Loading data...")
#     df = pd.read_csv("questions.csv")

# # Drop duplicates and empty rows
#     df.drop_duplicates(inplace=True)
#     df.dropna(inplace=True)

# # Normalize text
#     def normalize_text(text):
#     if isinstance(text, str):
#         text = text.lower()  # lowercase
#         text = re.sub(r"[^a-z0-9\s]", "", text)  # remove special chars
#         text = text.strip()
#     return text

#     df["Question"] = df["Question"].apply(normalize_text)

#     print("✅ Data cleaned!")

# # ========= Step 2: Embeddings + Clustering =========
#     print("🔄 Generating embeddings...")
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(df["Question"].tolist(), show_progress_bar=True)

#     print("🔍 Running clustering...")
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
#     cluster_labels = clusterer.fit_predict(embeddings)

#     df["cluster_id"] = cluster_labels

# # ========= Save Results =========
#     df.to_csv("clustered_questions.csv", index=False)
#     print("✅ Clustering complete! Saved as clustered_questions.csv")

import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import hdbscan

def clean_data(input_file="questions.csv", output_file="clustered_questions.csv"):
    # ========= Step 1: Data Preparation =========
    print("📂 Loading data...")
    df = pd.read_csv(input_file)

    # Drop duplicates and empty rows
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Normalize text
    def normalize_text(text):
        if isinstance(text, str):
            text = text.lower()  # lowercase
            text = re.sub(r"[^a-z0-9\s]", "", text)  # remove special chars
            text = text.strip()
            return text
        return ""

    df["Question"] = df["Question"].apply(normalize_text)

    print("✅ Data cleaned!")

    # ========= Step 2: Embeddings + Clustering =========
    print("🔄 Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["Question"].tolist(), show_progress_bar=True)

    print("🔍 Running clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = clusterer.fit_predict(embeddings)

    df["cluster_id"] = cluster_labels

    # ========= Save Results =========
    df.to_csv(output_file, index=False)
    print(f"✅ Clustering complete! Saved as {output_file}")

