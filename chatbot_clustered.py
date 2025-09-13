# # chatbot_clustered.py

# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # ---------------------------
# # Step 1: Load Data
# # ---------------------------
# df_questions = pd.read_csv("clustered_questions.csv")  # Expected columns: question, cluster_id
# df_laws = pd.read_csv("automated_cluster_laws.csv")    # Expected columns: cluster_id, matched_laws

# # Normalize column names: lowercase + strip spaces
# df_questions.columns = [c.strip().lower() for c in df_questions.columns]
# df_laws.columns = [c.strip().lower() for c in df_laws.columns]

# # Safety check
# required_q_cols = ['question', 'cluster_id']
# required_law_cols = ['cluster_id', 'matched_laws']

# for col in required_q_cols:
#     if col not in df_questions.columns:
#         raise KeyError(f"Column '{col}' not found in clustered_questions.csv. Available columns: {df_questions.columns.tolist()}")
# for col in required_law_cols:
#     if col not in df_laws.columns:
#         raise KeyError(f"Column '{col}' not found in automated_cluster_laws.csv. Available columns: {df_laws.columns.tolist()}")

# print(f"Loaded {len(df_questions)} questions and {len(df_laws)} clusters of laws.")

# # ---------------------------
# # Step 2: Load Embedding Model and Compute Question Embeddings
# # ---------------------------
# model = SentenceTransformer("all-MiniLM-L6-v2")
# question_embeddings = model.encode(df_questions['question'].tolist(), show_progress_bar=True)

# # ---------------------------
# # Step 3: Compute Cluster Centroids
# # ---------------------------
# cluster_ids = df_questions['cluster_id'].unique()
# cluster_centroids = {}

# for cid in cluster_ids:
#     idxs = df_questions[df_questions['cluster_id'] == cid].index.tolist()
#     cluster_centroids[cid] = np.mean(question_embeddings[idxs], axis=0)

# print("Computed cluster centroids.")

# # ---------------------------
# # Step 4: Define Chatbot Function
# # ---------------------------
# def get_answer(user_question):
#     q_embed = model.encode([user_question])
    
#     # Compare with cluster centroids
#     sims = []
#     for cid, centroid in cluster_centroids.items():
#         sim = cosine_similarity([q_embed[0]], [centroid])[0][0]
#         sims.append((cid, sim))
    
#     # Get best cluster
#     best_cluster_id = max(sims, key=lambda x: x[1])[0]
    
#     # Retrieve matched laws/notes for that cluster
#     matched_laws = df_laws[df_laws['cluster_id'] == best_cluster_id]['matched_laws'].values
#     matched_laws_text = matched_laws[0] if len(matched_laws) > 0 else "No laws found"
    
#     return f"Closest cluster: {best_cluster_id}\nLaws/Notes: {matched_laws_text}"

# # ---------------------------
# # Step 5: Test the Chatbot (Interactive)
# # ---------------------------
# print("\nChatbot ready! Type 'exit' or 'quit' to quit.")
# while True:
#     user_question = input("\nAsk a question: ")
#     if user_question.lower() in ["exit", "quit"]:
#         break
#     answer = get_answer(user_question)
#     print("\n" + answer)

# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # ---------------------------
# # Step 1: Load Data
# # ---------------------------
# df_questions = pd.read_csv("clustered_questions.csv")  # Expected columns: question, cluster_id
# df_laws = pd.read_csv("automated_cluster_laws.csv")    # Expected columns: cluster_id, matched_laws

# # Normalize column names: lowercase + strip spaces
# df_questions.columns = [c.strip().lower() for c in df_questions.columns]
# df_laws.columns = [c.strip().lower() for c in df_laws.columns]

# # Safety check
# required_q_cols = ['question', 'cluster_id']
# required_law_cols = ['cluster_id', 'matched_laws']

# for col in required_q_cols:
#     if col not in df_questions.columns:
#         raise KeyError(f"Column '{col}' not found in clustered_questions.csv. Available columns: {df_questions.columns.tolist()}")
# for col in required_law_cols:
#     if col not in df_laws.columns:
#         raise KeyError(f"Column '{col}' not found in automated_cluster_laws.csv. Available columns: {df_laws.columns.tolist()}")

# print(f"Loaded {len(df_questions)} questions and {len(df_laws)} clusters of laws.")

# # ---------------------------
# # Step 2: Load Embedding Model and Compute Question Embeddings
# # ---------------------------
# model = SentenceTransformer("all-MiniLM-L6-v2")
# question_embeddings = model.encode(df_questions['question'].tolist(), show_progress_bar=True)

# # ---------------------------
# # Step 3: Compute Cluster Centroids
# # ---------------------------
# cluster_ids = df_questions['cluster_id'].unique()
# cluster_centroids = {}

# for cid in cluster_ids:
#     idxs = df_questions[df_questions['cluster_id'] == cid].index.tolist()
#     cluster_centroids[cid] = np.mean(question_embeddings[idxs], axis=0)

# print("Computed cluster centroids.")

# # ---------------------------
# # Step 4: Define Chatbot Function
# # ---------------------------
# def get_answer(user_question):
#     q_embed = model.encode([user_question])
    
#     # Compare with cluster centroids
#     sims = []
#     for cid, centroid in cluster_centroids.items():
#         sim = cosine_similarity([q_embed[0]], [centroid])[0][0]
#         sims.append((cid, sim))
    
#     # Get best cluster
#     best_cluster_id = max(sims, key=lambda x: x[1])[0]
    
#     # Retrieve matched laws/notes for that cluster
#     matched_laws = df_laws[df_laws['cluster_id'] == best_cluster_id]['matched_laws'].values
#     matched_laws_text = matched_laws[0] if len(matched_laws) > 0 else "No laws found"
    
#     return f"Closest cluster: {best_cluster_id}\nLaws/Notes: {matched_laws_text}"

# # ---------------------------
# # Step 5: Interactive Chatbot Wrapper
# # ---------------------------
# def run_chatbot():
#     print("\nChatbot ready! Type 'exit' or 'quit' to quit.")
#     while True:
#         user_question = input("\nAsk a question: ")
#         if user_question.lower() in ["exit", "quit"]:
#             print("Chatbot stopped.")
#             break
#         answer = get_answer(user_question)
#         print("\n" + answer)


import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Step 1: Load Data
# ---------------------------
df_questions = pd.read_csv("clustered_questions.csv")  # Expected columns: question, cluster_id
df_laws = pd.read_csv("automated_cluster_laws.csv")    # Expected columns: cluster_id, matched_laws

# Normalize column names: lowercase + strip spaces
df_questions.columns = [c.strip().lower() for c in df_questions.columns]
df_laws.columns = [c.strip().lower() for c in df_laws.columns]

# Safety check
required_q_cols = ['question', 'cluster_id']
required_law_cols = ['cluster_id', 'matched_laws']

for col in required_q_cols:
    if col not in df_questions.columns:
        raise KeyError(f"Column '{col}' not found in clustered_questions.csv. Available columns: {df_questions.columns.tolist()}")
for col in required_law_cols:
    if col not in df_laws.columns:
        raise KeyError(f"Column '{col}' not found in automated_cluster_laws.csv. Available columns: {df_laws.columns.tolist()}")

print(f"Loaded {len(df_questions)} questions and {len(df_laws)} clusters of laws.")

# ---------------------------
# Step 2: Load Embedding Model and Compute Question Embeddings
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = model.encode(df_questions['question'].tolist(), show_progress_bar=True)

# ---------------------------
# Step 3: Compute Cluster Centroids
# ---------------------------
cluster_ids = df_questions['cluster_id'].unique()
cluster_centroids = {}

for cid in cluster_ids:
    idxs = df_questions[df_questions['cluster_id'] == cid].index.tolist()
    cluster_centroids[cid] = np.mean(question_embeddings[idxs], axis=0)

print("Computed cluster centroids.")

# ---------------------------
# Step 4: Define Chatbot Function
# ---------------------------
def get_answer(user_question: str) -> str:
    if len(df_questions) == 0 or len(cluster_centroids) == 0:
        return "Chatbot data not loaded properly."
    
    q_embed = model.encode([user_question])
    
    # Compare with cluster centroids
    sims = [(cid, cosine_similarity([q_embed[0]], [centroid])[0][0]) for cid, centroid in cluster_centroids.items()]
    
    # Get best cluster
    best_cluster_id = max(sims, key=lambda x: x[1])[0]
    
    # Retrieve matched laws/notes for that cluster
    matched_laws = df_laws[df_laws['cluster_id'] == best_cluster_id]['matched_laws'].values
    matched_laws_text = matched_laws[0] if len(matched_laws) > 0 else "No laws found"
    
    return f"Closest cluster: {best_cluster_id}\nLaws/Notes: {matched_laws_text}"

# ---------------------------
# Step 5: Optional CLI Chatbot Wrapper
# ---------------------------
def run_chatbot():
    print("\nChatbot ready! Type 'exit' or 'quit' to quit.")
    while True:
        user_question = input("\nAsk a question: ")
        if user_question.lower() in ["exit", "quit"]:
            print("Chatbot stopped.")
            break
        answer = get_answer(user_question)
        print("\n" + answer)
