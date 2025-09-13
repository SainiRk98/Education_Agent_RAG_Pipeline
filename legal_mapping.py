# # fully_automated_legal_mapping.py

# import os
# import pandas as pd
# import PyPDF2
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # ---------------------------
# # Step 1: Load Clustered Questions
# # ---------------------------
# df = pd.read_csv("clustered_questions.csv")

# # Normalize column names
# df.columns = [c.strip().lower() for c in df.columns]

# # Detect the question column
# possible_cols = ["question", "questions", "text", "query"]
# question_col = None
# for col in possible_cols:
#     if col in df.columns:
#         question_col = col
#         break

# if question_col is None:
#     raise ValueError(f"❌ Could not find a 'question' column in CSV. Found: {df.columns.tolist()}")

# print(f"✅ Loaded {len(df)} questions using column '{question_col}'.")

# # ---------------------------
# # Step 2: Extract Legal Texts from PDFs
# # ---------------------------
# def extract_text_from_pdf(file_path):
#     text = ""
#     with open(file_path, "rb") as f:
#         reader = PyPDF2.PdfReader(f)
#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#     return text

# legal_docs_folder = "legal_docs"
# if not os.path.exists(legal_docs_folder):
#     os.makedirs(legal_docs_folder)
#     print(f"⚠️ Created empty folder '{legal_docs_folder}'. Please add legal PDF files and rerun.")
#     exit()

# legal_texts = []
# for file in os.listdir(legal_docs_folder):
#     if file.endswith(".pdf"):
#         print(f"📄 Reading {file} ...")
#         legal_texts.append(extract_text_from_pdf(os.path.join(legal_docs_folder, file)))

# print(f"✅ Extracted text from {len(legal_texts)} legal documents.")

# # ---------------------------
# # Step 3: Split Legal Texts into Chunks
# # ---------------------------
# def split_into_chunks(text, chunk_size=500):
#     words = text.split()
#     chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
#     return chunks

# all_chunks = []
# for text in legal_texts:
#     all_chunks.extend(split_into_chunks(text))

# print(f"✅ Split legal documents into {len(all_chunks)} chunks.")

# # ---------------------------
# # Step 4: Generate Embeddings
# # ---------------------------
# print("🔄 Generating embeddings...")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# question_embeddings = model.encode(df[question_col].tolist(), show_progress_bar=True)
# legal_embeddings = model.encode(all_chunks, show_progress_bar=True)

# # ---------------------------
# # Step 5: Compute Similarity & Match
# # ---------------------------
# print("🤝 Matching questions to legal text...")
# similarities = cosine_similarity(question_embeddings, legal_embeddings)

# top_n = 3  # top N matches per question
# matches = []
# for i, sims in enumerate(similarities):
#     top_indices = sims.argsort()[-top_n:][::-1]
#     matched_texts = [all_chunks[j] for j in top_indices]
#     matches.append(matched_texts)

# df["matched_laws"] = matches

# # ---------------------------
# # Step 6: Save Output
# # ---------------------------
# output_file = "automated_cluster_laws.csv"
# df.to_csv(output_file, index=False)
# print(f"✅ Saved automated legal mapping to {output_file}")


# legal_mapping.py

import os
import pandas as pd
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def process_legal_mapping(
    clustered_file="clustered_questions.csv",
    legal_docs_folder="legal_docs",
    output_file="automated_cluster_laws.csv"
):
    # ---------------------------
    # Step 1: Load Clustered Questions
    # ---------------------------
    df = pd.read_csv(clustered_file)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect the question column
    possible_cols = ["question", "questions", "text", "query"]
    question_col = None
    for col in possible_cols:
        if col in df.columns:
            question_col = col
            break

    if question_col is None:
        raise ValueError(f"❌ Could not find a 'question' column in CSV. Found: {df.columns.tolist()}")

    print(f"✅ Loaded {len(df)} questions using column '{question_col}'.")

    # ---------------------------
    # Step 2: Extract Legal Texts from PDFs
    # ---------------------------
    def extract_text_from_pdf(file_path):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    if not os.path.exists(legal_docs_folder):
        os.makedirs(legal_docs_folder)
        print(f"⚠️ Created empty folder '{legal_docs_folder}'. Please add legal PDF files and rerun.")
        return  # stop if no PDFs

    legal_texts = []
    for file in os.listdir(legal_docs_folder):
        if file.endswith(".pdf"):
            print(f"📄 Reading {file} ...")
            legal_texts.append(extract_text_from_pdf(os.path.join(legal_docs_folder, file)))

    print(f"✅ Extracted text from {len(legal_texts)} legal documents.")

    # ---------------------------
    # Step 3: Split Legal Texts into Chunks
    # ---------------------------
    def split_into_chunks(text, chunk_size=500):
        words = text.split()
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    all_chunks = []
    for text in legal_texts:
        all_chunks.extend(split_into_chunks(text))

    print(f"✅ Split legal documents into {len(all_chunks)} chunks.")

    # ---------------------------
    # Step 4: Generate Embeddings
    # ---------------------------
    print("🔄 Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    question_embeddings = model.encode(df[question_col].tolist(), show_progress_bar=True)
    legal_embeddings = model.encode(all_chunks, show_progress_bar=True)

    # ---------------------------
    # Step 5: Compute Similarity & Match
    # ---------------------------
    print("🤝 Matching questions to legal text...")
    similarities = cosine_similarity(question_embeddings, legal_embeddings)

    top_n = 3  # top N matches per question
    matches = []
    for i, sims in enumerate(similarities):
        top_indices = sims.argsort()[-top_n:][::-1]
        matched_texts = [all_chunks[j] for j in top_indices]
        matches.append(matched_texts)

    df["matched_laws"] = matches

    # ---------------------------
    # Step 6: Save Output
    # ---------------------------
    df.to_csv(output_file, index=False)
    print(f"✅ Saved automated legal mapping to {output_file}")

