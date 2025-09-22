Education Agent RAG Pipeline
Project Overview

This project builds a Retrieval-Augmented Generation (RAG) pipeline to process and analyze data from education agents.
The workflow includes:

Data Cleaning → preprocess raw education agent/student data.

Data Storage → store structured data in MongoDB.

Rule Matching → validate and match dataset records against legal & compliance rules.

RAG-based Q&A → use LLMs to answer user queries by retrieving relevant records from MongoDB and applying rules.

The goal is to create a system that ensures compliance and provides intelligent, rule-aware answers.

Tech Stack

Python

MongoDB (for structured storage & retrieval)

LangChain / HuggingFace / OpenAI API (for RAG pipeline)

Pandas (for data cleaning)

Pipeline Steps

Data Cleaning: Handle missing values, normalize fields, remove duplicates.

MongoDB Storage: Insert cleaned dataset into a NoSQL collection.

Rule Matching: Compare data fields with legal/compliance rules.

RAG Pipeline:

Retrieve relevant records from MongoDB.

Augment prompt with rules & context.

Generate final answer using LLM.

 Workflow

Upload raw agent/student dataset.

Run cleaning script → generates structured data.

Store records into MongoDB.

Run rule-check module → highlights non-compliant cases.

Ask queries like:

“List agents compliant with Rule X.”

Future Improvements

Add a web dashboard for compliance monitoring.

Automate rule updates from legal documents.

Support multiple databases (Postgres, Qdrant, etc.)

“Which students need additional documents?”

“Summarize compliance issues in dataset.”
