# # main.py
# import clean
# import legal_mapping
# import chatbot_clustered
# import pandas as pd

# def main():
#     print("Step 1: Cleaning Data")
#     clean.clean_data()  # assuming clean.py has a function called clean_data()

#     print("Step 2: Legal Mapping")
#     legal_mapping.process_legal_mapping()  # assuming legal_mapping.py has this function

#     print("Step 3: Running Chatbot")
#     chatbot_clustered.run_chatbot()  # assuming chatbot_clustered.py has a run_chatbot() function

# if __name__ == "__main__":
#     main()

# # main.py
# import clean
# import legal_mapping
# import chatbot_clustered
# from fastapi import FastAPI
# from pydantic import BaseModel

# # ---------------------------
# # Step 0: Initialize FastAPI
# # ---------------------------
# app = FastAPI()

# # ---------------------------
# # Step 1: Run pipeline on startup
# # ---------------------------
# @app.on_event("startup")
# def startup_event():
#     print("Step 1: Cleaning Data")
#     clean.clean_data()

#     print("Step 2: Legal Mapping")
#     legal_mapping.process_legal_mapping()

#     print("✅ Pipeline completed. Chatbot API ready!")

# # ---------------------------
# # Step 2: Define request model
# # ---------------------------
# class QuestionRequest(BaseModel):
#     question: str

# # ---------------------------
# # Step 3: Define API endpoint
# # ---------------------------
# @app.post("/chat")
# def chat(request: QuestionRequest):
#     answer = chatbot_clustered.get_answer(request.question)
#     return {"question": request.question, "answer": answer}

# main.py
import clean
import legal_mapping
import chatbot_clustered
from fastapi import FastAPI
from pydantic import BaseModel
import threading

# ---------------------------
# Step 0: Initialize FastAPI
# ---------------------------
app = FastAPI()

# ---------------------------
# Step 1: Health check route
# ---------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Chatbot API is running 🚀"}

# ---------------------------
# Step 2: Run pipeline in background on startup
# ---------------------------
def run_pipeline():
    print("Step 1: Cleaning Data")
    clean.clean_data()

    print("Step 2: Legal Mapping")
    legal_mapping.process_legal_mapping()

    print("✅ Pipeline completed. Chatbot API ready!")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=run_pipeline).start()

# ---------------------------
# Step 3: Define request model
# ---------------------------
class QuestionRequest(BaseModel):
    question: str

# ---------------------------
# Step 4: Define API endpoint
# ---------------------------
@app.post("/chat")
def chat(request: QuestionRequest):
    # Ensure chatbot_clustered resources are loaded before calling get_answer
    # chatbot_clustered.load_resources()  # Add this function in chatbot_clustered.py
    answer = chatbot_clustered.get_answer(request.question)
    return {"question": request.question, "answer": answer}

