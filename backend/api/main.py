from fastapi import FastAPI, Query
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForQuestionAnswering
import torch
import os


T5_MODEL_PATH = "E:/project/ai_medical_chatbot/model/t5-medical-chatbot"
BIOBERT_MODEL_PATH = "E:/project/ai_medical_chatbot/model/biobert"


t5_tokenizer=AutoTokenizer.from_pretrained(T5_MODEL_PATH)
biobert_tokenizer=AutoTokenizer.from_pretrained(BIOBERT_MODEL_PATH)

t5_model=AutoTokenizer.from_pretrained(T5_MODEL_PATH)
biobert_tokenizer=AutoTokenizer.from_pretrained(BIOBERT_MODEL_PATH)


DEVICE="cuda" if torch.cuda.is_available() else "cpu"
t5_model.to(DEVICE)
biobert_model.to(DEVICE)

app=FastAPI()

def extract_fact_with_biobert(question:str, context: str) ->str:

    """
    Uses BioBERT to extract factual information from a given context"""
    inputs= biobert_tokenizer(question, context, return_tensors="pt",truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs=biobert_model(**inputs)

    start_scores=outputs.start_logits
    end_scores=outputs.end_logits
    answer_start= torch.argmax(start_scores)
    answer_end=torch.argmax(end_scores) +1
    answer =biobert_tokenizer.convert_tokens_to_string(
        biobert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])

    )
    return answer if answer.strip() else "Sorry"


def generate_response_with_t5(fact:str, question:str) -> str:
    """
    Uses T5 to generate a user-friendly response based on the factual answer."""

    prompt=f"Question : {question}\n Fact: {fact}\n Answer:"
    inputs=t5_tokenizer(prompt, return_tensors="pt",truncation=True).to(DEVICE)
    with torch.no_grad():
        output=t5_model.generate(**inputs)
    return t5_tokenizer.decode(output[0], skip_special_tokens=True)

@app.get("/chat")

async def chat(
    query: str= Query(..., description = "Enter your medical question"),
    context: str=Query(
        "Diabetes is a chronic condition that affects blood sugar"
    )
):


    response = get_medical_response(query, context)
    return {"response": response}

@app.get("/")
async def root():
    return {"message" : "Medical Chatbot API is running"}