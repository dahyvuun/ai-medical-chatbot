from fastapi import APIRouter, Query
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import torch

T5_MODEL_PATH="E:/project/ai_medical_chatbot/model/t5-medical-chatbot"
BIOBERT_MODEL_PATH = "E:/project/ai_medical_chatbot/model/biobert"


t5_tokenizer=AutoTokenizer.from_pretrained(T5_MODEL_PATH)
t5_model=AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_PATH)

biobert_tokenizer=AutoTokenizer.from_pretrained(BIOBERT_MODEL_PATH)
biobert_model=AutoModelForQuestionAnswering.from_pretrained(BIOBERT_MODEL_PATH)

router=APIRouter()


def get_medical_response_t5(query: str) -> str:
    """Generate medical response using fine_tuned T5 model"""
    inputs=t5_tokenizer(query, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs=t5_model.generate(**inputs)

    response=t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def get_medical_response_biobert(question:str, context: str="Medical knowledge base.")-> str:
    """Generate medical response using BioBERT for fact extraction"""

    inputs=biobert_tokenizer(question, context, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs=biobert_model(**inputs)
        answer_start=torch.argmax(outputs.start_logits)
        answer_end=torch.argmax(outputs.end_logits) +1
        response=biobert_tokenizer.convert_tokens_to_string(
            biobrt_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )

    return response


@router.get("/chat")
def chat(
    query: str=Query(..., description="User medical question"),
    use_t5: bool = Query(False, description="Use T5 for chatbot response"),
    context: str=Query(None, description="Medical context for BioBERT (optional)")

):
    """
    API to get chatbot response dynamically :
    -If 'use_t5=True', uses T5 model for chatbot-style responses.
    -Otherwise, uses BioBERT for fact extraction."""

    if use_t5:
        response=get_medical_response_t5(query)
    else:
        response=get_medical_response_biobert(query, context or "Medical knowledge base.")

    return {"response": response}