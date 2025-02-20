import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoTokenizer
from backend.services.config import MODEL_CONFIG

t5_path=MODEL_CONFIG["t5"]["model_name"]
biobert_path=MODEL_CONFIG["biobert"]["model_name"]

t5_tokenizer=AutoTokenizer.from_pretrained(t5_path)
biobert_tokenizer=AutoTokenizer.from_pretrained(biobert_path)

t5_model=AutoModelForSeq2SeqLM.from_pretrained(t5_path).to(MODEL_CONFIG["biobert"]["device"])
biobert_model=AutoModelForQuestionAnswering.from_pretrained(biobert_path).to(MODEL_CONFIG["biobert"]["device"])


def get_medical_response(query: str, use_t5:bool=False, context: str="Medical knowledge base.")-> str:
    """
    Generates a response using either:
    -**T5 model** (chatbot-like response) if 'use_t5=True'
    -**BioBERT model** (fact extraction) otherwise
    """

    if use_t5:
        inputs=t5_tokenzier(query, return_tensors="pt", truncation=True).to(MODEL_CONFIG["t5"]["device"])
        with torch.no_grad():
            outputs=t5_model.generate(**inputs)
        response=t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        inputs=biobert_tokenizer(query, context, return_tensors="pt", truncation=True).to(MODEL_CONFIG["biobert"]["device"])
        with torch.no_grad():
            outputs=biobert_model(**inputs)
            answer_start=torch.argmax(outputs.start_logits)
            answer_end=torch.argmax(outputs.end_logits)+1
            response=biobert_tokenizer.convert_tokens_to_string(
                biobert_toknizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])

            )

    return response


if __name__=="__main__":
    test_query="What are the symptoms of diabetes"
    print("\n🔹 Using **BioBERT** for Fact Extraction:")
    print(get_medical_response(test_query, use_t5=False))

    print("\n🔹 Using **T5** for Chatbot Response:")
    print(get_medical_response(test_query, use_t5=True))