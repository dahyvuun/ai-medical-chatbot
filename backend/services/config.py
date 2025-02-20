import os
import torch


BASE_DIR="E:/project/ai_medical_chatbot"

MODEL_CONFIG = {
    "t5" : {
        "model_name" : os.path.join(BASE_DIR, "model/t5-medical-chatbot"),
        "tokenizer_name" : os.path.join(BASE_DIR, "model/t5-medical-chatbot"),
        "device" : "cuda" if torch.cuda.is_available() else "cpu"
    },
    "biobert":{
        "model_name": "dmis-lab/biobert-base-cased-v1.1",
        "tokenizer_name" : "dmis-lab/biobert-base-cased-v1.1",
        "device" : "cuda" if torch.cuda.is_available() else "cpu"
    }
}

API_CONFIG={
    "host" : "127.0.0.1",
    "port" : 8000,
    "debug" : True
}

print(f"✅ Config Loaded: T5 at {MODEL_CONFIG['t5']['model_name']} | BioBERT at {MODEL_CONFIG['biobert']['model_name']}")