import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/chat"

def query_medical_bot(user_input):
    response = requests.get(API_URL, params={"query": user_input}).json()
    return response.get("response", "Error")

iface=gr.Interface(fn=query_medical_bot, inputs="text", outputs="text", title="AI Medical Assistant")
iface.launch()