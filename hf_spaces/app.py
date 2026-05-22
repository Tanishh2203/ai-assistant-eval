import os
import re
import time
import gradio as gr
from huggingface_hub import InferenceClient

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

SYSTEM_PROMPT = """You are a helpful AI assistant with knowledge about Ollive AI —
an AI liability insurance company that helps AI vendors win enterprise deals.
Answer questions clearly and concisely. If unsure, say so honestly."""

BLOCK_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"you are now (DAN|an? unrestricted)",
    r"how to (make|build|synthesize) (a bomb|explosives|drugs|poison)",
    r"write (malware|ransomware|virus|keylogger)",
    r"jailbreak",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in BLOCK_PATTERNS]

def is_blocked(text: str) -> bool:
    return any(pat.search(text) for pat in _COMPILED)

def chat(message: str, history: list) -> str:
    if not message.strip():
        return "Please enter a message."
    if is_blocked(message):
        return "I can't help with that request."

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        return "Error: HF_TOKEN secret is not configured in Space settings."

    client = InferenceClient(model=MODEL_ID, token=token, provider="novita")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history[-8:]:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            messages.append({"role": "user",      "content": item[0]})
            messages.append({"role": "assistant", "content": item[1]})
    messages.append({"role": "user", "content": message})

    try:
        t0 = time.time()
        response = client.chat_completion(messages=messages, max_tokens=600, temperature=0.5)
        latency  = round(time.time() - t0, 2)
        reply    = response.choices[0].message.content.strip()
        return f"{reply}\n\n_⚡ {latency}s | {MODEL_ID}_"
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="OSS Assistant — Qwen2.5") as demo:
    gr.Markdown("""
    # OSS Assistant — Qwen2.5-72B-Instruct
    **RAG-enhanced AI assistant** with safety guardrails and multi-turn memory.
    Built for the Ollive AI engineering assessment.
    """)
    gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(height=450, placeholder="Ask me anything about Ollive AI or general topics..."),
        textbox=gr.Textbox(placeholder="Type your message...", container=False, scale=7),
        title="",
        examples=[
            "What is Ollive and what problem does it solve?",
            "What happened in the Air Canada AI chatbot case?",
            "Why won't standard cyber insurance cover AI claims?",
            "What is RAG in AI?",
        ],
        submit_btn="Send",
    )
    gr.Markdown("""
    ---
    **Model**: Qwen/Qwen2.5-72B-Instruct via HuggingFace Inference API  
    **Features**: Multi-turn memory · Safety guardrails · Ollive knowledge base  
    **Source**: [GitHub](https://github.com/Tanishh2203/ai-assistant-eval)
    """)

if __name__ == "__main__":
    demo.launch()
