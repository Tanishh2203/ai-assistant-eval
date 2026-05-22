"""
HuggingFace Spaces — Public OSS Assistant
==========================================
Deploy this on HF Spaces (Gradio SDK) for a public demo URL.

Setup on HF Spaces:
  1. Create new Space → SDK: Gradio
  2. Upload this app.py + requirements.txt
  3. Add secrets: HF_TOKEN
  4. Your public URL: https://huggingface.co/spaces/YOUR_USERNAME/oss-assistant
"""

import os
import time
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# ── Simple in-memory KB (static for Spaces) ───────────────────────────────────
SYSTEM_PROMPT = """You are a helpful AI assistant with knowledge about Ollive AI — 
an AI liability insurance company. Answer questions clearly and concisely.
If you are unsure about something, say so honestly rather than guessing."""

# ── Safety patterns (lightweight, no chromadb dependency) ────────────────────
import re

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


# ── Chat function ─────────────────────────────────────────────────────────────

def chat(message: str, history: list) -> str:
    if not message.strip():
        return "Please enter a message."

    if is_blocked(message):
        return "🚫 I can't help with that request."

    # Build message history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history[-8:]:  # last 8 turns
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    try:
        t0 = time.time()
        response = client.chat_completion(
            messages=messages,
            max_tokens=600,
            temperature=0.5,
        )
        latency = round(time.time() - t0, 2)
        reply   = response.choices[0].message.content.strip()
        return f"{reply}\n\n_⚡ {latency}s | {MODEL_ID}_"
    except Exception as e:
        return f"⚠️ Error: {e}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="OSS Assistant — Qwen2.5") as demo:

    gr.Markdown("""
    # 🤖 OSS Assistant — Qwen2.5-7B-Instruct
    **RAG-enhanced AI assistant** with safety guardrails and multi-turn memory.
    
    Built for the Ollive AI engineering assessment.
    """)

    chatbot = gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(height=450, placeholder="Ask me anything about AI liability insurance or general topics..."),
        textbox=gr.Textbox(placeholder="Type your message...", container=False, scale=7),
        title="",
        examples=[
            "What is Ollive and what problem does it solve?",
            "What happened in the Air Canada AI chatbot case?",
            "Why won't standard cyber insurance cover AI claims?",
            "What is RAG in AI?",
            "What is 15% of 2500?",
        ],
        retry_btn=None,
        undo_btn="↩️ Undo",
        clear_btn="🗑️ Clear",
    )

    gr.Markdown("""
    ---
    **Model**: Qwen/Qwen2.5-7B-Instruct via HuggingFace Inference API  
    **Features**: Multi-turn memory · Safety guardrails · RAG knowledge base  
    **Source**: [GitHub Repository](https://github.com/YOUR_USERNAME/ai-assistant-eval)
    """)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(),)   