import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from huggingface_hub import InferenceClient

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared.rag_engine import RAGEngine

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv(ROOT / ".env")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("oss_bot")

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN", "")
BOT_TOKEN  = os.getenv("OSS_BOT_TOKEN", "")
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
MAX_HISTORY = 10          # conversation turns to keep in memory
TOP_K_RAG   = 3           # chunks to retrieve per query
CHROMA_DB   = str(ROOT / "chroma_db" / "oss")

# ── Clients ───────────────────────────────────────────────────────────────────
hf_client = InferenceClient(
    model=MODEL_ID,
    token=HF_TOKEN
)
rag = RAGEngine(collection_name="oss_kb", db_path=CHROMA_DB)
_loaded = rag.load_knowledge_base(str(ROOT / "knowledge_base"))
logger.info(f"OSS RAG ready — {_loaded} chunks loaded from knowledge_base/")

# ── Per-user conversation memory ──────────────────────────────────────────────
# { user_id: [ {"user": str, "assistant": str}, ... ] }
user_history: dict[int, list[dict]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_messages(user_id: int, user_msg: str, rag_context: str) -> list[dict]:
    system = (
        "You are a helpful, concise AI assistant. "
        "Answer clearly. If you are unsure, say so honestly."
    )
    if rag_context:
        system += (
            "\n\nRelevant context retrieved from the knowledge base:\n"
            f"---\n{rag_context}\n---\n"
            "Use this context to answer the user's question when applicable."
        )

    messages = [{"role": "system", "content": system}]

    for turn in user_history.get(user_id, [])[-MAX_HISTORY:]:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": user_msg})
    return messages


def _safe_reply(text: str, limit: int = 4000) -> str:
    """Truncate to Telegram's message limit."""
    return text if len(text) <= limit else text[:limit] + "\n…*(truncated)*"


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🤖 *OSS Assistant* — powered by Qwen2\.5\-0\.5B\n\n"
        "I'm a RAG\-enhanced assistant\. I search my knowledge base before "
        "every reply to give you grounded, accurate answers\.\n\n"
        "*Commands:*\n"
        "/help — usage guide\n"
        "/clear — clear your conversation history\n"
        "/stats — knowledge base & memory stats\n\n"
        "📎 Send any *\.txt* or *\.pdf* file to add it to my knowledge base\.\n"
        "Then just ask me anything\!",
        parse_mode="MarkdownV2",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📖 *How to use:*\n\n"
        "1\\. Type a message to start a conversation\n"
        "2\\. Upload a *\\.txt* or *\\.pdf* file to expand my knowledge base\n"
        "3\\. I'll automatically search the KB before each reply\n"
        "4\\. Use /clear to reset your conversation history\n\n"
        "_Tip: The more docs you upload, the better my answers\._",
        parse_mode="MarkdownV2",
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_history[user_id] = []
    await update.message.reply_text("✅ Conversation history cleared!")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id  = update.effective_user.id
    rag_stat = rag.get_stats()
    hist_len = len(user_history.get(user_id, []))
    await update.message.reply_text(
        f"📊 *Stats*\n\n"
        f"🗄 KB chunks: `{rag_stat['total_chunks']}`\n"
        f"💬 Your turns in memory: `{hist_len}` / `{MAX_HISTORY}`\n"
        f"🧠 Model: `{MODEL_ID}`",
        parse_mode="Markdown",
    )


# ── Document ingestion ────────────────────────────────────────────────────────

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc  = update.message.document
    name = doc.file_name or ""

    if not (name.lower().endswith(".txt") or name.lower().endswith(".pdf")):
        await update.message.reply_text(
            "⚠️ Unsupported file type.\n"
            "Please upload a `.txt` or `.pdf` file."
        )
        return

    status_msg = await update.message.reply_text(f"📄 Processing `{name}`…", parse_mode="Markdown")

    tg_file  = await context.bot.get_file(doc.file_id)
    raw      = await tg_file.download_as_bytearray()
    suffix   = Path(name).suffix.lower()
    tmp_path = Path(f"/tmp/{doc.file_id}{suffix}")
    tmp_path.write_bytes(raw)

    try:
        chunks = rag.add_file(str(tmp_path))
        await status_msg.edit_text(
            f"✅ Added *{chunks}* chunks from `{name}` to the knowledge base.",
            parse_mode="Markdown",
        )
    except Exception as exc:
        logger.error(f"File ingestion error: {exc}")
        await status_msg.edit_text(f"❌ Failed to process `{name}`: {exc}", parse_mode="Markdown")
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Chat handler ──────────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id  = update.effective_user.id
    user_msg = update.message.text.strip()

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    # 1. Retrieve relevant KB chunks
    chunks, metas = rag.retrieve(user_msg, top_k=TOP_K_RAG)
    rag_context   = "\n\n".join(chunks)

    # 2. Build conversation messages
    messages = _build_messages(user_id, user_msg, rag_context)

    is_error = False
    try:
        hf_resp = hf_client.chat_completion(
            messages=messages, max_tokens=600, temperature=0.5,
        )
        reply = hf_resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.error(f"HF API error: {exc}")
        reply = f"⚠️ Model error: {exc}"
        is_error = True

    if not is_error:
        if user_id not in user_history:
            user_history[user_id] = []
        user_history[user_id].append({"user": user_msg, "assistant": reply})

    if is_error:
        await update.message.reply_text(_safe_reply(reply))
        return

    footer = ""
    if chunks:
        sources = sorted({m.get("source", "?") for m in metas})
        footer = f"\n\n📚 Sources: {', '.join(sources)}"

    await update.message.reply_text(_safe_reply(reply) + footer)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("OSS_BOT_TOKEN is not set. Check your .env file.")
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set. Check your .env file.")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.Document.ALL,             handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,  handle_message))

    logger.info(f"🚀 OSS Bot starting — model={MODEL_ID}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
