import os
import sys
import logging
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

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
logger = logging.getLogger("frontier_bot")

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
BOT_TOKEN     = os.getenv("FRONTIER_BOT_TOKEN", "")
MODEL_ID      = "claude-sonnet-4-5"
MAX_HISTORY   = 10
TOP_K_RAG     = 3
CHROMA_DB     = str(ROOT / "chroma_db" / "frontier")

# ── Clients ───────────────────────────────────────────────────────────────────
claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

rag = RAGEngine(collection_name="frontier_kb", db_path=CHROMA_DB)
_loaded = rag.load_knowledge_base(str(ROOT / "knowledge_base"))
logger.info(f"Frontier RAG ready — {_loaded} chunks loaded from knowledge_base/")

# ── Per-user conversation memory ──────────────────────────────────────────────
user_history: dict[int, list[dict]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_system(rag_context: str) -> str:
    system = (
        "You are a helpful, concise AI assistant. "
        "Answer clearly and honestly. "
        "If you are unsure about something, acknowledge the uncertainty."
    )
    if rag_context:
        system += (
            "\n\nRelevant context retrieved from the knowledge base:\n"
            f"---\n{rag_context}\n---\n"
            "Use this context to ground your answers when applicable. "
            "If the answer isn't in the context, rely on your own knowledge "
            "and say so."
        )
    return system


def _build_messages(user_id: int, user_msg: str) -> list[dict]:
    messages = []
    for turn in user_history.get(user_id, [])[-MAX_HISTORY:]:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": user_msg})
    return messages


def _safe_reply(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…*(truncated)*"


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🤖 *Frontier Assistant* — powered by Claude Sonnet\n\n"
        "I'm a RAG-enhanced assistant from Anthropic. I search my knowledge "
        "base before every reply to give you accurate, grounded answers.\n\n"
        "*Commands:*\n"
        "/help — usage guide\n"
        "/clear — clear your conversation history\n"
        "/stats — knowledge base & memory stats\n\n"
        "📎 Send any `.txt` or `.pdf` file to expand my knowledge base.\n"
        "Then just ask me anything!",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📖 *How to use:*\n\n"
        "1. Type a message to start a conversation\n"
        "2. Upload a `.txt` or `.pdf` file to expand my knowledge base\n"
        "3. I'll automatically search the KB before each reply\n"
        "4. Use /clear to reset your conversation history\n\n"
        "_Tip: Upload your own docs and ask specific questions about them._",
        parse_mode="Markdown",
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
            "⚠️ Unsupported file type.\nPlease upload a `.txt` or `.pdf` file."
        )
        return

    status_msg = await update.message.reply_text(
        f"📄 Processing `{name}`…", parse_mode="Markdown"
    )

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
        await status_msg.edit_text(
            f"❌ Failed to process `{name}`: {exc}", parse_mode="Markdown"
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Chat handler ──────────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id  = update.effective_user.id
    user_msg = update.message.text.strip()

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    # 1. RAG retrieval
    chunks, metas = rag.retrieve(user_msg, top_k=TOP_K_RAG)
    rag_context   = "\n\n".join(chunks)

    # 2. Build system + messages
    system   = _build_system(rag_context)
    messages = _build_messages(user_id, user_msg)

    # 3. Call Claude
    try:
        resp  = claude.messages.create(
            model=MODEL_ID,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        reply = resp.content[0].text.strip()
    except Exception as exc:
        logger.error(f"Claude API error: {exc}")
        reply = f"⚠️ Model error: {exc}"

    # 4. Persist to memory
    if user_id not in user_history:
        user_history[user_id] = []
    user_history[user_id].append({"user": user_msg, "assistant": reply})

    # 5. Footer
    footer = ""
    if chunks:
        sources = sorted({m.get("source", "?") for m in metas})
        footer = f"\n\n📚 Sources: {', '.join(sources)}"

    await update.message.reply_text(_safe_reply(reply) + footer)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("FRONTIER_BOT_TOKEN is not set. Check your .env file.")
    if not ANTHROPIC_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Check your .env file.")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.Document.ALL,            handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info(f"🚀 Frontier Bot starting — model={MODEL_ID}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
