#!/bin/bash
# Run OSS Bot (Qwen2.5)
cd "$(dirname "$0")"

if [ ! -f ".env" ]; then
  echo "❌  .env not found. Copy .env.example → .env and fill in your values."
  exit 1
fi

source .env

if [ -z "$OSS_BOT_TOKEN" ] || [ "$OSS_BOT_TOKEN" = "your_oss_telegram_bot_token_here" ]; then
  echo "❌  OSS_BOT_TOKEN is not set in .env"
  exit 1
fi

if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
  echo "❌  HF_TOKEN is not set in .env"
  exit 1
fi

echo "🚀 Starting OSS Bot (Qwen2.5-0.5B-Instruct)..."
python oss_bot/bot.py
