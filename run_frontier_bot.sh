#!/bin/bash
# Run Frontier Bot (Claude Sonnet)
cd "$(dirname "$0")"

if [ ! -f ".env" ]; then
  echo "❌  .env not found. Copy .env.example → .env and fill in your values."
  exit 1
fi

source .env

if [ -z "$FRONTIER_BOT_TOKEN" ] || [ "$FRONTIER_BOT_TOKEN" = "your_frontier_telegram_bot_token_here" ]; then
  echo "❌  FRONTIER_BOT_TOKEN is not set in .env"
  exit 1
fi

if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your_anthropic_api_key_here" ]; then
  echo "❌  ANTHROPIC_API_KEY is not set in .env"
  exit 1
fi

echo "🚀 Starting Frontier Bot (Claude Sonnet)..."
python frontier_bot/bot.py
