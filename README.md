# Kraken Trading Bot

This repo runs a small Kraken trading setup from GitHub Actions.

Important: this can place real margin orders on Kraken. Use dry-run when testing.

## What Is Live

### AI Strategy Router

File: `strategy_router.py`

Workflow: `.github/workflows/strategy-router.yml`

Runs every 15 minutes. It checks signals from:

- old V4 swing bot
- ML V2 strategy
- ML V3 strategy

If there are valid candidates, OpenAI chooses:

- which strategy to use
- which symbol to trade
- how much budget each bot version should get
- leverage

The router uses `gpt-5.5` by default because this is a live-money decision layer. The smaller mini model is cheaper, but the router only calls OpenAI when there is a real candidate to evaluate.

The code still enforces hard safety limits before any order:

- max 1 open position
- max 25% free margin per trade
- max 35% total margin exposure
- max 2x leverage
- no new entries if there is already an open position or open order

### Daily Telegram Portfolio Report

File: `daily_portfolio_report.py`

Workflow: `.github/workflows/daily-portfolio-report.yml`

Runs every day at 09:00 Asia/Seoul and sends a Telegram portfolio update.

### Manual Workflows

These are not scheduled automatically:

- `.github/workflows/trading-bot-v4.yml`
- `.github/workflows/ml-live-trade.yml`
- paper-trading workflows

Use them manually only when needed.

## Required GitHub Secrets

Set these in GitHub:

`Settings -> Secrets and variables -> Actions`

Required:

```text
KRAKEN_API_KEY
KRAKEN_API_SECRET
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
OPENAI_API_KEY
```

Optional:

```text
NEWSDATA_API_KEY
CRYPTOCOMPARE_API_KEY
```

Do not commit real keys to the repo.

## Common Commands

Run the AI router safely in GitHub dry-run mode:

```bash
gh workflow run strategy-router.yml \
  --repo JKang78/Trading-Bot-V4 \
  --ref main \
  -f dry_run=true \
  -f ai_enabled=true
```

Run the daily portfolio report manually:

```bash
gh workflow run daily-portfolio-report.yml \
  --repo JKang78/Trading-Bot-V4 \
  --ref main
```

Run locally in dry-run mode:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
ROUTER_DRY_RUN=true venv/bin/python strategy_router.py
```

## Main Files

```text
strategy_router.py              AI strategy and budget router
kraken_bot_v4_advanced.py       old V4 swing bot
ml_live_trade.py                ML live trader
ml_strategy.py                  ML V2/V3 strategy logic
daily_portfolio_report.py       Telegram portfolio report
.github/workflows/              GitHub schedules and manual workflows
.env.example                    local environment template
```

## State Files

These files are generated and should not be committed:

```text
strategy_router_state.json
ml_live_state.json
v4_position_state.json
rl_state.json
.env
```

## Current Operating Model

GitHub Actions is the live runtime.

The AI router is the only scheduled trading controller. The old V4 bot and ML bot can still be run manually, but their automatic schedules are paused to avoid two bots trading the same Kraken account at the same time.
