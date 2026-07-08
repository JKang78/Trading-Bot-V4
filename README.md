# Kraken Trading Bot

This repo runs a small Kraken trading setup from GitHub Actions.

Important: this can place real margin orders on Kraken. Use dry-run when testing.

## What Is Live

### ML V2 Live Trader

File: `ml_live_trade.py`

Workflow: `.github/workflows/ml-live-trade.yml`

Runs every hour in live mode with real money. This is the only scheduled
trading controller. It trades the ML V2 strategy, which was the most profitable
profile in our backtests.

Coins: `XRP-USD, ADA-USD, SOL-USD, LINK-USD, DOGE-USD` (each passed
walk-forward validation with positive after-cost expectancy).

How it trades:

- Long-only by default (the backtested short side barely covered its costs).
- Each trade uses 20% of usable margin at 2x leverage.
- Hold ~3 days (72 x 1h bars), then close on a time-based exit.
- At most one position per coin, up to 5 open positions.
- Entries try a post-only maker limit order first, then fall back to a market
  order only if the expected value still survives taker fees and slippage.
- Exits are always market orders (a time-boxed strategy must be able to get out).

State (which positions we opened and when to close them) is saved to
`ml_live_state.json` and cached between runs so it survives independent cron runs.

Fresh V2/V3 research is captured by `research_v2_v3_profitability.py`.

To test safely, run the workflow manually with `dry_run=true` — it runs the full
logic without placing real orders.

### Daily Telegram Portfolio Report

File: `daily_portfolio_report.py`

Workflow: `.github/workflows/daily-portfolio-report.yml`

Runs every day at 09:00 Asia/Seoul and sends a Telegram portfolio update.

### Manual Workflows

These are not scheduled automatically:

- `.github/workflows/trading-bot-v4.yml` (old V4 swing bot)
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
```

Optional:

```text
NEWSDATA_API_KEY
CRYPTOCOMPARE_API_KEY
```

Do not commit real keys to the repo.

## Common Commands

Run the ML V2 live trader manually in live mode (real money):

```bash
gh workflow run ml-live-trade.yml \
  --repo JKang78/Trading-Bot-V4 \
  --ref main \
  -f dry_run=false \
  -f strategy=v2
```

Run the ML V2 live trader safely in GitHub dry-run mode (no real orders):

```bash
gh workflow run ml-live-trade.yml \
  --repo JKang78/Trading-Bot-V4 \
  --ref main \
  -f dry_run=true \
  -f strategy=v2
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
ML_LIVE_DRY_RUN=true venv/bin/python ml_live_trade.py
```

Run the standard ML V2 validation backtest that matches the live profile:

```bash
venv/bin/python ml_strategy_backtest.py \
  --live-profile v2 \
  --period 720d \
  --out ml_strategy_trades_live_v2.csv
```

Run the V2/V3 fee-sensitivity research sweep:

```bash
venv/bin/python research_v2_v3_profitability.py
```

Research the old V4 bear-market short profile with margin fees included:

```bash
venv/bin/python backtest.py \
  --symbols BTC-USD,ETH-USD \
  --period 720d \
  --interval 1h \
  --leverage 2 \
  --directions short \
  --trend-ema 200 \
  --fee 0.0040 \
  --exit-fee 0.0080 \
  --margin-open-fee 0.0004 \
  --rollover-fee-4h 0.0004 \
  --spread-slippage-buffer 0.0015 \
  --min-confidence 0.80 \
  --max-signal-age-hours 12 \
  --min-expectancy-pct 0.25 \
  --out v4_short_bear_research.csv
```

## Main Files

```text
ml_live_trade.py                ML V2 live trader (scheduled, real money)
ml_strategy.py                  ML V2/V3 strategy logic
kraken_bot_v4_advanced.py       old V4 swing bot (manual only)
research_v2_v3_profitability.py V2/V3 walk-forward fee-sensitivity research
daily_portfolio_report.py       Telegram portfolio report
.github/workflows/              GitHub schedules and manual workflows
.env.example                    local environment template
```

## State Files

These files are generated and should not be committed:

```text
ml_live_state.json
v4_position_state.json
rl_state.json
.env
```

## Current Operating Model

GitHub Actions is the scheduled runtime.

The ML V2 live trader (`ml-live-trade.yml`) is the only scheduled trading
controller and runs hourly as live by default. The old V4 bot and paper-trading
workflows can still be run manually, but their automatic schedules are paused to
avoid two bots trading the same Kraken account at the same time.
