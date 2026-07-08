# 🚀 Kraken Trading Bot V4 - Advanced AI System

Next-generation trading bot with **Machine Learning**, **On-Chain Analysis**, **Sentiment Analysis**, **Multi-Strategy Ensemble**, and **Reinforcement Learning**.

## 🆕 What's New in V4

### 🎯 New Advanced Features

#### 1. **📊 Sentiment Analysis** (CryptoCompare API)
- Real-time news analysis
- Social media metrics (Twitter, Reddit)
- Sentiment score (-1 to +1)
- Signal filtering based on market sentiment

#### 2. **🔗 On-Chain Metrics**
- Exchange flows (deposits/withdrawals)
- Active addresses trend
- Transaction volume analysis
- Whale activity detection
- Crypto-specific blockchain signals

#### 3. **🤝 Multi-Strategy Ensemble**
- **4 strategies** running in parallel:
  - Swing Trading (pivot detection)
  - Momentum Strategy (RSI + ROC)
  - Mean Reversion (Bollinger Bands)
  - Trend Following (MA crossovers)
- **Weighted voting** system
- **Consensus** level across strategies
- More robust final decision

#### 4. **🤖 Reinforcement Learning Position Sizing**
- **Dynamic** position sizing based on RL
- Learns from previous trades
- Adjusts allocation and leverage based on conditions
- Q-Learning with discretized states
- Persists knowledge between runs

---

## 📋 Requirements

### Required APIs

1. **Kraken API**
   - Account with margin trading
   - API key + secret with trading permissions

2. **CryptoCompare API** (NEW)
   - Free account at [CryptoCompare](https://www.cryptocompare.com/)
   - API key for sentiment and on-chain data
   - Free tier: 100,000 calls/month

3. **Telegram Bot** (optional)
   - Bot token
   - Chat ID

---

## 🚀 Quick Setup

### 1. Configure Secrets in GitHub

**Settings → Secrets and variables → Actions**

```
KRAKEN_API_KEY=your_kraken_key
KRAKEN_API_SECRET=your_kraken_secret
CRYPTOCOMPARE_API_KEY=your_cryptocompare_key  # NEW
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 2. File Structure

```
your-repo/
├── .github/
│   └── workflows/
│       └── trading-bot-v4.yml
├── kraken_bot_v3_multi_asset.py  # Base bot (V3)
├── sentiment_analyzer.py          # NEW
├── onchain_metrics.py             # NEW
├── ensemble_strategies.py         # NEW
├── rl_position_sizing.py          # NEW
├── requirements.txt               # Updated
└── README_V4.md
```

### 3. Update requirements.txt

```txt
requests>=2.31.0
yfinance>=1.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0  # For additional ML
```

---

## ⚙️ Configuration

### Environment Variables

In `trading-bot-v4.yml`:

```yaml
# APIs
KRAKEN_API_KEY: ${{ secrets.KRAKEN_API_KEY }}
KRAKEN_API_SECRET: ${{ secrets.KRAKEN_API_SECRET }}
CRYPTOCOMPARE_API_KEY: ${{ secrets.CRYPTOCOMPARE_API_KEY }}  # NEW

# Sentiment Analysis
USE_SENTIMENT_ANALYSIS: 'true'           # NEW
MIN_SENTIMENT_CONFIDENCE: '0.5'          # NEW

# On-Chain Analysis  
USE_ONCHAIN_ANALYSIS: 'true'             # NEW
MIN_ONCHAIN_STRENGTH: '0.5'              # NEW

# Ensemble System
USE_ENSEMBLE_SYSTEM: 'true'              # NEW
MIN_ENSEMBLE_CONSENSUS: '0.5'            # NEW
MIN_ENSEMBLE_CONFIDENCE: '0.5'           # NEW

# Reinforcement Learning
USE_RL_POSITION_SIZING: 'true'           # NEW
RL_LEARNING_RATE: '0.1'                  # NEW
RL_EPSILON: '0.1'                        # NEW

# Strategy Weights (Ensemble)
WEIGHT_SWING: '0.30'                     # NEW
WEIGHT_MOMENTUM: '0.25'                  # NEW
WEIGHT_MEAN_REVERSION: '0.25'            # NEW
WEIGHT_TREND_FOLLOWING: '0.20'           # NEW

# Trading (existing)
MAX_POSITIONS: '3'
LEVERAGE: '3'
STOP_LOSS_PCT: '4.0'
TAKE_PROFIT_PCT: '8.0'
# ... rest of V3 configuration
```

---

## 🎮 Using the System

### V4 Decision Flow

```
1. Fetch market data (yfinance)
   ↓
2. Multi-Layer Analysis:
   ├─ Swing Detection (V3)
   ├─ Sentiment Analysis (NEWS + SOCIAL) ← NEW
   ├─ On-Chain Metrics (BLOCKCHAIN) ← NEW
   └─ Ensemble Strategies (4 strategies) ← NEW
   ↓
3. Aggregated Decision:
   - All layers must be aligned
   - Minimum consensus required
   - Minimum confidence required
   ↓
4. RL Position Sizing: ← NEW
   - Calculate market state
   - Select best action
   - Determine optimal capital and leverage
   ↓
5. Execute Trade
   ↓
6. Update RL Agent:
   - Calculate reward
   - Update Q-values
   - Save knowledge
```

### Example Output

```
═══════════════════════════════════════════════════════════
KRAKEN TRADING BOT V4 - ADVANCED AI SYSTEM
═══════════════════════════════════════════════════════════
📅 2024-12-28 15:30:00
Mode: 🧪 SIMULATION

💰 Balance: 1,250.50 EUR
   Available margin: 1,100.00 EUR
   Open positions: 1/3

🔍 Analyzing BTC-USD...

   💭 Sentiment: Overall=0.45 (BULLISH)
      News=0.40, Social=0.50, Confidence=1.0
   
   🔗 OnChain: BULLISH (strength: 0.65)
      Metrics: exchange_flow=0.3, address_trend=0.4

   📊 ENSEMBLE DECISION
   Signal: BUY
   Confidence: 78%
   Consensus: 75% (3/4 strategies agree)
   
   Individual votes:
   ✓ swing: BUY (0.85)
   ✓ momentum: BUY (0.70)
   ✗ mean_reversion: NONE (0.00)
   ✓ trend_following: BUY (0.65)

   🤖 RL Position Sizing:
      State: vol=0.025, trend=0.08, wr=0.60
      Decision: $400.00 @ 4x leverage
      (Alloc: 40%, Lev mult: 1.3x)

🟢 Opening BUY on BTC-USD
   Price: $42,350.00
   Volume: 0.0094
   Leverage: 4x
   Margin: 400.00 EUR
   
   Ensemble Confidence: 78%
   Sentiment: BULLISH (0.45)
   OnChain: BULLISH (0.65)

✅ Order executed
```

---

## 📊 Ensemble System in Detail

### Included Strategies

#### 1. **Swing Trading**
- Intermediate highs/lows detection
- Volume validation
- ML quality scoring

#### 2. **Momentum Strategy**
- RSI (Relative Strength Index)
- Rate of Change (ROC)
- Volume confirmation
- Detects breakouts and reversals

#### 3. **Mean Reversion**
- Bollinger Bands
- Oversold/Overbought zones
- Ideal for sideways markets

#### 4. **Trend Following**
- Moving Average crossovers
- Golden/Death crosses
- Trend strength measurement

### Default Weights

| Strategy | Weight | When It Works Best |
|------------|------|------------------|
| Swing | 30% | Markets with defined ranges |
| Momentum | 25% | Breakouts and strong trends |
| Mean Reversion | 25% | Sideways markets |
| Trend Following | 20% | Sustained trends |

**Customizable** in configuration.

---

## 🤖 Reinforcement Learning

### How Does It Work?

1. **Market State** (6 features):
   - Current volatility
   - Trend strength
   - Recent win rate
   - Current drawdown
   - Open positions
   - Signal confidence

2. **Available Actions** (15 combinations):
   - Allocation: 20%, 33%, 50%, 70%, 100%
   - Leverage multiplier: 0.5x, 1.0x, 1.5x

3. **Rewards**:
   - Based on trade PnL%
   - Bonus for large winning trades
   - Penalty for large losses
   - Bonus for correct stop usage

4. **Learning**:
   - Q-Learning with discretized states
   - Epsilon-greedy exploration (10%)
   - Learning rate: 0.1
   - Discount factor: 0.95

### Persistence

- Q-table is saved in `rl_state.json`
- Loaded automatically on each run
- Improves over time based on results

---

## 🔬 Backtesting

The V4 system is compatible with the existing V3 backtester. To integrate the new features:

### Backtesting with Ensemble

```python
# In backtest_v3_walkforward.py, modify:

from ensemble_strategies import EnsembleSystem

class BacktesterV4(BacktesterV3):
    def __init__(self, config, market_data):
        super().__init__(config, market_data)
        self.ensemble = EnsembleSystem()
    
    def _look_for_signals(self, current_date, current_prices):
        # ... existing code ...
        
        for symbol, data in data_up_to_date.items():
            # Swing signal
            signal, signal_price, confidence = self.detectors[symbol].get_signal_at_date(current_date)
            
            if signal:
                # Verify with ensemble
                ensemble_decision = self.ensemble.get_ensemble_decision(
                    data, (signal, signal_price, confidence)
                )
                
                if (ensemble_decision.final_signal == signal and 
                    ensemble_decision.consensus_level > 0.5):
                    # Proceed with the trade
                    pass
```

---

## 📈 Metrics and Monitoring

### New V4 Metrics

1. **Sentiment Accuracy**
   - Sentiment vs performance correlation
   - True positive rate of bullish/bearish signals

2. **Ensemble Performance**
   - Win rate per individual strategy
   - Average consensus in winning vs losing trades
   - Contribution of each strategy to total PnL

3. **RL Learning Progress**
   - Q-values evolution
   - Exploration vs exploitation ratio
   - Average reward per episode

4. **On-Chain Signal Accuracy**
   - Exchange flow signal accuracy
   - Active addresses correlation with price movement

### Suggested Dashboard

```python
import matplotlib.pyplot as plt

def plot_v4_metrics(trades_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Sentiment vs Performance
    axes[0,0].scatter(trades_df['sentiment_score'], 
                     trades_df['pnl_pct'])
    axes[0,0].set_title('Sentiment vs PnL')
    
    # 2. Ensemble Consensus Distribution
    axes[0,1].hist(trades_df['ensemble_consensus'], bins=20)
    axes[0,1].set_title('Ensemble Consensus')
    
    # 3. RL Allocation Over Time
    axes[1,0].plot(trades_df['date'], 
                  trades_df['rl_allocation'])
    axes[1,0].set_title('RL Position Sizing Evolution')
    
    # 4. Strategy Win Rates
    strategy_wr = trades_df.groupby('winning_strategy')['is_win'].mean()
    axes[1,1].bar(strategy_wr.index, strategy_wr.values)
    axes[1,1].set_title('Win Rate by Strategy')
    
    plt.tight_layout()
    plt.savefig('v4_metrics.png')
```

---

## 🎯 Recommended Strategies

### Conservative Profile

```yaml
# Conservative configuration with AI
MAX_POSITIONS: '2'
LEVERAGE: '2'

# Very strict Sentiment & OnChain
MIN_SENTIMENT_CONFIDENCE: '0.7'
MIN_ONCHAIN_STRENGTH: '0.7'

# Ensemble requires high consensus
MIN_ENSEMBLE_CONSENSUS: '0.75'  # 3/4 strategies must agree
MIN_ENSEMBLE_CONFIDENCE: '0.7'

# RL will explore less
RL_EPSILON: '0.05'

# Balanced weights
WEIGHT_SWING: '0.25'
WEIGHT_MOMENTUM: '0.25'
WEIGHT_MEAN_REVERSION: '0.30'  # Higher in sideways markets
WEIGHT_TREND_FOLLOWING: '0.20'
```

### Aggressive Profile (Experimental AI)

```yaml
MAX_POSITIONS: '4'
LEVERAGE: '5'

# Less restrictive
MIN_SENTIMENT_CONFIDENCE: '0.3'
MIN_ONCHAIN_STRENGTH: '0.3'

# Lower consensus required
MIN_ENSEMBLE_CONSENSUS: '0.5'  # 2/4 is enough
MIN_ENSEMBLE_CONFIDENCE: '0.5'

# RL explores more
RL_EPSILON: '0.2'  # 20% exploration

# Weights favor momentum and trend
WEIGHT_SWING: '0.20'
WEIGHT_MOMENTUM: '0.35'  # More aggressive
WEIGHT_MEAN_REVERSION: '0.15'
WEIGHT_TREND_FOLLOWING: '0.30'
```

### Balanced Profile (Recommended)

```yaml
MAX_POSITIONS: '3'
LEVERAGE: '3'

MIN_SENTIMENT_CONFIDENCE: '0.5'
MIN_ONCHAIN_STRENGTH: '0.5'

MIN_ENSEMBLE_CONSENSUS: '0.6'
MIN_ENSEMBLE_CONFIDENCE: '0.6'

RL_EPSILON: '0.1'

# Default weights
WEIGHT_SWING: '0.30'
WEIGHT_MOMENTUM: '0.25'
WEIGHT_MEAN_REVERSION: '0.25'
WEIGHT_TREND_FOLLOWING: '0.20'
```

---

## 🔧 Troubleshooting

### CryptoCompare API Issues

**Error: API key invalid**
- Verify that the secret is configured correctly
- Make sure the key is active on CryptoCompare

**Error: Rate limit exceeded**
- Free tier: 100k calls/month
- Add more cache in the modules
- Increase `cache_duration` in analyzers

### RL Not Improving

**Symptoms**: Q-values don't change, always the same action
- Verify that `rl_state.json` is being saved
- Increase `RL_EPSILON` for more exploration
- Check that rewards are being calculated correctly

**Solution**: Delete `rl_state.json` for a complete reset

### Ensemble Never Trades

**Symptoms**: Always no signal or very low consensus
- Reduce `MIN_ENSEMBLE_CONSENSUS` (e.g.: 0.5)
- Adjust strategy weights
- Verify that data has sufficient history (180 days)

---

## 📚 API Documentation

### Sentiment Analyzer

```python
from sentiment_analyzer import SentimentAnalyzer, should_trade_based_on_sentiment

analyzer = SentimentAnalyzer(api_key)
sentiment = analyzer.get_sentiment('BTC')

print(f"Overall: {sentiment.overall_score}")
print(f"Bullish: {sentiment.is_bullish()}")
print(f"Bearish: {sentiment.is_bearish()}")

# Use in decision
can_trade = should_trade_based_on_sentiment(
    sentiment, 'BUY', min_confidence=0.5
)
```

### On-Chain Analyzer

```python
from onchain_metrics import OnChainAnalyzer, should_trade_based_on_onchain

analyzer = OnChainAnalyzer(api_key)
signal = analyzer.get_onchain_signal('BTC')

print(f"Signal: {signal.signal_type}")
print(f"Strength: {signal.strength}")
print(f"Metrics: {signal.metrics}")

can_trade = should_trade_based_on_onchain(
    signal, 'BUY', min_strength=0.5
)
```

### Ensemble System

```python
from ensemble_strategies import EnsembleSystem

ensemble = EnsembleSystem(weights={...})
decision = ensemble.get_ensemble_decision(data, swing_signal)

print(f"Final: {decision.final_signal}")
print(f"Confidence: {decision.confidence}")
print(f"Consensus: {decision.consensus_level}")

ensemble.print_decision_summary(decision)
```

### RL Position Sizer

```python
from rl_position_sizing import RLPositionSizer, PositionSizeCalculator

rl_sizer = RLPositionSizer()
calculator = PositionSizeCalculator(rl_sizer)

capital, leverage = calculator.get_optimal_size(
    data=market_data,
    signal_confidence=0.75,
    available_capital=1000,
    base_leverage=3,
    open_positions=1,
    recent_trades=trades_history,
    training=True
)

# After the trade
reward = rl_sizer.calculate_reward(trade_result)
rl_sizer.update_q_value(state, action_idx, reward, next_state)
rl_sizer.save_state()
```

---

## 🚦 Future Roadmap

### V4.1 - Short Term
- [ ] Integration with more sentiment sources (Twitter API v2)
- [ ] On-chain metrics from Glassnode/CryptoQuant
- [ ] Real-time web dashboard
- [ ] Full backtesting with all V4 features

### V4.2 - Medium Term  
- [ ] Deep RL (DQN/PPO) instead of Q-Learning
- [ ] Ensemble with meta-learning
- [ ] Bayesian hyperparameter optimization
- [ ] Multi-timeframe analysis

### V5.0 - Long Term
- [ ] LLM integration for news analysis
- [ ] Volatility prediction with LSTM
- [ ] Portfolio optimization with Markowitz
- [ ] Full system auto-tuning

---

## 📄 License

MIT License - Use at your own risk

---

## ⚠️ Disclaimer

**This bot is experimental and educational.**

- Markets are unpredictable
- RL and ML can fail in unseen conditions
- ALWAYS start in simulation mode
- Do not invest more than you can afford to lose
- Monitor the bot constantly
- External APIs can fail or change
- Past performance does not guarantee future results

**The V4 system is more complex = more potential points of failure.**

---

## 🆘 Support

### Logs and Debug

1. GitHub Actions → Workflow runs → View logs
2. Look for errors in each module:
   - `sentiment_analyzer`
   - `onchain_metrics`
   - `ensemble_strategies`
   - `rl_position_sizing`

### Common Issues

- **No signals generated**: Adjust thresholds, verify data
- **RL always explores**: Normal at the start, converges after ~50 trades
- **Sentiment always neutral**: Verify API key, check rate limits
- **Ensemble indecisive**: Reduce `MIN_ENSEMBLE_CONSENSUS`

---

## 🙏 Contributions

Pull requests welcome for:
- New strategies for ensemble
- RL agent improvements
- More sentiment/on-chain sources
- Performance optimizations
- Documentation

---

## 📞 Contact

For technical questions or to report bugs, open an Issue on GitHub.

---

**🚀 Happy AI Trading!**

*V4.0 - Advanced AI System - December 2024*

# 🔄 Key Changes in trading-bot-v4.yml

## ✅ Updated File

Replace your current `trading-bot-v4.yml` with the new version.

---

## 📝 Main Changes

### 1. **Corrected executable file name**

```yaml
# BEFORE (in your original file):
python kraken_bot_v4_advanced.py

# NOW (updated):
python kraken_bot_v4_advanced.py  # ✓ Correct name
```

### 2. **Improved RL State cache**

```yaml
# BEFORE: Basic cache
- uses: actions/cache@v3

# NOW: Cache with improved restore-keys
- name: 📂 Load RL State
  uses: actions/cache@v3
  with:
    path: rl_state.json
    key: rl-state-${{ github.run_number }}
    restore-keys: |
      rl-state-    # ✓ Searches for previous states
```

**Benefit:** The RL agent retains its learning between runs.

### 3. **Organized environment variables**

```yaml
# Now with clear sections:
# ═══ APIS ═══
# ═══ SENTIMENT ANALYSIS (V4) ═══
# ═══ ON-CHAIN ANALYSIS (V4) ═══
# ═══ ENSEMBLE (V4) ═══
# ═══ RL (V4) ═══
# ═══ MULTI-ASSET (V3) ═══
# ═══ RISK MANAGEMENT ═══
# ═══ STRATEGY (V3) ═══
# ═══ MODE ═══
```

**Benefit:** Easier to understand and modify.

### 4. **Manual inputs for testing**

```yaml
workflow_dispatch:
  inputs:
    dry_run: 'true'
    use_sentiment: 'true'   # ✓ Enable/disable individually
    use_onchain: 'true'
    use_ensemble: 'true'
    use_rl: 'true'
```

**Benefit:** You can test each feature separately.

### 5. **Improved echo with bot info**

```yaml
run: |
  echo "🤖 AI Features V4:"
  echo "   Sentiment Analysis: $USE_SENTIMENT_ANALYSIS"
  echo "   On-Chain Metrics: $USE_ONCHAIN_ANALYSIS"
  echo "   Ensemble System: $USE_ENSEMBLE_SYSTEM"
  echo "   RL Position Sizing: $USE_RL_POSITION_SIZING"
```

**Benefit:** You immediately see which features are active.

---

## 🎯 Recommended Configurations

### For Initial Testing

```yaml
# In environment variables, adjust:
DRY_RUN: 'true'              # ✓ SIMULATION
MAX_POSITIONS: '1'           # ✓ Only 1 position
LEVERAGE: '2'                # ✓ Low leverage
MIN_ENSEMBLE_CONSENSUS: '0.75'  # ✓ Very conservative
```

### For Conservative Production

```yaml
DRY_RUN: 'false'             # ⚠️ REAL
MAX_POSITIONS: '2'
LEVERAGE: '3'
MIN_ENSEMBLE_CONSENSUS: '0.6'
MIN_SENTIMENT_CONFIDENCE: '0.6'
MIN_ONCHAIN_STRENGTH: '0.6'
```

### For Aggressive Production

```yaml
DRY_RUN: 'false'             # ⚠️ REAL
MAX_POSITIONS: '3'
LEVERAGE: '4'
MIN_ENSEMBLE_CONSENSUS: '0.5'
MIN_SENTIMENT_CONFIDENCE: '0.4'
MIN_ONCHAIN_STRENGTH: '0.4'
RL_EPSILON: '0.15'           # More exploration
```

---

## 🔍 How to Run Manually

### 1. Go to your repository on GitHub

### 2. Click on "Actions"

### 3. Select "Kraken Trading Bot V4 - Advanced AI"

### 4. Click "Run workflow"

### 5. Configure options:

```
dry_run: true               ← Start with simulation
use_sentiment: true         ← Enable sentiment
use_onchain: true          ← Enable on-chain
use_ensemble: true         ← Enable ensemble
use_rl: true               ← Enable RL
```

### 6. Click "Run workflow" (green)

### 7. Wait 2-3 minutes

### 8. Review the logs:
- Click on the workflow that just ran
- Click on "trade"
- You will see all bot logs

---

## 📊 What to Expect in the Logs

### Startup:
```
════════════════════════════════════════════════════════════════════
🚀 KRAKEN TRADING BOT V4 - ADVANCED AI SYSTEM
════════════════════════════════════════════════════════════════════
📅 2024-12-28 15:30:00
🎯 Mode: SIMULATION

🤖 AI Features V4:
   Sentiment Analysis: true
   On-Chain Metrics: true
   Ensemble System: true
   RL Position Sizing: true
════════════════════════════════════════════════════════════════════
```

### During execution:
```
🚀 INITIALIZING KRAKEN TRADING BOT V4
   ✓ Sentiment Analyzer enabled
   ✓ On-Chain Analyzer enabled
   ✓ Ensemble System enabled
   ✓ RL Position Sizing enabled

💰 Balance: 1,250.50 EUR
   Available margin: 1,100.00 EUR

📊 Downloading multi-asset data...
   ✓ BTC-USD: 4320 candles
   ✓ ETH-USD: 4320 candles
   ...

🔍 Searching for signals with V4 analysis...

   🎯 BTC-USD: BUY signal detected

   📊 Layer 1: Sentiment Analysis
   ✓ Sentiment confirms

   🔗 Layer 2: On-Chain Metrics
   ✓ On-Chain confirms

   🎯 Layer 3: Ensemble Strategies
   ✓ Ensemble confirms with 75% consensus

   🤖 Layer 4: RL Position Sizing
   RL: $400.00 @ 3x

✅ DECISION: BUY
   Final confidence: 78%

🟢 Opening BUY on BTC-USD
   ...
```

---

## ⚠️ Common Errors

### Error: "CRYPTOCOMPARE_API_KEY not set"

**Solution:**
1. Go to Settings → Secrets → Actions
2. Add `CRYPTOCOMPARE_API_KEY`
3. Value: your CryptoCompare key

### Error: "Module 'sentiment_analyzer' not found"

**Solution:**
```bash
# Verify that all V4 files are in the repo:
git add sentiment_analyzer.py
git add onchain_metrics.py
git add ensemble_strategies.py
git add rl_position_sizing.py
git commit -m "Add V4 modules"
git push
```

### Error: "kraken_bot_v4_advanced.py: No such file"

**Solution:**
```bash
# Make sure the main file is present:
git add kraken_bot_v4_advanced.py
git commit -m "Add V4 main bot"
git push
```

---

## 🔄 Automatic Execution

The workflow runs **automatically every 15 minutes**.

To change the frequency:

```yaml
schedule:
  - cron: '*/30 * * * *'  # Every 30 minutes
  # Or
  - cron: '0 * * * *'     # Every hour
  # Or
  - cron: '0 */2 * * *'   # Every 2 hours
```

---

## 📬 Daily Portfolio Telegram Report

`daily_portfolio_report.py` sends a read-only account summary to Telegram:

```bash
python daily_portfolio_report.py
```

It reads Kraken balances, trade balance, open positions, open orders, the ML live state file, and tracked market prices. It does not place or cancel orders.

The scheduled workflow `.github/workflows/daily-portfolio-report.yml` runs every day at `00:00 UTC` (`09:00 Asia/Seoul`) and keeps account details out of Actions logs by default.

Useful settings:

```bash
PORTFOLIO_REPORT_TIMEZONE=Asia/Seoul
PORTFOLIO_REPORT_SYMBOLS=BTC-USD,ETH-USD,XRP-USD,ADA-USD,SOL-USD,LINK-USD,DOGE-USD
PORTFOLIO_REPORT_PRINT_STDOUT=false
```

---

## 🧭 AI Strategy Router

`strategy_router.py` runs the old V4 swing strategy plus ML V2/V3 in signal-only mode, asks OpenAI for a structured routing and per-version budget decision, then enforces hard local risk caps before any order can be placed.

GitHub workflow: `.github/workflows/strategy-router.yml`

Required GitHub secret:

```bash
OPENAI_API_KEY
```

Safety defaults:

```bash
ROUTER_MAX_OPEN_POSITIONS=1
ROUTER_MAX_TRADE_MARGIN_FRACTION=0.25
ROUTER_MAX_TOTAL_MARGIN_FRACTION=0.35
ROUTER_MAX_LEVERAGE=2
ROUTER_ML_STRATEGIES=v2,v3
```

The old V4 and ML live workflows remain manually runnable, but their automatic schedules are paused so the router is the only scheduled live controller.

---

## ✅ Final Checklist

Before pushing:

- [ ] `trading-bot-v4.yml` updated in `.github/workflows/`
- [ ] `kraken_bot_v4_advanced.py` in the repo root
- [ ] All V4 modules present:
  - [ ] `sentiment_analyzer.py`
  - [ ] `onchain_metrics.py`
  - [ ] `ensemble_strategies.py`
  - [ ] `rl_position_sizing.py`
- [ ] `requirements.txt` updated
- [ ] Secrets configured in GitHub:
  - [ ] `KRAKEN_API_KEY`
  - [ ] `KRAKEN_API_SECRET`
  - [ ] `CRYPTOCOMPARE_API_KEY`
  - [ ] `TELEGRAM_BOT_TOKEN` (optional)
  - [ ] `TELEGRAM_CHAT_ID` (optional)

---

## 🚀 Command to push everything

```bash
# 1. Add all new files
git add .github/workflows/trading-bot-v4.yml
git add kraken_bot_v4_advanced.py
git add sentiment_analyzer.py
git add onchain_metrics.py
git add ensemble_strategies.py
git add rl_position_sizing.py
git add requirements.txt

# 2. Commit
git commit -m "Add complete V4 system with AI features"

# 3. Push
git push

# 4. Go to GitHub Actions and run manually
```

---

**🎉 Workflow updated and ready for V4!**
