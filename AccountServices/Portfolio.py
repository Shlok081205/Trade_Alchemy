class PortfolioRiskManager:
    def __init__(self, initial_capital=100000, max_position_size=0.2, max_portfolio_risk=0.02):
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size  # Max 20% per position
        self.max_portfolio_risk = max_portfolio_risk  # Max 2% total risk
        self.positions = {}
        self.capital = initial_capital
        self.trade_history = []
        self.wins = []
        self.losses = []

    def calculate_position_size(self, ticker, signal_confidence, volatility, price):
        """Calculate position size based on ACTUAL historical performance"""

        # Validate inputs
        if volatility <= 0 or price <= 0:
            raise ValueError(f"Invalid volatility ({volatility}) or price ({price})")

        # ✅ Calculate Kelly fraction properly
        total_trades = len(self.wins) + len(self.losses)

        if total_trades < 30:
            # ✅ Not enough data - use conservative fixed fraction
            kelly_fraction = 0.02  # 2% per trade until we have history
        else:
            # ✅ Calculate ACTUAL win rate from historical trades
            win_rate = len(self.wins) / total_trades

            # ✅ Calculate ACTUAL average win vs average loss
            avg_win = sum(self.wins) / len(self.wins) if self.wins else 0.01
            avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0.01
            win_loss_ratio = avg_win / avg_loss  # This is your "b"

            # ✅ CORRECT Kelly formula using real data
            kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

            # Safety caps
            kelly = max(0, kelly)  # No negative
            kelly = min(kelly, 0.25)  # Max 25% even if Kelly says more

            # ✅ Use fractional Kelly (conservative)
            kelly_fraction = kelly * 0.25  # Use only 25% of Kelly recommendation

            # ✅ Adjust by confidence (but don't rely solely on it)
            # Scale from 0.5x to 1.0x based on confidence
            confidence_multiplier = 0.5 + (signal_confidence * 0.5)
            kelly_fraction = kelly_fraction * confidence_multiplier

        # Calculate dollar amount
        position_value = self.capital * kelly_fraction

        # Apply maximum position size constraint
        max_position_value = self.capital * self.max_position_size
        position_value = min(position_value, max_position_value)

        # Calculate shares
        shares = int(position_value / price)

        # Calculate risk per share
        risk_per_share = volatility * price

        # Apply portfolio risk limit
        max_shares_by_risk = int((self.capital * self.max_portfolio_risk) / risk_per_share)
        shares = min(shares, max_shares_by_risk)

        return shares

    def should_trade(self, confidence, min_confidence=0.6):
        """Determine if we should trade based on confidence threshold"""
        return confidence >= min_confidence

    def execute_trade(self, ticker, direction, confidence, price, volatility):
        """Execute a trade with position sizing"""
        if not self.should_trade(confidence):
            return None

        shares = self.calculate_position_size(ticker, confidence, volatility, price)

        if shares == 0:
            return None

        trade = {
            'ticker': ticker,
            'direction': direction,
            'confidence': confidence,
            'shares': shares,
            'price': price,
            'value': shares * price,
            'weight': (shares * price) / self.capital
        }

        self.positions[ticker] = trade
        self.trade_history.append(trade)

        return trade

    def get_portfolio_summary(self):
        """Get current portfolio summary"""
        if not self.positions:
            return None

        total_value = sum(pos['value'] for pos in self.positions.values())

        return {
            'num_positions': len(self.positions),
            'total_value': total_value,
            'capital_deployed': total_value / self.capital * 100,
            'positions': self.positions
        }


    def update_trade_result(self, entry_price, exit_price):
        """Call this after closing each trade"""
        profit_pct = (exit_price - entry_price) / entry_price

        if profit_pct > 0:
            self.wins.append(profit_pct)
        else:
            self.losses.append(abs(profit_pct))