class PortfolioRiskManager:
    def __init__(self, initial_capital=100000, max_position_size=0.2, max_portfolio_risk=0.02):
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size  # Max 20% per position
        self.max_portfolio_risk = max_portfolio_risk  # Max 2% total risk
        self.positions = {}
        self.capital = initial_capital
        self.trade_history = []

    def calculate_position_size(self, ticker, signal_confidence, volatility, price):
        """Calculate position size based on Kelly Criterion and risk limits"""

        # Kelly Criterion: f = (p*b - q) / b
        # where p = win probability, q = 1-p, b = win/loss ratio

        # Assume 1:1 risk/reward for simplicity
        p = signal_confidence
        q = 1 - p
        b = 1.0  # Risk/reward ratio

        kelly_fraction = (p * b - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% Kelly

        # Apply fractional Kelly (use 25% of Kelly to be conservative)
        fractional_kelly = kelly_fraction * 0.25

        # Calculate dollar amount
        position_value = self.capital * fractional_kelly

        # Apply maximum position size constraint
        max_position_value = self.capital * self.max_position_size
        position_value = min(position_value, max_position_value)

        # Calculate shares
        shares = int(position_value / price)

        # Calculate risk per share (use ATR or volatility)
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