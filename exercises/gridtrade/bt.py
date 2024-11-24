import pandas as pd
import numpy as np

class GridTrader:
    def __init__(self, grid_size_percent, num_grids):
        self.grid_size_percent = grid_size_percent  # Grid size in percentage
        self.num_grids = num_grids
        self.positions = []
        self.cash = 0
        self.initial_cash = 0
        
    def initialize_grids(self, initial_price):
        # Calculate grid levels above and below initial price
        self.grid_levels = []
        for i in range(-self.num_grids//2, self.num_grids//2 + 1):
            level = initial_price * (1 + i * self.grid_size_percent/100)
            self.grid_levels.append(level)
        self.grid_levels.sort()
        
    def backtest(self, df, initial_cash=100000):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions = []
        results = []
        
        # Initialize grids based on first closing price
        self.initialize_grids(df['close'].iloc[0])
        
        for idx, row in df.iterrows():
            price = row['close']
            
            # Check if price crosses any grid levels
            for i in range(len(self.grid_levels)-1):
                lower_grid = self.grid_levels[i]
                upper_grid = self.grid_levels[i+1]
                
                # Buy signal - price crosses grid level from above
                if price <= lower_grid and self.cash > 0:
                    position_size = (self.initial_cash / self.num_grids)
                    if position_size <= self.cash:
                        self.positions.append({
                            'entry_price': price,
                            'size': position_size/price,
                            'grid_level': lower_grid
                        })
                        self.cash -= position_size
                
                # Sell signal - price crosses grid level from below
                elif price >= upper_grid and self.positions:
                    for position in self.positions[:]:
                        if position['grid_level'] < upper_grid:
                            self.cash += position['size'] * price
                            self.positions.remove(position)
            
            # Calculate current portfolio value
            portfolio_value = self.cash + sum(pos['size'] * price for pos in self.positions)
            results.append({
                'date': row['date'],
                'price': price,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })
            
        return pd.DataFrame(results)

# Example usage
def run_backtest(csv_file, num_grids=10, start_date=None, days=None):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Convert date column to datetime if needed
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data based on start_date and days if provided
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df['date'] >= start_date]
        if days is not None:
            df = df.iloc[:days]
    # Initialize and run grid trader
    grid_size_percent = 1.0  # 1% grid size
    initial_cash = 100000  # Initial capital
    trader = GridTrader(grid_size_percent, num_grids)
    results = trader.backtest(df, initial_cash)
    
    # Calculate performance metrics
    total_return = (results['portfolio_value'].iloc[-1] - initial_cash) / initial_cash * 100
    max_drawdown = ((results['portfolio_value'].cummax() - results['portfolio_value']) / 
                   results['portfolio_value'].cummax()).max() * 100
    
    print(f"Total Return: {total_return:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    
    return results

# Example execution
results = run_backtest('hs300_index.csv', num_grids=5, start_date='2024-01-01', days=200)
