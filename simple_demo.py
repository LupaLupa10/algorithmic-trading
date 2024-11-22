# Simple Demostration from https://pypi.org/project/vectorbt/

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
import vectorbt as vbt
import numpy as np
import pandas as pd

@dataclass
class MarketData:
    """Handles market data acquisition and preprocessing"""
    symbols: List[str]
    period: Optional[str] = None
    
    def __post_init__(self):
        """Automatically fetch data after initialization"""
        self.data = self._fetch_data()
    
    def _fetch_data(self) -> pd.DataFrame:
        """Fetch close prices for specified symbols"""
        return vbt.YFData.download(
            self.symbols, 
            period=self.period, 
            missing_index='drop'
        ).get('Close')

class Strategy:
    """Base class for trading strategies"""
    def __init__(self, price_data: pd.DataFrame):
        self.price = price_data
        self.entries = None 
        self.exits = None
    
    def generate_signals(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate entry and exit signals"""
        raise NotImplementedError("Subclasses must implement generate_signals")

class MovingAverageCrossover(Strategy):
    """Moving Average Crossover strategy implementation"""
    def __init__(self, price_data: pd.DataFrame, fast_window: int, slow_window: int):
        super().__init__(price_data)
        self.fast_window = fast_window
        self.slow_window = slow_window
    
    def generate_signals(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate signals based on MA crossovers"""
        fast_ma = vbt.MA.run(self.price, self.fast_window)
        slow_ma = vbt.MA.run(self.price, self.slow_window)
        self.entries = fast_ma.ma_crossed_above(slow_ma)
        self.exits = fast_ma.ma_crossed_below(slow_ma)
        return self.entries, self.exits

class MultiMAOptimizer(Strategy):
    """Optimize multiple MA combinations"""
    def __init__(self, price_data: pd.DataFrame, window_range: np.ndarray):
        super().__init__(price_data)
        self.window_range = window_range
    
    def generate_signals(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate signals for all MA combinations"""
        fast_ma, slow_ma = vbt.MA.run_combs(
            self.price, 
            window=self.window_range, 
            r=2, 
            short_names=['fast', 'slow']
        )
        self.entries = fast_ma.ma_crossed_above(slow_ma)
        self.exits = fast_ma.ma_crossed_below(slow_ma)
        return self.entries, self.exits

class RandomizedStrategy(Strategy):
    """Monte Carlo simulation with random signals"""
    def __init__(self, price_data: pd.DataFrame, n_signals: int = 1000, 
                 min_window: int = 10, max_window: int = 101, seed: int = 42):
        super().__init__(price_data)
        self.n_signals = n_signals
        self.min_window = min_window
        self.max_window = max_window
        self.seed = seed
        
    def generate_signals(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate random entry/exit signals"""
        np.random.seed(self.seed)
        n = np.random.randint(self.min_window, self.max_window, size=self.n_signals)
        portfolio = vbt.Portfolio.from_random_signals(
            self.price, n=n.tolist(), init_cash=100, seed=self.seed
        )
        return portfolio.entries, portfolio.exits

    def analyze_expectancy(self, portfolio) -> object:
        """Analyze trade expectancy from Monte Carlo simulation"""
        mean_expectancy = portfolio.trades.expectancy().groupby(['randnx_n', 'symbol']).mean()
        return mean_expectancy.unstack().vbt.scatterplot(
            xaxis_title='randnx_n', 
            yaxis_title='mean_expectancy'
        )

class PortfolioManager:
    """Portfolio backtesting and analysis"""
    def __init__(self, price_data: pd.DataFrame, init_cash: float = 100):
        self.price = price_data
        self.init_cash = init_cash
        self.portfolio = None
    
    def backtest_signals(self, entries: pd.DataFrame, exits: pd.DataFrame, 
                        fees: float = 0.001, freq: str = '1D') -> None:
        """Run backtest with given signals"""
        self.portfolio = vbt.Portfolio.from_signals(
            self.price, entries, exits,
            size=np.inf, fees=fees, freq=freq,
            init_cash=self.init_cash
        )
    
    def get_stats(self, windows: Tuple[int, int], symbol: str) -> pd.Series:
        """Get performance statistics for specific parameters"""
        return self.portfolio[(windows[0], windows[1], symbol)].stats()
    
    def plot_returns_heatmap(self) -> None:
        """Visualize returns across parameter combinations"""
        fig = self.portfolio.total_return().vbt.heatmap(
            x_level='fast_window',
            y_level='slow_window',
            slider_level='symbol',
            symmetric=True,
            trace_kwargs=dict(
                colorbar=dict(
                    title='Total return',
                    tickformat='%'
                )
            )
        )
        fig.show()

class BollingerBandsVisualizer:
    """Bollinger Bands analysis and visualization"""
    def __init__(self, price_data: pd.DataFrame, period: str = '6mo'):
        self.price = price_data
        self.bbands = vbt.BBANDS.run(self.price)
    
    def create_heatmap_figure(self, index: pd.Index, bbands: object, **kwargs) -> object:
        """Create BB analysis heatmap"""
        bbands_slice = bbands.loc[index]
        fig = vbt.make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.15,
            subplot_titles=('%B', 'Bandwidth')
        )
        
        fig.update_layout(
            template='vbt_dark', 
            showlegend=False, 
            width=750, 
            height=400
        )
        
        bbands_slice.percent_b.vbt.ts_heatmap(
            trace_kwargs=dict(
                zmin=0, zmid=0.5, zmax=1, 
                colorscale='Spectral',
                colorbar=dict(
                    y=(fig.layout.yaxis.domain[0] + fig.layout.yaxis.domain[1]) / 2,
                    len=0.5
                )
            ),
            add_trace_kwargs=dict(row=1, col=1),
            fig=fig
        )
        
        bbands_slice.bandwidth.vbt.ts_heatmap(
            trace_kwargs=dict(
                colorbar=dict(
                    y=(fig.layout.yaxis2.domain[0] + fig.layout.yaxis2.domain[1]) / 2,
                    len=0.5
                )
            ),
            add_trace_kwargs=dict(row=2, col=1),
            fig=fig
        )
        return fig
    
    def save_animation(self, filename: str = 'bbands.gif', delta: int = 90, 
                      step: int = 3, fps: int = 3) -> None:
        """Save animated BB analysis"""
        vbt.save_animation(
            filename,
            self.bbands.wrapper.index,
            self.create_heatmap_figure,
            self.bbands,
            delta=delta,
            step=step,
            fps=fps
        )


if __name__ == "__main__":
    # Initialize market data
    market_data = MarketData(symbols=["BTC-USD", "ETH-USD", "LTC-USD"])
    
    # Test MA crossover strategy
    ma_strategy = MovingAverageCrossover(market_data.data, fast_window=10, slow_window=50)
    entries, exits = ma_strategy.generate_signals()
    portfolio = PortfolioManager(market_data.data)
    portfolio.backtest_signals(entries, exits)
    
    # Optimize MA parameters
    windows = np.arange(2, 101)
    optimizer = MultiMAOptimizer(market_data.data, windows)
    entries, exits = optimizer.generate_signals()
    portfolio = PortfolioManager(market_data.data)
    portfolio.backtest_signals(entries, exits)
    portfolio.plot_returns_heatmap()
    
    # Run Monte Carlo simulation
    random_strategy = RandomizedStrategy(market_data.data)
    entries, exits = random_strategy.generate_signals()
    portfolio = PortfolioManager(market_data.data)
    portfolio.backtest_signals(entries, exits)
    fig = random_strategy.analyze_expectancy(portfolio.portfolio)
    fig.show()
    
    # Analysis for specific parameters
    stats = portfolio.get_stats((10, 20), 'ETH-USD')
    print(stats)
    
    # Bollinger Bands visualization
    bb_symbols = ["BTC-USD", "ETH-USD", "ADA-USD"]
    bb_data = MarketData(bb_symbols, period='6mo')
    bb_visualizer = BollingerBandsVisualizer(bb_data.data)
    bb_visualizer.save_animation()