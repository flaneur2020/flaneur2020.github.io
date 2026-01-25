"""
回测引擎模块
"""
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, List, Optional
import logging

from core.portfolio import Portfolio
from core.rebalance import RebalanceStrategy
from core.metrics import PerformanceMetrics
from data.database import DatabaseManager

logger = logging.getLogger(__name__)


class BacktestEngine:
    """回测引擎"""

    def __init__(self, db_manager: DatabaseManager):
        """
        初始化回测引擎

        Args:
            db_manager: 数据库管理器
        """
        self.db = db_manager
        self.portfolio = None
        self.results = None

    def run_backtest(
        self,
        symbols: List[str],
        target_weights: Dict[str, float],
        start_date: date,
        end_date: date,
        rebalance_strategy: RebalanceStrategy,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 5.0
    ) -> dict:
        """
        执行回测

        Args:
            symbols: 资产代码列表
            target_weights: 目标权重字典
            start_date: 开始日期
            end_date: 结束日期
            rebalance_strategy: 再平衡策略
            initial_capital: 初始资金
            transaction_cost_bps: 交易成本（基点）

        Returns:
            回测结果字典
        """
        logger.info("=" * 60)
        logger.info("Starting backtest")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        logger.info("=" * 60)

        # 1. 加载价格数据
        logger.info("Loading price data...")
        prices_df = self._load_price_data(symbols, start_date, end_date)

        if prices_df.empty:
            logger.error("No price data available")
            return {}

        # 2. 初始化投资组合
        logger.info("Initializing portfolio...")
        self.portfolio = Portfolio(
            symbols=symbols,
            target_weights=target_weights,
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps
        )

        # 3. 获取所有交易日
        trading_days = sorted(prices_df['date'].unique())
        logger.info(f"Total trading days: {len(trading_days)}")

        # 4. 按日期迭代执行回测
        for i, trading_date in enumerate(trading_days):
            # 获取当日价格
            today_prices = self._get_prices_for_date(prices_df, trading_date)

            # 首日初始化
            if not self.portfolio.is_initialized:
                self.portfolio.initialize(today_prices, trading_date)

            # 获取当前权重
            current_weights = self.portfolio.get_current_weights(today_prices)

            # 检查是否需要再平衡
            if rebalance_strategy.should_rebalance(
                trading_date,
                current_weights,
                target_weights
            ):
                self.portfolio.rebalance(today_prices, trading_date)

            # 记录当日价值
            self.portfolio.record_daily_value(trading_date, today_prices)

            # 定期输出进度
            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i + 1}/{len(trading_days)} days...")

        logger.info(f"Backtest completed: {len(trading_days)} days processed")

        # 5. 计算性能指标
        logger.info("Calculating performance metrics...")
        self.results = self._calculate_metrics()

        return self.results

    def _load_price_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        从数据库加载价格数据

        Args:
            symbols: 资产代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            价格DataFrame
        """
        try:
            df = self.db.get_prices(symbols, start_date, end_date)

            if df.empty:
                logger.warning("No data returned from database")
                return df

            # 确保有adjusted_close列
            if 'adjusted_close' not in df.columns:
                logger.error("Missing 'adjusted_close' column in price data")
                return pd.DataFrame()

            # 数据预处理：前向填充缺失的交易日
            df = self._fill_missing_trading_days(df, symbols)

            return df

        except Exception as e:
            logger.error(f"Failed to load price data: {e}")
            return pd.DataFrame()

    def _fill_missing_trading_days(
        self,
        df: pd.DataFrame,
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        填充缺失的交易日（前向填充）

        Args:
            df: 原始价格DataFrame
            symbols: 资产代码列表

        Returns:
            填充后的DataFrame
        """
        # 获取所有唯一日期
        all_dates = sorted(df['date'].unique())

        # 为每个资产创建完整的日期序列
        filled_dfs = []

        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df = symbol_df.sort_values('date')

            # 创建完整日期范围
            date_range = pd.DataFrame({'date': all_dates})

            # 合并并前向填充
            merged = date_range.merge(symbol_df, on='date', how='left')
            merged['symbol'] = symbol

            # 前向填充价格
            price_columns = ['open', 'high', 'low', 'close', 'adjusted_close']
            for col in price_columns:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(method='ffill')

            # 成交量填充为0
            if 'volume' in merged.columns:
                merged['volume'] = merged['volume'].fillna(0)

            filled_dfs.append(merged)

        # 合并所有资产
        result = pd.concat(filled_dfs, ignore_index=True)

        return result

    def _get_prices_for_date(
        self,
        prices_df: pd.DataFrame,
        trading_date: date
    ) -> Dict[str, float]:
        """
        获取指定日期的价格字典

        Args:
            prices_df: 价格DataFrame
            trading_date: 交易日期

        Returns:
            {symbol: price} 字典
        """
        day_data = prices_df[prices_df['date'] == trading_date]

        prices = {}
        for _, row in day_data.iterrows():
            prices[row['symbol']] = row['adjusted_close']

        return prices

    def _calculate_metrics(self) -> dict:
        """
        计算所有性能指标

        Returns:
            结果字典
        """
        if self.portfolio is None or not self.portfolio.history:
            logger.error("No portfolio history to calculate metrics")
            return {}

        # 获取历史数据
        history_df = self.portfolio.get_history_df()

        # 计算性能指标
        metrics = PerformanceMetrics(
            portfolio_values=history_df['value'],
            dates=history_df['date']
        )

        # 获取指标摘要
        summary = metrics.get_summary()

        # 按月/年的表现
        monthly_metrics = metrics.metrics_by_period('M')
        yearly_metrics = metrics.metrics_by_period('Y')

        # 回撤序列
        drawdown_series = metrics.get_drawdown_series()

        # 再平衡历史
        rebalance_summary = self.portfolio.get_rebalance_summary()

        # 组合结果
        results = {
            'summary': summary,
            'daily_values': history_df,
            'monthly_metrics': monthly_metrics,
            'yearly_metrics': yearly_metrics,
            'drawdown_series': drawdown_series,
            'rebalance_history': rebalance_summary,
            'portfolio_history': self.portfolio.history,
            'metrics_object': metrics
        }

        # 打印摘要
        logger.info("\n" + "=" * 60)
        logger.info("Backtest Results Summary")
        logger.info("=" * 60)
        logger.info(f"Total Return:       {summary['total_return']:.2%}")
        logger.info(f"Annualized Return:  {summary['annualized_return']:.2%}")
        logger.info(f"Volatility:         {summary['volatility']:.2%}")
        logger.info(f"Sharpe Ratio:       {summary['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown:       {summary['max_drawdown']:.2%}")
        logger.info(f"Calmar Ratio:       {summary['calmar_ratio']:.2f}")
        logger.info(f"Start Value:        ${summary['start_value']:,.2f}")
        logger.info(f"End Value:          ${summary['end_value']:,.2f}")
        logger.info(f"Total Days:         {summary['total_days']}")
        logger.info("=" * 60)

        return results

    def compare_strategies(
        self,
        symbols: List[str],
        target_weights: Dict[str, float],
        start_date: date,
        end_date: date,
        strategies: Dict[str, RebalanceStrategy],
        initial_capital: float = 100000.0
    ) -> pd.DataFrame:
        """
        比较多个再平衡策略

        Args:
            symbols: 资产代码列表
            target_weights: 目标权重字典
            start_date: 开始日期
            end_date: 结束日期
            strategies: 策略字典 {name: strategy}
            initial_capital: 初始资金

        Returns:
            比较结果DataFrame
        """
        results = []

        for strategy_name, strategy in strategies.items():
            logger.info(f"\nTesting strategy: {strategy_name}")

            result = self.run_backtest(
                symbols=symbols,
                target_weights=target_weights,
                start_date=start_date,
                end_date=end_date,
                rebalance_strategy=strategy,
                initial_capital=initial_capital
            )

            if result and 'summary' in result:
                summary = result['summary']
                summary['strategy'] = strategy_name
                summary['rebalance_count'] = len(result.get('rebalance_history', []))
                results.append(summary)

        return pd.DataFrame(results)
