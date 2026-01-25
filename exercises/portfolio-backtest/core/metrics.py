"""
性能指标计算模块
"""
import numpy as np
import pandas as pd
from datetime import date
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """投资组合性能指标计算器"""

    def __init__(self, portfolio_values: pd.Series, dates: pd.Series):
        """
        初始化性能指标计算器

        Args:
            portfolio_values: 每日组合价值序列
            dates: 对应的日期序列
        """
        self.values = portfolio_values.values if isinstance(portfolio_values, pd.Series) else portfolio_values
        self.dates = dates.values if isinstance(dates, pd.Series) else dates

        # 计算日收益率
        self.returns = self._calculate_returns()

    def _calculate_returns(self) -> np.ndarray:
        """计算日收益率"""
        if len(self.values) < 2:
            return np.array([])

        returns = np.diff(self.values) / self.values[:-1]
        return returns

    def total_return(self) -> float:
        """
        计算累计收益率

        Returns:
            总收益率（例如：0.25表示25%收益）
        """
        if len(self.values) < 1:
            return 0.0

        return (self.values[-1] / self.values[0]) - 1

    def annualized_return(self) -> float:
        """
        计算年化收益率

        Returns:
            年化收益率
        """
        if len(self.dates) < 2:
            return 0.0

        # 确保日期是可以计算差值的类型
        start_date = pd.Timestamp(self.dates[0])
        end_date = pd.Timestamp(self.dates[-1])

        total_days = (end_date - start_date).days
        years = total_days / 365.25

        if years <= 0:
            return 0.0

        total_ret = self.total_return()
        return (1 + total_ret) ** (1 / years) - 1

    def volatility(self, annualized: bool = True) -> float:
        """
        计算波动率（收益率标准差）

        Args:
            annualized: 是否年化

        Returns:
            波动率
        """
        if len(self.returns) < 2:
            return 0.0

        vol = np.std(self.returns, ddof=1)

        if annualized:
            # 假设252个交易日/年
            vol *= np.sqrt(252)

        return vol

    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        计算夏普比率

        Args:
            risk_free_rate: 无风险利率（年化）

        Returns:
            夏普比率
        """
        ann_return = self.annualized_return()
        ann_vol = self.volatility(annualized=True)

        if ann_vol == 0:
            return 0.0

        return (ann_return - risk_free_rate) / ann_vol

    def max_drawdown(self) -> float:
        """
        计算最大回撤

        Returns:
            最大回撤（正数，例如：0.20表示20%的回撤）
        """
        if len(self.values) < 2:
            return 0.0

        # 计算累计最高点
        cummax = np.maximum.accumulate(self.values)

        # 计算回撤
        drawdown = (self.values - cummax) / cummax

        return abs(drawdown.min())

    def calmar_ratio(self) -> float:
        """
        计算卡尔玛比率（年化收益/最大回撤）

        Returns:
            卡尔玛比率
        """
        max_dd = self.max_drawdown()

        if max_dd == 0:
            return 0.0

        return self.annualized_return() / max_dd

    def get_drawdown_series(self) -> pd.Series:
        """
        获取回撤序列

        Returns:
            回撤时间序列
        """
        cummax = np.maximum.accumulate(self.values)
        drawdown = (self.values - cummax) / cummax
        return pd.Series(drawdown, index=self.dates)

    def metrics_by_period(self, frequency: str = 'M') -> pd.DataFrame:
        """
        按时间段计算指标

        Args:
            frequency: 'D' (日), 'M' (月), 'Y' (年)

        Returns:
            包含各时段指标的DataFrame
        """
        # 创建DataFrame
        df = pd.DataFrame({
            'date': self.dates,
            'value': self.values
        })

        # 计算收益率
        df['return'] = df['value'].pct_change()

        # 按频率分组
        df['period'] = pd.to_datetime(df['date']).dt.to_period(frequency)

        # 聚合统计
        results = []

        for period, group in df.groupby('period'):
            if len(group) < 2:
                continue

            period_start_value = group['value'].iloc[0]
            period_end_value = group['value'].iloc[-1]
            period_return = (period_end_value / period_start_value) - 1

            results.append({
                'period': str(period),
                'start_value': period_start_value,
                'end_value': period_end_value,
                'return': period_return,
                'volatility': group['return'].std() * np.sqrt(len(group)),
                'max_value': group['value'].max(),
                'min_value': group['value'].min()
            })

        return pd.DataFrame(results)

    def rolling_metrics(self, window_days: int = 252) -> pd.DataFrame:
        """
        计算滚动指标

        Args:
            window_days: 滚动窗口天数

        Returns:
            包含滚动指标的DataFrame
        """
        if len(self.values) < window_days:
            return pd.DataFrame()

        df = pd.DataFrame({
            'date': self.dates,
            'value': self.values
        })

        # 计算滚动收益率
        df['rolling_return'] = df['value'].pct_change(window_days)

        # 计算滚动波动率
        df['return'] = df['value'].pct_change()
        df['rolling_volatility'] = df['return'].rolling(window_days).std() * np.sqrt(252)

        # 计算滚动夏普比率
        df['rolling_sharpe'] = (
            (df['rolling_return'] / window_days * 252) / df['rolling_volatility']
        )

        return df[window_days:][['date', 'rolling_return', 'rolling_volatility', 'rolling_sharpe']]

    def get_summary(self, risk_free_rate: float = 0.02) -> dict:
        """
        获取所有关键指标的摘要

        Args:
            risk_free_rate: 无风险利率

        Returns:
            指标字典
        """
        return {
            'total_return': self.total_return(),
            'annualized_return': self.annualized_return(),
            'volatility': self.volatility(),
            'sharpe_ratio': self.sharpe_ratio(risk_free_rate),
            'max_drawdown': self.max_drawdown(),
            'calmar_ratio': self.calmar_ratio(),
            'total_days': len(self.dates),
            'start_date': self.dates[0] if len(self.dates) > 0 else None,
            'end_date': self.dates[-1] if len(self.dates) > 0 else None,
            'start_value': self.values[0] if len(self.values) > 0 else None,
            'end_value': self.values[-1] if len(self.values) > 0 else None
        }
