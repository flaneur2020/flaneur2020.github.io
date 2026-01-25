"""
投资组合核心模块
"""
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Portfolio:
    """投资组合类"""

    def __init__(
        self,
        symbols: List[str],
        target_weights: Dict[str, float],
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 5.0
    ):
        """
        初始化投资组合

        Args:
            symbols: 资产代码列表
            target_weights: 目标权重字典 {symbol: weight}
            initial_capital: 初始资金
            transaction_cost_bps: 交易成本（基点，例如5表示0.05%）
        """
        self.symbols = symbols
        self.target_weights = target_weights
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps

        # 验证权重总和为1
        total_weight = sum(target_weights.values())
        if not np.isclose(total_weight, 1.0, atol=0.01):
            logger.warning(f"Target weights sum to {total_weight:.4f}, not 1.0")

        # 投资组合状态
        self.cash = initial_capital
        self.shares = {symbol: 0.0 for symbol in symbols}

        # 历史记录
        self.history = []
        self.rebalance_history = []

        # 初始化标志
        self.is_initialized = False

    def initialize(self, prices: Dict[str, float], current_date: date):
        """
        初始化投资组合（首次建仓）

        Args:
            prices: 当前价格字典 {symbol: price}
            current_date: 当前日期
        """
        logger.info(f"Initializing portfolio on {current_date}")

        total_value = self.cash

        # 按目标权重分配资金
        for symbol in self.symbols:
            target_value = total_value * self.target_weights[symbol]
            price = prices[symbol]

            # 计算可购买的股数
            shares = target_value / price
            cost = shares * price

            # 扣除交易成本
            transaction_cost = cost * (self.transaction_cost_bps / 10000.0)

            self.shares[symbol] = shares
            self.cash -= (cost + transaction_cost)

        self.is_initialized = True

        # 记录初始化
        self.rebalance_history.append({
            'date': current_date,
            'type': 'initialization',
            'portfolio_value': total_value,
            'weights': self.target_weights.copy()
        })

        logger.info(f"Portfolio initialized with ${total_value:,.2f}")

    def get_current_value(self, prices: Dict[str, float]) -> float:
        """
        计算当前总价值

        Args:
            prices: 当前价格字典

        Returns:
            投资组合总价值
        """
        portfolio_value = self.cash

        for symbol, shares in self.shares.items():
            portfolio_value += shares * prices[symbol]

        return portfolio_value

    def get_current_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        计算当前权重

        Args:
            prices: 当前价格字典

        Returns:
            当前权重字典
        """
        total_value = self.get_current_value(prices)

        if total_value == 0:
            return {symbol: 0.0 for symbol in self.symbols}

        weights = {}
        for symbol, shares in self.shares.items():
            asset_value = shares * prices[symbol]
            weights[symbol] = asset_value / total_value

        return weights

    def rebalance(self, prices: Dict[str, float], current_date: date) -> Dict[str, float]:
        """
        执行再平衡

        Args:
            prices: 当前价格字典
            current_date: 当前日期

        Returns:
            交易详情字典
        """
        # 计算当前总价值
        total_value = self.get_current_value(prices)

        # 计算目标持仓金额
        target_amounts = {
            symbol: total_value * weight
            for symbol, weight in self.target_weights.items()
        }

        # 计算当前持仓金额
        current_amounts = {
            symbol: self.shares[symbol] * prices[symbol]
            for symbol in self.symbols
        }

        # 计算需要交易的金额
        trades = {}
        total_trade_cost = 0.0

        for symbol in self.symbols:
            trade_amount = target_amounts[symbol] - current_amounts[symbol]
            trades[symbol] = trade_amount

            # 计算交易成本
            if trade_amount != 0:
                cost = abs(trade_amount) * (self.transaction_cost_bps / 10000.0)
                total_trade_cost += cost

        logger.info(
            f"Rebalancing on {current_date}: "
            f"Total value=${total_value:,.2f}, "
            f"Transaction cost=${total_trade_cost:,.2f}"
        )

        # 执行交易
        for symbol, trade_amount in trades.items():
            price = prices[symbol]

            if trade_amount > 0:
                # 买入
                shares_to_buy = trade_amount / price
                self.shares[symbol] += shares_to_buy
                self.cash -= trade_amount
            elif trade_amount < 0:
                # 卖出
                shares_to_sell = -trade_amount / price
                self.shares[symbol] -= shares_to_sell
                self.cash += -trade_amount

        # 扣除交易成本
        self.cash -= total_trade_cost

        # 记录再平衡
        self.rebalance_history.append({
            'date': current_date,
            'type': 'rebalance',
            'portfolio_value': total_value,
            'transaction_cost': total_trade_cost,
            'trades': trades.copy(),
            'weights_before': current_amounts,
            'weights_after': self.get_current_weights(prices)
        })

        return trades

    def record_daily_value(
        self,
        current_date: date,
        prices: Dict[str, float]
    ):
        """
        记录每日价值

        Args:
            current_date: 当前日期
            prices: 当前价格字典
        """
        total_value = self.get_current_value(prices)
        current_weights = self.get_current_weights(prices)

        self.history.append({
            'date': current_date,
            'value': total_value,
            'cash': self.cash,
            'weights': current_weights.copy(),
            'shares': self.shares.copy()
        })

    def get_history_df(self) -> pd.DataFrame:
        """
        获取历史记录DataFrame

        Returns:
            包含历史数据的DataFrame
        """
        if not self.history:
            return pd.DataFrame()

        records = []
        for record in self.history:
            row = {
                'date': record['date'],
                'value': record['value'],
                'cash': record['cash']
            }

            # 添加每个资产的权重
            for symbol in self.symbols:
                row[f'{symbol}_weight'] = record['weights'].get(symbol, 0.0)

            records.append(row)

        return pd.DataFrame(records)

    def get_rebalance_summary(self) -> pd.DataFrame:
        """
        获取再平衡历史摘要

        Returns:
            再平衡历史DataFrame
        """
        if not self.rebalance_history:
            return pd.DataFrame()

        records = []
        for rebalance in self.rebalance_history:
            records.append({
                'date': rebalance['date'],
                'type': rebalance['type'],
                'portfolio_value': rebalance['portfolio_value'],
                'transaction_cost': rebalance.get('transaction_cost', 0.0)
            })

        return pd.DataFrame(records)
