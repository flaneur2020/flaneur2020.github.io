"""
再平衡策略模块
"""
from abc import ABC, abstractmethod
from datetime import date
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class RebalanceStrategy(ABC):
    """再平衡策略抽象基类"""

    @abstractmethod
    def should_rebalance(
        self,
        current_date: date,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> bool:
        """
        判断是否需要再平衡

        Args:
            current_date: 当前日期
            current_weights: 当前权重字典
            target_weights: 目标权重字典

        Returns:
            是否需要再平衡
        """
        pass


class NoRebalance(RebalanceStrategy):
    """不进行再平衡"""

    def should_rebalance(
        self,
        current_date: date,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> bool:
        """永远不再平衡"""
        return False


class AnnualRebalance(RebalanceStrategy):
    """年度再平衡策略"""

    def __init__(self, month: int = 1, day: int = 1):
        """
        初始化年度再平衡策略

        Args:
            month: 再平衡月份（1-12）
            day: 再平衡日期（1-31）
        """
        self.month = month
        self.day = day
        self.last_rebalance_year = None

    def should_rebalance(
        self,
        current_date: date,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> bool:
        """
        每年指定日期再平衡

        Args:
            current_date: 当前日期
            current_weights: 当前权重
            target_weights: 目标权重

        Returns:
            是否需要再平衡
        """
        # 检查是否到达再平衡日期
        if current_date.month == self.month and current_date.day >= self.day:
            # 避免同一年多次再平衡
            if self.last_rebalance_year != current_date.year:
                self.last_rebalance_year = current_date.year
                logger.info(f"Annual rebalance triggered on {current_date}")
                return True

        return False


class QuarterlyRebalance(RebalanceStrategy):
    """季度再平衡策略"""

    def __init__(self, months: list = None):
        """
        初始化季度再平衡策略

        Args:
            months: 再平衡月份列表，默认 [1, 4, 7, 10]（每季度第一个月）
        """
        self.months = months if months else [1, 4, 7, 10]
        self.last_rebalance = None

    def should_rebalance(
        self,
        current_date: date,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> bool:
        """
        每季度第一个交易日再平衡

        Args:
            current_date: 当前日期
            current_weights: 当前权重
            target_weights: 目标权重

        Returns:
            是否需要再平衡
        """
        # 检查是否是再平衡月份
        if current_date.month in self.months:
            # 创建当前月份的标识
            period_id = (current_date.year, current_date.month)

            # 避免同一季度多次再平衡
            if self.last_rebalance != period_id:
                self.last_rebalance = period_id
                logger.info(f"Quarterly rebalance triggered on {current_date}")
                return True

        return False


class ThresholdRebalance(RebalanceStrategy):
    """阈值触发再平衡策略"""

    def __init__(self, threshold: float = 0.05):
        """
        初始化阈值再平衡策略

        Args:
            threshold: 偏离阈值（例如：0.05表示5%的偏离）
        """
        self.threshold = threshold

    def should_rebalance(
        self,
        current_date: date,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> bool:
        """
        当任何资产偏离目标权重超过阈值时再平衡

        Args:
            current_date: 当前日期
            current_weights: 当前权重
            target_weights: 目标权重

        Returns:
            是否需要再平衡
        """
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            deviation = abs(current_weight - target_weight)

            if deviation > self.threshold:
                logger.info(
                    f"Threshold rebalance triggered on {current_date}: "
                    f"{symbol} deviation={deviation:.2%} (threshold={self.threshold:.2%})"
                )
                return True

        return False


class MonthlyRebalance(RebalanceStrategy):
    """月度再平衡策略"""

    def __init__(self):
        """初始化月度再平衡策略"""
        self.last_rebalance = None

    def should_rebalance(
        self,
        current_date: date,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> bool:
        """
        每月第一个交易日再平衡

        Args:
            current_date: 当前日期
            current_weights: 当前权重
            target_weights: 目标权重

        Returns:
            是否需要再平衡
        """
        # 创建当前月份的标识
        period_id = (current_date.year, current_date.month)

        # 避免同一月多次再平衡
        if self.last_rebalance != period_id:
            self.last_rebalance = period_id
            logger.info(f"Monthly rebalance triggered on {current_date}")
            return True

        return False


def get_rebalance_strategy(strategy_name: str, config: dict = None) -> RebalanceStrategy:
    """
    根据策略名称获取再平衡策略实例

    Args:
        strategy_name: 策略名称 ('annual', 'quarterly', 'threshold', 'monthly', 'none')
        config: 策略配置参数

    Returns:
        再平衡策略实例
    """
    config = config or {}

    if strategy_name == 'annual':
        return AnnualRebalance()
    elif strategy_name == 'quarterly':
        months = config.get('months', [1, 4, 7, 10])
        return QuarterlyRebalance(months=months)
    elif strategy_name == 'threshold':
        threshold = config.get('threshold', 0.05)
        return ThresholdRebalance(threshold=threshold)
    elif strategy_name == 'monthly':
        return MonthlyRebalance()
    elif strategy_name == 'none' or strategy_name == 'never':
        return NoRebalance()
    else:
        logger.warning(f"Unknown strategy '{strategy_name}', using NoRebalance")
        return NoRebalance()
