"""
数据存储模块 - 协调下载和数据库存储
"""
import logging
import time
import random
from datetime import date, datetime, timedelta
from typing import List, Optional
from pathlib import Path

from data.downloader import DataDownloader
from data.database import DatabaseManager
from config.settings import get_config

logger = logging.getLogger(__name__)

# 资产之间的延迟配置
ASSET_DELAY_MIN = 2.0  # 最小延迟（秒）
ASSET_DELAY_MAX = 4.0  # 最大延迟（秒）


class DataStorage:
    """数据存储管理器"""

    def __init__(self, db_manager: DatabaseManager):
        """
        初始化数据存储

        Args:
            db_manager: 数据库管理器实例
        """
        self.db = db_manager
        self.downloader = DataDownloader()

    def sync_asset_data(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> bool:
        """
        同步单个资产数据：如果本地没有，则下载；如果有，则检查是否需要更新

        Args:
            symbol: 资产代码
            start_date: 开始日期（如果为None，则从资产成立日开始）
            end_date: 结束日期（如果为None，则到今天）

        Returns:
            同步是否成功
        """
        try:
            # 确定日期范围
            if end_date is None:
                end_date = date.today()

            if start_date is None:
                # 尽可能下载全历史：从很早的日期开始，请求会自动返回可用范围
                start_date = date(1900, 1, 1)

            # 检查本地已有的数据
            local_start, local_end = self.db.get_date_range(symbol)

            if local_start is not None:
                # 有本地数据，检查是否需要更新
                logger.info(f"{symbol}: Local data from {local_start} to {local_end}")

                # 下载缺失的后续数据（到今天）
                if local_end and local_end < end_date:
                    # 从本地最新日期的下一天开始下载
                    download_start = local_end + timedelta(days=1)
                    logger.info(f"{symbol}: Downloading new data from {download_start} to {end_date}")

                    df = self.downloader.download_asset_data(symbol, download_start, end_date)
                    if df is not None:
                        self.db.insert_prices(symbol, df)
                        return True
                    else:
                        logger.warning(f"Failed to download new data for {symbol}")
                        return False
                else:
                    logger.info(f"{symbol}: Data is up to date")
                    return True
            else:
                # 没有本地数据，完整下载
                logger.info(f"{symbol}: No local data, downloading from {start_date} to {end_date}")
                df = self.downloader.download_asset_data(symbol, start_date, end_date)

                if df is not None:
                    self.db.insert_prices(symbol, df)
                    return True
                else:
                    logger.error(f"Failed to download data for {symbol}")
                    return False

        except Exception as e:
            logger.error(f"Error syncing {symbol}: {e}")
            return False

    def sync_all_assets(
        self,
        portfolio_name: str = "permanent_portfolio",
        end_date: Optional[date] = None
    ) -> dict:
        """
        同步指定投资组合的所有资产数据

        Args:
            portfolio_name: 投资组合名称
            end_date: 结束日期

        Returns:
            同步结果字典 {symbol: success}
        """
        try:
            config = get_config()

            if portfolio_name not in config.get('portfolios', {}):
                logger.error(f"Portfolio '{portfolio_name}' not found in config")
                return {}

            portfolio = config['portfolios'][portfolio_name]
            assets = portfolio.get('assets', [])
            symbols = [asset['symbol'] for asset in assets]

            results = {}

            logger.info(f"Syncing portfolio '{portfolio_name}' with {len(symbols)} assets")

            for i, asset in enumerate(assets):
                symbol = asset['symbol']
                logger.info(f"\nProcessing {symbol}...")

                # 除第一个资产外，添加随机延迟防止限速
                if i > 0:
                    delay = random.uniform(ASSET_DELAY_MIN, ASSET_DELAY_MAX)
                    logger.info(f"Waiting {delay:.1f}s before next request...")
                    time.sleep(delay)

                success = self.sync_asset_data(symbol, end_date=end_date)
                results[symbol] = success

                if success:
                    coverage = self.db.check_data_completeness(symbol)
                    logger.info(f"✓ {symbol}: Data coverage {coverage:.1%}")
                else:
                    logger.warning(f"✗ {symbol}: Sync failed")

            return results

        except Exception as e:
            logger.error(f"Error syncing assets: {e}")
            return {}

    def get_backtest_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Optional[dict]:
        """
        获取用于回测的数据

        Args:
            symbols: 资产代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            {symbol: DataFrame} 字典，如果数据不足则返回 None
        """
        try:
            # 检查每个资产的数据可用性
            for symbol in symbols:
                local_start, local_end = self.db.get_date_range(symbol)

                if local_start is None:
                    logger.error(f"No data available for {symbol}")
                    return None

                if local_start > start_date or local_end < end_date:
                    logger.error(
                        f"Insufficient data for {symbol}: "
                        f"need [{start_date}, {end_date}], "
                        f"have [{local_start}, {local_end}]"
                    )
                    return None

            # 获取所有数据
            result = {}
            for symbol in symbols:
                df = self.db.get_prices([symbol], start_date, end_date)
                if df is None or df.empty:
                    logger.error(f"Failed to retrieve data for {symbol}")
                    return None
                result[symbol] = df

            return result

        except Exception as e:
            logger.error(f"Error getting backtest data: {e}")
            return None

    @staticmethod
    def _get_asset_start_date(symbol: str, config: dict) -> Optional[date]:
        """
        兼容旧逻辑：配置文件不再维护 inception/inception_date 字段。
        保留该方法作为兼容入口，返回尽可能早的日期以请求全历史。
        """
        return date(1900, 1, 1)
