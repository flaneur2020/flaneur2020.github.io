"""
数据下载模块 - 从yfinance下载历史数据
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataDownloader:
    """数据下载器"""

    def __init__(self):
        """初始化下载器"""
        pass

    @staticmethod
    def download_asset_data(
        symbol: str,
        start_date: date,
        end_date: date,
        progress_desc: str = None
    ) -> Optional[pd.DataFrame]:
        """
        下载单个资产的历史数据

        Args:
            symbol: 资产代码 (e.g., 'SPY', 'TLT')
            start_date: 开始日期
            end_date: 结束日期
            progress_desc: 进度条描述

        Returns:
            包含 OHLCV 数据的 DataFrame，如果下载失败则返回 None
        """
        try:
            logger.info(f"Downloading {symbol} from {start_date} to {end_date}")

            # 使用yfinance下载数据
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None

            # 重置索引，使日期成为列
            df.reset_index(inplace=True)
            df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

            # yfinance现在默认返回调整后的价格
            # 将close复制为adjusted_close
            df['adjusted_close'] = df['close']

            # 选择必要的列
            df = df[['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']]

            # 处理日期格式
            df['date'] = pd.to_datetime(df['date']).dt.date

            logger.info(f"Downloaded {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
            return None

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """
        验证数据质量

        Args:
            df: 要验证的数据框

        Returns:
            数据是否有效
        """
        if df is None or df.empty:
            return False

        # 检查必要的列
        required_columns = ['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.warning("Missing required columns")
            return False

        # 检查是否有NaN值
        if df[['open', 'high', 'low', 'close', 'adjusted_close']].isna().any().any():
            logger.warning("Found NaN values in price data")
            return False

        # 检查price逻辑：high >= low, high >= close, low <= close
        invalid_rows = (
            (df['high'] < df['low']) |
            (df['high'] < df['close']) |
            (df['low'] > df['close'])
        )

        if invalid_rows.any():
            logger.warning(f"Found {invalid_rows.sum()} rows with invalid OHLC relationships")
            return False

        # 检查异常波动（单日波动>20%）
        daily_return = (df['close'] - df['open']) / df['open']
        extreme_volatility = abs(daily_return) > 0.20

        if extreme_volatility.any():
            extreme_count = extreme_volatility.sum()
            logger.warning(f"Found {extreme_count} days with >20% volatility (possible splits/dividends)")
            # 这不是致命错误，因为adjusted_close应该已处理

        return True

    @staticmethod
    def fill_missing_data(df: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """
        填补缺失的交易日数据

        Args:
            df: 包含缺失值的数据框
            method: 填充方法 ('forward' 或 'interpolate')

        Returns:
            填充后的数据框
        """
        if df is None or df.empty:
            return df

        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)

        # 生成完整的交易日日期范围
        start_date = df['date'].min()
        end_date = df['date'].max()
        all_dates = pd.bdate_range(start=start_date, end=end_date)
        all_dates_df = pd.DataFrame({'date': all_dates})

        # 合并数据
        df_filled = all_dates_df.merge(df, on='date', how='left')

        # 前向填充价格数据
        price_cols = ['open', 'high', 'low', 'close', 'adjusted_close']
        df_filled[price_cols] = df_filled[price_cols].fillna(method='ffill')

        # 成交量用0填充
        df_filled['volume'] = df_filled['volume'].fillna(0)

        # 移除没有价格数据的行（通常是最后几个未来的日期）
        df_filled = df_filled.dropna(subset=['adjusted_close'])

        logger.info(f"Filled missing data: {len(df)} -> {len(df_filled)} records")
        return df_filled

    @staticmethod
    def download_all_assets(
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> dict:
        """
        批量下载多个资产的数据

        Args:
            symbols: 资产代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            {symbol: DataFrame} 的字典
        """
        results = {}

        for symbol in tqdm(symbols, desc="Downloading asset data"):
            df = DataDownloader.download_asset_data(symbol, start_date, end_date)

            if df is not None and DataDownloader.validate_data(df):
                results[symbol] = df
                logger.info(f"✓ {symbol}: {len(df)} records")
            else:
                logger.warning(f"✗ {symbol}: Failed validation or no data")

        return results

    @staticmethod
    def get_asset_inception_date(symbol: str) -> Optional[date]:
        """
        获取资产的成立日期（通过下载早期数据）

        Args:
            symbol: 资产代码

        Returns:
            成立日期
        """
        try:
            # 从1990年开始尝试
            ticker = yf.Ticker(symbol)
            df = ticker.history(start='1990-01-01', end=datetime.now())

            if not df.empty:
                inception = df.index.min().date()
                logger.info(f"{symbol} inception date: {inception}")
                return inception
            return None
        except Exception as e:
            logger.error(f"Failed to get inception date for {symbol}: {e}")
            return None
