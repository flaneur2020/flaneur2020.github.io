"""
数据下载模块 - 支持多数据源（yfinance, stooq）
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List
import logging
import time
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Rate limiting 配置
REQUEST_DELAY_MIN = 1.0
REQUEST_DELAY_MAX = 2.0


class DataDownloader:
    """数据下载器 - 支持多数据源"""

    # 数据源优先级
    DATA_SOURCES = ['stooq', 'yfinance']

    def __init__(self):
        """初始化下载器"""
        pass

    @staticmethod
    def _download_from_stooq(
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """从 Stooq 下载数据（免费，无需 API key）"""
        try:
            import requests
            from io import StringIO

            logger.info(f"[Stooq] Downloading {symbol}...")

            # Stooq CSV 下载 URL
            # 格式：d=开始日期, d1=结束日期 (YYYYMMDD)
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&d1={start_str}&d2={end_str}&i=d"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # 读取 CSV
            df = pd.read_csv(StringIO(response.text))

            if df.empty or len(df) < 10:
                logger.warning(f"[Stooq] No data for {symbol}")
                return None

            # 统一列名（Stooq 列名：Date,Open,High,Low,Close,Volume）
            df.columns = [c.lower() for c in df.columns]
            df['adjusted_close'] = df['close']
            df['date'] = pd.to_datetime(df['date']).dt.date

            # 选择需要的列
            df = df[['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']]
            df = df.dropna()
            df = df.sort_values('date')  # Stooq 可能是倒序

            logger.info(f"[Stooq] Downloaded {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.warning(f"[Stooq] Failed for {symbol}: {e}")
            return None

    @staticmethod
    def _download_from_yfinance(
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """从 yfinance 下载数据"""
        try:
            logger.info(f"[yfinance] Downloading {symbol}...")

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"[yfinance] No data for {symbol}")
                return None

            df.reset_index(inplace=True)
            df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

            df['adjusted_close'] = df['close']
            df = df[['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']]
            df['date'] = pd.to_datetime(df['date']).dt.date

            logger.info(f"[yfinance] Downloaded {len(df)} records for {symbol}")
            return df

        except Exception as e:
            error_msg = str(e)
            if "Rate limited" in error_msg or "Too Many Requests" in error_msg:
                logger.warning(f"[yfinance] Rate limited for {symbol}")
            else:
                logger.warning(f"[yfinance] Failed for {symbol}: {e}")
            return None

    @staticmethod
    def download_asset_data(
        symbol: str,
        start_date: date,
        end_date: date,
        progress_desc: str = None
    ) -> Optional[pd.DataFrame]:
        """
        下载单个资产的历史数据（自动尝试多个数据源）

        Args:
            symbol: 资产代码 (e.g., 'SPY', 'TLT')
            start_date: 开始日期
            end_date: 结束日期
            progress_desc: 进度条描述

        Returns:
            包含 OHLCV 数据的 DataFrame，如果下载失败则返回 None
        """
        logger.info(f"Downloading {symbol} from {start_date} to {end_date}")

        for source in DataDownloader.DATA_SOURCES:
            if source == 'stooq':
                df = DataDownloader._download_from_stooq(symbol, start_date, end_date)
            elif source == 'yfinance':
                df = DataDownloader._download_from_yfinance(symbol, start_date, end_date)
            else:
                continue

            if df is not None and DataDownloader.validate_data(df):
                logger.info(f"✓ {symbol}: Got {len(df)} records from {source}")
                return df

            # 数据源之间稍微等一下
            time.sleep(0.5)

        logger.error(f"✗ {symbol}: All data sources failed")
        return None

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """验证数据质量"""
        if df is None or df.empty:
            return False

        required_columns = ['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.warning("Missing required columns")
            return False

        # 检查价格列是否有 NaN
        price_cols = ['open', 'high', 'low', 'close', 'adjusted_close']
        if df[price_cols].isna().any().any():
            logger.warning("Found NaN values in price data")
            return False

        return True

    @staticmethod
    def fill_missing_data(df: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """填补缺失的交易日数据"""
        if df is None or df.empty:
            return df

        df = df.sort_values('date').reset_index(drop=True)

        start_date = df['date'].min()
        end_date = df['date'].max()
        all_dates = pd.bdate_range(start=start_date, end=end_date)
        all_dates_df = pd.DataFrame({'date': all_dates})

        df_filled = all_dates_df.merge(df, on='date', how='left')

        price_cols = ['open', 'high', 'low', 'close', 'adjusted_close']
        df_filled[price_cols] = df_filled[price_cols].ffill()
        df_filled['volume'] = df_filled['volume'].fillna(0)
        df_filled = df_filled.dropna(subset=['adjusted_close'])

        logger.info(f"Filled missing data: {len(df)} -> {len(df_filled)} records")
        return df_filled

    @staticmethod
    def download_all_assets(
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> dict:
        """批量下载多个资产的数据"""
        results = {}

        for i, symbol in enumerate(tqdm(symbols, desc="Downloading asset data")):
            if i > 0:
                delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
                time.sleep(delay)

            df = DataDownloader.download_asset_data(symbol, start_date, end_date)

            if df is not None and DataDownloader.validate_data(df):
                results[symbol] = df
            else:
                logger.warning(f"✗ {symbol}: Failed to download")

        return results

    @staticmethod
    def get_asset_inception_date(symbol: str) -> Optional[date]:
        """获取资产的成立日期"""
        # 尝试从 Stooq 获取（从 1990 年开始）
        df = DataDownloader._download_from_stooq(symbol, date(1990, 1, 1), datetime.now().date())
        if df is not None and not df.empty:
            inception = df['date'].min()
            logger.info(f"{symbol} inception date: {inception}")
            return inception

        # 回退到 yfinance
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start='1990-01-01', end=datetime.now())

            if not hist.empty:
                inception = hist.index.min().date()
                logger.info(f"{symbol} inception date: {inception}")
                return inception
        except Exception as e:
            logger.error(f"Failed to get inception date for {symbol}: {e}")

        return None
