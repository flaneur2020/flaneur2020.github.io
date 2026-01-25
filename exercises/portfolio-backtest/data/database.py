"""
DuckDB数据库操作模块
"""
import duckdb
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """DuckDB数据库管理类"""

    def __init__(self, db_path: Path):
        """
        初始化数据库连接

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.connection = None
        self.connect()
        self.create_tables()

    def connect(self):
        """创建数据库连接"""
        try:
            self.connection = duckdb.connect(str(self.db_path))
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def disconnect(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def create_tables(self):
        """创建必要的数据库表"""
        try:
            # 资产价格表
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS asset_prices (
                    symbol VARCHAR NOT NULL,
                    date DATE NOT NULL,
                    open DECIMAL(10, 4),
                    high DECIMAL(10, 4),
                    low DECIMAL(10, 4),
                    close DECIMAL(10, 4),
                    adjusted_close DECIMAL(10, 4),
                    volume BIGINT,
                    data_source VARCHAR DEFAULT 'yfinance',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date)
                )
            """)

            # 创建索引
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_asset_prices_date ON asset_prices(date)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_asset_prices_symbol ON asset_prices(symbol)"
            )

            logger.info("Tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def insert_prices(self, symbol: str, df: pd.DataFrame):
        """
        插入资产价格数据

        Args:
            symbol: 资产代码
            df: 包含 date, open, high, low, close, adjusted_close, volume 的DataFrame
        """
        try:
            # 确保日期格式正确
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['symbol'] = symbol

            # 使用 REPLACE 实现 upsert（如果记录存在则更新）
            self.connection.execute(
                """
                INSERT INTO asset_prices
                (symbol, date, open, high, low, close, adjusted_close, volume)
                SELECT symbol, date, open, high, low, close, adjusted_close, volume
                FROM df
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adjusted_close = EXCLUDED.adjusted_close,
                    volume = EXCLUDED.volume
                """
            )
            logger.info(f"Inserted {len(df)} records for {symbol}")
        except Exception as e:
            logger.error(f"Failed to insert prices for {symbol}: {e}")
            raise

    def get_latest_date(self, symbol: str) -> Optional[date]:
        """
        获取某个资产的最新数据日期

        Args:
            symbol: 资产代码

        Returns:
            最新日期，如果没有数据则返回 None
        """
        try:
            result = self.connection.execute(
                f"SELECT MAX(date) as latest_date FROM asset_prices WHERE symbol = '{symbol}'"
            ).fetchall()

            if result and result[0][0]:
                return result[0][0]
            return None
        except Exception as e:
            logger.error(f"Failed to get latest date for {symbol}: {e}")
            return None

    def get_prices(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        获取价格数据

        Args:
            symbols: 资产代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            包含价格数据的DataFrame
        """
        try:
            symbols_list = ','.join([f"'{s}'" for s in symbols])
            query = f"""
                SELECT symbol, date, open, high, low, close, adjusted_close, volume
                FROM asset_prices
                WHERE symbol IN ({symbols_list})
                  AND date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date, symbol
            """
            df = self.connection.execute(query).fetch_df()
            logger.info(f"Retrieved {len(df)} records for {len(symbols)} symbols")
            return df
        except Exception as e:
            logger.error(f"Failed to get prices: {e}")
            raise

    def get_symbols(self) -> List[str]:
        """获取数据库中所有可用的资产代码"""
        try:
            result = self.connection.execute(
                "SELECT DISTINCT symbol FROM asset_prices ORDER BY symbol"
            ).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []

    def get_date_range(self, symbol: str) -> Tuple[Optional[date], Optional[date]]:
        """
        获取某个资产的数据日期范围

        Args:
            symbol: 资产代码

        Returns:
            (开始日期, 结束日期) 元组
        """
        try:
            result = self.connection.execute(
                f"""
                SELECT MIN(date) as start_date, MAX(date) as end_date
                FROM asset_prices
                WHERE symbol = '{symbol}'
                """
            ).fetchall()

            if result and result[0][0]:
                return result[0][0], result[0][1]
            return None, None
        except Exception as e:
            logger.error(f"Failed to get date range for {symbol}: {e}")
            return None, None

    def check_data_completeness(self, symbol: str) -> float:
        """
        检查数据完整性（返回数据覆盖的交易日占比）

        Args:
            symbol: 资产代码

        Returns:
            数据覆盖率（0-1之间的浮点数）
        """
        try:
            result = self.connection.execute(
                f"""
                SELECT COUNT(*) as data_count,
                       DATEDIFF('day', MIN(date), MAX(date)) as total_days
                FROM asset_prices
                WHERE symbol = '{symbol}'
                """
            ).fetchall()

            if result and result[0][1]:
                data_count = result[0][0]
                total_days = result[0][1]
                # 假设每年约252个交易日
                expected_count = (total_days / 365.0) * 252
                coverage = data_count / expected_count if expected_count > 0 else 0
                return min(coverage, 1.0)
            return 0.0
        except Exception as e:
            logger.error(f"Failed to check data completeness: {e}")
            return 0.0

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
