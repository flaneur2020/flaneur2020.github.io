#!/usr/bin/env python3
"""
数据初始化脚本 - 下载所有资产的历史数据到DuckDB
"""
import logging
from datetime import date
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DATABASE_PATH, get_config
from data.database import DatabaseManager
from data.storage import DataStorage

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Portfolio Backtest - Data Initialization")
    logger.info("=" * 60)

    try:
        # 初始化数据库
        logger.info(f"Database path: {DATABASE_PATH}")
        db_manager = DatabaseManager(DATABASE_PATH)

        # 初始化存储管理器
        storage = DataStorage(db_manager)

        # 同步永久投资组合的数据
        logger.info("\nSyncing Permanent Portfolio data...")
        results = storage.sync_all_assets(portfolio_name="permanent_portfolio")

        # 显示结果
        logger.info("\n" + "=" * 60)
        logger.info("Sync Results:")
        logger.info("=" * 60)

        for symbol, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"{symbol}: {status}")

        # 显示数据统计
        logger.info("\n" + "=" * 60)
        logger.info("Data Statistics:")
        logger.info("=" * 60)

        config = get_config()
        portfolio = config['portfolios']['permanent_portfolio']

        for asset in portfolio['assets']:
            symbol = asset['symbol']
            start_date, end_date = db_manager.get_date_range(symbol)
            coverage = db_manager.check_data_completeness(symbol)

            if start_date:
                logger.info(
                    f"{symbol}: {start_date} to {end_date} "
                    f"({(end_date - start_date).days} days, {coverage:.1%} coverage)"
                )
            else:
                logger.warning(f"{symbol}: No data")

        db_manager.disconnect()
        logger.info("\n" + "=" * 60)
        logger.info("Data initialization complete!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
