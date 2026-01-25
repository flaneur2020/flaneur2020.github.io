#!/usr/bin/env python3
"""
数据初始化脚本 - 下载所有资产的历史数据到DuckDB
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DATABASE_PATH, get_config
from data.database import DatabaseManager
from data.storage import DataStorage

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def download_portfolio(storage: DataStorage, portfolio_name: str, portfolio: dict) -> dict:
    """下载单个投资组合的数据

    Args:
        storage: 数据存储管理器
        portfolio_name: 投资组合名称
        portfolio: 投资组合配置

    Returns:
        同步结果字典 {symbol: success}
    """
    logger.info(f"\nSyncing data for: {portfolio['name']} ({portfolio_name})...")
    results = storage.sync_all_assets(portfolio_name=portfolio_name)
    return results


def get_portfolios_to_sync(config: dict, portfolio_name: str | None) -> dict:
    """获取需要同步的投资组合

    Args:
        config: 配置字典
        portfolio_name: 指定的投资组合名称，None 表示全部

    Returns:
        待同步的投资组合字典 {name: portfolio}

    Raises:
        SystemExit: 当指定的投资组合不存在时
    """
    if portfolio_name:
        if portfolio_name not in config["portfolios"]:
            logger.error(f"Portfolio '{portfolio_name}' not found in config.")
            logger.info(f"Available portfolios: {list(config['portfolios'].keys())}")
            sys.exit(1)
        return {portfolio_name: config["portfolios"][portfolio_name]}
    return config["portfolios"]


def show_sync_results(sync_results: dict) -> None:
    """显示同步结果

    Args:
        sync_results: 同步结果字典 {symbol: success}
    """
    logger.info("\n" + "=" * 60)
    logger.info("Sync Results:")
    logger.info("=" * 60)

    for symbol, success in sync_results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{symbol}: {status}")


def show_data_statistics(db_manager: DatabaseManager, symbols: set) -> None:
    """显示数据统计信息

    Args:
        db_manager: 数据库管理器
        symbols: 资产代码集合
    """
    logger.info("\n" + "=" * 60)
    logger.info("Data Statistics:")
    logger.info("=" * 60)

    for symbol in sorted(symbols):
        start_date, end_date = db_manager.get_date_range(symbol)
        coverage = db_manager.check_data_completeness(symbol)

        if start_date:
            logger.info(
                f"{symbol}: {start_date} to {end_date} "
                f"({(end_date - start_date).days} days, {coverage:.1%} coverage)"
            )
        else:
            logger.warning(f"{symbol}: No data")


def main(portfolio_name: str | None = None):
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

        # 获取配置
        config = get_config()

        # 确定要同步的 portfolios
        portfolios_to_sync = get_portfolios_to_sync(config, portfolio_name)

        # 遍历所有要同步的 portfolios
        all_symbols = set()
        sync_results = {}

        for p_name, portfolio in portfolios_to_sync.items():
            results = download_portfolio(storage, p_name, portfolio)
            sync_results.update(results)

            # 收集所有 symbols 用于后续统计显示
            for asset in portfolio["assets"]:
                all_symbols.add(asset["symbol"])

        # 显示结果
        show_sync_results(sync_results)
        show_data_statistics(db_manager, all_symbols)

        db_manager.disconnect()
        logger.info("\n" + "=" * 60)
        logger.info("Data initialization complete!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载投资组合的历史数据")
    parser.add_argument(
        "--portfolio",
        "-p",
        default=None,
        help="要下载的投资组合名称 (不指定则同步所有组合)",
    )
    args = parser.parse_args()
    main(portfolio_name=args.portfolio)
