#!/usr/bin/env python3
"""
回测测试脚本 - 测试永久投资组合回测引擎
"""
import logging
from datetime import date
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DATABASE_PATH, get_config
from data.database import DatabaseManager
from core.backtest import BacktestEngine
from core.rebalance import get_rebalance_strategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Portfolio Backtest - Engine Test")
    logger.info("=" * 60)

    try:
        # 加载配置
        config = get_config()
        portfolio_config = config['portfolios']['permanent_portfolio']

        # 提取资产列表和权重
        symbols = [asset['symbol'] for asset in portfolio_config['assets']]
        target_weights = {
            asset['symbol']: asset['weight']
            for asset in portfolio_config['assets']
        }

        logger.info(f"\nPortfolio: {portfolio_config['name']}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Target weights: {target_weights}")

        # 初始化数据库
        db_manager = DatabaseManager(DATABASE_PATH)

        # 创建回测引擎
        engine = BacktestEngine(db_manager)

        # 设置回测参数
        # 从2007年开始（所有ETF都可用）
        start_date = date(2007, 1, 11)  # SHV的第一个有数据的交易日
        end_date = date(2026, 1, 23)  # 最新数据日期

        initial_capital = 100000.0

        # 测试年度再平衡策略
        logger.info("\n" + "=" * 60)
        logger.info("Testing Annual Rebalance Strategy")
        logger.info("=" * 60)

        annual_strategy = get_rebalance_strategy('annual')

        results = engine.run_backtest(
            symbols=symbols,
            target_weights=target_weights,
            start_date=start_date,
            end_date=end_date,
            rebalance_strategy=annual_strategy,
            initial_capital=initial_capital
        )

        if results:
            # 保存结果
            logger.info("\nSaving results...")

            # 保存每日价值
            daily_df = results['daily_values']
            daily_df.to_csv('backtest_daily_values.csv', index=False)
            logger.info("Saved: backtest_daily_values.csv")

            # 保存月度指标
            if not results['monthly_metrics'].empty:
                results['monthly_metrics'].to_csv('backtest_monthly_metrics.csv', index=False)
                logger.info("Saved: backtest_monthly_metrics.csv")

            # 保存年度指标
            if not results['yearly_metrics'].empty:
                results['yearly_metrics'].to_csv('backtest_yearly_metrics.csv', index=False)
                logger.info("Saved: backtest_yearly_metrics.csv")

            # 保存再平衡历史
            if not results['rebalance_history'].empty:
                results['rebalance_history'].to_csv('backtest_rebalance_history.csv', index=False)
                logger.info("Saved: backtest_rebalance_history.csv")

            logger.info("\n" + "=" * 60)
            logger.info("Yearly Performance:")
            logger.info("=" * 60)

            if not results['yearly_metrics'].empty:
                yearly = results['yearly_metrics']
                print("\n", yearly[['period', 'return', 'volatility']].to_string(index=False))

        # 关闭数据库
        db_manager.disconnect()

        logger.info("\n" + "=" * 60)
        logger.info("Test completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
