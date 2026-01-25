"""
Streamlit应用 - 永久投资组合回测系统前端
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
from pathlib import Path
import sys
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATABASE_PATH, get_config
from data.database import DatabaseManager
from core.backtest import BacktestEngine
from core.rebalance import get_rebalance_strategy

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="永久投资组合回测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_manager():
    """获取数据库管理器（缓存）"""
    return DatabaseManager(DATABASE_PATH)


@st.cache_data
def load_config():
    """加载配置"""
    return get_config()


def format_currency(value):
    """格式化货币"""
    return f"${value:,.2f}"


def format_percentage(value):
    """格式化百分比"""
    return f"{value:.2%}"


def main():
    """主应用"""
    st.title("📊 永久投资组合回测系统")
    st.markdown("Harry Browne Permanent Portfolio Backtesting Engine")

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 回测配置")

        # 加载配置
        config = load_config()

        # 选择投资组合
        portfolio_names = list(config['portfolios'].keys())
        selected_portfolio = st.selectbox(
            "投资组合",
            portfolio_names,
            format_func=lambda x: config['portfolios'][x]['name']
        )

        portfolio_config = config['portfolios'][selected_portfolio]
        assets = portfolio_config['assets']
        symbols = [asset['symbol'] for asset in assets]
        target_weights = {asset['symbol']: asset['weight'] for asset in assets}

        st.markdown("### 资产配置")
        for asset in assets:
            st.text(f"{asset['symbol']}: {asset['weight']:.0%} - {asset['name']}")

        # 时间范围选择
        st.markdown("### 时间范围")
        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "开始日期",
                value=date(2007, 1, 11),
                min_value=date(2007, 1, 1),
                max_value=date.today()
            )

        with col2:
            end_date = st.date_input(
                "结束日期",
                value=date.today(),
                min_value=date(2007, 1, 1),
                max_value=date.today()
            )

        # 验证日期范围
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期")
            return

        # 再平衡策略
        st.markdown("### 再平衡策略")
        strategy_options = {
            'annual': '年度再平衡',
            'quarterly': '季度再平衡',
            'threshold': '阈值触发（5%）',
            'monthly': '月度再平衡',
            'none': '不再平衡'
        }

        selected_strategy = st.selectbox(
            "再平衡频率",
            list(strategy_options.keys()),
            format_func=lambda x: strategy_options[x]
        )

        # 初始资金
        st.markdown("### 初始投资")
        initial_capital = st.number_input(
            "初始资金（美元）",
            min_value=1000,
            value=100000,
            step=10000
        )

        # 运行回测按钮
        run_button = st.button("🚀 运行回测", type="primary", use_container_width=True)

    # 主区域
    if run_button:
        with st.spinner("正在运行回测..."):
            try:
                # 初始化数据库和引擎
                db_manager = get_db_manager()
                engine = BacktestEngine(db_manager)

                # 获取再平衡策略
                strategy_config = config.get('rebalance_strategies', {}).get(selected_strategy, {})
                rebalance_strategy = get_rebalance_strategy(selected_strategy, strategy_config)

                # 运行回测
                results = engine.run_backtest(
                    symbols=symbols,
                    target_weights=target_weights,
                    start_date=start_date,
                    end_date=end_date,
                    rebalance_strategy=rebalance_strategy,
                    initial_capital=initial_capital
                )

                if results:
                    # 显示结果
                    display_results(results, symbols, selected_portfolio, config)

            except Exception as e:
                st.error(f"回测失败: {str(e)}")
                logger.error(f"Error: {e}", exc_info=True)


def display_results(results, symbols, portfolio_name, config):
    """显示回测结果"""

    summary = results['summary']

    # 关键指标卡片
    st.markdown("## 关键指标")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "累计收益率",
            format_percentage(summary['total_return']),
            delta=None
        )

    with col2:
        st.metric(
            "年化收益率",
            format_percentage(summary['annualized_return']),
            delta=None
        )

    with col3:
        st.metric(
            "夏普比率",
            f"{summary['sharpe_ratio']:.2f}",
            delta=None
        )

    with col4:
        st.metric(
            "最大回撤",
            format_percentage(-summary['max_drawdown']),
            delta=None
        )

    # 额外指标
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("波动率（年化）", format_percentage(summary['volatility']))

    with col2:
        st.metric("卡尔玛比率", f"{summary['calmar_ratio']:.2f}")

    with col3:
        st.metric(
            "最终价值",
            format_currency(summary['end_value']),
            delta=format_currency(summary['end_value'] - summary['start_value'])
        )

    # 投资组合净值曲线
    st.markdown("## 投资组合净值曲线")

    daily_values = results['daily_values']
    fig = create_value_chart(daily_values)
    st.plotly_chart(fig, use_container_width=True)

    # 回撤分析
    st.markdown("## 回撤分析")

    drawdown_fig = create_drawdown_chart(results['drawdown_series'])
    st.plotly_chart(drawdown_fig, use_container_width=True)

    # 分时段表现
    st.markdown("## 分时段表现")

    tab1, tab2, tab3 = st.tabs(["按年度", "按月度", "再平衡历史"])

    with tab1:
        if not results['yearly_metrics'].empty:
            yearly = results['yearly_metrics'].copy()
            yearly['period'] = yearly['period'].astype(str)
            yearly['return'] = yearly['return'].apply(lambda x: f"{x:.2%}")
            yearly['volatility'] = yearly['volatility'].apply(lambda x: f"{x:.2%}")
            st.dataframe(yearly[['period', 'return', 'volatility']], use_container_width=True)

            # 年度收益图
            yearly_data = results['yearly_metrics'].copy()
            fig_annual = px.bar(
                yearly_data,
                x='period',
                y='return',
                title='年度收益率',
                labels={'period': '年份', 'return': '收益率'}
            )
            fig_annual.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_annual, use_container_width=True)

    with tab2:
        if not results['monthly_metrics'].empty:
            monthly = results['monthly_metrics'].copy()
            monthly['period'] = monthly['period'].astype(str)
            monthly['return'] = monthly['return'].apply(lambda x: f"{x:.2%}")
            # 显示最近12个月
            st.dataframe(
                monthly[['period', 'return']].tail(12),
                use_container_width=True
            )

    with tab3:
        if not results['rebalance_history'].empty:
            rebalance = results['rebalance_history'].copy()
            rebalance['date'] = rebalance['date'].astype(str)
            rebalance['portfolio_value'] = rebalance['portfolio_value'].apply(
                lambda x: format_currency(x)
            )
            rebalance['transaction_cost'] = rebalance['transaction_cost'].apply(
                lambda x: format_currency(x) if x > 0 else "-"
            )
            st.dataframe(rebalance, use_container_width=True)

            st.info(f"总再平衡次数: {len(rebalance)}")
            if rebalance['transaction_cost'].dtype == object:
                # 计算总交易成本
                costs = results['rebalance_history']['transaction_cost'].sum()
                st.info(f"总交易成本: {format_currency(costs)}")

    # 资产权重变化
    st.markdown("## 资产权重变化")

    daily_df = daily_values.copy()

    # 提取权重列
    weight_data = []
    for symbol in symbols:
        col = f'{symbol}_weight'
        if col in daily_df.columns:
            weight_data.append({
                'date': daily_df['date'],
                'symbol': symbol,
                'weight': daily_df[col]
            })

    if weight_data:
        weight_df = pd.concat(
            [pd.DataFrame(d) for d in weight_data],
            ignore_index=True
        )

        fig_weight = px.area(
            weight_df,
            x='date',
            y='weight',
            color='symbol',
            title='资产权重变化',
            labels={'date': '日期', 'weight': '权重', 'symbol': '资产'},
            groupnorm='percent'
        )
        st.plotly_chart(fig_weight, use_container_width=True)

    # 下载数据
    st.markdown("## 导出数据")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = daily_values.to_csv(index=False)
        st.download_button(
            label="📥 下载每日数据 (CSV)",
            data=csv,
            file_name="backtest_daily_values.csv",
            mime="text/csv"
        )

    with col2:
        if not results['yearly_metrics'].empty:
            csv = results['yearly_metrics'].to_csv(index=False)
            st.download_button(
                label="📥 下载年度指标 (CSV)",
                data=csv,
                file_name="backtest_yearly_metrics.csv",
                mime="text/csv"
            )

    with col3:
        if not results['rebalance_history'].empty:
            csv = results['rebalance_history'].to_csv(index=False)
            st.download_button(
                label="📥 下载再平衡历史 (CSV)",
                data=csv,
                file_name="backtest_rebalance_history.csv",
                mime="text/csv"
            )


def create_value_chart(daily_values):
    """创建净值曲线图"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_values['date'],
        y=daily_values['value'],
        mode='lines',
        name='投资组合价值',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))

    fig.update_layout(
        title='投资组合净值曲线',
        xaxis_title='日期',
        yaxis_title='价值 (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    fig.update_yaxes(tickformat='$,.0f')

    return fig


def create_drawdown_chart(drawdown_series):
    """创建回撤曲线图"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values,
        mode='lines',
        name='回撤',
        line=dict(color='#d62728', width=1),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.3)'
    ))

    fig.update_layout(
        title='回撤曲线',
        xaxis_title='日期',
        yaxis_title='回撤率',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    fig.update_yaxes(tickformat='.0%')

    return fig


if __name__ == "__main__":
    main()
