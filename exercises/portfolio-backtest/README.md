# 永久投资组合回测系统

基于 Harry Browne 永久投资组合策略的交互式回测系统。使用 Streamlit 前端、DuckDB 数据存储和 Python 回测引擎。

## 功能特性

✅ **完整的投资组合回测** - 从 1993 年至今的历史数据
✅ **多种再平衡策略** - 年度、季度、月度、阈值触发
✅ **灵活的资产配置** - 支持自定义资产列表和权重
✅ **关键性能指标** - 总收益率、年化收益率、夏普比率、最大回撤等
✅ **交互式可视化** - Plotly 图表和 Streamlit 界面
✅ **数据导出** - CSV 格式下载

## 快速开始

### 1. 环境设置

使用 `uv` 管理依赖：

```bash
# 创建虚拟环境并安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

### 2. 初始化数据

下载历史数据到 DuckDB 数据库：

```bash
python download_data.py
```

这将从 yfinance 下载所有资产的历史数据并存储到 `database/portfolio.duckdb`。

**数据覆盖范围：**
- SPY (S&P 500): 1993-01-29 至今
- TLT (20年期国债): 2002-07-26 至今
- GLD (黄金): 2004-11-18 至今
- SHV (短期国债): 2007-01-03 至今

### 3. 运行 Streamlit 应用

```bash
streamlit run visualization/streamlit_app.py
```

应用将在 http://localhost:8501 打开。

### 4. 测试回测引擎

运行测试脚本验证系统功能：

```bash
python test_backtest.py
```

## 项目结构

```
portfolio-backtest/
├── config/
│   ├── settings.py              # 全局配置管理
│   └── portfolios.yaml          # 投资组合配置
├── data/
│   ├── downloader.py            # yfinance 数据下载
│   ├── database.py              # DuckDB 操作
│   └── storage.py               # 数据存储协调
├── core/
│   ├── portfolio.py             # 投资组合类
│   ├── rebalance.py             # 再平衡策略
│   ├── metrics.py               # 性能指标计算
│   └── backtest.py              # 回测引擎
├── visualization/
│   └── streamlit_app.py         # Streamlit 应用
├── database/
│   └── portfolio.duckdb         # 数据库文件
├── download_data.py             # 数据初始化脚本
├── test_backtest.py             # 测试脚本
└── pyproject.toml               # 项目配置
```

## 投资组合配置

### 传统永久投资组合

经典的 Harry Browne 永久投资组合：

- **25% 股票** (SPY) - 市场增长
- **25% 长期国债** (TLT) - 衰退时期表现好
- **25% 黄金** (GLD) - 通胀对冲
- **25% 短期国债** (SHV) - 现金替代

配置文件：`config/portfolios.yaml`

```yaml
portfolios:
  permanent_portfolio:
    name: "传统永久投资组合"
    assets:
      - symbol: SPY
        weight: 0.25
        asset_class: equity
```

## 再平衡策略

### 支持的策略

1. **年度再平衡** - 每年 1 月 1 日再平衡
2. **季度再平衡** - 每季度第一个交易日
3. **月度再平衡** - 每月第一个交易日
4. **阈值触发** - 任何资产偏离目标权重超过 5% 时再平衡
5. **不再平衡** - 仅首次建仓，不进行再平衡

### 性能指标

- **总收益率** - 投资期间的累计收益
- **年化收益率** - 按年计算的平均收益
- **波动率** - 收益率的标准差（风险度量）
- **夏普比率** - 风险调整收益率
- **最大回撤** - 从高点到低点的最大跌幅
- **卡尔玛比率** - 年化收益/最大回撤

## API 文档

### BacktestEngine

```python
from core.backtest import BacktestEngine
from data.database import DatabaseManager

# 初始化
db = DatabaseManager("database/portfolio.duckdb")
engine = BacktestEngine(db)

# 运行回测
results = engine.run_backtest(
    symbols=['SPY', 'TLT', 'GLD', 'SHV'],
    target_weights={'SPY': 0.25, 'TLT': 0.25, 'GLD': 0.25, 'SHV': 0.25},
    start_date=date(2007, 1, 11),
    end_date=date(2026, 1, 23),
    rebalance_strategy=annual_strategy,
    initial_capital=100000.0
)

# 访问结果
print(results['summary']['total_return'])  # 总收益率
print(results['summary']['sharpe_ratio'])  # 夏普比率
```

## 历史表现示例

2007年以来年度回测结果（年度再平衡）：

```
总收益率:       290.83%
年化收益率:     7.42%
波动率:         7.21%
夏普比率:       0.75
最大回撤:       17.10%（2022年）
```

## 关键特点

### 数据处理

- 自动从 yfinance 下载历史数据
- 支持增量更新（只下载缺失的日期）
- 前复权价格处理（调整分红和分拆）
- 缺失交易日前向填充

### 性能优化

- DuckDB 高效查询
- 向量化 NumPy 计算
- Streamlit 数据缓存
- 支持大规模数据回测

### 交易成本建模

- 可配置的交易佣金
- 买卖价差（基点）
- 再平衡时自动扣除成本

## 常见问题

### Q: 如何修改资产配置？

编辑 `config/portfolios.yaml` 文件，添加新的投资组合配置：

```yaml
portfolios:
  my_portfolio:
    name: "我的自定义投资组合"
    inception_date: "2007-01-11"
    assets:
      - symbol: VTI
        weight: 0.60
        asset_class: equity
      - symbol: BND
        weight: 0.40
        asset_class: bond
```

### Q: 如何添加新的资产？

1. 在 `config/portfolios.yaml` 中添加资产定义
2. 运行 `python download_data.py` 下载该资产的历史数据
3. 在回测中使用新资产

### Q: 支持哪些资产？

任何 yfinance 支持的资产，包括：
- 股票：SPY, QQQ, IVV 等
- ETF：所有主流 ETF
- 期货：黄金（GC=F）、石油等

### Q: 如何用我的数据？

1. 在 `data/downloader.py` 中修改数据源
2. 调整 `data/database.py` 中的表结构
3. 更新 `data/storage.py` 的数据加载逻辑

## 未来改进方向

- [ ] 支持更复杂的再平衡规则
- [ ] 添加税收影响计算
- [ ] 蒙特卡洛模拟
- [ ] 多投资组合对比
- [ ] 滑点模型
- [ ] 实时数据更新
- [ ] 风险指标（VaR, CVaR）

## 注意事项

1. **历史数据** - 某些 ETF 成立晚于 1950 年，故无法完全回测到 1950 年
2. **成本建模** - 当前使用简化的交易成本模型，实际成本可能不同
3. **税收** - 回测未考虑税收影响
4. **生存偏差** - 当前 ETF 都存活至今，历史数据不包含倒闭的基金

## 许可证

MIT License

## 作者

创建于 2026 年 1 月
