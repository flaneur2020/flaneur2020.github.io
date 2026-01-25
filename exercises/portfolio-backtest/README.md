# 永久投资组合回测系统

基于 Harry Browne 永久投资组合策略的交互式回测系统。使用 Streamlit 前端、DuckDB 数据存储和 Python 回测引擎。

## Quick Start

Initialize the venv:


```bash
uv sync

source .venv/bin/activate
```

Run:

```bash
# download data to local duckdb
uv run python3 download_data.py

# display the visualization
uv run streamlit run visualization/streamlit_app.py
```

应用将在 http://localhost:8501 打开。

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
2. 运行 `uv python3 download_data.py` 下载该资产的历史数据
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

## License

MIT License
