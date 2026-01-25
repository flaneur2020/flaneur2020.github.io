#!/bin/bash
# 快速启动脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}永久投资组合回测系统${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查数据库是否有数据
check_database() {
    uv run python3 -c "
import duckdb
from pathlib import Path
db_path = Path('database/portfolio.duckdb')
if not db_path.exists():
    exit(1)
conn = duckdb.connect(str(db_path))
count = conn.execute('SELECT COUNT(*) FROM asset_prices').fetchone()[0]
exit(0 if count > 0 else 1)
" 2>/dev/null
}

# 步骤1: 确保数据库有数据
if ! check_database; then
    echo -e "${YELLOW}数据库为空或不存在，正在下载历史数据...${NC}"
    echo -e "${YELLOW}这可能需要几分钟时间...${NC}"

    if ! uv run python3 download_data.py; then
        echo -e "${RED}数据下载失败！${NC}"
        exit 1
    fi

    # 再次检查数据库
    if ! check_database; then
        echo -e "${RED}数据下载后数据库仍为空，请检查网络或稍后重试${NC}"
        exit 1
    fi

    echo -e "${GREEN}数据下载完成！${NC}"
else
    echo -e "${GREEN}✓ 数据库已有数据${NC}"
fi

# 步骤2: 启动 Streamlit
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}启动 Streamlit 应用...${NC}"
echo -e "${GREEN}打开浏览器访问: http://localhost:8501${NC}"
echo -e "${GREEN}========================================${NC}"

uv run streamlit run visualization/streamlit_app.py
