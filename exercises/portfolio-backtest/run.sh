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

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}虚拟环境不存在，正在创建...${NC}"
    python3 -m venv .venv
fi

# 激活虚拟环境
echo -e "${YELLOW}激活虚拟环境...${NC}"
source .venv/bin/activate

# 检查数据库
if [ ! -f "database/portfolio.duckdb" ]; then
    echo -e "${YELLOW}数据库不存在，正在下载历史数据...${NC}"
    echo -e "${YELLOW}这可能需要几分钟时间...${NC}"
    python download_data.py
    echo -e "${GREEN}数据下载完成！${NC}"
else
    echo -e "${GREEN}✓ 数据库已存在${NC}"
fi

# 启动 Streamlit
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}启动 Streamlit 应用...${NC}"
echo -e "${GREEN}打开浏览器访问: http://localhost:8501${NC}"
echo -e "${GREEN}========================================${NC}"

streamlit run visualization/streamlit_app.py
