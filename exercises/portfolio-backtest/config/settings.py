"""
全局配置管理
"""
import os
from pathlib import Path
import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据库路径
DATABASE_PATH = PROJECT_ROOT / "database" / "portfolio.duckdb"

# 配置文件路径
CONFIG_FILE = PROJECT_ROOT / "config" / "portfolios.yaml"

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"


def get_config() -> dict:
    """加载投资组合配置"""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"配置文件未找到: {CONFIG_FILE}")

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def ensure_directories():
    """确保必要的目录存在"""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# 初始化
ensure_directories()
