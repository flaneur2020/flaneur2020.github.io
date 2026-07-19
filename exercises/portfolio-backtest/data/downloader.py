"""
数据下载模块 - 支持多数据源（yfinance, stooq）

注意：Yahoo Finance 和 Stooq 都对自动化请求做了反爬和限速处理。
- yfinance: 2025 年起 Yahoo 加强了反爬，"Expecting value / 429 Too Many
  Requests" 通常是被限速或命中了 consent 墙。这里通过 curl_cffi 的浏览器
  伪装 session 绕过 TLS 指纹检测，并对 429 做指数退避重试。
- Stooq: 现在会先返回一段 JS 浏览器校验（SHA-256 proof-of-work）再下发
  auth cookie；只有带上该 cookie 才能拿到真正的 CSV。这里实现了解 PoW 的
  逻辑，并复用同一个 curl_cffi session（带 cookie + 浏览器 TLS 指纹）。
"""
import hashlib
import re

import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List
import logging
import time
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Rate limiting 配置
REQUEST_DELAY_MIN = 1.0
REQUEST_DELAY_MAX = 2.0

# 伪装的浏览器 User-Agent（普通 requests 的默认 UA 很容易被识别为爬虫）
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _make_impersonated_session():
    """创建一个伪装成浏览器的 requests session。

    优先用 curl_cffi（与 yfinance 同源依赖，能伪造 TLS/JA3 指纹，绕过 Yahoo /
    Stooq 的反爬）；不可用时回退到普通 requests Session 并设置浏览器 UA 和
    cookie jar。返回 (session, backend) 便于调用方区分行为。
    """
    try:
        from curl_cffi import requests as cf_requests

        session = cf_requests.Session(impersonate="chrome")
        session.headers.update({"User-Agent": BROWSER_UA})
        return session, "curl_cffi"
    except ImportError:
        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": BROWSER_UA})
        return session, "requests"


def _is_stooq_challenge(text: str) -> bool:
    """判断返回内容是否是 Stooq 的 JS 浏览器校验页而非 CSV。"""
    if not text:
        return False
    return text.lstrip().lower().startswith(("<!doctype", "<html"))


def _solve_stooq_challenge(session, challenge_html: str, referer: str) -> bool:
    """解 Stooq 的 SHA-256 proof-of-work 校验并写入 auth cookie。

    页面内联 JS 的逻辑：找到 nonce n，使 SHA256(challenge + str(n)) 的十六进制
    表示以 d 个 '0' 开头，然后 POST 到 /__verify；服务端校验通过后下发 auth
    cookie。返回是否成功（后续带 cookie 请求就能拿到 CSV）。
    """
    m_c = re.search(r'const c="([^"]+)"', challenge_html)
    m_d = re.search(r',d=(\d+),', challenge_html)
    if not m_c or not m_d:
        logger.warning("[Stooq] Could not parse challenge page")
        return False

    challenge = m_c.group(1)
    difficulty = int(m_d.group(1))
    prefix = "0" * difficulty

    nonce = 0
    while not hashlib.sha256(f"{challenge}{nonce}".encode()).hexdigest().startswith(prefix):
        nonce += 1
        if nonce > 50_000_000:  # 兜底，防止异常难度卡死
            logger.warning(f"[Stooq] PoW difficulty {difficulty} too high, giving up")
            return False

    try:
        rv = session.post(
            "https://stooq.com/__verify",
            data={"c": challenge, "n": str(nonce)},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": referer,
                "Origin": "https://stooq.com",
            },
            timeout=30,
        )
        ok = rv.status_code == 200 and rv.text.strip().lower() == "ok"
        if not ok:
            logger.warning(f"[Stooq] Verify returned {rv.status_code}: {rv.text[:80]}")
        return ok
    except Exception as e:
        logger.warning(f"[Stooq] Verify request failed: {e}")
        return False


class DataDownloader:
    """数据下载器 - 支持多数据源"""

    # 数据源优先级
    DATA_SOURCES = ['stooq', 'yfinance']

    def __init__(self):
        """初始化下载器"""
        pass

    @staticmethod
    def _download_from_stooq(
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """从 Stooq 下载数据（免费，无需 API key）

        Stooq 现在会先返回一段 JS 浏览器校验（SHA-256 proof-of-work）再下发
        auth cookie，所以必须用带 cookie / 浏览器指纹的 session 请求，并在
        首次拿到校验页时解出 PoW、提交拿到 cookie，之后才能拿到 CSV 内容。
        """
        session, backend = _make_impersonated_session()
        try:
            from io import StringIO

            logger.info(f"[Stooq] Downloading {symbol} (via {backend})...")

            # Stooq CSV 下载 URL
            # 注意：参数顺序是 d1=开始日期, d2=结束日期 (YYYYMMDD)
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&d1={start_str}&d2={end_str}&i=d"

            # 首次请求可能命中 JS 校验页（proof-of-work）。解出 PoW 提交拿到
            # auth cookie 后，再次请求即可拿到 CSV。challenge cookie 有时效，
            # 最多重试几次。
            text = ""
            for attempt in range(3):
                response = session.get(url, timeout=30)
                text = getattr(response, "text", "")
                if not _is_stooq_challenge(text):
                    break
                logger.debug(f"[Stooq] {symbol}: got JS challenge page (attempt {attempt+1}/3), solving PoW")
                if not _solve_stooq_challenge(session, text, url):
                    break
                time.sleep(1 + attempt)

            if not text or _is_stooq_challenge(text):
                logger.warning(f"[Stooq] No CSV data for {symbol} (still JS challenge after attempts)")
                return None

            if text.strip().lower() == "access denied":
                logger.warning(f"[Stooq] Access denied for {symbol} (IP may be blocked by Stooq)")
                return None

            # 读取 CSV
            df = pd.read_csv(StringIO(text))

            if df.empty or len(df) < 10:
                logger.warning(f"[Stooq] No data for {symbol}")
                return None

            # 统一列名（Stooq 列名：Date,Open,High,Low,Close,Volume）
            df.columns = [c.lower() for c in df.columns]
            df['adjusted_close'] = df['close']
            df['date'] = pd.to_datetime(df['date']).dt.date

            # 选择需要的列
            df = df[['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']]
            df = df.dropna()
            df = df.sort_values('date')  # Stooq 可能是倒序

            logger.info(f"[Stooq] Downloaded {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.warning(f"[Stooq] Failed for {symbol}: {e}")
            return None

    @staticmethod
    def _download_from_yfinance(
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """从 yfinance 下载数据

        用 curl_cffi 伪装的浏览器 session 喂给 yfinance，绕过 Yahoo 的 TLS 指纹 /
        consent 墙；遇到 429 限速时做指数退避重试。
        """
        session, backend = _make_impersonated_session()
        max_retries = 4
        last_err = None
        for attempt in range(max_retries):
            try:
                logger.info(f"[yfinance] Downloading {symbol} (via {backend}, attempt {attempt+1}/{max_retries})...")

                ticker = yf.Ticker(symbol, session=session)
                df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

                if df.empty:
                    logger.warning(f"[yfinance] No data for {symbol}")
                    return None

                df.reset_index(inplace=True)
                df.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }, inplace=True)

                df['adjusted_close'] = df['close']
                df = df[['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']]
                df['date'] = pd.to_datetime(df['date']).dt.date

                logger.info(f"[yfinance] Downloaded {len(df)} records for {symbol}")
                return df

            except Exception as e:
                last_err = e
                error_msg = str(e)
                if "Rate limited" in error_msg or "Too Many Requests" in error_msg:
                    wait = 5 * (2 ** attempt) + random.uniform(0, 2)
                    logger.warning(
                        f"[yfinance] Rate limited for {symbol}, retrying in {wait:.1f}s "
                        f"({attempt+1}/{max_retries})"
                    )
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(f"[yfinance] Failed for {symbol}: {e}")
                    return None

        logger.warning(f"[yfinance] Rate limited for {symbol} after {max_retries} retries: {last_err}")
        return None

    @staticmethod
    def download_asset_data(
        symbol: str,
        start_date: date,
        end_date: date,
        progress_desc: str = None
    ) -> Optional[pd.DataFrame]:
        """
        下载单个资产的历史数据（自动尝试多个数据源）

        Args:
            symbol: 资产代码 (e.g., 'SPY', 'TLT')
            start_date: 开始日期
            end_date: 结束日期
            progress_desc: 进度条描述

        Returns:
            包含 OHLCV 数据的 DataFrame，如果下载失败则返回 None
        """
        logger.info(f"Downloading {symbol} from {start_date} to {end_date}")

        for source in DataDownloader.DATA_SOURCES:
            if source == 'stooq':
                df = DataDownloader._download_from_stooq(symbol, start_date, end_date)
            elif source == 'yfinance':
                df = DataDownloader._download_from_yfinance(symbol, start_date, end_date)
            else:
                continue

            if df is not None and DataDownloader.validate_data(df):
                logger.info(f"✓ {symbol}: Got {len(df)} records from {source}")
                return df

            # 数据源之间稍微等一下
            time.sleep(0.5)

        logger.error(f"✗ {symbol}: All data sources failed")
        return None

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """验证数据质量"""
        if df is None or df.empty:
            return False

        required_columns = ['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.warning("Missing required columns")
            return False

        # 检查价格列是否有 NaN
        price_cols = ['open', 'high', 'low', 'close', 'adjusted_close']
        if df[price_cols].isna().any().any():
            logger.warning("Found NaN values in price data")
            return False

        return True

    @staticmethod
    def fill_missing_data(df: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """填补缺失的交易日数据"""
        if df is None or df.empty:
            return df

        df = df.sort_values('date').reset_index(drop=True)

        start_date = df['date'].min()
        end_date = df['date'].max()
        all_dates = pd.bdate_range(start=start_date, end=end_date)
        all_dates_df = pd.DataFrame({'date': all_dates})

        df_filled = all_dates_df.merge(df, on='date', how='left')

        price_cols = ['open', 'high', 'low', 'close', 'adjusted_close']
        df_filled[price_cols] = df_filled[price_cols].ffill()
        df_filled['volume'] = df_filled['volume'].fillna(0)
        df_filled = df_filled.dropna(subset=['adjusted_close'])

        logger.info(f"Filled missing data: {len(df)} -> {len(df_filled)} records")
        return df_filled

    @staticmethod
    def download_all_assets(
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> dict:
        """批量下载多个资产的数据"""
        results = {}

        for i, symbol in enumerate(tqdm(symbols, desc="Downloading asset data")):
            if i > 0:
                delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
                time.sleep(delay)

            df = DataDownloader.download_asset_data(symbol, start_date, end_date)

            if df is not None and DataDownloader.validate_data(df):
                results[symbol] = df
            else:
                logger.warning(f"✗ {symbol}: Failed to download")

        return results

    @staticmethod
    def get_asset_inception_date(symbol: str) -> Optional[date]:
        """获取资产的成立日期"""
        # 尝试从 Stooq 获取（从 1990 年开始）
        df = DataDownloader._download_from_stooq(symbol, date(1990, 1, 1), datetime.now().date())
        if df is not None and not df.empty:
            inception = df['date'].min()
            logger.info(f"{symbol} inception date: {inception}")
            return inception

        # 回退到 yfinance
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start='1990-01-01', end=datetime.now())

            if not hist.empty:
                inception = hist.index.min().date()
                logger.info(f"{symbol} inception date: {inception}")
                return inception
        except Exception as e:
            logger.error(f"Failed to get inception date for {symbol}: {e}")

        return None
