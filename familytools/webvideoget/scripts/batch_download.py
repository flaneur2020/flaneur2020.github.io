#!/usr/bin/env python3
"""
批量视频下载脚本
从URL列表下载多个视频并按指定格式命名
"""

import argparse
import subprocess
import os
import time
import re
from urllib.parse import urlparse, unquote, parse_qs

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def create_driver():
    """创建Chrome浏览器驱动"""
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = uc.Chrome(options=options)
    return driver


def get_cookies_string(driver):
    """获取浏览器的cookie并转换为curl格式"""
    cookies = driver.get_cookies()
    cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])
    return cookie_str


def extract_from_iframe(driver, iframe_element, iframe_index):
    """从iframe中提取视频源"""
    video_sources = []

    try:
        driver.switch_to.frame(iframe_element)
        time.sleep(2)

        videos = driver.find_elements(By.TAG_NAME, "video")
        for video in videos:
            src = video.get_attribute("src")
            if src:
                video_sources.append(src)

            sources = video.find_elements(By.TAG_NAME, "source")
            for source in sources:
                src = source.get_attribute("src")
                if src:
                    video_sources.append(src)

        if not video_sources:
            html = driver.page_source
            patterns = [
                r'https?://[^\s"\'<>]+\.mp4[^\s"\'<>]*',
                r'https?://[^\s"\'<>]+\.m3u8[^\s"\'<>]*',
                r'src=["\']([^"\']+(?:\.mp4|\.m3u8|video|stream)[^"\']*)["\']',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    if match and match not in video_sources:
                        video_sources.append(match)

    except Exception as e:
        print(f"  处理iframe内部时出错: {e}")
    finally:
        driver.switch_to.default_content()

    return video_sources


def extract_iframe_sources(driver):
    """提取iframe中的视频源"""
    video_sources = []
    embed_iframes = []

    iframes = driver.find_elements(By.TAG_NAME, "iframe")

    for i, iframe in enumerate(iframes):
        try:
            src = iframe.get_attribute("src")
            if not src:
                embed_iframes.append((iframe, i+1))
                continue

            if any(keyword in src.lower() for keyword in ['embed', 'video', 'player', 'stream', '.mp4', '.m3u8']):
                embed_iframes.append((iframe, i+1))
            elif 'ad' not in src.lower() and 'ads' not in src.lower():
                embed_iframes.append((iframe, i+1))

        except Exception:
            pass

    for iframe, index in embed_iframes:
        sources = extract_from_iframe(driver, iframe, index)
        video_sources.extend(sources)

    return video_sources


def download_video(url, cookies, output_path, user_agent, referer=None):
    """使用curl下载视频"""
    curl_cmd = [
        "curl",
        "-L",
        "-o", output_path,
        "-H", f"Cookie: {cookies}",
        "-H", f"User-Agent: {user_agent}",
        "--progress-bar",
    ]

    if referer:
        curl_cmd.extend(["-H", f"Referer: {referer}"])

    curl_cmd.append(url)

    try:
        subprocess.run(curl_cmd, check=True)

        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size < 1024:
                with open(output_path, 'r', errors='ignore') as f:
                    content = f.read(500)
                    if '403' in content or 'forbidden' in content.lower():
                        os.remove(output_path)
                        return False
            return True
        return False
    except subprocess.CalledProcessError:
        return False


def process_episode(driver, url, episode_num, title, output_dir):
    """处理单个剧集下载"""
    # 生成文件名: s03eXX_title_in_lowercase.mp4
    safe_title = re.sub(r'[^\w\s-]', '', title.lower())
    safe_title = re.sub(r'[-\s]+', '_', safe_title)
    filename = f"s03e{episode_num:02d}_{safe_title}.mp4"
    output_path = os.path.join(output_dir, filename)

    # 检查文件是否已存在
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 1024 * 1024:  # 大于1MB
            print(f"[跳过] {filename} 已存在 ({file_size // 1024 // 1024}MB)")
            return True

    print(f"\n{'='*60}")
    print(f"正在处理: S03E{episode_num:02d} - {title}")
    print(f"URL: {url}")
    print(f"保存为: {filename}")

    try:
        driver.get(url)
        time.sleep(5)

        page_title = driver.title
        if "Just a moment" in page_title or "稍候" in page_title:
            print("等待验证...")
            for i in range(60):
                time.sleep(1)
                page_title = driver.title
                if "Just a moment" not in page_title and "稍候" not in page_title:
                    break

        time.sleep(10)  # 等待页面加载

        cookies = get_cookies_string(driver)
        user_agent = driver.execute_script("return navigator.userAgent")

        # 提取视频源
        video_sources = extract_iframe_sources(driver)
        video_sources = list(set(video_sources))

        if not video_sources:
            print(f"[失败] 未找到视频源")
            return False

        video_url = video_sources[0]
        print(f"找到视频源: {video_url[:80]}...")

        # 下载
        print(f"开始下载...")
        success = download_video(video_url, cookies, output_path, user_agent, url)

        if success:
            file_size = os.path.getsize(output_path)
            print(f"[成功] 下载完成: {filename} ({file_size // 1024 // 1024}MB)")
            return True
        else:
            print(f"[失败] 下载失败")
            return False

    except Exception as e:
        print(f"[错误] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="批量视频下载工具")
    parser.add_argument("-o", "--output", default="~/emby/media", help="下载目录")
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # 剧集列表
    episodes = [
        (26, "Junior Jack", "https://m.wcostream.tv/geronimo-stilton-season-3-episode-26-junior-jack"),
        (25, "Lights, Camera, Action", "https://m.wcostream.tv/geronimo-stilton-season-3-episode-25-lights-camera-action"),
        (24, "The Mouse Sitter", "https://m.wcostream.tv/geronimo-stilton-season-3-episode-24-the-mouse-sitter"),
        (23, "The Mystery of Mancheco Island", "https://m.wcostream.tv/geronimo-stilton-season-3-episode-23-the-mystery-of-mancheco-island"),
        (22, "Ghost Bashers", "https://m.wcostream.tv/geronimo-stilton-season-3-episode-22-ghost-bashers"),
        (21, "Skateboarding Championship", "https://m.wcostream.tv/geronimo-stilton-season-3-episode-21-skateboarding-championship"),
        (20, "Virtual Vacation", "https://m.wcostream.tv/geronimo-stilton-season-3-episode-20-virtual-vacation"),
        (19, "The False Teeth Caper", "https://m.wcostream.tv/geronimo-stilton-season-3-episode-19-the-false-teeth-caper"),
        (17, "A Tall Order", "https://m.wcostream.tv/geronimo-stilton-season-3-episode-17-a-tall-order"),
        (16, "The Sword of Mousitomo", "https://m.wcostream.tv/geronimo-stilton-season-3-episode-16-the-sword-of-mousitomo"),
        (15, "Snow Job", "https://m.wcostream.tv/geronimo-stilton-season-3-episode-15-snow-job"),
    ]

    print(f"下载目录: {output_dir}")
    print(f"共 {len(episodes)} 个剧集待下载")
    print(f"剧集列表:")
    for ep_num, title, _ in episodes:
        print(f"  S03E{ep_num:02d} - {title}")

    driver = None
    results = []

    try:
        driver = create_driver()

        for ep_num, title, url in episodes:
            success = process_episode(driver, url, ep_num, title, output_dir)
            results.append((ep_num, title, success))

    finally:
        if driver:
            driver.quit()

    # 打印结果汇总
    print(f"\n{'='*60}")
    print("下载结果汇总:")
    print("-" * 60)
    success_count = 0
    for ep_num, title, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"S03E{ep_num:02d} - {title}: {status}")
        if success:
            success_count += 1

    print("-" * 60)
    print(f"总计: {success_count}/{len(episodes)} 成功")


if __name__ == "__main__":
    main()
