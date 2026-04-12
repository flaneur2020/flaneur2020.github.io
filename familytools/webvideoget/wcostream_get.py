#!/usr/bin/env python3
"""
浏览器视频下载脚本
打开浏览器访问URL，读取cookie，解析video标签，使用curl下载视频
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


def create_driver(headless=False):
    """创建Chrome浏览器驱动（使用undetected_chromedriver绕过检测）"""
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # 禁用图片加载以加快速度
    # prefs = {"profile.managed_default_content_settings.images": 2}
    # options.add_experimental_option("prefs", prefs)

    driver = uc.Chrome(options=options)
    return driver


def get_cookies_string(driver):
    """获取浏览器的cookie并转换为curl格式"""
    cookies = driver.get_cookies()
    cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])
    return cookie_str


def get_cookies_dict(driver):
    """获取浏览器的cookie并转换为字典格式"""
    return {c['name']: c['value'] for c in driver.get_cookies()}


def extract_video_sources(driver, url):
    """提取页面中的视频源地址"""
    video_sources = []

    # 等待页面加载
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "video"))
        )
    except Exception:
        print("等待video标签超时，尝试继续...")

    # 查找所有video标签
    videos = driver.find_elements(By.TAG_NAME, "video")

    for video in videos:
        # 获取src属性
        src = video.get_attribute("src")
        if src:
            video_sources.append(src)
            print(f"找到视频源: {src}")

        # 检查source子标签
        sources = video.find_elements(By.TAG_NAME, "source")
        for source in sources:
            src = source.get_attribute("src")
            if src:
                video_sources.append(src)
                print(f"找到视频源: {src}")

    # 处理相对URL
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

    absolute_sources = []
    for src in video_sources:
        if src.startswith("//"):
            src = parsed_url.scheme + ":" + src
        elif src.startswith("/"):
            src = base_url + src
        elif not src.startswith("http"):
            src = base_url + "/" + src
        absolute_sources.append(src)

    return absolute_sources


def extract_from_iframe(driver, iframe_element, iframe_index, parent_url):
    """从iframe中提取视频源"""
    video_sources = []

    try:
        # 切换到iframe
        driver.switch_to.frame(iframe_element)
        time.sleep(2)  # 等待iframe内容加载

        print(f"  进入 iframe {iframe_index} 内部...")

        # 在iframe内查找video标签
        videos = driver.find_elements(By.TAG_NAME, "video")
        for video in videos:
            src = video.get_attribute("src")
            if src:
                video_sources.append(src)
                print(f"  iframe内找到视频: {src}")

            sources = video.find_elements(By.TAG_NAME, "source")
            for source in sources:
                src = source.get_attribute("src")
                if src:
                    video_sources.append(src)
                    print(f"  iframe内找到视频: {src}")

        # 如果没找到video标签，尝试从HTML中提取
        if not video_sources:
            html = driver.page_source
            # 搜索常见的视频URL模式
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
                        print(f"  iframe HTML中找到: {match}")

    except Exception as e:
        print(f"  处理iframe内部时出错: {e}")
    finally:
        # 切换回主文档
        driver.switch_to.default_content()

    return video_sources


def extract_iframe_sources(driver, url):
    """提取iframe中的视频源"""
    video_sources = []
    embed_iframes = []  # 存储需要深入访问的iframe

    # 获取所有iframe
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    print(f"找到 {len(iframes)} 个iframe")

    for i, iframe in enumerate(iframes):
        try:
            src = iframe.get_attribute("src")
            print(f"检查 iframe {i+1}: {src[:100] if src else '(empty)'}...")

            if not src:
                # 没有src的iframe可能是动态加载的，尝试进入查看
                embed_iframes.append((iframe, i+1))
                continue

            # 检查是否是视频相关的iframe
            if any(keyword in src.lower() for keyword in ['embed', 'video', 'player', 'stream', '.mp4', '.m3u8']):
                embed_iframes.append((iframe, i+1))
                print(f"  -> 标记为视频iframe")
            elif 'ad' not in src.lower() and 'ads' not in src.lower():
                # 非广告iframe也可能包含视频
                embed_iframes.append((iframe, i+1))

        except Exception as e:
            print(f"处理iframe {i+1}时出错: {e}")

    # 深入访问每个可能包含视频的iframe
    for iframe, index in embed_iframes:
        sources = extract_from_iframe(driver, iframe, index, url)
        video_sources.extend(sources)

    return video_sources


def extract_video_urls_from_html(driver):
    """从HTML源码中提取视频URL"""
    video_urls = []

    # 使用JavaScript搜索页面中的视频URL
    potential_urls = driver.execute_script("""
        var urls = [];
        var html = document.documentElement.outerHTML;

        // 搜索mp4链接
        var mp4Regex = /https?:\\/\\/[^"\\s'<>]+\\.mp4[^"\\s'<>]*/gi;
        var mp4Matches = html.match(mp4Regex);
        if (mp4Matches) urls = urls.concat(mp4Matches);

        // 搜索m3u8链接
        var m3u8Regex = /https?:\\/\\/[^"\\s'<>]+\\.m3u8[^"\\s'<>]*/gi;
        var m3u8Matches = html.match(m3u8Regex);
        if (m3u8Matches) urls = urls.concat(m3u8Matches);

        // 搜索webm链接
        var webmRegex = /https?:\\/\\/[^"\\s'<>]+\\.webm[^"\\s'<>]*/gi;
        var webmMatches = html.match(webmRegex);
        if (webmMatches) urls = urls.concat(webmMatches);

        return [...new Set(urls)];
    """)

    if potential_urls:
        video_urls.extend(potential_urls)
        print(f"从HTML中提取到 {len(potential_urls)} 个可能的视频链接")

    return video_urls


def get_filename_from_url(url, title="video"):
    """从URL或标题提取文件名"""
    parsed = urlparse(url)

    # 尝试从URL参数中获取文件名
    query_params = parse_qs(parsed.query)
    if 'file' in query_params:
        file_param = query_params['file'][0]
        filename = os.path.basename(unquote(file_param))
        if filename.endswith('.mp4'):
            return filename

    path = unquote(parsed.path)

    if path and "." in os.path.basename(path) and not path.endswith('.php'):
        filename = os.path.basename(path)
    else:
        # 清理标题作为文件名
        safe_title = re.sub(r'[^\w\s-]', '', title)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)[:50]
        filename = f"{safe_title}.mp4"

    # 清理文件名中的特殊字符
    filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
    return filename.strip()


def download_video(url, cookies, output_dir, filename, user_agent, referer=None):
    """使用curl下载视频"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

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

    print(f"\n正在下载: {filename}")
    print(f"保存到: {output_path}")

    try:
        result = subprocess.run(curl_cmd, check=True)

        # 检查下载的文件是否有效
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size < 1024:  # 小于1KB可能是错误信息
                with open(output_path, 'r', errors='ignore') as f:
                    content = f.read(500)
                    if '403' in content or 'forbidden' in content.lower() or 'error' in content.lower():
                        print(f"下载失败: 服务器返回错误页面")
                        os.remove(output_path)
                        return False
            print(f"下载完成: {output_path} ({file_size} bytes)")
            return True
        return False
    except subprocess.CalledProcessError as e:
        print(f"下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="浏览器视频下载工具")
    parser.add_argument("url", help="要访问的URL地址")
    parser.add_argument("-o", "--output", default="./downloads", help="下载目录 (默认: ./downloads)")
    parser.add_argument("--headless", action="store_true", help="无头模式运行浏览器")
    parser.add_argument("--wait", type=int, default=10, help="页面加载后等待时间(秒)，用于处理动态加载")
    parser.add_argument("--debug", action="store_true", help="调试模式，保存页面源码")

    args = parser.parse_args()

    print(f"正在访问: {args.url}")

    driver = None
    try:
        driver = create_driver(headless=args.headless)
        driver.get(args.url)

        # 等待Cloudflare挑战完成
        print("等待页面加载...")
        time.sleep(5)

        # 检查是否还在Cloudflare挑战页面
        current_url = driver.current_url
        page_title = driver.title
        print(f"当前页面标题: {page_title}")

        if "Just a moment" in page_title or "稍候" in page_title or "challenge" in current_url.lower():
            print("检测到安全验证，等待完成...")
            for i in range(60):
                time.sleep(1)
                page_title = driver.title
                if "Just a moment" not in page_title and "稍候" not in page_title:
                    print(f"验证已完成: {page_title}")
                    break
                print(f"等待验证... {i+1}/60")

        print(f"额外等待 {args.wait} 秒...")
        time.sleep(args.wait)

        # 获取页面标题用于文件命名
        page_title = driver.title.split('|')[0].strip()

        # 获取cookie
        cookies = get_cookies_string(driver)
        print(f"\n获取到 {len(cookies)} 字节的Cookie")

        # 获取User-Agent
        user_agent = driver.execute_script("return navigator.userAgent")

        # 调试模式保存页面
        if args.debug:
            debug_file = "debug_main_page.html"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print(f"主页面源码已保存到: {debug_file}")

        # 提取视频源
        print("\n正在解析video标签...")
        video_sources = extract_video_sources(driver, args.url)

        # 提取iframe中的视频
        if not video_sources:
            print("\n检查iframe...")
            iframe_sources = extract_iframe_sources(driver, args.url)
            video_sources.extend(iframe_sources)

        # 从HTML中搜索视频链接
        if not video_sources:
            print("\n从页面源码搜索视频链接...")
            html_sources = extract_video_urls_from_html(driver)
            video_sources.extend(html_sources)

        # 去重
        video_sources = list(set(video_sources))

        if not video_sources:
            print("\n未找到视频源")
            print("提示: 该网站可能使用加密视频流或需要手动操作")
            print("尝试用 --debug 参数查看页面内容")
            return

        print(f"\n共找到 {len(video_sources)} 个视频源")
        print("-" * 50)

        # 下载所有视频
        for i, src in enumerate(video_sources, 1):
            filename = get_filename_from_url(src, page_title)
            download_video(src, cookies, args.output, filename, user_agent, args.url)

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if driver:
            driver.quit()
            print("\n浏览器已关闭")


if __name__ == "__main__":
    main()
