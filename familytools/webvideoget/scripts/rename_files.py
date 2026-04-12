#!/usr/bin/env python3
"""
文件名标准化脚本
将视频文件重命名为标准格式: sXXeYY_title.mp4
"""

import os
import re
from pathlib import Path

def extract_episode_info(filename):
    """从文件名提取季数、集数和标题"""
    # 移除扩展名
    name = Path(filename).stem

    # 格式1: s03e15_snow_job.mp4 (已标准化)
    match = re.match(r'^s(\d+)e(\d+)_(.+)$', name, re.IGNORECASE)
    if match:
        season = int(match.group(1))
        episode = int(match.group(2))
        title = match.group(3)
        return season, episode, title, True

    # 格式2: geronimo_stilton_s03e03.mp4
    match = re.match(r'^geronimo_stilton_s(\d+)e(\d+)$', name, re.IGNORECASE)
    if match:
        season = int(match.group(1))
        episode = int(match.group(2))
        title = "unknown_title"  # 需要手动确认
        return season, episode, title, False

    # 格式3: Geronimo Stilton Season 2 Episode 10 The Cave Mouse Watch cartoo
    match = re.match(r'^Geronimo Stilton Season (\d+) Episode (\d+) (.+?) Watch cart', name)
    if match:
        season = int(match.group(1))
        episode = int(match.group(2))
        title = match.group(3).strip()
        return season, episode, title, False

    return None, None, None, False

def standardize_title(title):
    """标准化标题: 转小写，替换空格和特殊字符为下划线"""
    # 转小写
    title = title.lower()
    # 替换空格和特殊字符为下划线
    title = re.sub(r"[^\w\s-]", "", title)
    title = re.sub(r"[-\s]+", "_", title)
    # 移除前后下划线
    title = title.strip("_")
    return title

def generate_new_name(season, episode, title):
    """生成标准文件名"""
    safe_title = standardize_title(title)
    return f"s{season:02d}e{episode:02d}_{safe_title}.mp4"

def main():
    media_dir = Path.home() / "emby" / "media"

    if not media_dir.exists():
        print(f"目录不存在: {media_dir}")
        return

    # 获取所有mp4文件
    mp4_files = sorted(media_dir.glob("*.mp4"))

    print(f"找到 {len(mp4_files)} 个mp4文件")
    print("=" * 70)

    rename_plan = []

    for filepath in mp4_files:
        filename = filepath.name
        season, episode, title, is_standard = extract_episode_info(filename)

        if season is None:
            print(f"[跳过] 无法解析: {filename}")
            continue

        if is_standard:
            print(f"[已标准化] {filename}")
            continue

        new_name = generate_new_name(season, episode, title)
        new_path = media_dir / new_name

        rename_plan.append((filepath, new_path, filename, new_name))
        print(f"[待重命名] {filename}")
        print(f"      -> {new_name}")

    print("=" * 70)
    print(f"\n共 {len(rename_plan)} 个文件需要重命名")

    if not rename_plan:
        print("所有文件已标准化!")
        return

    print("\n执行重命名...")
    for old_path, new_path, old_name, new_name in rename_plan:
        try:
            old_path.rename(new_path)
            print(f"[OK] {old_name} -> {new_name}")
        except Exception as e:
            print(f"[错误] {old_name}: {e}")

    print("\n完成!")

if __name__ == "__main__":
    main()
