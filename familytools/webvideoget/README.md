# Browser Video Download

A Python script that opens a browser, visits a URL, extracts video sources from the page, and downloads them using curl.

## Requirements

- Python 3.12+
- Chrome/Chromium browser
- [uv](https://docs.astral.sh/uv/) - Python package manager

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone or download this project
cd browserdownload

# Install dependencies
uv sync
```

## Usage

### Single Video Download

```bash
# Basic usage (recommended with xvfb-run for servers without display)
xvfb-run -a uv run wcostream_get.py "https://example.com/video-page"

# Specify output directory
xvfb-run -a uv run wcostream_get.py "https://example.com/video-page" -o ./my_videos

# Increase wait time for slow-loading pages
xvfb-run -a uv run wcostream_get.py "https://example.com/video-page" --wait 20

# Debug mode (saves page source for inspection)
xvfb-run -a uv run wcostream_get.py "https://example.com/video-page" --debug

# Headless mode (for local machines with display)
uv run wcostream_get.py "https://example.com/video-page" --headless
```

### Batch Download

```bash
# Download multiple episodes
xvfb-run -a uv run scripts/batch_download.py -o ~/emby/media
```

### File Rename Utility

```bash
# Standardize file names to sXXeYY_title.mp4 format
python3 scripts/rename_files.py
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `url` | URL to visit (required) | - |
| `-o, --output` | Download directory | `./downloads` |
| `--headless` | Run browser in headless mode | `False` |
| `--wait` | Wait time after page load (seconds) | `10` |
| `--debug` | Save page source for debugging | `False` |

## File Naming Convention

Downloaded files are automatically named in the format:
```
sXXeYY_episode_title.mp4
```

Example: `s03e15_snow_job.mp4`

## How It Works

1. Opens Chrome browser using `undetected-chromedriver`
2. Waits for Cloudflare/security challenges to complete
3. Extracts cookies and user-agent from the browser
4. Searches for video sources in:
   - `<video>` and `<source>` tags
   - Embedded iframes (enters iframe to find real video URLs)
   - HTML source code regex patterns
5. Downloads videos using `curl` with proper headers

## System Dependencies

For servers without a display, install `xvfb`:

```bash
# Ubuntu/Debian
sudo apt install xvfb

# Fedora/RHEL
sudo dnf install xorg-x11-server-Xvfb
```

## Project Structure

```
browserdownload/
├── wcostream_get.py      # Main video download script
├── pyproject.toml        # Python project config
├── README.md             # This file
└── scripts/
    ├── batch_download.py # Batch download multiple episodes
    └── rename_files.py   # File name standardization utility
```

## Troubleshooting

### Cloudflare Not Bypassed

If Cloudflare still blocks access:
- Try without `--headless` mode using `xvfb-run`
- Increase `--wait` time to allow full page load
- Use `--debug` to inspect the page content

### Video Not Found

- The site may use encrypted video streams (HLS with DRM)
- Try increasing the wait time with `--wait 30`
- Check `debug_main_page.html` with `--debug` flag

### 403 Forbidden on Download

- Some sites require specific referer headers
- The script automatically includes referer, but some sites may need additional headers

## License

MIT
