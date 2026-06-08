"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng requests + BeautifulSoup nếu Playwright không có browser.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install requests beautifulsoup4
"""

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://vnexpress.net/dien-vien-hai-huu-tin-su-dung-ma-tuy-vi-to-mo-4599355.html",
    "https://vnexpress.net/dem-su-dung-ma-tuy-cuong-loan-cua-ca-si-chau-viet-cuong-3863999.html",
    "https://ngoisao.vnexpress.net/cuoc-song-chi-dan-truoc-khi-bi-dieu-tra-vi-lien-quan-ma-tuy-4814636.html",
    "https://vnexpress.net/227-nguoi-bi-truy-to-trong-vu-4-tiep-vien-hang-khong-xach-ma-tuy-5057648.html",
    "https://vnexpress.net/trum-ma-tuy-dung-sau-duong-day-lien-quan-4-tiep-vien-hang-khong-5059153.html",
]


def sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "-_.").rstrip("-_.")


def extract_article_text(soup: BeautifulSoup) -> str:
    article = soup.find("article")
    if article:
        blocks = article.find_all(["h1", "h2", "h3", "p"])
    else:
        blocks = soup.find_all(["h1", "h2", "h3", "p"])

    lines = []
    for block in blocks:
        text = block.get_text(separator=" ", strip=True)
        if text:
            lines.append(text)

    return "\n\n".join(lines).strip()


def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo đơn giản với requests and BeautifulSoup.
    """
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    content = extract_article_text(soup)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    }


def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = crawl_article(url)

        parsed = urlparse(url)
        slug = sanitize_filename(Path(parsed.path).stem)
        filename = f"{i:02d}_{slug}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
    else:
        crawl_all()
