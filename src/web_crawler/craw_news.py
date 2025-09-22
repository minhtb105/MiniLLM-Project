import time 
import re
import os
import time
import json
import logging
import requests
from bs4 import BeautifulSoup, Tag
from pathlib import Path
from urllib.parse import urljoin


headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36",
    'Accept': "text/html,application/xhtml+xml,application/xml;",
}

def get_article_urls(list_url: str, base="https://congdankhuyenhoc.vn", limit=20):
    global headers
    try:
        resp = requests.get(list_url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to fetch list page {list_url}: {e}")
        return []

    soup = BeautifulSoup(resp.content, "html.parser")
    urls = []

    for a in soup.select("div.box-category-content a"):
        href = a.get("href")
        if not href:
            continue
        href = urljoin(base, href)
        urls.append(href)

    seen, filtered = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            filtered.append(u)
        if len(filtered) >= limit:
            break

    logging.info(f"Collected {len(filtered)} article URLs from {list_url}")
    
    return filtered

def craw_news(url: str):
    global headers    
    response = requests.get(
        url,
        headers=headers,
        timeout=30)

    soup = BeautifulSoup(response.content, 'html.parser')
    
    for junk in soup.select(".detail_comment, .footer__top, .modal, .box-comment, .detail_related"):
        junk.decompose()
    
    title = soup.find("h1", class_="detail-title").get_text()
    print(title)

    data = {"title": title, "sections": []}

    main_content = soup.find("div", class_="detail__cmain")
    if not main_content:
        return data

    current_section = None
    for tag in main_content.select("h2, p"):
        if tag.name == "h2":
            header_text = tag.get_text(" ", strip=True)
            current_section = {"header": header_text, "content": ""}
            data["sections"].append(current_section)
        elif tag.name == "p" and current_section:
            blacklist = {"title", "author", "source", "footer", "comment"}
            if tag.get("class") and any(cls in blacklist for cls in tag.get("class")):
                continue
            
            text = tag.get_text(" ", strip=True)
            if text:
                current_section["content"] += text + " "

    return data

if __name__ == "__main__":
    category_urls = [
        "https://congdankhuyenhoc.vn/theo-dong-su-kien.htm",
        "https://congdankhuyenhoc.vn/khuyen-hoc.htm",
        "https://congdankhuyenhoc.vn/cong-dan-hoc-tap.htm",
        "https://congdankhuyenhoc.vn/giao-duc-phap-luat.htm",
        "https://congdankhuyenhoc.vn/kinh-te-xa-hoi.htm",
        "https://congdankhuyenhoc.vn/khoa-hoc-cong-nghe.htm",
        "https://congdankhuyenhoc.vn/goc-nhin-cong-dan.htm",
    ]
    
    all_data = []
    for cat in category_urls:
        article_urls = get_article_urls(cat, limit=10)
        for url in article_urls:
            try:
                data = craw_news(url)
                if data and data.get("sections"):
                    all_data.append(data)
                time.sleep(1) 
            except Exception as e:
                logging.error(f"Error crawling {url}: {e}")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "news")
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "news_data.json"), "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
