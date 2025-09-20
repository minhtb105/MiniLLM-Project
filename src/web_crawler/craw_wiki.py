import time 
import re
import os
import json
import logging
import requests
from bs4 import BeautifulSoup, Tag
from pathlib import Path


def extract_intro_text(soup: BeautifulSoup) -> str:
    content = soup.find("div", class_="mw-content-ltr mw-parser-output")
    if not content:
        content = soup.find("div", id="mw-content-text")
        
    if not content:
        return ""

    for p in content.find_all("p"):
        text = p.get_text(" ", strip=True)
        if not text:
            continue

        # reject if contains edit prompts or "citation needed"
        lowered = text.lower()
        if any(kw in lowered for kw in ["bài này cần", "vui lòng chỉnh sửa", "không có nguồn", "mời bạn", "đọc thêm", "xem thêm"]):
            continue

        # remove citation numbers like [1], [2], ...
        text = re.sub(r'\[\d+\]', '', text)
        text = text.replace('[sửa | sửa mã nguồn]', '')
        return text  # return the first valid paragraph

    return ""
    

IGNORE_HEADERS = [
    "Nội dung", 
    "Xem thêm",
    "Tham khảo",
    "Liên kết ngoài",
    "Chú thích",
    "Cước chú",
    "Sách",
    "Giáo trình",
    "Tạp chí nghiên cứu",
    "Luận văn",
    "Báo chí",
    "Thư mục",
    "Tài nguyên chung",
    "Tạp chí và Hội thảo",
    "Nhóm nghiên cứu",
    "Phần mềm",
    "Đọc thêm",
    "Nghiên cứu thêm",
]
def crawl_wikipedia(page_title: str):
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36",
        'Accept': "text/html,application/xhtml+xml,application/xml;",
        'Referer': "https://vi.wikipedia.org"
    }

    response = requests.get(
        f'https://vi.wikipedia.org/wiki/{page_title}', 
        headers=headers,
        timeout=30)

    soup = BeautifulSoup(response.content, 'html.parser')

    title = soup.find("span", class_="mw-page-title-main").get_text()

    toc = soup.find("div", id="toc")
    if toc:
        toc.decompose()

    data = {"title": title, "sections": []}
    data["intro"] = extract_intro_text(soup)

    current_section = None
    for tag in soup.select("h2, h3, p, ul, ol"):
        if tag.name in ["h2", "h3"]:
            header_text = tag.get_text(" ", strip=True)
            current_section = {"header": header_text, "content": ""}
            data["sections"].append(current_section)
        elif current_section is not None:
            # paragraphs
            if tag.name == "p":
                text = tag.get_text(" ", strip=True)
                if text:
                    current_section["content"] += text + " "

            # lists
            elif tag.name in ["ul", "ol"]:
                items = [li.get_text(" ", strip=True) for li in tag.find_all("li")]
                if items:
                    list_text = "; ".join(items)
                    current_section["content"] += list_text + " "
                    
    sections = [
        sec for sec in data["sections"]
        if (sec["content"].strip() and sec["header"] not in IGNORE_HEADERS)
    ]
    data["sections"] = sections

    return data

PAGE_TITLES = [
    "Trí_tuệ_nhân_tạo",
    "Máy_học",
    "Học_sâu",
    "Xử_lý_ngôn_ngữ_tự_nhiên",
    "Thị_giác_máy_tính",
    "Khoa_học_dữ_liệu",
    "Robot",
    "Mạng_nơ-ron_nhân_tạo",
    "Học_tăng_cường",
    "ChatGPT"
]
results = []

for title in PAGE_TITLES:
    try:
        data = crawl_wikipedia(title)
        results.append(data)
        logging.info(f"Crawled: {title}")
        time.sleep(1)  
    except Exception as e:
        logging.error(f"Error crawling {title}: {e}")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)
output_file = os.path.join(DATA_DIR, "wiki_demo.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

logging.info(f"Done! Saved {len(results)} pages to {output_file}")
