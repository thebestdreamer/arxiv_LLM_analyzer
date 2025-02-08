import os
import time
import requests
import feedparser
from tqdm import tqdm
import arxiv

# 配置参数
ARXIV_URL = "http://export.arxiv.org/api/query?"
SEARCH_QUERY = "abs:large+language+models"  # 搜索关键词
MAX_RESULTS = 10  # 每次获取的最大文章数量
SAVE_PATH = "./arxiv_papers"  # 文章保存路径
UPDATE_INTERVAL = 86400  # 更新间隔时间（秒），24小时

#https://arxiv.org/search/?query=large+language+models&searchtype=all&source=header&start=100
def fetch_latest_papers():
    # 获取arXiv API响应
    # response = requests.get(query_url)
    response = arxiv.Search(
        query="LLMs on reinforcement learning",
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order = arxiv.SortOrder.Descending
    )
    for result in response.results():
        print(result.entry_id, '->', result.title)

    # 解析Atom feed
    # feed = feedparser.parse(response.content)
    papers = []
    for entry in response.results():
        paper = {
            "title": entry.title,
            "summary": entry.summary,
            "authors": [author.name for author in entry.authors],
            "published": entry.published,
            "pdf_url": entry.pdf_url,
            "id": entry.entry_id.split("/abs/")[-1]
        }
        papers.append(paper)
    return papers


def download_paper(pdf_url, save_path):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download: {pdf_url}")


def save_papers(papers, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for paper in tqdm(papers, desc="Downloading papers"):
        pdf_filename = f"{paper['id']}.pdf"
        pdf_path = os.path.join(save_dir, pdf_filename)
        if not os.path.exists(pdf_path):
            download_paper(paper["pdf_url"], pdf_path)
            time.sleep(1)  # 避免请求过于频繁


def main():
    while True:
        print("Fetching latest papers from arXiv...")
        papers = fetch_latest_papers()
        if papers:
            save_papers(papers, SAVE_PATH)
            print(f"Successfully saved {len(papers)} papers to {SAVE_PATH}")
        else:
            print("No new papers found.")

        print(f"Waiting for next update in {UPDATE_INTERVAL // 3600} hours...")
        time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    main()