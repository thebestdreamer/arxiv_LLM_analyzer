import os
import time
import logging
import yaml
from typing import List, Dict
from tqdm import tqdm
import arxiv
import requests
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


# --------------------------
# 配置管理模块
# --------------------------
class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path,encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化目录
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    @property
    def arxiv_config(self) -> Dict:
        return self.config["arxiv"]

    @property
    def model_config(self) -> Dict:
        return self.config["model"]

    @property
    def pdf_dir(self) -> str:
        return self.config["paths"]["pdf_storage"]

    @property
    def summary_dir(self) -> str:
        return self.config["paths"]["summaries"]


# --------------------------
# arXiv下载模块
# --------------------------
class ArxivDownloader:
    def __init__(self, config: Dict):
        # 将多个关键词转换为arxiv支持的OR查询格式
        queries = [f'all:"{q}"' for q in config["search_query"]]
        self.search_query = " OR ".join(queries)
        self.max_results = config["max_results"]
        self.update_interval = config["update_interval"]
        self.downloaded_ids = set()
    def _load_downloaded(self) -> None:
        """加载已下载论文ID记录"""
        record_file = os.path.join(config_manager.pdf_dir, ".downloaded")
        if os.path.exists(record_file):
            with open(record_file) as f:
                self.downloaded_ids = set(f.read().splitlines())

    def _save_downloaded(self, paper_id: str) -> None:
        """保存下载记录"""
        record_file = os.path.join(config_manager.pdf_dir, ".downloaded")
        with open(record_file, "a") as f:
            f.write(f"{paper_id}\n")

    def fetch_new_papers(self) -> List[Dict]:
        """获取最新论文（自动过滤已下载）"""
        self._load_downloaded()
        client = arxiv.Client()
        search = arxiv.Search(
            query=self.search_query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        new_papers = []
        for result in client.results(search=search):
            if result.entry_id.split('/')[-1] not in self.downloaded_ids:
                paper_data = {
                    "arxiv_id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "pdf_url": result.pdf_url,
                    "published": result.published.date(),
                    "authors": [a.name for a in result.authors],
                    "abstract": result.summary,
                }
                new_papers.append(paper_data)
        return new_papers

    def download_paper(self, paper: Dict) -> bool:
        """下载单篇论文PDF"""
        try:
            response = requests.get(paper["pdf_url"], timeout=10)
            if response.status_code == 200:
                filename = f"{paper['arxiv_id']}.pdf"
                save_path = os.path.join(config_manager.pdf_dir, filename)

                with open(save_path, "wb") as f:
                    f.write(response.content)

                self._save_downloaded(paper["arxiv_id"])
                return True
        except Exception as e:
            logging.error(f"下载失败 {paper['arxiv_id']}: {str(e)}")
            return False


# --------------------------
# 论文处理模块
# --------------------------
class PaperProcessor:
    def __init__(self, model_cfg: Dict):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.converter = PdfConverter(
                artifact_dict=create_model_dict(),
            )
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.chunk_size = model_cfg.get("chunk_size", 2000)
        self.max_length = model_cfg.get("max_length", 1024)

    def extract_content(self, pdf_path: str) -> Dict:
        """高效提取PDF内容"""
        try:
            text = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text.append(page.filter(
                        lambda obj: obj["object_type"] == "char"
                    ).extract_text(
                        layout=True,
                        x_tolerance=1,
                        y_tolerance=3
                    ))
            return {"full_text": "\n".join(text), "pages": len(text)}
        except Exception as e:
            logging.error(f"PDF解析失败 {pdf_path}: {str(e)}")
            return None

    def marker_PDF_text(self,pdf_path):
        # 第一次运行需要导入huggingface token下载部分文件
        from huggingface_hub import login
        login("your huggingface token")

        try:

            rendered = self.converter(pdf_path)
            text, _, images = text_from_rendered(rendered)
            return {"full_text": text, "pages": 1}

        except Exception as e:
            torch.cuda.empty_cache()
            logging.error(f"PDF解析失败 {pdf_path}: {str(e)}")
            print(f"PDF解析失败: {str(e)}")
            return None

    def generate_summary(self, text: str) -> str:
        """生成结构化解读"""
        # system_prompt = """作为AI研究员，请生成知乎风格的论文解读：
        # - 用中文撰写，语言通俗
        # - 包含3-5个技术亮点
        # - 分析实际应用场景
        # - 说明对LLM研究的贡献
        # - 最后给出对本文创新程度和整体可靠性的评分，最低0分，最高10分
        # - 使用Markdown格式，采用中文回答"""
        system_prompt = \
"""请按照以下结构化框架将用户给到的部分arXiv论文内容转化为中文技术报告。要求：
- 格式要求：使用Markdown语法；二级标题用##标识；技术亮点使用编号列表。采用中文知乎作答风格进行回复。评分部分采用表格形式
- 标题保留英文原题（开头添加#号）
- 核心研究（约150字）：提炼研究背景、对象、方法、样本量,突出比较研究的核心结论
- 技术亮点（分点陈述）：列出3-5个关键技术要素,包含模型选择、系统架构、实验设计,说明数据收集方式与分析方法
- 应用场景（具体领域）：指明可直接应用的学科领域,描述具体使用场景
- 创新与可靠性评估（10分制，给出0-10分的评分）：创新性：方法论/应用场景的新颖度。可靠性：实验设计的严谨性与数据支持强度。各维度需附1句评分依据
再次注意：你必须采用中文进行回复，以面向中文用户, 请勿提供任何参考文献的相关信息
"""

        inputs = self.tokenizer.encode(
            f"<|user|>{text[:4096]}</s><|system|>{system_prompt}</s><|assistant|>",
            return_tensors="pt"
        )

        outputs = self.model.generate(
            inputs.to(self.model.device),
            max_new_tokens = self.max_length,
            min_new_tokens = 64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        return self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )


# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    # 初始化配置
    config_manager = ConfigManager()

    # 配置日志
    logging.basicConfig(
        filename="logs/processor.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 初始化模块
    downloader = ArxivDownloader(config_manager.arxiv_config)
    processor = PaperProcessor(config_manager.model_config)

    while True:
        try:
            # 下载新论文
            new_papers = downloader.fetch_new_papers()
            if new_papers:
                logging.info(f"发现 {len(new_papers)} 篇新论文")
                for paper in tqdm(new_papers, desc="下载中"):
                    if downloader.download_paper(paper):
                        # 处理新论文
                        pdf_path = os.path.join(
                            config_manager.pdf_dir,
                            f"{paper['arxiv_id']}.pdf"
                        )
                        # 目前content仍然存在不同PDF格式下\n兼容性的问题，所以demo采用统一格式的abs进行分析

                        # 载入全文进行分析
                        # content = processor.marker_PDF_text(pdf_path)
                        #content = processor.extract_content(pdf_path)
                        # 载入摘要进行分析
                        content= {"full_text": paper["abstract"], "pages": 1}

                        if content:
                            summary = processor.generate_summary(content["full_text"])
                            output_path = os.path.join(
                                config_manager.summary_dir,
                                f"{paper['arxiv_id']}.md"
                            )
                            with open(output_path, "w", encoding='utf-8') as f:
                                f.write(f"# {paper['title']}\n\n")
                                f.write(summary)
                            logging.info(f"已生成摘要：{paper['arxiv_id']}")

            # 等待下次检查
            time.sleep(config_manager.arxiv_config["update_interval"])

        except KeyboardInterrupt:
            logging.info("用户中断程序")
            break
        except Exception as e:
            logging.error(f"主循环错误: {str(e)}")
            time.sleep(60)  # 错误后等待1分钟