import json
import os
import threading
from typing import Dict, List, Optional, Tuple


class RAG_Database:
    def __init__(self, rag_db_path: str = "rag_db.json") -> None:
        """从 JSON 文件加载 RAG 数据库。

        JSON 文件格式：
            {
                "http://example.com/doc1": "文档1的文本内容...",
                "http://example.com/doc2": "文档2的文本内容..."
            }

        加载顺序即 url 的索引顺序，必须与子进程 LMCacheLookupServer
        读取同一个 rag_db.json 时的顺序一致（引导匹配靠索引反查）。

        :param rag_db_path: JSON 文件路径，默认 rag_db.json
        """
        self.rag_db_path = rag_db_path
        self.url_to_text: Dict[str, str] = {}

        if os.path.exists(rag_db_path):
            with open(rag_db_path, "r", encoding="utf-8") as f:
                self.url_to_text = json.load(f)
        else:
            print(f"[RAG_Database] Warning: '{rag_db_path}' not found, starting empty.")

    def get_text(self, url: str) -> Optional[str]:
        """根据 url 查询对应的文本内容。

        :param url: 要查询的文档 url
        :return: 对应的文本内容；url 不存在时返回 None
        """
        return self.url_to_text.get(url, None)

    def get_urls(self) -> List[str]:
        """按 JSON 加载顺序返回全部 url（供索引编码/反查使用）。"""
        return list(self.url_to_text.keys())

    def __contains__(self, url: str) -> bool:
        return url in self.url_to_text

    def __len__(self) -> int:
        return len(self.url_to_text)


class DocCacheManager:
    def __init__(self, rag_database):
        self.rag_database = rag_database
        self.request_list = []
        self.request_cnt = 0
        self.url_list = list(rag_database.url_to_text.keys())

    def generate_request_id(self, urls, prefix="rag_req") -> str:
        idx_list = [self.url_list.index(u) for u in urls]
        request_id = f"{prefix}_{self.request_cnt}|{','.join(map(str, idx_list))}"
        self.request_cnt += 1
        self.request_list.append(request_id)
        return request_id
