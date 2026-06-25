import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .intent import HashingTextEmbedder, cosine
'''模仿人类的三层记忆结构:感官记忆\短期记忆\长期记忆'''

@dataclass
class MemoryRecord:
    text: str
    kind: str
    idx: Optional[int] = None
    turn_id: int = 0
    metadata: Dict = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class BaseMemory:
    def __init__(self, kind: str, embedder=None):
        self.kind = kind
        self.embedder = embedder or HashingTextEmbedder()
        self.records: List[MemoryRecord] = []

    def add(self, text: str, idx=None, turn_id: int = 0, metadata=None):
        record = MemoryRecord(
            text=text,
            kind=self.kind,
            idx=idx,
            turn_id=turn_id,
            metadata=metadata or {},
            embedding=self.embedder.encode(text),
        )
        self.records.append(record)
        return record

    def retrieve(self, query: str, topk: int = 3) -> List[MemoryRecord]:
        query_embedding = self.embedder.encode(query)
        ranked = sorted(
            self.records,
            key=lambda record: cosine(query_embedding, record.embedding),
            reverse=True,
        )
        return ranked[:topk]

    def to_list(self):
        return [record.to_dict() for record in self.records]

    def load_list(self, rows):
        self.records = [MemoryRecord(**row) for row in rows]


class SensoryMemory(BaseMemory):
    '''存储原始输入，不加工'''
    def __init__(self, embedder=None):
        super().__init__("sensory", embedder)


class ShortTermMemory(BaseMemory):
    '''存储最近几轮对话的压缩表示'''
    def __init__(self, embedder=None):
        super().__init__("short_term", embedder)


class LongTermMemory(BaseMemory):
    '''存储跨用户的长期画像'''
    def __init__(self, embedder=None):
        super().__init__("long_term", embedder)


class IntentMemoryManager:
    '''统一接口，管理三层记忆'''
    def __init__(self, embedder=None):
        self.embedder = embedder or HashingTextEmbedder()
        self.sensory = SensoryMemory(self.embedder)
        self.short_term = ShortTermMemory(self.embedder)
        self.long_term = LongTermMemory(self.embedder)

    def add_memory(self, text: str, kind: str = "short_term", idx=None, turn_id: int = 0, metadata=None):
        return self._memory(kind).add(text, idx=idx, turn_id=turn_id, metadata=metadata)

    def retrieve_memory(self, query: str, topk: int = 3, kinds=None) -> List[MemoryRecord]:
        kinds = kinds or ["sensory", "short_term", "long_term"]
        records = []
        for kind in kinds:
            records.extend(self._memory(kind).retrieve(query, topk=topk))
        query_embedding = self.embedder.encode(query)
        records = sorted(records, key=lambda record: cosine(query_embedding, record.embedding), reverse=True)
        return records[:topk]

    def summarize_memory(self, query: str = "", topk: int = 3) -> str:
        records = self.retrieve_memory(query, topk=topk) if query else self.short_term.records[-topk:]
        if not records:
            return "No retrieved intent memory."
        return "\n".join(f"- [{record.kind}] {record.text}" for record in records)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "sensory": self.sensory.to_list(),
            "short_term": self.short_term.to_list(),
            "long_term": self.long_term.to_list(),
        }
        with open(path, "w", encoding="utf-8") as fout:
            json.dump(payload, fout, indent=2, ensure_ascii=False)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as fin:
            payload = json.load(fin)
        self.sensory.load_list(payload.get("sensory", []))
        self.short_term.load_list(payload.get("short_term", []))
        self.long_term.load_list(payload.get("long_term", []))

    def _memory(self, kind: str) -> BaseMemory:
        if kind == "sensory":
            return self.sensory
        if kind == "long_term":
            return self.long_term
        return self.short_term
