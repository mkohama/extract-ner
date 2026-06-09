"""マスキング検出（UI 非依存）。

公開 API:
    MaskingEngine : マスキング検出エンジン（候補生成→確信度→ルーティング→マスク）
    MaskResult / Candidate / MaskEntry : 結果の型
    MaskDictionary : マスク辞書（社名・商標・人名の登録リスト）
"""

from src.masking.dictionary import DictMatch, MaskDictionary, normalize
from src.masking.engine import (
    AUTO_MASK_CONFIDENCE,
    Candidate,
    CandidateGroup,
    MaskAnalysis,
    MaskEntry,
    MaskingEngine,
    MaskResult,
    vote_category,
)

__all__ = [
    "MaskingEngine",
    "MaskAnalysis",
    "MaskResult",
    "Candidate",
    "CandidateGroup",
    "MaskEntry",
    "AUTO_MASK_CONFIDENCE",
    "MaskDictionary",
    "DictMatch",
    "normalize",
    "vote_category",
]
