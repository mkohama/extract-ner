"""LLM 検出のための窓化（J1 本文 → ~6-8k トークンの窓）。

LLM には SudachiPy の 49KB 制約が無いので、NER 用の細かいチャンク（xlsx≈1000 トークン）を
そのまま投げる必要はない。文脈を効かせるため、**`build_body` が作った本文 `text` を
`CHUNK_SEPARATOR` 境界でまとめて ~6-8k トークンの窓に切る**（① の決定）。窓は `text` の連続スパン
なので、窓内 locate の結果に ``window_start`` を足すだけで全文（merge）座標に一致する。

spaCy / pii-masker / Azure に非依存（トークン計測は既存の tiktoken cl100k_base を流用）。
"""

from __future__ import annotations

from src.core.document.splitters.token_utils import tiktoken_len
from src.ner.preprocess import CHUNK_SEPARATOR

# 窓の既定サイズ（① の決定: 容量でなく「信頼できる長さ」で切る。mini の長文 recall 劣化を避ける）。
# これはコミット済みのベースライン。実運用の上書きは app.py 側で env（LLM_WINDOW_MAX_TOKENS /
# LLM_WINDOW_OVERLAP_TOKENS）から行い、値は detector_version の win… に自動反映される（windows.py は純粋に保つ）。
DEFAULT_MAX_TOKENS = 7000
# 窓間 overlap（境界で先行文脈が切れるのを緩和。重複検出は detect_layer の解決で潰れる）。**既定 0＝重なり無し**
# （窓化は CHUNK_SEPARATOR 境界で割るので実体は文字単位に切れない。継ぎ目の先行文脈が要るなら 100〜200 へ）。
DEFAULT_OVERLAP_TOKENS = 0


def _segments(text: str) -> list[tuple[int, int, int]]:
    """``text`` を ``CHUNK_SEPARATOR`` 境界で分け、各セグメントの (start, end, tokens) を返す。

    セグメント間の区切り（``CHUNK_SEPARATOR``）自体はどのセグメントにも含めない（区切りに PII は無い）。
    空白のみのセグメントは除外する（窓に入れても無意味）。
    """
    out: list[tuple[int, int, int]] = []
    pos = 0
    step = len(CHUNK_SEPARATOR)
    for part in text.split(CHUNK_SEPARATOR):
        start, end = pos, pos + len(part)
        if part.strip():
            out.append((start, end, tiktoken_len(part)))
        pos = end + step
    return out


def iter_windows(
    text: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
) -> list[tuple[int, int]]:
    """``text`` を ~``max_tokens`` の窓に切り、各窓の (start, end) 文字オフセットを返す。

    - 切れ目は ``CHUNK_SEPARATOR`` 境界のみ（セグメントの途中では切らない）。
    - 各窓は連続セグメントの貪欲詰め込み（合計トークンが ``max_tokens`` を超えない範囲）。
      1 セグメントだけで超える場合はそれ単独で 1 窓（さらに分割しない＝LLM は大容量を許容）。
    - ``overlap`` > 0 なら、2 窓目以降は直前のセグメントを ~``overlap`` トークン分さかのぼって含める
      （窓境界で文脈が切れるのを緩和。重複検出は :func:`~src.llm.detect_layer.detect_document` 側で解決）。

    返すのは ``text`` の連続スパン。``text[start:end]`` が窓本文（pii-masker へ渡す `document`）。
    """
    segs = _segments(text)
    if not segs:
        return []

    # ① 貪欲詰め込みでセグメント index 範囲 [i0, i1]（inclusive）を作る。
    groups: list[tuple[int, int]] = []
    i0 = 0
    tok = 0
    for i, (_s, _e, t) in enumerate(segs):
        if i == i0:
            tok = t
        elif tok + t <= max_tokens:
            tok += t
        else:
            groups.append((i0, i - 1))
            i0, tok = i, t
    groups.append((i0, len(segs) - 1))

    # ② overlap: 2 窓目以降の開始を直前セグメント側へ ~overlap トークン分さかのぼる。
    windows: list[tuple[int, int]] = []
    for gi, (a, b) in enumerate(groups):
        start_idx = a
        if overlap > 0 and gi > 0:
            acc = 0
            j = a - 1
            while j >= 0 and acc < overlap:
                acc += segs[j][2]
                start_idx = j
                j -= 1
        windows.append((segs[start_idx][0], segs[b][1]))
    return windows


__all__ = ["iter_windows", "DEFAULT_MAX_TOKENS", "DEFAULT_OVERLAP_TOKENS"]
