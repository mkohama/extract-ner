"""
テキスト処理ユーティリティ

エンコーディング判定、Unicode正規化など、テキスト処理に関する共通機能を提供。
"""

import unicodedata
from pathlib import Path
from typing import Tuple

from src.logger import log


def normalize_nfkc(text: str) -> str:
    """テキストをNFKC正規化する

    NFKC (Normalization Form Compatibility Composition) 正規化により、
    全角英数字を半角に変換するなど、互換性のある文字を統一する。

    ドキュメント読み込み時と検索クエリ処理時の両方で使用し、
    インデックスとクエリの一貫性を保つ。

    Args:
        text: 正規化前のテキスト

    Returns:
        NFKC正規化されたテキスト

    Example:
        >>> normalize_nfkc("ＳＭ２５")
        "SM25"
        >>> normalize_nfkc("①②③")
        "123"
    """
    return unicodedata.normalize("NFKC", text)


def detect_encoding_and_read(
    file_path: str | Path, encoding: str | None = None
) -> Tuple[str, str]:
    """
    ファイルを読み込み、テキスト内容と使用されたエンコーディングを返す。

    Args:
        file_path: ファイルパス
        encoding: 明示的に試行したいエンコーディング (オプション)

    Returns:
        (text_content, encoding_name)

    Raises:
        RuntimeError: どのエンコーディングでも読み込めなかった場合
        FileNotFoundError: ファイルが存在しない場合
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 優先順位: 指定エンコーディング -> utf-8 -> cp932 (Shift_JIS) -> euc-jp -> utf-16
    encodings_to_try = []
    if encoding:
        encodings_to_try.append(encoding)

    # 一般的な日本語環境のエンコーディング
    # chardet等は遅いため、まずはこれらを高速に試す
    encodings_to_try.extend(["utf-8", "cp932", "euc-jp", "utf-16"])

    # 試行
    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as f:
                text = f.read()
            return text, enc
        except UnicodeDecodeError:
            continue
        except Exception as e:
            log(f"Warning: Failed to read {file_path} with {enc}: {e}")
            continue

    # 全て失敗した場合は、chardetによる自動検出を試みる
    log(f"Fast encoding detection failed for {file_path}. Falling back to chardet...")

    raw_content = file_path.read_bytes()

    try:
        import chardet

        detected = chardet.detect(raw_content)

        if detected and detected.get("encoding"):
            detected_enc = detected["encoding"]
            confidence = detected.get("confidence", 0)

            # 信頼度が低い場合は警告
            if confidence < 0.5:
                log(
                    f"Warning: Low confidence ({confidence}) for detected encoding {detected_enc} on {file_path}"
                )

            try:
                text = raw_content.decode(detected_enc)
                return text, detected_enc
            except UnicodeDecodeError:
                pass  # 検出されたエンコーディングでも失敗

    except ImportError:
        log("chardet module not found. Skipping auto-detection.")
    except Exception as e:
        log(f"Error during chardet auto-detection: {e}")

    # 最終手段: latin-1 (必ず読み込めるが文字化けの可能性大)
    # または errors='replace' で無理やり読む
    log(f"All detection methods failed for {file_path}. Using 'utf-8' with replace.")
    return raw_content.decode("utf-8", errors="replace"), "utf-8-replace"
