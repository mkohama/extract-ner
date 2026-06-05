"""extract-ner の統一コマンドラインインタフェース（薄い表示層）。

低レベルな ``uv run main.py ...`` / ``uv run streamlit run app.py`` の代わりに、
1 つのエントリポイント ``extract-ner`` にサブコマンドをぶら下げる。

    uv run extract-ner ui                 # Streamlit UI を起動
    uv run extract-ner ner <file>         # ファイル/テキストを NER → HTML 表示
    uv run extract-ner debug <file>       # トークンの品詞 / NER ラベルを観察
    uv run extract-ner check              # 品質ゲート（ruff + mypy）

実際の抽出は src.ner.NerEngine が担当する。本ファイルは入力取得・引数処理・
コンソール出力・displaCy / Streamlit への受け渡しだけを行う。
"""

from __future__ import annotations

import subprocess
import sys
import webbrowser
from collections import Counter
from pathlib import Path

import click
from spacy import displacy

from src.ner import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    NerEngine,
    TokenInfo,
    build_color_map,
    render_html,
    to_displacy_data,
)
from src.sources import SAMPLE_TEXT, load_chunks_from_file

# プロジェクトルート（app.py や品質ゲート対象の解決に使う）
_ROOT = Path(__file__).resolve().parent.parent


def _ensure_utf8_stdout() -> None:
    """Windows コンソールでの日本語 UTF-8 出力の文字化けを避ける。

    端末が対応していれば stdout を UTF-8 に切り替える（未対応なら
    debug の --out でファイル出力すれば確実に読める）。
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def _load_chunks(file: Path | None, text: str | None) -> list[str]:
    """ファイル or テキスト or サンプルから解析対象チャンク（テキストのリスト）を得る。

    ファイルは kb-mcp と同じ単位でチャンク化する（長文でも SudachiPy 上限で
    落ちず、検索ヒット単位と結果が揃う）。テキスト/サンプルは 1 チャンク扱い。
    """
    if text:
        return [text]
    if file:
        click.echo(f"ファイルをテキスト化・チャンク化中: {file}")
        chunks = load_chunks_from_file(file)
        click.echo(f"チャンク数: {len(chunks)}")
        return chunks
    return [SAMPLE_TEXT]


# --------------------------------------------------------------------------- #
# debug 用ヘルパー（トークンの品詞 / NER ラベルを並べて recall の穴を観察する）
# --------------------------------------------------------------------------- #
def _proper_subtype(tag: str) -> str | None:
    """SudachiPy 品詞から固有名詞のサブタイプを返す（固有名詞でなければ None）。

    例: ``名詞-固有名詞-人名-姓`` → ``人名`` / ``名詞-固有名詞-一般`` → ``一般``。
    """
    parts = tag.split("-")
    if "固有名詞" not in parts:
        return None
    i = parts.index("固有名詞")
    return parts[i + 1] if i + 1 < len(parts) else "一般"


def _is_interesting(info: TokenInfo) -> bool:
    """既定表示の対象か（NER 検出 / 固有名詞 / 語彙外のいずれか）。"""
    return bool(info.ent_type) or _proper_subtype(info.tag) is not None or info.is_oov


def _token_table_lines(infos: list[TokenInfo], *, show_all: bool) -> list[str]:
    """トークン診断テーブルの行を組み立てて返す。"""
    rows = infos if show_all else [i for i in infos if _is_interesting(i)]
    lines = [
        f"{'表層':<14}{'Sudachi品詞':<26}{'UD':<7}"
        f"{'NERラベル':<18}{'IOB':<5}{'OOV':<4}",
        "-" * 74,
    ]
    for info in rows:
        oov = "OOV" if info.is_oov else ""
        lines.append(
            f"{info.text:<14}{info.tag:<26}{info.pos:<7}"
            f"{(info.ent_type or '-'):<18}{info.ent_iob:<5}{oov:<4}"
        )
    if not show_all:
        hidden = len(infos) - len(rows)
        lines.append(f"\n（一般語など {hidden} トークンを非表示。全件は --all-tokens）")
    return lines


def _token_summary_lines(infos: list[TokenInfo]) -> list[str]:
    """recall の穴を測る要約の行を返す（実値を含まず共有可）。"""
    proper = [i for i in infos if _proper_subtype(i.tag) is not None]
    by_sub = Counter(_proper_subtype(i.tag) for i in proper)
    ner_tagged = [i for i in infos if i.ent_type]
    # 固有名詞だが NER が拾えなかった = Sudachi 品詞でのみ救える候補（recall の穴）
    missed = [i for i in proper if not i.ent_type]

    sub = " / ".join(f"{k}:{v}" for k, v in sorted(by_sub.items())) or "なし"
    lines = [
        "\n--- 要約（この数値は実値を含まないので共有可） ---",
        f"トークン総数: {len(infos)}",
        f"固有名詞(名詞-固有名詞-*): {len(proper)}  内訳 [{sub}]",
        f"NER がラベル付与したトークン: {len(ner_tagged)}",
        f"★ 固有名詞だが NER 未検出（Sudachi 品詞でのみ拾える候補）: {len(missed)}",
    ]
    missed_sub = Counter(_proper_subtype(i.tag) for i in missed)
    if missed_sub:
        detail = " / ".join(f"{k}:{v}" for k, v in sorted(missed_sub.items()))
        lines.append(f"   内訳 [{detail}]")
    return lines


# --------------------------------------------------------------------------- #
# コマンド定義
# --------------------------------------------------------------------------- #
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """GiNZA 日本語固有表現抽出（NER）ツール。

    LLM に渡す前の機密情報マスキングを目的に、ファイル/テキスト/kb-mcp の文書を
    テキスト化・チャンク化して NER し、displaCy で表示する。
    """


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.argument("streamlit_args", nargs=-1, type=click.UNPROCESSED)
def ui(streamlit_args: tuple[str, ...]) -> None:
    """Streamlit UI を起動する（`streamlit run app.py` のラッパ）。

    追加引数はそのまま streamlit に渡す。例: `extract-ner ui --server.port 8502`
    """
    app = _ROOT / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *streamlit_args]
    raise SystemExit(subprocess.call(cmd))


@cli.command()
@click.argument(
    "file", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--text", help="ファイルの代わりに解析するテキストを直接指定する。")
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    type=click.Choice(list(AVAILABLE_MODELS)),
    show_default=True,
    help="使用する GiNZA モデル（electra=高精度・低速 / ja_ginza=軽量・高速）。",
)
@click.option(
    "--labels",
    multiple=True,
    metavar="LABEL",
    help="抽出するカテゴリを限定する（複数指定可）。例: --labels Person --labels Company",
)
@click.option(
    "--flatten", is_flag=True, help="Markdown テーブルを平文化してから解析する。"
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("ner.html"),
    show_default=True,
    help="出力する HTML ファイル名。",
)
@click.option(
    "--open", "open_browser", is_flag=True, help="生成した HTML をブラウザで開く。"
)
@click.option("--serve", is_flag=True, help="ブラウザ表示用のサーバーを起動する。")
def ner(
    file: Path | None,
    text: str | None,
    model: str,
    labels: tuple[str, ...],
    flatten: bool,
    output: Path,
    open_browser: bool,
    serve: bool,
) -> None:
    """ファイルやテキストを NER し、displaCy で表示する。

    FILE を省略すると --text、それも無ければサンプルテキストを解析する。
    """
    chunks = _load_chunks(file, text)

    click.echo(f"GiNZA モデル ({model}) を読み込み中 ...")
    engine = NerEngine(model)
    result = engine.extract_chunks(
        chunks, labels=list(labels) or None, flatten_tables=flatten
    )

    # --- コンソール出力 ---
    click.echo(f"\n抽出された固有表現: {len(result.entities)} 件\n")
    click.echo(f"{'テキスト':<16}{'ラベル':<22}{'開始':>5}{'終了':>5}")
    click.echo("-" * 50)
    for ent in result.entities:
        click.echo(f"{ent.text:<16}{ent.label:<22}{ent.start:>5}{ent.end:>5}")

    # --- displaCy 表示 ---
    # 色は（フィルタに関わらず安定させるため）モデルの全ラベルから作る
    colors = build_color_map(engine.available_labels())

    if serve:
        click.echo("\nhttp://localhost:5000 で表示します (Ctrl+C で終了)")
        displacy.serve(
            to_displacy_data(result),
            style="ent",
            manual=True,
            options={"colors": colors},
            auto_select_port=True,
        )
        return

    html = render_html(result, colors, page=True)
    output.write_text(html, encoding="utf-8")
    click.echo(f"\nHTML を書き出しました: {output.resolve()}")

    if open_browser:
        webbrowser.open(output.resolve().as_uri())


@cli.command()
@click.argument(
    "file", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--text", help="ファイルの代わりに解析するテキストを直接指定する。")
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    type=click.Choice(list(AVAILABLE_MODELS)),
    show_default=True,
    help="使用する GiNZA モデル。",
)
@click.option(
    "--both-models",
    is_flag=True,
    help="ja_ginza と ja_ginza_electra を両方流して比較する。",
)
@click.option(
    "--all-tokens",
    is_flag=True,
    help="全トークンを表示する（既定は固有名詞 / NER 検出 / OOV のみ）。",
)
@click.option(
    "--flatten", is_flag=True, help="Markdown テーブルを平文化してから解析する。"
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="結果を UTF-8 テキストにも書き出す（コンソール文字化け対策）。",
)
def debug(
    file: Path | None,
    text: str | None,
    model: str,
    both_models: bool,
    all_tokens: bool,
    flatten: bool,
    out: Path | None,
) -> None:
    """各トークンの SudachiPy 品詞 / GiNZA NER ラベルを並べて観察する。

    NER が逃した固有名詞を、文脈非依存な Sudachi の品詞で拾えるか確認する
    （マスキングの recall の穴を実データで特定するため）。
    """
    chunks = _load_chunks(file, text)
    models = list(AVAILABLE_MODELS) if both_models else [model]

    lines: list[str] = []
    for model_name in models:
        lines += [
            "=" * 74,
            f"=== model: {model_name}（flatten={flatten}） ===",
            "=" * 74,
        ]
        engine = NerEngine(model_name)
        infos = engine.debug_tokens(chunks, flatten_tables=flatten)
        lines += _token_table_lines(infos, show_all=all_tokens)
        lines += _token_summary_lines(infos)
        lines.append("")

    report = "\n".join(lines)
    click.echo(report)
    if out is not None:
        out.write_text(report, encoding="utf-8")
        click.echo(f"\nレポートを書き出しました（UTF-8）: {out.resolve()}")


@cli.command()
def check() -> None:
    """品質ゲート（ruff + mypy）を実行する。"""
    targets = ["src", "main.py", "app.py"]
    click.echo("$ ruff check " + " ".join(targets))
    rc_ruff = subprocess.call(["ruff", "check", *targets], cwd=_ROOT)
    click.echo("\n$ mypy " + " ".join(targets))
    rc_mypy = subprocess.call(["mypy", *targets], cwd=_ROOT)
    raise SystemExit(rc_ruff or rc_mypy)


def main() -> None:
    """エントリポイント。click の解析・ヘルプ出力より前に stdout を UTF-8 化する。"""
    _ensure_utf8_stdout()
    cli()


if __name__ == "__main__":
    main()
