"""data-redactor の統一コマンドラインインタフェース（薄い表示層）。

低レベルな ``uv run main.py ...`` / ``uv run streamlit run app.py`` の代わりに、
1 つのエントリポイント ``data-redactor`` にサブコマンドをぶら下げる。

    uv run data-redactor ui                 # Streamlit UI を起動
    uv run data-redactor ner <file>         # ファイル/テキストを NER → HTML 表示
    uv run data-redactor debug <file>       # トークンの品詞 / NER ラベルを観察
    uv run data-redactor check              # 品質ゲート（ruff + mypy）

実際の抽出は src.ner.NerEngine が担当する。本ファイルは入力取得・引数処理・
コンソール出力・displaCy / Streamlit への受け渡しだけを行う。
"""

from __future__ import annotations

import re
import subprocess
import sys
import webbrowser
from collections import Counter
from pathlib import Path

import click
from spacy import displacy

from src.masking import (
    CandidateGroup,
    MaskAnalysis,
    MaskDictionary,
    MaskingEngine,
    tally_votes,
)
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
# マスク辞書の既定パス
_DEFAULT_DICT = _ROOT / "data" / "mask_dict.yaml"
# pii-masker（submodule）追従に使う場所。detector_version は app.py に焼き込まれている。
_SUBMODULE = _ROOT / "external" / "pii-masker"
_APP_PY = _ROOT / "app.py"
_DETECTOR_HASH_RE = re.compile(r"pii-masker@([0-9a-fA-F]+)")


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

    追加引数はそのまま streamlit に渡す。例: `data-redactor ui --server.port 8502`
    """
    app = _ROOT / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *streamlit_args]
    try:
        code = subprocess.call(cmd)
    except KeyboardInterrupt:
        code = 0  # Ctrl+C はサーバ停止の通常手順。正常終了として扱う（"Aborted!" を出さない）
    raise SystemExit(code)


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


def _audit_lines(groups: list[CandidateGroup], *, with_surface: bool) -> list[str]:
    """各実体について「解決結果・票の分布・全票」を 1 行にする（確信度づけの監査用）。

    票の分布 (by-cat) は engine.tally_votes と同じ集計（カテゴリ別チャネル数・折衷ルール適用）を使う
    ＝確信度づけの実ロジックを監査できる。解決カテゴリが最多票でない／票が複数カテゴリに割れている
    実体には ``split⚠`` を付ける。with_surface=True のときだけ表層を末尾に付ける（機密。共有禁止）。
    """
    lines: list[str] = []
    for i, g in enumerate(groups, start=1):
        by_cat = tally_votes(g.votes)
        cat_str = "{" + ", ".join(f"{c}:{n}" for c, n in by_cat.most_common()) + "}"
        votes_str = " ".join(f"{ch}={label}" for ch, label in g.votes)
        split = len(by_cat) > 1 or (
            by_cat and by_cat.most_common(1)[0][0] != g.category
        )
        flag = "split⚠" if split else "      "
        line = (
            f"#{i:03d}  resolved={g.category}/{g.confidence}  {flag}"
            f"  by-cat={cat_str}  count={g.count}  votes=({votes_str})"
        )
        if with_surface:
            line += f"  surface={g.surface!r}"
        lines.append(line)
    return lines


def _embedded_leak_lines(
    dictionary: MaskDictionary, analysis: MaskAnalysis, *, with_surface: bool
) -> list[str]:
    """辞書語を部分文字列として内包する未一致トークン（SmashMark/SonyXXX 型）を列挙する。

    トークン単位辞書では取りこぼす＝**真の漏れ候補**。Stage 3（部分一致）の要否を実データで
    見極めるための監査。表層・canonical は機密なので、redacted ではカテゴリ別件数のみ出す。

    対象は **商標・社名のみ**（部分一致したいのはこの distinctive 語）。人名・地名は意図的に
    token 単位のままなので除外する（`小浜市⊃小浜` のような `〇〇市/〇〇区` ノイズを出さない）。
    """
    target = {"商標", "社名"}
    hits = [
        h
        for h in dictionary.embedded([t.surface for t in analysis.tokens])
        if h[2] in target
    ]
    if not hits:
        return ["  （なし。対象=商標/社名。人名・地名は token 単位のため対象外）"]
    if not with_surface:
        by_cat = Counter(cat for _, _, cat in hits)
        return [
            "  ※表層・辞書語は伏せています（--audit-surface で表示）。カテゴリ別件数:",
            *(f"    {cat}: {n} 箇所" for cat, n in by_cat.most_common()),
        ]
    agg = Counter(
        (analysis.tokens[i].surface, canonical, cat) for i, canonical, cat in hits
    )
    return [
        f"  {tok!r} ⊃ {canonical!r} [{cat}]  ×{n}"
        for (tok, canonical, cat), n in agg.most_common()
    ]


@cli.command()
@click.argument(
    "file", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--text", help="ファイルの代わりにマスクするテキストを直接指定する。")
@click.option(
    "--dict",
    "dict_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=f"マスク辞書(YAML)。既定: {_DEFAULT_DICT}（あれば自動で読む）。",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    type=click.Choice(list(AVAILABLE_MODELS)),
    help="使うモデル（複数指定可。既定は両モデル併用）。",
)
@click.option(
    "--flatten/--no-flatten",
    default=True,
    show_default=True,
    help="テーブルを平文化して検出（検出専用。マスクは | 入り原文に当てる＝既定 ON・表が無ければ無影響）。",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="マスク済みテキストを UTF-8 で書き出す。",
)
@click.option(
    "--audit",
    is_flag=True,
    help="全候補（確定/強も含む）の票の分布と解決結果を出す（表層なし＝共有OK）。",
)
@click.option(
    "--audit-surface",
    is_flag=True,
    help="監査出力に表層も付ける（機密。ローカル照合用。共有禁止）。--audit を含意。",
)
@click.option(
    "--audit-out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="監査出力を UTF-8 で書き出す（--audit-surface 併用時は表層入り＝gitignore 下へ）。",
)
def mask(
    file: Path | None,
    text: str | None,
    dict_path: Path | None,
    models: tuple[str, ...],
    flatten: bool,
    out: Path | None,
    audit: bool,
    audit_surface: bool,
    audit_out: Path | None,
) -> None:
    """機密情報を検出してマスク（伏せ字）する。確定/強は自動マスク、弱はレビュー候補。"""
    chunks = _load_chunks(file, text)

    path = dict_path or (_DEFAULT_DICT if _DEFAULT_DICT.exists() else None)
    dictionary = (
        MaskDictionary.load(path) if path is not None else MaskDictionary.empty()
    )
    click.echo(f"マスク辞書: {path or '（なし）'}（{len(dictionary)} 表層）")

    used = list(models) or list(AVAILABLE_MODELS)
    click.echo(f"モデル: {', '.join(used)} を読み込み中 ...")
    engine = MaskingEngine(dictionary=dictionary, models=used)
    analysis = engine.analyze(chunks, flatten_tables=flatten)
    # 実体（表層）ごとに集約。確定/強を自動マスク（全出現に展開）。
    groups = engine.group_candidates(analysis.candidates)
    selected = [m for g in groups if g.confidence in ("確定", "強") for m in g.members]
    result = engine.apply(analysis, selected)

    click.echo("\n===== マスク済みテキスト（自動マスク＝確定/強） =====")
    click.echo(result.masked_text)

    click.echo(f"\n===== 対応表（{len(result.mapping)} 種・自動マスク） =====")
    for entry in result.mapping:
        click.echo(
            f"{entry.placeholder}\t[{entry.category}]\t{' / '.join(entry.surfaces)}"
        )

    review = [g for g in groups if g.confidence in ("中", "弱")]
    click.echo(f"\n===== レビュー候補（中/弱・{len(review)} 実体） =====")
    for g in review:
        detail = " ".join(f"{ch}={label}" for ch, label in g.votes)
        click.echo(
            f"・{g.surface}\t[{g.category}/{g.confidence}]\t出現{g.count}\t({detail})"
        )

    if out is not None:
        out.write_text(result.masked_text, encoding="utf-8")
        click.echo(f"\nマスク済みテキストを書き出しました（UTF-8）: {out.resolve()}")

    if audit or audit_surface or audit_out is not None:
        lines = _audit_lines(groups, with_surface=audit_surface)
        leak_lines = _embedded_leak_lines(
            dictionary, analysis, with_surface=audit_surface
        )
        report = (
            "監査（全候補・票の分布）\n"
            + "\n".join(lines)
            + "\n\n部分文字列の取りこぼし候補（辞書語を内包する未一致トークン）\n"
            + "\n".join(leak_lines)
        )
        if audit_surface:
            click.echo(
                "\n⚠ 表層を含む監査出力です（機密）。チャット等に貼らないでください。"
            )
        click.echo("\n===== 監査（全候補・票の分布） =====")
        for line in lines:
            click.echo(line)
        click.echo(
            "\n===== 部分文字列の取りこぼし候補"
            "（辞書語を内包する未一致トークン＝SmashMark/SonyXXX 型） ====="
        )
        for line in leak_lines:
            click.echo(line)
        if audit_out is not None:
            audit_out.write_text(report + "\n", encoding="utf-8")
            click.echo(f"\n監査出力を書き出しました（UTF-8）: {audit_out.resolve()}")


@cli.command()
def check() -> None:
    """品質ゲート（ruff + mypy）を実行する。"""
    targets = ["src", "main.py", "app.py"]
    click.echo("$ ruff check " + " ".join(targets))
    rc_ruff = subprocess.call(["ruff", "check", *targets], cwd=_ROOT)
    click.echo("\n$ mypy " + " ".join(targets))
    rc_mypy = subprocess.call(["mypy", *targets], cwd=_ROOT)
    raise SystemExit(rc_ruff or rc_mypy)


def _git_out(args: list[str], cwd: Path) -> str:
    """git をサブプロセスで実行し標準出力（strip 済み）を返す。失敗時は CalledProcessError。"""
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _pii_masker_ene_types(sub: Path) -> set[str] | None:
    """pii-masker が宣言する ENE type 名の集合を返す（取れなければ None）。

    出所は ``schema.py`` の ``PII_TYPES``（detector のプロンプトと GT で揃える語彙の正本。
    全 target を通じて LLM が出しうる type の宣言一覧で、個別 target 専用の ``Trademark`` も含む）。
    かつては汎用検出プロンプト本文を正規表現で走査していたが、(1) プロンプトが ``targets.py`` へ移り、
    (2) 汎用プロンプト(pii)には ``Trademark`` が無いため「マップにあるがプロンプトに無い」と毎回
    誤警告が出た。そこで宣言一覧（PII_TYPES）を直接読む方式に変えた（import せず list リテラルを
    走査＝依存・副作用なし）。``PII_TYPES = [...]`` の書式が大きく変わると None を返す（手動確認を促す）。
    """
    schema = sub / "src" / "pii_masker" / "schema.py"
    if not schema.exists():
        return None
    block = re.search(
        r"PII_TYPES.*?=\s*\[(.*?)\]", schema.read_text(encoding="utf-8"), re.S
    )
    if not block:
        return None
    return set(re.findall(r'"([A-Z][A-Za-z_]+)"', block.group(1))) or None


@cli.command(name="sync-pii-masker")
@click.argument("ref", required=False)
@click.option(
    "--no-update",
    is_flag=True,
    help="submodule を更新せず、現在の HEAD で検証だけ行う。",
)
@click.option(
    "--skip-tests", is_flag=True, help="ruff/mypy/pytest をスキップする（速い確認用）。"
)
def sync_pii_masker(ref: str | None, no_update: bool, skip_tests: bool) -> None:
    """pii-masker（submodule）を更新し、detector_version の追従と検証をまとめて行う。

    機械的な手順を自動化する：① submodule のポインタ更新（REF 省略時は追跡ブランチの最新／
    REF 指定でそのコミット・タグへ）→ ② 新 HEAD の短縮ハッシュ取得 → ③ app.py の
    _DETECTOR_STATIC の `pii-masker@<hash>` を書き換え（= LLM 検出キャッシュを自動ミスさせる）
    → ④ ENE type ドリフト警告と submodule の変更点表示 → ⑤ ruff/mypy/pytest。

    **コミットはしない**（submodule ポインタと app.py を stage するだけ）。インターフェース契約・
    ENE マップ更新・実機 e2e（az login）は人手で確認してからコミットすること（ENE マップは版バンプ不要）。
    """
    from src.masking.engine import _ENE_TO_CATEGORY

    if not _SUBMODULE.is_dir():
        raise SystemExit(
            f"submodule が見つかりません: {_SUBMODULE}\n"
            "先に `git submodule update --init` を実行してください。"
        )

    app_text = _APP_PY.read_text(encoding="utf-8")
    m = _DETECTOR_HASH_RE.search(app_text)
    old_hash = m.group(1) if m else None
    click.echo(f"現在の detector_version ハッシュ: {old_hash or '（不明）'}")

    # ① submodule 更新
    try:
        if no_update:
            click.echo("--no-update: submodule は更新せず現在の HEAD で検証します。")
        elif ref:
            click.echo(f"submodule を {ref} に更新中 ...")
            subprocess.check_call(["git", "fetch"], cwd=_SUBMODULE)
            subprocess.check_call(["git", "checkout", ref], cwd=_SUBMODULE)
        else:
            click.echo("submodule を追跡ブランチの最新に更新中 ...")
            subprocess.check_call(
                ["git", "submodule", "update", "--remote", "external/pii-masker"],
                cwd=_ROOT,
            )
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"submodule の更新に失敗しました: {e}") from e

    # ② 新ハッシュ
    new_hash = _git_out(["rev-parse", "--short", "HEAD"], cwd=_SUBMODULE)
    click.echo(f"pii-masker HEAD: {new_hash}")

    # ③ detector_version の書き換え（ハッシュ部分のみ。win… は env 由来で自動・type-map は版に含めない）
    if old_hash == new_hash:
        click.echo("detector_version のハッシュは最新です（書き換え不要）。")
    elif old_hash is None:
        click.echo(
            "⚠ app.py に pii-masker@<hash> が見つからず書き換えできませんでした。手動で確認してください。"
        )
    else:
        app_text = _DETECTOR_HASH_RE.sub(f"pii-masker@{new_hash}", app_text, count=1)
        _APP_PY.write_text(app_text, encoding="utf-8")
        click.echo(
            f"app.py の detector_version を pii-masker@{old_hash} → pii-masker@{new_hash} "
            "に書き換えました（LLM キャッシュが自動ミス→再取得になります）。"
        )

    # ④-a submodule の変更点（契約 / プロンプトを目視するための手がかり）
    if old_hash and old_hash != new_hash:
        click.echo("\n===== pii-masker の変更（要目視：契約 / プロンプト） =====")
        try:
            stat = _git_out(["diff", "--stat", f"{old_hash}..HEAD"], cwd=_SUBMODULE)
            click.echo(stat or "（差分なし）")
        except subprocess.CalledProcessError:
            click.echo(f"（{old_hash}..HEAD の差分を取得できませんでした）")
        click.echo(
            "→ targets.py（プロンプト/型一覧＝target 別指示）・detector_llm.py（detect/extract）・"
            "schema.py・locate.py の変更は src/llm のアダプタ契約（detect/locate_all の戻り値）に影響します。"
        )

    # ④-b ENE type ドリフト（schema.py の PII_TYPES vs _ENE_TO_CATEGORY）
    types = _pii_masker_ene_types(_SUBMODULE)
    click.echo("\n===== ENE type ドリフト検査 =====")
    if types is None:
        click.echo(
            "⚠ schema.py の PII_TYPES から type 一覧を抽出できませんでした。schema.py を手動確認してください。"
        )
    else:
        mapped = set(_ENE_TO_CATEGORY)
        unmapped = sorted(types - mapped)
        extra = sorted(mapped - types)
        if unmapped:
            click.echo(
                "⚠ マップに無い ENE type（『その他』に落ちる＝recall 漏れの恐れ）: "
                + ", ".join(unmapped)
                + "\n  → src/masking/engine.py の _ENE_TO_CATEGORY に追加してください"
                "（detector_version のバンプは不要＝マップは解析時に毎回当たる後段変換。次の解析で反映）。"
            )
        else:
            click.echo("マップ漏れの新 type はありません。")
        if extra:
            click.echo(
                "・マップにあるが pii-masker の PII_TYPES に無い type（先取り/廃止の可能性。多くは無害）: "
                + ", ".join(extra)
            )

    # ⑤ stage（コミットはしない）
    subprocess.call(["git", "add", "external/pii-masker", "app.py"], cwd=_ROOT)
    click.echo(
        "\nstage しました（external/pii-masker, app.py）。コミットは目視確認後に手動で。"
    )

    # ⑥ 検証
    if skip_tests:
        click.echo("--skip-tests: ruff/mypy/pytest はスキップしました。")
    else:
        click.echo("\n===== 検証（ruff / mypy / pytest） =====")
        targets = ["src", "main.py", "app.py"]
        rc_ruff = subprocess.call(["ruff", "check", *targets], cwd=_ROOT)
        rc_mypy = subprocess.call(["mypy", *targets], cwd=_ROOT)
        rc_test = subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=_ROOT)
        if rc_ruff or rc_mypy or rc_test:
            click.echo("⚠ 検証で失敗があります。修正してからコミットしてください。")

    # 残りの手動チェックリスト（pii-masker 更新で発生する作業のみ。窓ポリシー win… は env で調整する
    # こちら都合の設定＝pii-masker 更新とは別の作業なので、ここには載せない）。
    click.echo(
        "\n===== 残りの手動ステップ =====\n"
        "1. 上の ENE ドリフト・submodule 差分を確認し、必要なら _ENE_TO_CATEGORY を更新（版バンプ不要）\n"
        "2. 契約変更（detect / locate_all の戻り値）があれば src/llm のアダプタを修正\n"
        "3. 実機（az login 済み）で 🤖 LLM検出 を回し件数/カテゴリを目視\n"
        "4. docs-dev/insight-memo.md に日付つきで記録\n"
        "5. 問題なければ git commit"
    )


def main() -> None:
    """エントリポイント。click の解析・ヘルプ出力より前に stdout を UTF-8 化する。"""
    _ensure_utf8_stdout()
    cli()


if __name__ == "__main__":
    main()
