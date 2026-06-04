import sys

# MCPプロトコルが stdio を介して JSON-RPC をやり取りする場合、
# デバッグログが標準出力(stdout)に混入すると、MCPクライアントが正しく動作しなくなる可能性がある.
# そのため、標準エラー出力(stderr)にログを出力する.


def log(message: str) -> None:
    print(message, file=sys.stderr)
