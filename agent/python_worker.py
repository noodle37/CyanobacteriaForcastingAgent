"""python 工具的子进程入口。任务从 stdin 读入，结果写到 argv[1]。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent.python_exec import run_user_code


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        result = run_user_code(
            payload.get("code") or "",
            root=Path(payload["root"]),
            series=list(payload.get("series") or []),
            horizon=int(payload.get("horizon") or 0),
            freq=str(payload.get("freq") or "D"),
            actual=payload.get("actual"),
            forecast=payload.get("forecast"),
            baseline=payload.get("baseline"),
            metrics=payload.get("metrics"),
            read_roots=tuple(payload.get("read_roots") or ()),
            write_roots=tuple(payload.get("write_roots") or ()),
        )
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "files_written": []}
    Path(sys.argv[1]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
