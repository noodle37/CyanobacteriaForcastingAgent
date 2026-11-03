"""在子进程里跑模型写的 Python。写文件只能落到允许的目录。"""

from __future__ import annotations

import ast
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_ALLOWED_IMPORTS = {
    "math",
    "statistics",
    "json",
    "copy",
    "collections",
    "itertools",
    "datetime",
    "re",
    "decimal",
    "fractions",
    "functools",
    "random",
    "numpy",
    "pandas",
}
_WRITE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".txt"}
_REPO = Path(__file__).resolve().parents[1]


def execute_python(
    code: str,
    *,
    root: Path,
    series: list,
    horizon: int,
    freq: str,
    actual,
    forecast,
    baseline,
    metrics,
    read_roots: tuple[str, ...],
    write_roots: tuple[str, ...],
    timeout_s: float,
) -> dict:
    payload = {
        "code": code,
        "root": str(root),
        "series": series,
        "horizon": horizon,
        "freq": freq,
        "actual": actual,
        "forecast": forecast,
        "baseline": baseline,
        "metrics": metrics,
        "read_roots": list(read_roots),
        "write_roots": list(write_roots),
    }
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO) if not env.get("PYTHONPATH") else str(_REPO) + os.pathsep + env["PYTHONPATH"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "result.json"
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "agent.python_worker", str(result_path)],
                input=json.dumps(payload, ensure_ascii=False).encode(),
                cwd=str(_REPO),
                env=env,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"python 超过 {timeout_s:g} 秒", "files_written": []}
        if not result_path.exists():
            err = proc.stderr.decode(errors="replace")[-2000:]
            return {"ok": False, "error": err or f"python 退出码 {proc.returncode}", "files_written": []}
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"python 结果不是 JSON: {exc}", "files_written": []}


def run_user_code(
    code: str,
    *,
    root: Path,
    series: list,
    horizon: int,
    freq: str,
    actual,
    forecast,
    baseline,
    metrics,
    read_roots: tuple[str, ...],
    write_roots: tuple[str, ...],
) -> dict:
    written: list[str] = []
    stdout = io.StringIO()
    original = _as_floats(series)
    namespace = _namespace(
        root=root,
        series=list(original),
        horizon=int(horizon),
        freq=str(freq or "D"),
        actual=actual,
        forecast=forecast,
        baseline=baseline,
        metrics=metrics,
        read_roots=read_roots,
        write_roots=write_roots,
        stdout=stdout,
        written=written,
    )
    text = _strip_fence(code)
    if len(text) > 100_000:
        return {"ok": False, "error": "代码超过 100000 字符", "files_written": written}
    try:
        tree = ast.parse(text or "pass", filename="<python>", mode="exec")
        _reject_imports(tree)
        exec(compile(tree, "<python>", "exec"), namespace, namespace)  # noqa: S102
    except SyntaxError as exc:
        return _failed(f"SyntaxError: {exc}", stdout, written)
    except Exception as exc:
        return _failed(f"{type(exc).__name__}: {exc}", stdout, written)

    try:
        new_series = _as_floats(namespace.get("series"))
    except (TypeError, ValueError) as exc:
        return _failed(f"series 不是有限数值: {exc}", stdout, written)
    new_horizon = namespace.get("horizon", horizon)
    new_freq = str(namespace.get("freq") or freq or "D")
    return {
        "ok": True,
        "result": _jsonable(namespace.get("result")),
        "stdout": stdout.getvalue()[:4000],
        "series": new_series,
        "series_changed": new_series != original,
        "horizon": new_horizon,
        "freq": new_freq[:16],
        "files_written": written,
    }


def _namespace(root, series, horizon, freq, actual, forecast, baseline, metrics, read_roots, write_roots, stdout, written):
    def read_file(path: str) -> str:
        dest = _resolve(root, path, read_roots)
        if not dest.is_file():
            raise FileNotFoundError(f"找不到文件: {path}")
        text = dest.read_text(encoding="utf-8")
        if len(text) > 20_000:
            return text[:20_000] + "\n…(文件被截断)"
        return text

    def write_file(path: str, content: str) -> str:
        if not isinstance(content, str):
            raise TypeError("write_file 的内容必须是字符串")
        if len(content) > 200_000:
            raise ValueError("文件超过 200000 字符")
        dest = _resolve(root, path, write_roots)
        if dest.suffix not in _WRITE_SUFFIXES:
            raise ValueError(f"不允许写入这种文件: {dest.suffix or '(无后缀)'}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        written.append(dest.relative_to(root.resolve()).as_posix())
        return dest.relative_to(root.resolve()).as_posix()

    def _print(*args, **kwargs):
        kwargs["file"] = stdout
        print(*args, **kwargs)

    safe = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
        "round": round,
        "pow": pow,
        "divmod": divmod,
        "all": all,
        "any": any,
        "slice": slice,
        "isinstance": isinstance,
        "print": _print,
        "repr": repr,
        "type": type,
        "iter": iter,
        "next": next,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "ZeroDivisionError": ZeroDivisionError,
        "StopIteration": StopIteration,
        "RuntimeError": RuntimeError,
        "__import__": _limited_import,
    }
    return {
        "__builtins__": safe,
        "series": series,
        "horizon": horizon,
        "freq": freq,
        "actual": None if actual is None else list(actual),
        "forecast": forecast,
        "baseline": baseline,
        "metrics": metrics,
        "result": None,
        "read_file": read_file,
        "write_file": write_file,
    }


def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError("不允许相对导入")
    root_name = str(name).split(".", 1)[0]
    if root_name not in _ALLOWED_IMPORTS:
        raise ImportError(f"不允许导入 {name}")
    return __import__(name, globals, locals, fromlist, level)


def _reject_imports(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ImportError("不允许相对导入")
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root_name = name.split(".", 1)[0]
            if root_name not in _ALLOWED_IMPORTS:
                raise ImportError(f"不允许导入 {name}")


def _resolve(root: Path, raw: str, roots: tuple[str, ...]) -> Path:
    text = str(raw).replace("\\", "/").strip()
    if not text or "\x00" in text:
        raise ValueError("路径为空")
    path = Path(text)
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        rel = candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("路径超出允许的根目录") from exc
    if not rel.parts or rel.parts[0] not in roots:
        raise ValueError(f"路径不在允许目录内: {rel.as_posix()}")
    return candidate


def _strip_fence(code: str) -> str:
    text = str(code or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _as_floats(values) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    parsed = []
    for value in list(values or []):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("序列里出现了非数值")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("序列包含非有限数值")
        parsed.append(number)
    return parsed


def _failed(error: str, stdout: io.StringIO, written: list[str]) -> dict:
    return {
        "ok": False,
        "error": error,
        "stdout": stdout.getvalue()[:4000],
        "files_written": written,
        "series_changed": False,
    }


def _jsonable(value, depth: int = 0):
    if depth > 6:
        return repr(value)[:500]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in list(value.items())[:50]}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in list(value)[:200]]
    return repr(value)[:500]
