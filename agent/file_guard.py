"""模型代码执行期间的文件访问与模块护栏。

沙箱原本只靠 read_file / write_file 两个助手限制可访问目录,但模型能绕开它们:

  - numpy / pandas 自带的文件接口(np.save、pd.read_csv、DataFrame.to_csv……)
    直接走解释器或 C 层文件句柄,完全不受目录白名单约束;
  - numpy.ctypeslib 挂着 ctypes、numpy.f2py 能调 C 编译器,等于原生代码执行。

这里在模型代码执行期间把这两类口子一起收掉:文件读写收窄回同一套目录白名单,
危险模块替换成拿到就报错的占位对象。护栏只在执行期间生效,执行完立刻还原。
"""

from __future__ import annotations

import builtins
import io
import os
import sys
import sysconfig
from contextlib import contextmanager
from pathlib import Path

# 这些 numpy 子模块能拿到 ctypes 或 C 编译器,等于原生代码执行,直接禁掉。
BLOCKED_SUBMODULES = {
    "numpy.ctypeslib",
    "numpy.f2py",
    "numpy.distutils",
}

_WRITE_MODE_CHARS = set("wax+")


class Blocked:
    """占位对象:模型拿到它就等于知道这条路已经被封了。"""

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def __getattr__(self, item: str):
        raise ImportError(f"沙箱禁止访问 {object.__getattribute__(self, '_name')}")

    def __call__(self, *args, **kwargs):
        raise ImportError(f"沙箱禁止访问 {object.__getattribute__(self, '_name')}")

    def __repr__(self) -> str:
        return f"<blocked {object.__getattribute__(self, '_name')}>"


def is_write_mode(mode) -> bool:
    return bool(set(str(mode or "r")) & _WRITE_MODE_CHARS)


def runtime_read_prefixes() -> tuple[Path, ...]:
    """解释器自身的目录:解释器自己的库文件要放行,否则 import pandas 会被自己挡住。"""
    raw = {sys.prefix, sys.base_prefix}
    try:
        paths = sysconfig.get_paths()
    except Exception:  # noqa: BLE001 - 取不到就只用 prefix
        paths = {}
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        value = paths.get(key)
        if value:
            raw.add(value)
    return tuple(sorted((Path(p) for p in raw if p), key=str))


class Guard:
    """按 read_roots / write_roots 判定某个路径能不能读写。"""

    def __init__(self, *, root, read_roots, write_roots, resolve) -> None:
        self.root = Path(root)
        self.read_roots = tuple(read_roots or ())
        self.write_roots = tuple(write_roots or ())
        self._resolve = resolve
        self._runtime_prefixes = runtime_read_prefixes()

    def _inside_runtime(self, path: Path) -> bool:
        for prefix in self._runtime_prefixes:
            if path == prefix or prefix in path.parents:
                return True
        return False

    def check(self, target, mode) -> None:
        """target 是路径或文件描述符;mode 是打开模式。越界直接抛 PermissionError。"""
        if isinstance(target, int):
            return
        try:
            raw = os.fspath(target)
        except TypeError:
            return
        if not isinstance(raw, str) or not raw:
            return
        if is_write_mode(mode):
            try:
                self._resolve(self.root, raw, self.write_roots)
            except Exception as exc:  # noqa: BLE001 - 越界统一转成权限错误
                raise PermissionError(f"沙箱不允许写入该路径: {raw}") from exc
            return
        try:
            self._resolve(self.root, raw, self.read_roots)
            return
        except Exception:  # noqa: BLE001 - 不在白名单里,再看是不是解释器自己的目录
            pass
        try:
            resolved = Path(raw).resolve()
        except Exception:  # noqa: BLE001
            resolved = Path(raw)
        if self._inside_runtime(resolved):
            return
        raise PermissionError(f"沙箱不允许读取该路径: {raw}")


@contextmanager
def file_guard(guard: Guard):
    """执行期间把 builtins.open / io.open 换成受白名单约束的版本。"""
    real_open = builtins.open
    real_io_open = io.open

    def make(real):
        def guarded(file, mode="r", *args, **kwargs):
            guard.check(file, mode)
            return real(file, mode, *args, **kwargs)

        return guarded

    builtins.open = make(real_open)
    io.open = make(real_io_open)
    try:
        yield
    finally:
        builtins.open = real_open
        io.open = real_io_open


def patch_imported(name: str, module, guard: Guard) -> None:
    """模型在沙箱里 import numpy / pandas 之后,收掉它们自带的越界接口。"""
    root_name = name.split(".")[0]
    if root_name == "numpy":
        _patch_numpy(module, guard)
    elif root_name == "pandas":
        _patch_pandas(module, guard)


def _patch_numpy(module, guard: Guard) -> None:
    real = getattr(module, "fromfile", None)
    if callable(real) and not getattr(real, "_sandbox_guarded", False):

        def fromfile(file, *args, **kwargs):
            # np.fromfile 走 C 层文件句柄,拦不住 open 补丁,只能单独包一层。
            guard.check(file, "rb")
            return real(file, *args, **kwargs)

        fromfile._sandbox_guarded = True  # type: ignore[attr-defined]
        try:
            module.fromfile = fromfile
        except Exception:  # noqa: BLE001
            pass
    for attr in ("ctypeslib", "f2py"):
        try:
            # 用占位对象盖住,而不是 delattr:numpy 的 __getattr__ 会把删掉的属性再懒加载回来。
            setattr(module, attr, Blocked(f"numpy.{attr}"))
        except Exception:  # noqa: BLE001
            pass


def _patch_pandas(module, guard: Guard) -> None:
    # pandas 的读写都经过 get_handle -> open,open 已经被补丁覆盖,这里不用再包。
    return None
