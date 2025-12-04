"""
Lightweight OpenCV stub to keep MetaDrive working when the bundled `cv2` binary
is broken on macOS (e.g., missing `_avifDecoderCreate`).  This implements only
the tiny subset of functions/constants MetaDrive touches in headless mode.
"""
from __future__ import annotations

from typing import Iterable, Tuple, Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
try:
    from matplotlib.path import Path
except Exception:  # pragma: no cover - matplotlib may be unavailable
    Path = None  # type: ignore

# Basic constants used in MetaDrive
INTER_LINEAR = 1
INTER_NEAREST = 0
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 4
IMREAD_UNCHANGED = -1


def _to_array(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr


def imread(path: str, flags: int | None = None) -> np.ndarray:  # pragma: no cover - I/O wrapper
    try:
        from PIL import Image

        with Image.open(path) as im:
            return np.array(im)
    except Exception:
        return np.zeros((1, 1, 3), dtype=np.uint8)


def imwrite(path: str, img: Any) -> bool:  # pragma: no cover - I/O wrapper
    try:
        from PIL import Image

        arr = _to_array(img)
        Image.fromarray(arr.astype(np.uint8)).save(path)
        return True
    except Exception:
        return False


def imencode(ext: str, img: Any):  # pragma: no cover - compatibility shim
    arr = _to_array(img).astype(np.uint8)
    return True, arr.flatten()


def resize(img: Any, dsize: Tuple[int, int], interpolation: int | None = None) -> np.ndarray:
    arr = _to_array(img)
    if len(dsize) != 2:
        return arr
    width, height = int(dsize[0]), int(dsize[1])
    if width <= 0 or height <= 0:
        return arr
    try:
        from PIL import Image

        pil = Image.fromarray(arr)
        return np.array(pil.resize((width, height)))
    except Exception:
        y_idx = np.linspace(0, arr.shape[0] - 1, height).round().astype(int)
        x_idx = np.linspace(0, arr.shape[1] - 1, width).round().astype(int)
        return arr[y_idx][:, x_idx]


def fillPoly(img: Any, pts: Iterable[np.ndarray], color) -> np.ndarray:
    arr = _to_array(img).copy()
    pts = np.asarray(list(pts)).reshape(-1, 2)
    if Path is None or arr.size == 0:
        return arr
    h, w = arr.shape[:2]
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).reshape(-1, 2)
    mask = Path(pts).contains_points(grid).reshape(h, w)
    arr[mask] = color
    return arr


def polylines(img: Any, pts: Iterable[np.ndarray], isClosed: bool, color, thickness: int = 1) -> np.ndarray:
    arr = _to_array(img).copy()
    pts_arr = np.asarray(list(pts)).reshape(-1, 2)
    if arr.size == 0 or pts_arr.shape[0] < 2:
        return arr
    for start, end in zip(pts_arr[:-1], pts_arr[1:]):
        num = max(abs(int(end[0] - start[0])), abs(int(end[1] - start[1])), 1) * max(thickness, 1)
        xs = np.linspace(start[0], end[0], num=num).round().astype(int)
        ys = np.linspace(start[1], end[1], num=num).round().astype(int)
        xs = np.clip(xs, 0, arr.shape[1] - 1)
        ys = np.clip(ys, 0, arr.shape[0] - 1)
        arr[ys, xs] = color
    if isClosed and pts_arr.shape[0] > 2:
        arr = polylines(arr, [pts_arr[-1], pts_arr[0]], False, color, thickness)
    return arr


def dilate(img: Any, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    arr = _to_array(img)
    if arr.ndim == 2:
        arr = arr[..., None]
    out = arr
    for _ in range(max(iterations, 1)):
        pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
        padded = np.pad(out, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge")
        windows = sliding_window_view(padded, (kernel.shape[0], kernel.shape[1], out.shape[2]))
        out = windows.max(axis=(2, 3))
    return out


def cvtColor(img: Any, code: int) -> np.ndarray:
    arr = _to_array(img)
    if arr.ndim >= 3:
        return arr[..., ::-1]
    return arr


def VideoWriter_fourcc(*args) -> int:  # pragma: no cover - compatibility shim
    return 0


class VideoWriter:  # pragma: no cover - compatibility shim
    def __init__(self, *_, **__):
        self._frames = []

    def write(self, frame):
        self._frames.append(_to_array(frame))

    def release(self):
        self._frames.clear()


def waitKey(delay: int = 0) -> int:  # pragma: no cover - compatibility shim
    return -1


def imshow(winname: str, mat: Any) -> None:  # pragma: no cover - compatibility shim
    return


def destroyAllWindows() -> None:  # pragma: no cover - compatibility shim
    return


def __getattr__(name: str):
    # Fallback so unexpected attributes do not crash MetaDrive when using this stub.
    return 0
