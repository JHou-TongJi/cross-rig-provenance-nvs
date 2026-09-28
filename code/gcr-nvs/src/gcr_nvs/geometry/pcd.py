"""Small dependency-free PCD reader for the captured binary point clouds."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_pcd(path: Path, fields: tuple[str, ...] = ("x", "y", "z", "intensity", "ring", "time")) -> np.ndarray:
    with path.open("rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"invalid PCD header: {path}")
            decoded = line.decode("ascii", errors="strict").strip()
            header.append(decoded)
            if decoded.startswith("DATA"):
                break
        values = {line.split(maxsplit=1)[0]: line.split()[1:] for line in header if line and not line.startswith("#")}
        names = values["FIELDS"]
        counts = [int(value) for value in values.get("COUNT", ["1"] * len(names))]
        points = int(values["POINTS"][0])
        data_type = values["DATA"][0].lower()
        if data_type == "ascii":
            raw = np.loadtxt(handle, dtype=np.float32)
            raw = np.atleast_2d(raw)
            return raw[:, [names.index(name) for name in fields if name in names]]
        if data_type != "binary":
            raise ValueError(f"unsupported PCD DATA mode {data_type!r}: {path}")
        dtype_parts = []
        sizes = values["SIZE"]
        types = values["TYPE"]
        for name, size, type_code, count in zip(names, sizes, types, counts):
            if type_code == "F" and size == "4":
                dtype = "f4"
            elif type_code == "F" and size == "8":
                dtype = "f8"
            elif type_code == "U" and size == "2":
                dtype = "u2"
            elif type_code == "U" and size == "4":
                dtype = "u4"
            else:
                raise ValueError(f"unsupported PCD field {name}: TYPE={type_code}, SIZE={size}")
            dtype_parts.append((name, dtype, count))
        dtype = np.dtype(
            [(name, np.dtype(type_code) if count == 1 else (np.dtype(type_code), count)) for name, type_code, count in dtype_parts]
        )
        values_array = np.fromfile(handle, dtype=dtype, count=points)
        return np.column_stack([values_array[name].reshape(points, -1)[:, 0] for name in fields if name in names]).astype(np.float32)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    coordinates = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, indices = np.unique(coordinates, axis=0, return_index=True)
    return points[np.sort(indices)]
