"""Compute exact uint16 depth statistics for a training split."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath, PureWindowsPath

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--suffix", default="_depth.png")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--progress", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def sample_stem(line):
    line = line.strip().strip('"').strip("'")
    path_cls = PureWindowsPath if "\\" in line else PurePosixPath
    return path_cls(line).stem


def distribution_stats(histogram, minimum_value=0):
    values = np.arange(histogram.size, dtype=np.float64)
    selected = values >= minimum_value
    counts = histogram[selected].astype(np.float64)
    values = values[selected]
    count = int(counts.sum())
    if count == 0:
        return None

    total = float(np.dot(counts, values))
    total_squares = float(np.dot(counts, values * values))
    mean = total / count
    variance = max(total_squares / count - mean * mean, 0.0)
    cumulative = np.cumsum(counts, dtype=np.float64)

    def percentile(q):
        rank = q / 100.0 * max(count - 1, 0)
        return int(values[np.searchsorted(cumulative, rank + 1, side="left")])

    occupied = np.flatnonzero(histogram)
    occupied = occupied[occupied >= minimum_value]
    return {
        "count": count,
        "min": int(occupied[0]),
        "max": int(occupied[-1]),
        "mean": mean,
        "variance_population": variance,
        "std_population": variance**0.5,
        "percentiles": {
            str(q): percentile(q) for q in (0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100)
        },
    }


def process_chunk(worker_id, paths, progress):
    histogram = np.zeros(65536, dtype=np.uint64)
    missing = []
    wrong_dtype = []
    shapes = {}

    for index, depth_path in enumerate(paths, start=1):
        if not depth_path.is_file():
            missing.append(str(depth_path))
            continue
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            missing.append(str(depth_path))
            continue
        if depth.ndim != 2 or depth.dtype != np.uint16:
            wrong_dtype.append({"path": str(depth_path), "shape": list(depth.shape), "dtype": str(depth.dtype)})
            continue

        shapes[str(depth.shape)] = shapes.get(str(depth.shape), 0) + 1
        histogram += np.bincount(depth.ravel(), minlength=65536).astype(np.uint64)
        if progress > 0 and index % progress == 0:
            print(f"Worker {worker_id}: {index}/{len(paths)}", flush=True)

    return histogram, missing, wrong_dtype, shapes


def main(args):
    lines = [line for line in args.train_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    paths = [args.images_dir / f"{sample_stem(line)}{args.suffix}" for line in lines]
    histogram = np.zeros(65536, dtype=np.uint64)
    missing = []
    wrong_dtype = []
    shapes = {}

    workers = max(1, min(args.workers, len(paths)))
    chunks = [paths[index::workers] for index in range(workers)]
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_chunk, index, chunk, args.progress) for index, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            local_histogram, local_missing, local_wrong_dtype, local_shapes = future.result()
            histogram += local_histogram
            missing.extend(local_missing)
            wrong_dtype.extend(local_wrong_dtype)
            for shape, count in local_shapes.items():
                shapes[shape] = shapes.get(shape, 0) + count
            print(f"Completed {sum(shapes.values()) + len(missing) + len(wrong_dtype)}/{len(lines)}", flush=True)

    nonzero_count = int(histogram[1:].sum())
    result = {
        "train_list": str(args.train_list),
        "images_dir": str(args.images_dir),
        "listed_images": len(lines),
        "processed_images": len(lines) - len(missing) - len(wrong_dtype),
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "wrong_dtype_count": len(wrong_dtype),
        "wrong_dtype_examples": wrong_dtype[:20],
        "shapes": shapes,
        "zero_count": int(histogram[0]),
        "zero_ratio": float(histogram[0] / histogram.sum()) if histogram.sum() else None,
        "all_pixels": distribution_stats(histogram, minimum_value=0),
        "valid_nonzero_pixels": distribution_stats(histogram, minimum_value=1),
        "nonzero_ge_65000_count": int(histogram[65000:].sum()),
        "nonzero_ge_65000_ratio": float(histogram[65000:].sum() / nonzero_count) if nonzero_count else None,
    }

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(parse_args())
