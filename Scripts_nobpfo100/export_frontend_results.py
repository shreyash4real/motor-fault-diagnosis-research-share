"""Export the stored current-signal ensemble results for the static frontend.

This deliberately uses only the committed evaluation artefacts. It does not run
inference, read raw current recordings, or expose local Windows paths.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import struct
from array import array
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "Outputs_nobpfo100" / "training"
OUT_PATH = ROOT / "results-data.json"
CLASS_NAMES = ["healthy 1", "stator short 1", "bearing bpfo 3", "broken rotor bar"]
CLASS_KEYS = ["healthy", "stator_short", "bearing_bpfo", "broken_rotor_bar"]
CLASS_TO_KEY = dict(zip(CLASS_NAMES, CLASS_KEYS))


def read_npy(path: Path) -> tuple[tuple[int, ...], list[float | int]]:
    """Read the small little-endian numeric .npy files used by the outputs."""
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"Not an .npy file: {path}")
    major, minor = raw[6], raw[7]
    if major == 1:
        header_len = struct.unpack_from("<H", raw, 8)[0]
        offset = 10
    elif major in (2, 3):
        header_len = struct.unpack_from("<I", raw, 8)[0]
        offset = 12
    else:
        raise ValueError(f"Unsupported .npy version {major}.{minor}")

    header = ast.literal_eval(raw[offset : offset + header_len].decode("latin1"))
    dtype = header["descr"]
    shape = tuple(header["shape"])
    if header["fortran_order"] or dtype not in ("<f8", "<i8"):
        raise ValueError(f"Unsupported .npy layout in {path}: {header}")

    values = array("d" if dtype == "<f8" else "q")
    values.frombytes(raw[offset + header_len :])
    if dtype[0] == ">":
        values.byteswap()
    flat = list(values)
    expected = 1
    for dim in shape:
        expected *= dim
    if len(flat) != expected:
        raise ValueError(f"Unexpected value count in {path}: {len(flat)} != {expected}")
    return shape, flat


def matrix(path: Path) -> list[list[float]]:
    shape, flat = read_npy(path)
    if len(shape) != 2:
        raise ValueError(f"Expected a matrix: {path} -> {shape}")
    rows, cols = shape
    return [list(map(float, flat[i * cols : (i + 1) * cols])) for i in range(rows)]


def vector(path: Path) -> list[int]:
    shape, flat = read_npy(path)
    if len(shape) != 1:
        raise ValueError(f"Expected a vector: {path} -> {shape}")
    return [int(v) for v in flat]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_weights(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header = rows[0][1:]
    return {
        row[0]: [float(row[index + 1]) for index in range(len(header))]
        for row in rows[1:]
        if row and row[0]
    }


def read_temperatures(path: Path) -> dict[str, float]:
    return {row["member"]: float(row["T"]) for row in read_csv(path)}


def parse_summary(path: Path) -> tuple[int, float, float, int]:
    text = path.read_text(encoding="utf-8")
    samples = int(re.search(r"Test samples:\s*(\d+)", text).group(1))
    accuracy = float(re.search(r"Test accuracy:\s*([0-9.]+)", text).group(1))
    macro_f1 = float(re.search(r"Test macro-F1:\s*([0-9.]+)", text).group(1))
    errors = int(re.search(r"Misclassified:\s*(\d+)", text).group(1))
    return samples, accuracy, macro_f1, errors


def confusion(labels: list[int], predictions: list[int]) -> list[list[int]]:
    matrix_out = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for truth, prediction in zip(labels, predictions):
        matrix_out[truth][prediction] += 1
    return matrix_out


def gallery_index() -> dict[tuple[str, int, int, int], dict[str, str]]:
    result: dict[tuple[str, int, int, int], dict[str, str]] = {}
    manifest = ROOT / "sample_gallery" / "manifest.csv"
    for row in read_csv(manifest):
        source = row["source"].replace("\\", "/")
        match = re.search(
            r"/(healthy_1|stator_short_1|bearing_bpfo_3|broken_rotor_bar)/"
            r"speed_(50|75|100)/col(\d+)_seg(\d+)(?:_ch\d+)?\.png$",
            source,
        )
        if not match:
            continue
        class_key, speed, col, seg = match.groups()
        result.setdefault((class_key, int(speed), int(col), int(seg)), {})[
            row["representation"]
        ] = row["gallery_path"]
    return result


def sample_metadata(path: Path, labels: list[int]) -> list[dict]:
    rows = read_csv(path)
    if len(rows) != len(labels):
        raise ValueError(f"Metadata/label mismatch: {path}")
    gallery = gallery_index()
    output = []
    for index, (row, label) in enumerate(zip(rows, labels)):
        path_text = row["ch1_path"].replace("\\", "/")
        speed_match = re.search(r"/(50|75|100)/", path_text)
        class_match = next((name for name in CLASS_NAMES if f"/{name}/" in path_text), None)
        if not speed_match or class_match is None:
            raise ValueError(f"Could not derive sample identity: {row}")
        speed = int(speed_match.group(1))
        col = int(row["col_index"])
        seg = int(row["seg_idx"])
        key = CLASS_TO_KEY[class_match]
        output.append(
            {
                "index": index,
                "trueClass": CLASS_TO_KEY[CLASS_NAMES[label]],
                "speedPct": speed,
                "column": col,
                "segment": seg,
                "gallery": gallery.get((key, speed, col, seg), {}),
            }
        )
    return output


def per_class(path: Path) -> list[dict]:
    return [
        {
            "class": CLASS_TO_KEY[row["class"]],
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "support": int(row["support"]),
        }
        for row in read_csv(path)
    ]


def main() -> None:
    definitions = [
        {
            "id": "stft-dwt-temperature",
            "title": "STFT + DWT · Temperature calibrated",
            "run": "ENSEMBLE_stft_dwt_temperature",
            "members": ["stft", "dwt"],
            "kind": "temperature",
        },
        {
            "id": "stft-dwt-validation-f1",
            "title": "STFT + DWT · Validation-F1 weighted",
            "run": "ENSEMBLE_stft_dwt_perclass_f1",
            "members": ["stft", "dwt"],
            "kind": "validation-f1",
        },
        {
            "id": "stft-dwt-envelope-temperature",
            "title": "STFT + DWT + Envelope · Temperature calibrated",
            "run": "ENSEMBLE_stft_dwt_envelope_temperature",
            "members": ["stft", "dwt", "envelope_dilated"],
            "kind": "temperature",
        },
        {
            "id": "stft-dwt-envelope-validation-f1",
            "title": "STFT + DWT + Envelope · Validation-F1 weighted",
            "run": "ENSEMBLE_stft_dwt_envelope_perclass_f1",
            "members": ["stft", "dwt", "envelope_dilated"],
            "kind": "validation-f1",
        },
    ]

    labels = vector(OUTPUT_ROOT / definitions[0]["run"] / "labels.npy")
    metadata = sample_metadata(
        OUTPUT_ROOT / "E4_mod_alexnet_227" / "test_meta_sigs.csv", labels
    )
    experiments = []

    for definition in definitions:
        run_dir = OUTPUT_ROOT / definition["run"]
        arrays = []
        for member in definition["members"]:
            prefix = "softmax_calibrated_" if definition["kind"] == "temperature" else "softmax_"
            arrays.append(matrix(run_dir / f"{prefix}{member}.npy"))
        rows = len(arrays[0])
        if any(len(values) != rows for values in arrays) or rows != len(labels):
            raise ValueError(f"Prediction alignment mismatch in {definition['run']}")

        temperatures = (
            read_temperatures(run_dir / "temperatures.csv")
            if definition["kind"] == "temperature"
            else None
        )
        weights = (
            read_weights(run_dir / "weights.csv")
            if definition["kind"] == "validation-f1"
            else None
        )

        probabilities = []
        for row_index in range(rows):
            if weights is None:
                combined = [
                    sum(values[row_index][class_index] for values in arrays) / len(arrays)
                    for class_index in range(len(CLASS_NAMES))
                ]
            else:
                combined = [
                    sum(
                        arrays[member_index][row_index][class_index]
                        * weights[member][class_index]
                        for member_index, member in enumerate(definition["members"])
                    )
                    for class_index in range(len(CLASS_NAMES))
                ]
                total = sum(combined) or 1.0
                combined = [value / total for value in combined]
            probabilities.append(combined)

        predictions = [max(range(len(row)), key=row.__getitem__) for row in probabilities]
        samples = []
        for sample, probs, prediction in zip(metadata, probabilities, predictions):
            ranked = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
            samples.append(
                {
                    **sample,
                    "prediction": CLASS_KEYS[prediction],
                    "correct": CLASS_KEYS[prediction] == sample["trueClass"],
                    "confidence": probs[prediction],
                    "margin": probs[ranked[0]] - probs[ranked[1]],
                    "probabilities": {
                        CLASS_KEYS[class_index]: probs[class_index]
                        for class_index in range(len(CLASS_NAMES))
                    },
                }
            )

        sample_count, accuracy, macro_f1, errors = parse_summary(run_dir / "summary.txt")
        if sample_count != len(samples) or errors != sum(not item["correct"] for item in samples):
            raise ValueError(f"Stored summary does not match generated predictions: {definition['run']}")
        experiments.append(
            {
                "id": definition["id"],
                "title": definition["title"],
                "run": definition["run"],
                "representations": ["stft", "dwt"]
                + (["envelope"] if len(definition["members"]) == 3 else []),
                "fusion": definition["kind"],
                "members": definition["members"],
                "accuracy": accuracy,
                "macroF1": macro_f1,
                "errors": errors,
                "perClass": per_class(run_dir / "per_class_metrics.csv"),
                "temperatures": temperatures,
                "weights": weights,
                "confusion": confusion(labels, predictions),
                "samples": samples,
            }
        )

    payload = {
        "version": 1,
        "source": {
            "split": "nobpfo100",
            "testWindows": len(labels),
            "testColumnGroups": len({(sample["trueClass"], sample["speedPct"], sample["column"]) for sample in metadata}),
            "windowSeconds": 1.0,
            "windowStrideSeconds": 0.25,
            "note": "Precomputed motor-current evaluation. BPFO-3 at 100% speed is excluded from this test split by design.",
        },
        "classes": [
            {"key": key, "label": label}
            for key, label in zip(CLASS_KEYS, CLASS_NAMES)
        ],
        "experiments": experiments,
    }
    OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")
    for experiment in experiments:
        print(
            f"{experiment['id']}: {experiment['accuracy']:.6f} accuracy, "
            f"{experiment['macroF1']:.6f} macro-F1, {experiment['errors']} errors"
        )


if __name__ == "__main__":
    main()
