from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np


DEFAULT_DIMENSIONS = (8, 16, 32, 64, 128)
DEFAULT_SAMPLE_MULTIPLIERS = (2, 5, 10)
DEFAULT_NOISE_STD = 0.1
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_NUM_SEEDS = 20
DEFAULT_SEED_OFFSET = 0


@dataclass
class SyntheticSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    w_true: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


def _parse_int_list(raw: str, *, arg_name: str) -> List[int]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{arg_name} cannot be empty.")
    values = [int(token) for token in tokens]
    if any(value <= 0 for value in values):
        raise ValueError(f"All values in {arg_name} must be positive integers.")
    return values


def _resolve_seeds(
    *,
    num_seeds: int,
    seed_offset: int,
    explicit_seeds: str | None,
) -> List[int]:
    if explicit_seeds is not None:
        seeds = [int(token.strip()) for token in explicit_seeds.split(",") if token.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid integer was found.")
        return seeds

    if num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    return [seed_offset + i for i in range(num_seeds)]


def _next_power_of_two(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    return 1 if n == 1 else 2 ** int(np.ceil(np.log2(n)))


def _pad_features(X: np.ndarray, target_dim: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if target_dim < X.shape[1]:
        raise ValueError("target_dim must be >= number of columns in X.")
    if target_dim == X.shape[1]:
        return X.copy()
    pad_width = target_dim - X.shape[1]
    return np.pad(X, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


def _pad_vector(v: np.ndarray, target_dim: int) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    if target_dim < v.shape[0]:
        raise ValueError("target_dim must be >= vector length.")
    if target_dim == v.shape[0]:
        return v.copy()
    return np.pad(v, (0, target_dim - v.shape[0]), mode="constant", constant_values=0.0)


def _generate_single_split(
    *,
    n_samples: int,
    n_features: int,
    noise_std: float,
    train_ratio: float,
    seed: int,
) -> SyntheticSplit:
    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1.")
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1).")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n_samples, n_features))
    w_true = rng.standard_normal(size=n_features)
    epsilon = rng.normal(loc=0.0, scale=noise_std, size=n_samples)
    y = X @ w_true + epsilon

    order = rng.permutation(n_samples)
    n_train = int(np.floor(train_ratio * n_samples))
    n_train = min(max(1, n_train), n_samples - 1)
    train_indices = order[:n_train]
    test_indices = order[n_train:]

    return SyntheticSplit(
        X_train=X[train_indices],
        y_train=y[train_indices],
        X_test=X[test_indices],
        y_test=y[test_indices],
        w_true=w_true,
        train_indices=train_indices,
        test_indices=test_indices,
    )


def _save_npz(path: Path, *, X: np.ndarray, X_padded: np.ndarray, y: np.ndarray) -> None:
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=float),
        X_padded=np.asarray(X_padded, dtype=float),
        y=np.asarray(y, dtype=float).reshape(-1),
    )


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def generate_synthetic_dataset(
    *,
    output_dir: Path,
    dimensions: Sequence[int],
    sample_multipliers: Sequence[int],
    seeds: Sequence[int],
    noise_std: float,
    train_ratio: float,
) -> None:
    if not dimensions:
        raise ValueError("dimensions cannot be empty.")
    if not sample_multipliers:
        raise ValueError("sample_multipliers cannot be empty.")
    if not seeds:
        raise ValueError("seeds cannot be empty.")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_pairs: List[dict] = []
    for D in dimensions:
        padded_dim = _next_power_of_two(int(D))
        d_qubits = int(np.log2(padded_dim))
        for multiplier in sample_multipliers:
            N = int(multiplier) * int(D)
            pair_name = f"N{N}_D{D}"
            pair_dir = output_dir / pair_name
            pair_dir.mkdir(parents=True, exist_ok=True)

            manifest_pairs.append(
                {
                    "pair_name": pair_name,
                    "N": int(N),
                    "D": int(D),
                    "d": int(d_qubits),
                    "padded_dim": int(padded_dim),
                }
            )

            for seed in seeds:
                split = _generate_single_split(
                    n_samples=N,
                    n_features=int(D),
                    noise_std=float(noise_std),
                    train_ratio=float(train_ratio),
                    seed=int(seed),
                )

                seed_dir = pair_dir / f"seed_{int(seed):04d}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                X_train_padded = _pad_features(split.X_train, padded_dim)
                X_test_padded = _pad_features(split.X_test, padded_dim)
                w_true_padded = _pad_vector(split.w_true, padded_dim)

                _save_npz(
                    seed_dir / "train.npz",
                    X=split.X_train,
                    X_padded=X_train_padded,
                    y=split.y_train,
                )
                _save_npz(
                    seed_dir / "test.npz",
                    X=split.X_test,
                    X_padded=X_test_padded,
                    y=split.y_test,
                )

                np.savez_compressed(
                    seed_dir / "params.npz",
                    w_true=np.asarray(split.w_true, dtype=float),
                    w_true_padded=np.asarray(w_true_padded, dtype=float),
                    noise_std=float(noise_std),
                    N=int(N),
                    D=int(D),
                    d=int(d_qubits),
                    padded_dim=int(padded_dim),
                    seed=int(seed),
                )

                _write_json(
                    seed_dir / "split_meta.json",
                    {
                        "pair_name": pair_name,
                        "seed": int(seed),
                        "N": int(N),
                        "D": int(D),
                        "d": int(d_qubits),
                        "padded_dim": int(padded_dim),
                        "train_ratio": float(train_ratio),
                        "test_ratio": float(1.0 - train_ratio),
                        "n_train": int(split.X_train.shape[0]),
                        "n_test": int(split.X_test.shape[0]),
                        "noise_std": float(noise_std),
                        "train_file": "train.npz",
                        "test_file": "test.npz",
                        "params_file": "params.npz",
                    },
                )

    _write_json(
        output_dir / "manifest.json",
        {
            "dimensions": [int(x) for x in dimensions],
            "sample_multipliers": [int(x) for x in sample_multipliers],
            "pairs": manifest_pairs,
            "seeds": [int(seed) for seed in seeds],
            "noise_std": float(noise_std),
            "train_ratio": float(train_ratio),
            "test_ratio": float(1.0 - train_ratio),
            "notes": (
                "For each pair (N, D), train/test files are generated for the same seed set. "
                "X entries are i.i.d standard normal and y = X @ w_true + epsilon "
                "with epsilon ~ N(0, sigma^2 I)."
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic regression datasets over (N, D) sweeps, "
            "with train/test splits for shared random seeds."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "generated"),
        help="Output root directory for generated datasets.",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        default="8,16,32,64,128",
        help="Comma-separated feature dimensions D.",
    )
    parser.add_argument(
        "--sample-multipliers",
        type=str,
        default="2,5,10",
        help="Comma-separated sample multipliers to form N = multiplier * D.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=DEFAULT_NOISE_STD,
        help="Noise standard deviation sigma in y = X @ w_true + epsilon.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Train split ratio; test ratio is 1 - train_ratio.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help="Number of seeds to generate when --seeds is not provided.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=DEFAULT_SEED_OFFSET,
        help="Starting seed value used with --num-seeds.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional explicit comma-separated seed list. Overrides --num-seeds/--seed-offset.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dimensions = _parse_int_list(args.dimensions, arg_name="--dimensions")
    sample_multipliers = _parse_int_list(
        args.sample_multipliers, arg_name="--sample-multipliers"
    )
    seeds = _resolve_seeds(
        num_seeds=int(args.num_seeds),
        seed_offset=int(args.seed_offset),
        explicit_seeds=args.seeds,
    )

    output_dir = Path(args.output_dir).resolve()
    generate_synthetic_dataset(
        output_dir=output_dir,
        dimensions=dimensions,
        sample_multipliers=sample_multipliers,
        seeds=seeds,
        noise_std=float(args.noise_std),
        train_ratio=float(args.train_ratio),
    )

    print("Synthetic dataset generation complete.")
    print(f"Output directory: {output_dir}")
    print(f"Dimensions (D): {dimensions}")
    print(f"Sample multipliers (N = m * D): {sample_multipliers}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Noise std (sigma): {float(args.noise_std)}")
    print(f"Train/Test split: {float(args.train_ratio):.2f}/{1.0 - float(args.train_ratio):.2f}")


if __name__ == "__main__":
    main()
