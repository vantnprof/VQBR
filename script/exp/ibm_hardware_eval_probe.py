from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
ABLATION_REAL_HARDWARE_PATH = ROOT_DIR / "script" / "exp" / "ablation_real_hardware.py"
DEFAULT_RESULTS_ROOT = ROOT_DIR / "results" / "synthetic"


def _load_ablation_real_hardware_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_ablation_real_hardware_helpers",
        ABLATION_REAL_HARDWARE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load helper module from: {ABLATION_REAL_HARDWARE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ARH = _load_ablation_real_hardware_module()

ENFORCED_VQBR_OPTIMIZER = _ARH.ENFORCED_VQBR_OPTIMIZER
ENFORCED_VQBR_LOSS = _ARH.ENFORCED_VQBR_LOSS
ENFORCED_VQBR_SU2_GATES = _ARH.ENFORCED_VQBR_SU2_GATES
DEFAULT_IBM_CONFIG_PATH = _ARH.DEFAULT_IBM_CONFIG_PATH

_PrimitiveObjectiveEvaluator = _ARH._PrimitiveObjectiveEvaluator
_ZNEObjectiveEvaluator = _ARH._ZNEObjectiveEvaluator
_backend_name = _ARH._backend_name
_build_ibm_estimator = _ARH._build_ibm_estimator
_estimate_prior_from_prior_split = _ARH._estimate_prior_from_prior_split
_generate_synthetic_data = _ARH._generate_synthetic_data
_parse_float_list = _ARH._parse_float_list
_prepare_vqbr_model = _ARH._prepare_vqbr_model
_resolve_ibm_runtime_backend = _ARH._resolve_ibm_runtime_backend
_resolve_seeds = _ARH._resolve_seeds
_sanitize_args_for_results = _ARH._sanitize_args_for_results
_sanitize_argv = _ARH._sanitize_argv
_split_dataset_indices = _ARH._split_dataset_indices


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_run_dir(args: argparse.Namespace) -> Path:
    cached = getattr(args, "_resolved_probe_run_dir", None)
    if isinstance(cached, str) and cached.strip():
        return Path(cached).expanduser().resolve()
    if str(args.run_dir).strip():
        run_dir = Path(str(args.run_dir).strip()).expanduser().resolve()
        args._resolved_probe_run_dir = str(run_dir)
        return run_dir
    run_tag = (
        str(args.run_tag).strip()
        if str(args.run_tag).strip()
        else f"ibm_eval_timing_probe_N{int(args.n_samples)}_D{int(args.n_features)}_{_timestamp_tag()}"
    )
    run_dir = Path(args.results_root).expanduser().resolve() / run_tag
    args._resolved_probe_run_dir = str(run_dir)
    return run_dir


def _build_hadamard_test_pubs(
    evaluator: Any,
    *,
    theta_vec: np.ndarray,
    batch_indices: np.ndarray,
) -> List[Any]:
    theta_list = [np.asarray(theta_vec, dtype=float).reshape(-1).tolist()]
    pubs: List[Any] = []
    for sample_idx in np.asarray(batch_indices, dtype=int).reshape(-1).tolist():
        pubs.append(
            (
                evaluator._overlap_circuits[int(sample_idx)],
                evaluator._overlap_observables[int(sample_idx)],
                theta_list,
            )
        )
    has_uc = evaluator._uc_circuit is not None and evaluator._uc_observable is not None
    if has_uc:
        pubs.append((evaluator._uc_circuit, evaluator._uc_observable, theta_list))
    return pubs


def _build_d_term_pub(evaluator: Any, *, theta_vec: np.ndarray) -> List[Any]:
    theta_list = [np.asarray(theta_vec, dtype=float).reshape(-1).tolist()]
    return [(evaluator._d_circuit, evaluator._d_observable, theta_list)]


def _time_call(fn: Any) -> tuple[float, Any]:
    start = time.perf_counter()
    value = fn()
    elapsed = float(time.perf_counter() - start)
    return elapsed, value


def _format_seconds(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "nan"
    return f"{seconds:.3f}s"


def _component_breakdown_no_zne(
    *,
    evaluator: Any,
    theta_init: np.ndarray,
    full_batch: np.ndarray,
) -> Dict[str, Any]:
    hadamard_pubs = _build_hadamard_test_pubs(
        evaluator,
        theta_vec=theta_init,
        batch_indices=full_batch,
    )
    d_pub = _build_d_term_pub(evaluator, theta_vec=theta_init)

    hadamard_seconds, _ = _time_call(
        lambda: evaluator._run_estimator_with(evaluator.estimator, hadamard_pubs)
    )
    d_seconds, _ = _time_call(
        lambda: evaluator._run_estimator_with(evaluator.estimator, d_pub)
    )
    return {
        "hadamard_test_seconds": float(hadamard_seconds),
        "d_term_seconds": float(d_seconds),
        "hadamard_test_pub_count": int(len(hadamard_pubs)),
        "d_term_pub_count": int(len(d_pub)),
    }


def _component_breakdown_zne(
    *,
    evaluator: Any,
    theta_init: np.ndarray,
    full_batch: np.ndarray,
) -> Dict[str, Any]:
    hadamard_pubs = _build_hadamard_test_pubs(
        evaluator,
        theta_vec=theta_init,
        batch_indices=full_batch,
    )
    d_pub = _build_d_term_pub(evaluator, theta_vec=theta_init)

    hadamard_seconds, _ = _time_call(
        lambda: evaluator._run_zne_hadamard_test_estimator(hadamard_pubs)
    )
    d_seconds, _ = _time_call(
        lambda: evaluator._run_estimator_with(evaluator._baseline_estimator, d_pub)
    )
    return {
        "hadamard_test_seconds": float(hadamard_seconds),
        "d_term_seconds": float(d_seconds),
        "hadamard_test_pub_count": int(len(hadamard_pubs)),
        "d_term_pub_count": int(len(d_pub)),
    }


def _print_probe_summary(results: Dict[str, Any]) -> None:
    measured = dict(results.get("measured", {}))
    estimates = dict(results.get("estimated_runtime_seconds", {}))
    dataset = dict(results.get("dataset", {}))
    ibm = dict(results.get("ibm", {}))

    print("=== IBM Hardware Evaluation Timing Probe ===")
    print(
        f"N={dataset.get('n_samples')}, D={dataset.get('n_features')} | "
        f"train_size={dataset.get('train_size')} | backend={ibm.get('backend_name')} | "
        f"shots={ibm.get('shots')}"
    )
    print(
        f"no-ZNE objective evaluation: {_format_seconds(float(measured.get('objective_no_zne_seconds', np.nan)))} "
        f"[pubs={measured.get('objective_no_zne_pub_count')}]"
    )
    print(
        f"ZNE objective evaluation: {_format_seconds(float(measured.get('objective_zne_seconds', np.nan)))} "
        f"[hadamard pubs={measured.get('objective_zne_hadamard_pub_count')}, "
        f"d pubs={measured.get('objective_zne_d_pub_count')}]"
    )

    setup_seconds = float(measured.get("one_time_setup_seconds", np.nan))
    if np.isfinite(setup_seconds):
        print(f"one-time setup before timed evaluations: {_format_seconds(setup_seconds)}")

    component_breakdown = dict(measured.get("component_breakdown", {}))
    if component_breakdown:
        no_zne = dict(component_breakdown.get("no_zne", {}))
        zne = dict(component_breakdown.get("zne", {}))
        print(
            "breakdown (extra hardware jobs): "
            f"no-ZNE hadamard={_format_seconds(float(no_zne.get('hadamard_test_seconds', np.nan)))}, "
            f"no-ZNE d={_format_seconds(float(no_zne.get('d_term_seconds', np.nan)))} | "
            f"ZNE hadamard={_format_seconds(float(zne.get('hadamard_test_seconds', np.nan)))}, "
            f"ZNE d={_format_seconds(float(zne.get('d_term_seconds', np.nan)))}"
        )

    print(
        f"estimated no-ZNE branch runtime for maxiter={results.get('estimated_maxiter')}: "
        f"{_format_seconds(float(estimates.get('no_zne_branch_seconds', np.nan)))}"
    )
    print(
        f"estimated ZNE branch runtime for maxiter={results.get('estimated_maxiter')}: "
        f"{_format_seconds(float(estimates.get('zne_branch_seconds', np.nan)))}"
    )
    print(
        "estimated total IBM hardware runtime: "
        f"{_format_seconds(float(estimates.get('hardware_total_seconds', np.nan)))} "
        f"(excluding queue variability)"
    )


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = _resolve_seeds(
        num_seeds=int(args.num_seeds),
        seed_offset=int(args.seed_offset),
        explicit_seeds=args.seeds,
    )
    if len(seeds) != 1:
        raise ValueError("Pass exactly one seed for the IBM timing probe.")
    seed = int(seeds[0])
    zne_noise_factors = _parse_float_list(
        args.zne_noise_factors,
        default=(1.0, 3.0, 5.0),
    )

    run_dir = _build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "probe_results.json"

    overall_start = time.perf_counter()
    backend, backend_info = _resolve_ibm_runtime_backend(
        config_path=args.ibm_config,
        backend_override=args.ibm_backend,
        channel_override=args.ibm_channel,
        instance_override=args.ibm_instance,
        url_override=args.ibm_url,
        token_override=args.ibm_token,
    )

    X, y, _ = _generate_synthetic_data(
        n_samples=int(args.n_samples),
        n_features=int(args.n_features),
        noise_std=float(args.noise_std),
        w_true_std=float(args.w_true_std),
        seed=seed,
    )
    split = _split_dataset_indices(
        n_samples=int(args.n_samples),
        prior_ratio=float(args.prior_ratio),
        train_ratio=float(args.train_ratio),
        test_ratio=float(args.test_ratio),
        seed=seed,
    )
    X_prior = np.asarray(X[split.prior_indices], dtype=float)
    y_prior = np.asarray(y[split.prior_indices], dtype=float)
    X_train = np.asarray(X[split.train_indices], dtype=float)
    y_train = np.asarray(y[split.train_indices], dtype=float)

    m0, Sigma0, V, prior_info = _estimate_prior_from_prior_split(
        X_prior,
        y_prior,
        seed=seed,
        bootstrap_samples=int(args.prior_bootstrap_samples),
        ridge=float(args.prior_ridge),
        sigma_floor=float(args.prior_sigma_floor),
        v_floor=float(args.V_floor),
    )

    prepared = _prepare_vqbr_model(
        X=X_train,
        y=y_train,
        m0=m0,
        Sigma0=Sigma0,
        V=V,
        reps=int(args.vqbr_reps),
        entanglement=str(args.vqbr_entanglement),
        seed=seed,
    )
    theta_init = np.asarray(
        prepared.model._make_initial_point(prepared.model._num_params),
        dtype=float,
    ).reshape(-1)
    full_batch = np.arange(int(X_train.shape[0]), dtype=int)

    no_zne_estimator, no_zne_info = _build_ibm_estimator(
        backend=backend,
        shots=int(args.ibm_shots),
        seed=seed,
        use_zne=False,
        zne_noise_factors=zne_noise_factors,
        zne_extrapolator=str(args.zne_extrapolator),
        zne_measure_mitigation=bool(args.zne_measure_mitigation),
    )
    no_zne_evaluator = _PrimitiveObjectiveEvaluator(
        prepared=prepared,
        estimator=no_zne_estimator,
        method_key="vqbr_ibm_no_zne_probe",
        method_label="IBM objective probe (w/o ZNE)",
        backend_name=str(backend_info.get("backend_name", _backend_name(backend))),
        transpile_backend=backend,
        transpile_optimization_level=int(args.ibm_transpile_optimization_level),
    )

    zne_baseline_estimator, zne_baseline_info = _build_ibm_estimator(
        backend=backend,
        shots=int(args.ibm_shots),
        seed=seed,
        use_zne=False,
        zne_noise_factors=zne_noise_factors,
        zne_extrapolator=str(args.zne_extrapolator),
        zne_measure_mitigation=bool(args.zne_measure_mitigation),
    )
    zne_estimator, zne_estimator_info = _build_ibm_estimator(
        backend=backend,
        shots=int(args.ibm_shots),
        seed=seed,
        use_zne=True,
        zne_noise_factors=zne_noise_factors,
        zne_extrapolator=str(args.zne_extrapolator),
        zne_measure_mitigation=bool(args.zne_measure_mitigation),
    )
    zne_evaluator = _ZNEObjectiveEvaluator(
        prepared=prepared,
        estimators=[zne_estimator],
        baseline_estimator=zne_baseline_estimator,
        noise_factors=zne_noise_factors,
        zne_extrapolator=str(args.zne_extrapolator),
        method_key="vqbr_ibm_zne_probe",
        method_label="IBM objective probe (w/ ZNE)",
        backend_name=str(backend_info.get("backend_name", _backend_name(backend))),
        transpile_backend=backend,
        transpile_optimization_level=int(args.ibm_transpile_optimization_level),
    )
    setup_seconds = float(time.perf_counter() - overall_start)

    no_zne_seconds, no_zne_snapshot = _time_call(
        lambda: no_zne_evaluator.evaluate_snapshot(
            theta_init,
            full_batch,
            scale_batch_to_full=False,
        )
    )
    zne_seconds, zne_snapshot = _time_call(
        lambda: zne_evaluator.evaluate_snapshot(
            theta_init,
            full_batch,
            scale_batch_to_full=False,
        )
    )

    no_zne_pubs = _build_hadamard_test_pubs(
        no_zne_evaluator,
        theta_vec=theta_init,
        batch_indices=full_batch,
    )
    zne_pubs = _build_hadamard_test_pubs(
        zne_evaluator,
        theta_vec=theta_init,
        batch_indices=full_batch,
    )
    d_pub = _build_d_term_pub(no_zne_evaluator, theta_vec=theta_init)

    objective_eval_count = int(args.estimated_maxiter) + 1
    estimated_no_zne_branch = float(objective_eval_count * no_zne_seconds)
    estimated_zne_branch = float(objective_eval_count * zne_seconds)
    estimated_total = float(estimated_no_zne_branch + estimated_zne_branch)

    measured: Dict[str, Any] = {
        "one_time_setup_seconds": float(setup_seconds),
        "objective_no_zne_seconds": float(no_zne_seconds),
        "objective_zne_seconds": float(zne_seconds),
        "objective_no_zne_pub_count": int(len(no_zne_pubs) + len(d_pub)),
        "objective_zne_hadamard_pub_count": int(len(zne_pubs)),
        "objective_zne_d_pub_count": int(len(d_pub)),
        "objective_no_zne_snapshot": {
            "L_tilde": float(getattr(no_zne_snapshot, "L_tilde", np.nan)),
            "a_hat": float(getattr(no_zne_snapshot, "a_hat", np.nan)),
            "c_hat": float(getattr(no_zne_snapshot, "c_hat", np.nan)),
            "d_hat": float(getattr(no_zne_snapshot, "d_hat", np.nan)),
            "e_hat": float(getattr(no_zne_snapshot, "e_hat", np.nan)),
        },
        "objective_zne_snapshot": {
            "L_tilde": float(getattr(zne_snapshot, "L_tilde", np.nan)),
            "a_hat": float(getattr(zne_snapshot, "a_hat", np.nan)),
            "c_hat": float(getattr(zne_snapshot, "c_hat", np.nan)),
            "d_hat": float(getattr(zne_snapshot, "d_hat", np.nan)),
            "e_hat": float(getattr(zne_snapshot, "e_hat", np.nan)),
        },
    }
    if bool(args.component_breakdown):
        measured["component_breakdown"] = {
            "no_zne": _component_breakdown_no_zne(
                evaluator=no_zne_evaluator,
                theta_init=theta_init,
                full_batch=full_batch,
            ),
            "zne": _component_breakdown_zne(
                evaluator=zne_evaluator,
                theta_init=theta_init,
                full_batch=full_batch,
            ),
        }

    results = {
        "script": str(Path(__file__).resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "n_samples": int(args.n_samples),
            "n_features": int(args.n_features),
            "noise_std": float(args.noise_std),
            "w_true_std": float(args.w_true_std),
            "prior_ratio": float(args.prior_ratio),
            "train_ratio": float(args.train_ratio),
            "test_ratio": float(args.test_ratio),
            "seed": int(seed),
            "train_size": int(X_train.shape[0]),
            "prior_size": int(X_prior.shape[0]),
        },
        "ibm": {
            "backend_name": str(backend_info.get("backend_name", "")),
            "shots": int(args.ibm_shots),
            "transpile_optimization_level": int(args.ibm_transpile_optimization_level),
            "zne_noise_factors": [float(x) for x in zne_noise_factors],
            "zne_extrapolator": str(args.zne_extrapolator),
            "zne_measure_mitigation": bool(args.zne_measure_mitigation),
            "resolved_info": backend_info,
            "no_zne_estimator": no_zne_info,
            "zne_estimator": zne_estimator_info,
            "zne_baseline_estimator": zne_baseline_info,
        },
        "prior": {key: float(value) for key, value in prior_info.items()},
        "vqbr": {
            "optimizer": str(ENFORCED_VQBR_OPTIMIZER),
            "loss": str(ENFORCED_VQBR_LOSS),
            "reps": int(args.vqbr_reps),
            "entanglement": str(args.vqbr_entanglement),
            "su2_gates": list(ENFORCED_VQBR_SU2_GATES),
        },
        "measured": measured,
        "estimated_maxiter": int(args.estimated_maxiter),
        "estimated_runtime_seconds": {
            "no_zne_branch_seconds": float(estimated_no_zne_branch),
            "zne_branch_seconds": float(estimated_zne_branch),
            "hardware_total_seconds": float(estimated_total),
            "hardware_total_plus_setup_seconds": float(estimated_total + setup_seconds),
            "objective_evaluations_per_branch": int(objective_eval_count),
        },
        "artifacts": {
            "output_json": str(output_path),
            "run_dir": str(run_dir),
        },
        "config": {
            "argv": _sanitize_argv(sys.argv),
            "args": _sanitize_args_for_results(args),
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    _print_probe_summary(results)
    print(f"Saved probe results JSON to: {output_path}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe how long one real IBM hardware objective evaluation takes for the "
            "VQBR real-hardware ablation, with and without ZNE, then estimate the full "
            "hardware runtime from those measurements."
        )
    )
    parser.add_argument("--N", "--n-samples", dest="n_samples", type=int, default=50)
    parser.add_argument("--D", "--n-features", dest="n_features", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--w-true-std", type=float, default=1.0)
    parser.add_argument("--prior-ratio", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--prior-bootstrap-samples", type=int, default=64)
    parser.add_argument("--prior-ridge", type=float, default=1e-6)
    parser.add_argument("--prior-sigma-floor", type=float, default=1e-4)
    parser.add_argument("--V-floor", type=float, default=1e-8)
    parser.add_argument("--vqbr-reps", type=int, default=2)
    parser.add_argument("--vqbr-entanglement", type=str, default="linear")
    parser.add_argument("--estimated-maxiter", type=int, default=100)

    parser.add_argument("--ibm-config", type=str, default=str(DEFAULT_IBM_CONFIG_PATH))
    parser.add_argument("--ibm-backend", type=str, default="ibm_kingston")
    parser.add_argument("--ibm-channel", type=str, default="")
    parser.add_argument("--ibm-instance", type=str, default="")
    parser.add_argument("--ibm-url", type=str, default="")
    parser.add_argument("--ibm-token", type=str, default="")
    parser.add_argument("--ibm-shots", type=int, default=1024)
    parser.add_argument("--ibm-transpile-optimization-level", type=int, default=1)
    parser.add_argument("--zne-noise-factors", type=str, default="1,3,5")
    parser.add_argument("--zne-extrapolator", type=str, default="linear")
    parser.set_defaults(zne_measure_mitigation=False)
    parser.add_argument("--zne-measure-mitigation", action="store_true")

    parser.set_defaults(component_breakdown=False)
    parser.add_argument(
        "--component-breakdown",
        action="store_true",
        help=(
            "Run extra hardware jobs to separately time the Hadamard-test portion and the "
            "d-term portion for each evaluation type."
        ),
    )

    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--console-log-path", type=str, default="run.log")
    return parser


class _TeeStream:
    def __init__(self, *streams: object):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = _build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    console_log_raw = str(args.console_log_path).strip()
    if not console_log_raw:
        run_probe(args)
        return

    console_log_path = run_dir / Path(console_log_raw).name
    with console_log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        tee_stdout = _TeeStream(sys.__stdout__, log_file)
        tee_stderr = _TeeStream(sys.__stderr__, log_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print(f"Console log file: {console_log_path}")
            run_probe(args)


if __name__ == "__main__":
    main()
