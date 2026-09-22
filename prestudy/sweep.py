"""
Layer Duplication / Skip Sweep.

The mechanism is layer *rerouting* (contiguous duplication or skip of a
layer block), driven by `re_polar.core.layer_engine.LayerEngine`:

  - Duplication (i < j, upper triangle):  layers i..j-1 execute twice
  - Skip        (i > j, lower triangle):  layers j..i-1 are omitted

Both use the unified path  [0..j-1] + [i..N-1]  from LayerEngine.

For each (i, j) config:
  1. Reroute the model via LayerEngine.apply_layer_rerouting()
  2. Run the mmlu_pro_domains benchmark (the paper's own 6-domain subset)
  3. Record delta = score - baseline
  4. Restore the original layer stack

Results are saved as JSON (dup = upper triangle, skip = lower triangle of
one N+1 x N+1 delta matrix, diagonal = baseline) -- the numbers behind the
paper's motivation-section figure on repeat/skip accuracy by layer block
(Qwen3-8B, mmlu_pro_domains 6-domain subset); see `prestudy/run.py` for the
CLI.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from re_polar.core.layer_engine import LayerEngine
from re_polar.core.mmlu_pro_domain_eval import (
    load_mmlu_pro_domain_samples,
    prepare_mmlu_pro_domain_inputs,
    run_mmlu_pro_domains,
)

VALID_MODES = ("dup", "skip")


class LayerDuplicationSweep:
    """
    Sweeps all (i, j) duplication and/or skip configurations for a model,
    measures mmlu_pro_domains accuracy deltas vs. the unmodified baseline.

    Args:
        engine:          loaded LayerEngine (holds model/tokenizer/device/num_layers)
        output_dir:      where to write JSON results
        modes:           subset of ("dup", "skip")
        n_per_domain:    mmlu_pro_domains samples per domain (paper default: 200)
        batch_size:      mmlu_pro_domains forward-pass batch size
        no_think:        passed through to prompt formatting
        num_shards:      total number of workers (for distributed runs)
        shard_index:     this worker's index (0-based)
        stride:          keep every `stride`-th config (subsample a sweep); 1 = all
        pretokenize:     tokenize prompts once before the sweep
    """

    def __init__(
        self,
        engine: LayerEngine,
        output_dir: str = "./prestudy/results",
        modes: Tuple[str, ...] = VALID_MODES,
        n_per_domain: int = 200,
        batch_size: int = 8,
        no_think: bool = True,
        num_shards: int = 1,
        shard_index: int = 0,
        stride: int = 1,
        pretokenize: bool = True,
        explicit_configs: Optional[List[Tuple[str, int, int]]] = None,
    ):
        bad = set(modes) - set(VALID_MODES)
        if bad:
            raise ValueError(f"Unknown modes {bad}; valid: {VALID_MODES}")
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.modes = tuple(modes)
        self.n_per_domain = n_per_domain
        self.batch_size = batch_size
        self.no_think = no_think
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.stride = max(1, stride)
        self.explicit_configs = (
            [(str(c[0]), int(c[1]), int(c[2])) for c in explicit_configs]
            if explicit_configs is not None else None
        )

        self._prepared_inputs = None
        if pretokenize:
            print("Loading + pre-tokenizing mmlu_pro_domains prompts ...")
            samples = load_mmlu_pro_domain_samples(n_per_domain=self.n_per_domain)
            self._prepared_inputs = prepare_mmlu_pro_domain_inputs(
                engine.tokenizer, samples, no_think=self.no_think, device=engine.device)

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def measure_baseline(self) -> Dict[str, float]:
        print("Measuring baseline (original model) ...")
        result = self._run_benchmark()
        scores = self._scores_only(result)
        for name, score in scores.items():
            print(f"  Baseline {name}: {score:.4f}")
        return scores

    def _run_benchmark(self) -> Dict:
        return run_mmlu_pro_domains(
            self.engine.model, self.engine.tokenizer, batch_size=self.batch_size,
            no_think=self.no_think, prepared_inputs=self._prepared_inputs,
            return_details=False)

    def _scores_only(self, result: Dict) -> Dict[str, float]:
        """{"mmlu_pro_domains": aggregate, "mmlu_pro_domains_<domain>": per-domain, ...}."""
        scores = {"mmlu_pro_domains": float(result["average"])}
        for domain_key, domain_r in result.get("per_domain", {}).items():
            scores[f"mmlu_pro_domains_{domain_key}"] = float(domain_r["average"])
        return scores

    # ------------------------------------------------------------------
    # Config enumeration
    # ------------------------------------------------------------------

    def _iter_configs(self, mode: str) -> List[Tuple[int, int]]:
        n = self.engine.num_layers
        if mode == "dup":
            # generate_dup_configs yields the (0,0) baseline + all (i<j); drop the
            # i==j baseline here (delta is 0 by definition, no rerouting needed).
            return [(i, j) for (i, j) in LayerEngine.generate_dup_configs(n) if i != j]
        return list(LayerEngine.generate_skip_configs(n))

    def _apply(self, mode: str, i: int, j: int) -> None:
        path = self.engine.get_dup_path(i, j) if mode == "dup" else self.engine.get_skip_path(i, j)
        self.engine.apply_layer_rerouting(path)

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------

    def run(
        self,
        baseline_scores: Optional[Dict[str, float]] = None,
        quick_test: int = 0,
    ) -> Dict:
        if baseline_scores is None:
            baseline_scores = self.measure_baseline()

        # Build the full work list across requested modes, then shard / stride / quick-test.
        full: List[Tuple[str, int, int]] = []
        total_per_mode: Dict[str, int] = {}
        if self.explicit_configs is not None:
            full = list(self.explicit_configs)
            for mode, i, j in full:
                total_per_mode[mode] = total_per_mode.get(mode, 0) + 1
        else:
            for mode in self.modes:
                cfgs = self._iter_configs(mode)
                total_per_mode[mode] = len(cfgs)
                cfgs = cfgs[:: self.stride]
                full.extend((mode, i, j) for (i, j) in cfgs)

        my_work = [c for idx, c in enumerate(full) if idx % self.num_shards == self.shard_index]
        if quick_test > 0:
            my_work = random.sample(my_work, min(quick_test, len(my_work)))

        stride_note = f", stride={self.stride}" if self.stride > 1 else ""
        print(
            f"Sweep modes={self.modes} total={sum(total_per_mode.values())} "
            f"-> running {len(my_work)} configs "
            f"(shard {self.shard_index}/{self.num_shards}{stride_note}) ..."
        )

        sweep_results: Dict[str, Dict] = {}
        for idx, (mode, i, j) in enumerate(my_work):
            key = f"{mode}:({i},{j})"
            t0 = time.time()

            self._apply(mode, i, j)
            result = self._run_benchmark()
            self.engine.restore_original()

            scores = self._scores_only(result)
            entry: Dict[str, float] = {"mode": mode, "i": i, "j": j}
            deltas = []
            for name, score in scores.items():
                entry[f"{name}_score"] = score
                d = score - baseline_scores.get(name, 0.0)
                entry[f"{name}_delta"] = d
                if not name.startswith("mmlu_pro_domains_"):  # only the aggregate feeds combined_delta
                    deltas.append(d)
            entry["combined_delta"] = float(np.mean(deltas)) if deltas else 0.0
            elapsed = time.time() - t0
            entry["elapsed_s"] = round(elapsed, 1)
            sweep_results[key] = entry

            print(
                f"  [{idx+1}/{len(my_work)}] {key:18s} "
                f"mmlu_pro_domains:{scores['mmlu_pro_domains']:.4f}"
                f"(Δ{entry['mmlu_pro_domains_delta']:+.4f})  [{elapsed:.0f}s]"
            )
            self._save_partial(baseline_scores, sweep_results, total_per_mode)

        return self._build_output(baseline_scores, sweep_results, total_per_mode)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _meta(self, baseline_scores, sweep_results, total_per_mode, partial: bool) -> Dict:
        return {
            "model_id": self.engine.model_id,
            "num_layers": self.engine.num_layers,
            "modes": list(self.modes),
            "baseline_scores": baseline_scores,
            "benchmark": "mmlu_pro_domains",
            "n_per_domain": self.n_per_domain,
            "total_configs": int(sum(total_per_mode.values())),
            "total_per_mode": total_per_mode,
            "completed_configs": len(sweep_results),
            "stride": self.stride,
            "shard": f"{self.shard_index}/{self.num_shards}",
            "partial": partial,
        }

    def _tag(self) -> str:
        return self.engine.model_id.replace("/", "_").replace(".", "p")

    def _shard_tag(self) -> str:
        return f"_shard{self.shard_index}of{self.num_shards}" if self.num_shards > 1 else ""

    def _save_partial(self, baseline_scores, sweep_results, total_per_mode) -> None:
        out = {
            "metadata": self._meta(baseline_scores, sweep_results, total_per_mode, partial=True),
            "results": sweep_results,
        }
        partial_path = self.output_dir / f"layer_duplication_{self._tag()}{self._shard_tag()}_partial.json"
        tmp_path = partial_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(partial_path)  # atomic on POSIX

    def _build_output(self, baseline_scores, sweep_results, total_per_mode) -> Dict:
        out = {
            "metadata": self._meta(baseline_scores, sweep_results, total_per_mode, partial=False),
            "results": sweep_results,
        }
        tag, shard_tag = self._tag(), self._shard_tag()
        ts = time.strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"layer_duplication_{tag}{shard_tag}_{ts}.json"
        json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Results saved -> {json_path}")
        return out
