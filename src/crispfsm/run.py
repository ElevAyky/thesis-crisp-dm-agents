# src/crispfsm/run.py
from __future__ import annotations
import argparse, os, shutil, time, uuid, json
from dataclasses import asdict
from crispfsm.config import RunConfig, THRESHOLD_K
from crispfsm.scoring import baseline_f1_on_holdout, compare_part_to_baseline, score_part
from crispfsm.state import new_state
from crispfsm.datasets import prepare_from_manifest
from crispfsm.leakage import leak_reliance
from crispfsm.transcripts import audit_access, collect_transcripts
from crispfsm.backend import TokenUsage, ClaudeCliBackend, RateLimitError

# Exit-Code bei Abo-Kontingent-Ende (EX_TEMPFAIL): der Runner wartet und fährt
# denselben Run neu — der Run zählt nicht (Johannes 18.09.). 1 = sonstiger ERROR.
EXIT_RATE_LIMITED = 75
from crispfsm.pricing import cost_usd
from crispfsm.graph import build_graph
from crispfsm.manifests import load_manifest


def reset_workdir(workdir: str) -> None:
    """Neustart eines Runs (ERROR/crash oder rate_limited) muss leer beginnen:
    Artefakte des abgebrochenen Versuchs (bu/, du/, predictions ...) würden sonst
    vom neuen Worker vorgefunden. Ein gemessener Run (summary ohne ERROR) wird
    nie überschrieben."""
    if not os.path.exists(workdir):
        return
    summary = os.path.join(workdir, "summary.json")
    if os.path.exists(summary):
        try:
            with open(summary) as f:
                outcome = json.load(f).get("outcome")
        except (OSError, ValueError):
            outcome = None
        if outcome not in (None, "ERROR"):
            raise SystemExit(f"{workdir} ist bereits gemessen ({outcome}) — nicht überschrieben.")
    shutil.rmtree(workdir)


def outcome_label(final: dict) -> str:
    """Vier Terminal-Zustände. ERROR (Decker 14.09.): der Harness selbst ist
    gecrasht oder nicht terminiert — kein Agent-Befund, nicht in die SUCCESS-Rate
    mischen, sondern re-runnen (Runner: ERROR-Summaries gelten als nicht gemessen)."""
    if final.get("error"):
        return "ERROR"
    if final["success"]:
        return "SUCCESS"
    return "ABSTAIN" if final.get("abstained") else "FAIL"


def fail_reason(final: dict, *, safety_max_calls: int) -> str | None:
    """Benannte Ursache für jeden Nicht-SUCCESS (Decker 14.09.: „den Grund festhalten").
    Zerlegt die SUCCESS-Rate in: abstain · form_error · spec_mismatch · below_threshold ·
    gate_failed · safety_max_calls; ERROR-Gründe (nicht gewertet, re-run): crash ·
    recursion_limit · rate_limited. Reihenfolge = Präzedenz."""
    if final.get("success"):
        return None
    if final.get("error"):
        return final["error"].get("reason", "crash")
    if final.get("abstained"):
        return "abstain"
    if final.get("budget_exhausted"):       # S4: gemeinsames Run-Kontingent aufgebraucht
        return "budget_exhausted"
    if final.get("total_calls", 0) >= safety_max_calls:
        return "safety_max_calls"
    metrics = final.get("gate_metrics") or {}
    if metrics.get("form_error"):
        return "form_error"
    if metrics.get("spec_mismatch"):
        return "spec_mismatch"
    if final.get("current_phase") in ("eval", "react") and metrics.get("f1") is not None:
        return "below_threshold"
    return "gate_failed"


def predictions_path(cfg: RunConfig, workdir: str) -> str:
    """Final abgegebene Predictions: C/D im Arbeitsordner, A/B aus der Eval-Phase."""
    if cfg.arm in ("crisp_react", "naked_react"):
        return os.path.join(workdir, "work", "predictions.parquet")
    return os.path.join(workdir, "work", "eval", "predictions.parquet")


def final_verdict(cfg: RunConfig, final: dict, workdir: str, *, safety_max_calls: int) -> dict:
    """Endurteil — fuer ALLE Arme dieselbe Funktion, immer auf dem TEST-Anteil (S1, 23.09.).

    SUCCESS <=> (1) der Prozess ist regulaer zu Ende gelaufen (A/B: Evaluation-Gate
    bestanden; C/D: ein Deliverable abgegeben) und hat nicht abstiniert, (2) die finalen
    Predictions sind formal gueltig, (3) die Zielspalte wurde korrekt bestimmt (B3) und
    (4) sie sind auf den Testzeilen nicht signifikant schlechter als die Referenz-Baseline
    (gepaarter Bootstrap) bzw. — nur ohne Baseline-Regel — ueber der Punkt-Schwelle.
    f1_test ist der F1 der final abgegebenen Predictions (Stufe 2, Friedman), unabhaengig
    vom Urteil — nicht das Beste ueber mehrere Versuche (waere Auswahl auf dem Test-Set)."""
    free_arm = cfg.arm in ("crisp_react", "naked_react")
    pred_path = predictions_path(cfg, workdir)
    holdout_dir = os.path.join(workdir, "holdout")
    abstained = bool(final.get("abstained"))
    process_ok = not abstained and (
        free_arm or (final.get("gate_passed") and final.get("current_phase") == "eval"))

    score = score_part(pred_path, holdout_dir, "test", target=cfg.target)
    f1_test = score.f1 if score.ok else None
    comparison = None
    if score.ok and cfg.baseline_pred_path and cfg.eval_metric == "f1":
        comparison = compare_part_to_baseline(pred_path, holdout_dir, "test",
                                              cfg.baseline_pred_path, target=cfg.target)
        quality_ok = comparison.noninferior
    else:
        metric_val = {"f1": score.f1, "accuracy": score.accuracy}.get(cfg.eval_metric)
        quality_ok = score.ok and metric_val is not None and metric_val >= cfg.eval_f1_threshold
    target_ok = (final.get("target_correct") if free_arm else final.get("bu_correct")) is True

    success = bool(process_ok and score.ok and target_ok and quality_ok)
    if success:
        reason = None
    elif abstained or not process_ok:
        reason = fail_reason({**final, "success": False}, safety_max_calls=safety_max_calls)
    elif not score.ok:
        reason = "form_error"
    elif not target_ok:
        reason = "wrong_target"
    else:
        reason = "below_threshold"
    return {"success": success, "fail_reason": reason, "f1_test": f1_test,
            "final_f1": f1_test if success else None, "process_completed": bool(process_ok),
            "target_ok": target_ok,
            "test_comparison": asdict(comparison) if comparison is not None else None}


def harness_version() -> dict:
    """Stand des Harness, mit dem ein Run lief (S8, 23.09.): Git-Commit, ob src/ oder
    manifests/ uncommittete Aenderungen hatten, und ein Fingerabdruck aller Dateien mit
    LLM-seitigen Texten (Prompts, Pruefer, Gate-/Scoring-Feedback). So ist fuer jeden Run
    belegbar, welcher Wortlaut galt — auch ohne Git."""
    import hashlib, subprocess
    src_dir = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(src_dir))
    h = hashlib.sha256()
    for name in ("prompts.py", "softgates.py", "gates.py", "scoring.py"):
        with open(os.path.join(src_dir, name), "rb") as f:
            h.update(f.read())
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "manifests"],
                                    cwd=repo, capture_output=True, text=True,
                                    timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    # Befund 24.10. (docs/befund-auto-memory.md): CLI-Version + Gedaechtnis-Sperre pro Run
    # belegen — Paritaet zwischen Armen und Abgrenzung zu den Runs mit geteiltem Memory.
    from crispfsm.backend import WORKER_ENV, WORKER_SETTINGS
    try:
        claude_version = subprocess.run([shutil.which("claude") or "claude", "--version"],
                                        capture_output=True, text=True,
                                        timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        claude_version = None
    memory_locked = (WORKER_SETTINGS.get("autoMemoryEnabled") is False
                     and WORKER_ENV.get("CLAUDE_CODE_DISABLE_AUTO_MEMORY") == "1")
    git_locked = WORKER_ENV.get("CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS") == "1"
    return {"commit": commit, "dirty": dirty, "prompt_sha256": h.hexdigest(),
            "claude_version": claude_version,
            "auto_memory": "disabled" if memory_locked else "enabled",
            "git_context": "disabled" if git_locked else "enabled"}


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="CRISP-DM FSM Walking Skeleton")
    ap.add_argument("--dataset", default="breast_cancer")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--eval-f1-threshold", type=float, default=None,
                    help="expliziter Override der Erfolgsschwelle (Debug). Default: "
                         "THRESHOLD_K x baseline_f1 aus dem Manifest (k = 1, Decker 14.09.)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--poisoned", action="store_true",
                    help="poisoned_variant aus dem Manifest fahren (erwartet: ABSTAIN)")
    ap.add_argument("--leakage", action="store_true",
                    help="Leakage-Spalte injizieren (Train perfekt korreliert, Test "
                         "zufällig) — Messpunkt leakage_detected via DU-Report")
    ap.add_argument("--leakage-variant", choices=["perfect", "spurious"], default="perfect",
                    help="mit --leakage: perfect = Tarnname, perfekt korreliert (Batch 1); "
                         "spurious = fachfremder Name, ~85 %% (Batch 2, docs/leakage-runs.md §7)")
    ap.add_argument("--manifests-dir", default="manifests")
    ap.add_argument("--arm", choices=["hard_fsm", "soft_fsm", "crisp_react", "naked_react",
                                      "hard_fsm_single"],
                    default="hard_fsm",
                    help="Experiment-Arm: hard_fsm (A) | soft_fsm (B) | "
                         "crisp_react (C) | naked_react (D) | hard_fsm_single "
                         "(A-Single, H1: wie A, eine Worker-Session fuer alle Phasen)")
    ap.add_argument("--react-max-turns", type=int, default=None,
                    help="Arm C/D: Turn-Budget der Einzel-Session "
                         "(Default: Anzahl FSM-Phasen x max-turns)")
    args = ap.parse_args()

    # Manifest laden: task_prompt ans LLM, canonical_spec.metric nur als Eval-Config
    # (canonical_spec-Werte dürfen NIE in einen Prompt lecken).
    manifest = load_manifest(args.dataset, manifests_dir=args.manifests_dir)
    if args.poisoned:
        if manifest.poisoned_variant is None:
            ap.error(f"Manifest '{args.dataset}' hat keine poisoned_variant.")
        task_prompt = manifest.poisoned_variant.task_prompt
    else:
        task_prompt = manifest.task_prompt

    # Erfolgsschwelle: baseline-relativ aus dem Manifest (k = 1), CLI-Flag nur als
    # expliziter Override. Kein stilles 0.5 mehr — ohne baseline_f1 und ohne Flag
    # bricht der Run ab, sonst wäre SUCCESS je Dataset etwas anderes.
    # Erfolgsregel seit 23.09.: gepaarter Bootstrap gegen die Default-Baseline auf
    # demselben Holdout (Baseline-Predictions versioniert unter
    # <manifests-dir>/baseline_preds/, erzeugt von baseline_difficulty.py --holdout).
    # Die Punkt-Schwelle bleibt nur fuer --eval-f1-threshold (Debug/Skelett).
    baseline_src = None
    if args.eval_f1_threshold is not None:
        threshold, threshold_source = args.eval_f1_threshold, "cli"
    elif manifest.baseline_f1 is not None:
        threshold, threshold_source = THRESHOLD_K * manifest.baseline_f1, "manifest_baseline"
        # Je Anteil eine Datei (S1): test = Endurteil aller Arme, val = A/B-Gates
        baseline_src = {part: os.path.join(args.manifests_dir, "baseline_preds",
                                           f"{args.dataset}.{part}.parquet")
                        for part in ("test", "val")}
        missing = [p for p in baseline_src.values() if not os.path.exists(p)]
        if missing or manifest.baseline_f1_val is None:
            ap.error(f"Baseline-Predictions/baseline_f1_val fehlen ({missing}) — erzeugen mit "
                     f"`python scripts/baseline_difficulty.py --holdout --write-manifests`.")
    else:
        ap.error(f"Manifest '{args.dataset}' hat kein baseline_f1 und --eval-f1-threshold "
                 f"fehlt — Schwelle nicht bestimmbar (benchmarks.md §2 messen).")
    success_rule = "noninferior_paired" if baseline_src else "threshold"

    run_id = args.run_id or f"{args.dataset}-{uuid.uuid4().hex[:8]}"
    workdir = os.path.abspath(os.path.join(args.runs_dir, run_id))
    reset_workdir(workdir)
    os.makedirs(os.path.join(workdir, "work"))

    cfg = RunConfig(run_id=run_id, workdir=workdir, dataset=args.dataset,
                    model=args.model, eval_f1_threshold=threshold,
                    task_prompt=task_prompt, eval_metric=manifest.canonical_spec.metric,
                    arm=args.arm, react_max_turns=args.react_max_turns)

    # Target neutralisieren und den kanonischen Namen VOR build_graph/new_state in cfg
    # setzen — die Phasen-Nodes und der State-Snapshot lesen cfg.target genau hier ab.
    info = prepare_from_manifest(manifest, cfg.workdir, inject_leakage=args.leakage,
                                 leak_variant=args.leakage_variant)
    leakage_variant = args.leakage_variant if args.leakage else None
    cfg.target = info["target_column"]
    print(f"[setup] {cfg.dataset}: {info}")
    print(f"[setup] workdir = {workdir}")

    if baseline_src:
        # Baseline-Predictions in den Holdout des Runs (ausserhalb der Worker-cwd)
        # und Ausrichtung pruefen: ihr F1 auf DIESEM Holdout muss baseline_f1 treffen,
        # sonst stammen sie von einem anderen Split -> harter Abbruch, kein Run.
        cfg.baseline_pred_path = os.path.join(workdir, "holdout", "baseline_test.parquet")
        cfg.baseline_val_path = os.path.join(workdir, "holdout", "baseline_val.parquet")
        for part, dst, expected in (("test", cfg.baseline_pred_path, manifest.baseline_f1),
                                    ("val", cfg.baseline_val_path, manifest.baseline_f1_val)):
            shutil.copy(baseline_src[part], dst)
            aligned_f1 = baseline_f1_on_holdout(dst, os.path.join(workdir, "holdout", f"{part}.parquet"),
                                                target=cfg.target)
            if abs(aligned_f1 - expected) > 6e-4:
                raise SystemExit(f"Baseline-Predictions passen nicht zum Holdout ({part}): F1 "
                                 f"{aligned_f1:.4f} vs. Manifest {expected} ({baseline_src[part]})")

    # Kontaminations-Baseline: SHA-256 der Holdout-Dateien direkt nach dem Setup,
    # bevor irgendein Worker läuft (Spec-Nachtrag §0). Auch test_features.parquet:
    # ein Worker, der die Feature-Zeilen umsortiert, würde die Messung sonst still
    # korrumpieren (Predictions vs. Wahrheit zeilenweise fehlausgerichtet). Seit
    # 23.09. auch die Baseline-Predictions: wer sie ueberschreibt, senkt die Latte —
    # und Validierungsanteil + Zeilen-Index (S1).
    holdout_paths = [os.path.join(workdir, "holdout", name)
                     for name in ("test.parquet", "test_features.parquet", "val.parquet",
                                  "eval_index.json")
                     if os.path.exists(os.path.join(workdir, "holdout", name))]
    holdout_paths += [p for p in (cfg.baseline_pred_path, cfg.baseline_val_path) if p]
    holdout_hashes = [_sha256(p) for p in holdout_paths]

    backend = ClaudeCliBackend(model=cfg.model)
    if cfg.arm in ("crisp_react", "naked_react"):
        from crispfsm.react import build_react_graph
        graph = build_react_graph(backend, cfg)
    else:
        graph = build_graph(backend, cfg)
    # ERROR-State (Decker 14.09.): Crash im Backend/Harness oder Nicht-Terminierung
    # (LangGraph-recursion_limit) landen in einem expliziten Terminal-Zustand mit
    # Grund — kein fehlendes summary.json. Exit 1, damit der Runner den Run als
    # nicht gemessen erkennt (Harness-Fehler ≠ Agent-Fehler → re-run, nicht zählen).
    run_started = time.time()   # Transkripte dieses Runs = ab hier geaendert (transcripts.py)
    try:
        final = graph.invoke(new_state(cfg), config={"recursion_limit": cfg.recursion_limit})
    except Exception as e:  # noqa: BLE001 — bewusst breit: jeder Crash wird protokolliert
        import sys, traceback
        from langgraph.errors import GraphRecursionError
        if isinstance(e, RateLimitError):
            reason = "rate_limited"
        elif isinstance(e, GraphRecursionError):
            reason = "recursion_limit"
        else:
            reason = "crash"
        tb = traceback.format_exc()
        with open(os.path.join(workdir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(tb)
        summary = {
            "run_id": run_id, "arm": cfg.arm, "outcome": "ERROR",
            "success": False, "abstained": False,
            "fail_reason": reason, "fail_phase": None,
            "error": {"reason": reason, "type": type(e).__name__, "message": str(e)},
            "poisoned": args.poisoned,
            "expected_outcome": "ABSTAIN" if args.poisoned else "SUCCESS",
            "leakage": args.leakage, "leakage_variant": leakage_variant,
            "leakage_column": info["leakage_column"],
            "task_prompt": task_prompt, "final_f1": None, "best_f1_measured": None,
            "eval_f1_threshold": cfg.eval_f1_threshold,
            "threshold_source": threshold_source, "threshold_k": THRESHOLD_K,
            "success_rule": success_rule,
            "baseline_f1": manifest.baseline_f1,
            "holdout_intact": [_sha256(p) for p in holdout_paths] == holdout_hashes,
        }
        json.dump(summary, open(os.path.join(workdir, "summary.json"), "w"), indent=2)
        print(tb, file=sys.stderr)
        print(f"\n=== RUN ERROR ({reason}) === {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(EXIT_RATE_LIMITED if reason == "rate_limited" else 1)

    # Kontaminations-Detektion (Spec-Nachtrag §0): Worker könnten ../holdout
    # überschreiben. holdout_intact = BEIDE Dateien (test.parquet +
    # test_features.parquet) unverändert. Hash-Gleichheit belegt Write-Integrität
    # (Reads sind damit NICHT detektierbar — dokumentierte Limitation).
    holdout_intact = [_sha256(p) for p in holdout_paths] == holdout_hashes
    if not holdout_intact:
        print("WARNUNG: holdout/ wurde während des Runs verändert "
              "(test.parquet, test_features.parquet und/oder Baseline-Predictions) — "
              "Ergebnis als kontaminiert werten!")

    # Endurteil fuer alle Arme identisch, auf dem Test-Anteil (S1). Die Graph-Knoten
    # entscheiden nur ueber den PROZESS (A/B: Gate auf dem Validierungsanteil);
    # SUCCESS, F1 und Baseline-Vergleich im Summary stammen ausschliesslich von hier.
    verdict = final_verdict(cfg, final, workdir, safety_max_calls=cfg.safety_max_calls)
    final = {**final, "success": verdict["success"]}

    ut = final["usage_total"]
    usage = TokenUsage(ut["input_tokens"], ut["output_tokens"],
                       ut["cache_read_tokens"], ut["cache_creation_tokens"])
    try:
        cost = cost_usd(usage, cfg.model)
    except KeyError:
        cost = float("nan")

    # bu_correct misst BU-Zielspalte == kanonische Zielspalte — bei einem poisoned
    # (unlösbaren) Task ist das bedeutungslos: eine halluzinierte, aber schema-valide
    # Spec würde sonst true melden. Darum auf None zwingen (§4-Konfusionsmatrix).
    bu_correct = None if args.poisoned else final["bu_correct"]
    target_correct = None if args.poisoned else final["target_correct"]

    # leakage_detected (roadmap §1): injizierte Spalte ∈ excluded_columns des
    # DU-Reports? None = nicht messbar (keine Injektion, oder kein DU-Report —
    # die freien Arme C/D haben keine DU-Phase; Messung dort indirekt über den
    # Test-F1-Einbruch).
    du_report = final["du_report"]
    if not args.leakage:
        leakage_detected = None
    elif du_report:
        leakage_detected = info["leakage_column"] in [
            e.get("column") for e in du_report.get("excluded_columns", [])]
    else:
        leakage_detected = None
    # leakage_exploited (Leakage-Runs, Decker 09.10.): folgen die final abgegebenen
    # Predictions dem Leak? Fuer ALLE Arme aus Predictions + Holdout gemessen
    # (docs/leakage-runs.md) — getrennt von der DU-Erkennung oben.
    # Worker-Transkripte sichern + Zugriffs-Audit (Review 24.10., M5/M2): die CLI raeumt
    # Transkripte nach ~30 Tagen weg, holdout_intact sieht keine Lesezugriffe.
    transcripts = collect_transcripts(os.path.join(workdir, "work"),
                                      os.path.join(workdir, "transcripts"), since=run_started)
    access = audit_access([os.path.join(workdir, "transcripts", n) for n in transcripts],
                          run_dir_name=run_id)
    if access["flagged"]:
        print(f"WARNUNG: Zugriffs-Audit markiert Tool-Aufrufe: {access['counts']}")
    reliance = None
    if args.leakage:
        reliance = leak_reliance(predictions_path(cfg, workdir), os.path.join(workdir, "holdout"),
                                 target=cfg.target, leak_column=info["leakage_column"],
                                 leak_values=info["leakage_values"])
    summary = {
        "run_id": run_id, "arm": cfg.arm, "outcome": outcome_label(final),
        "harness": harness_version(),
        "success": final["success"], "abstained": final["abstained"],
        # Ursache jedes Nicht-SUCCESS (Decker 14.09.) + Phase, in der der Lauf endete
        "fail_reason": verdict["fail_reason"],
        "fail_phase": None if final["success"] else final["current_phase"],
        "process_completed": verdict["process_completed"], "target_ok": verdict["target_ok"],
        "error": None,
        "poisoned": args.poisoned,
        "expected_outcome": "ABSTAIN" if args.poisoned else "SUCCESS",
        "leakage": args.leakage, "leakage_variant": leakage_variant,
        "leakage_column": info["leakage_column"],
        "leakage_detected": leakage_detected, "du_report": du_report,
        "leakage_exploited": reliance.exploited if reliance else None,
        "leakage_reliance": asdict(reliance) if reliance else None,
        "leakage_values": info["leakage_values"],
        "bu_correct": bu_correct, "target_correct": target_correct,
        # B3: erste BU-Entscheidung; first=False und bu_correct=True heisst: ein
        # Ruecksprung zu BU hat eine falsche Zielspalte korrigiert
        "bu_correct_first": None if args.poisoned else final.get("bu_correct_first"),
        "task_spec": final["task_spec"],
        "task_prompt": task_prompt, "final_f1": verdict["final_f1"],
        # Seit S1 (23.09.): TEST-F1 der final abgegebenen Predictions (nicht mehr das Beste
        # ueber Gate-Versuche) — Stufe-2-Groesse (Friedman). Key-Name aus Kompatibilitaet
        # mit main_runs/analyze_main beibehalten; f1_test ist derselbe Wert.
        "best_f1_measured": verdict["f1_test"], "f1_test": verdict["f1_test"],
        # A/B: Messung des letzten Gate-Durchlaufs (Validierungsanteil) — nur Diagnose
        "last_gate_metrics": final.get("gate_metrics"),
        "eval_f1_threshold": cfg.eval_f1_threshold,
        "threshold_source": threshold_source, "threshold_k": THRESHOLD_K,
        "baseline_f1": manifest.baseline_f1,
        # success_rule: noninferior_paired (gepaarter Bootstrap, 23.09.) | threshold (CLI)
        "baseline_f1_val": manifest.baseline_f1_val,
        "success_rule": success_rule, "baseline_comparison": verdict["test_comparison"],
        "visited_phases": final["visited_phases"], "attempts": final["attempts"],
        "total_calls": final["total_calls"], "gate_calls": final["gate_calls"],
        # S4: verbrauchtes Worker-Kontingent (A/B; C/D: eine Session)
        "worker_cost_usd": final.get("worker_cost_usd"), "worker_turns": final.get("worker_turns"),
        "budget_exhausted": final.get("budget_exhausted"),
        "usage_total": ut,
        "total_cost_usd": cost, "holdout_intact": holdout_intact,
        "transcripts": transcripts, "access_audit": access,
    }
    json.dump(summary, open(os.path.join(workdir, "summary.json"), "w"), indent=2)
    print("\n=== RUN SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
