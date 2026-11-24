# src/crispfsm/config.py
from __future__ import annotations
from dataclasses import dataclass

# Kanonische FSM-Phasenfolge (Arm A/B). Single Source für die Arm-D-Parität:
# die eine ReAct-Session bekommt das Phasen-Äquivalent an Budget/Turns
# (Spec 2026-07-18 §2.5, seit der DU-Phase 5 statt 4).
FSM_PHASES = ("bu", "du", "prep", "model", "eval")

# Erfolgsschwelle baseline-relativ (Decker 14.09.): SUCCESS <=> macro-F1 >=
# THRESHOLD_K x manifest.baseline_f1. k = 1: "alles unter den primitiven Methoden
# wäre schlecht". Steuert in Arm A/B auch den eval->model-Rücksprung. Ändern =
# neue Semantik der SUCCESS-Rate -> nur vor einem kompletten Kern-Lauf.
THRESHOLD_K = 1.0

# Validierungsanteil des Harness (S1, 23.09.), als Anteil am GESAMTEN Datensatz: Split
# 70 % Train (Agent) / 10 % Validierung (nur Gates von Arm A/B, fuer keinen Agenten als
# solche erkennbar) / 20 % Test (finale Bewertung aller Arme). Grund: das Eval-Gate
# meldete vorher den Test-F1 an den Worker zurueck -> bis zu 3 Blicke auf das finale
# Test-Set nur fuer A/B. Aendern = neuer Kern (Baselines haengen am Split).
VAL_FRACTION = 0.1


@dataclass
class RunConfig:
    run_id: str
    workdir: str                       # runs/<run_id>
    dataset: str = "breast_cancer"
    target: str = "target"
    task_prompt: str = ""              # BU-Aufgabenbeschreibung (neutralisiert, ohne Zielspalte)
    eval_metric: str = "f1"            # Metrik fuer das Eval-Gate (Task 7)
    model: str = "claude-opus-4-8"
    max_retries: int = 3               # pro Phase
    safety_max_calls: int = 20         # globales Runaway-Notaus (Worker-Calls)
    # Budgets (S4, 23.09.): ALLE Arme bekommen dasselbe Gesamtkontingent pro Run
    # (run_budget_usd / run_max_turns, nur Worker-Calls; B's Pruefer zaehlen nicht — sie
    # sind der Mechanismus des Arms wie A's Code-Gates). C/D verbrauchen es in ihrer einen
    # Session; A/B ueber alle Phasen + Retries, jeder Phasen-Call zusaetzlich gedeckelt
    # (budget_usd_per_call / max_turns) und nie mehr als der Rest. Leer -> FSM stoppt
    # (fail_reason budget_exhausted). Sicherheitsnetz, kein experimenteller Hebel.
    budget_usd_per_call: float | None = 4.0
    max_turns: int = 60                # max agentische Schritte pro Worker-Call (A/B)
    run_budget_usd: float | None = 20.0
    run_max_turns: int = 300
    eval_f1_threshold: float = 0.5     # Sanity-Schwelle (Skelett bewusst niedrig)
    # Gesetzt (run.py aus dem Manifest) -> Erfolgsregel = gepaarter Bootstrap gegen
    # die Default-Baseline auf demselben Holdout (scoring.compare_to_baseline,
    # 23.09.); eval_f1_threshold ist dann nur noch der berichtete Baseline-Punktwert.
    # None -> Punkt-Schwelle eval_f1_threshold (Debug/--eval-f1-threshold).
    baseline_pred_path: str | None = None
    # Baseline-Predictions fuer den Validierungsanteil (S1): Gates von A/B vergleichen
    # dort, die finale Bewertung aller Arme auf dem Test-Anteil (baseline_pred_path).
    baseline_val_path: str | None = None
    recursion_limit: int = 50          # LangGraph-Backstop (gegen Version prüfen)
    arm: str = "hard_fsm"              # "hard_fsm" (A) | "soft_fsm" (B) | "crisp_react" (C)
                                       # | "naked_react" (D) | "hard_fsm_single" (A-Single,
                                       # H1: wie A, eine Session fuer alle Phasen)
    react_max_turns: int | None = None # Arm C/D: Default run_max_turns (gleiches Kontingent)
                                       # nach __post_init__ immer int (None nur als Sentinel)

    def __post_init__(self):
        if self.react_max_turns is None:
            self.react_max_turns = self.run_max_turns
        if self.arm not in ("hard_fsm", "soft_fsm", "crisp_react", "naked_react",
                            "hard_fsm_single"):
            raise ValueError(
                f"arm '{self.arm}' unbekannt "
                f"(erlaubt: hard_fsm, soft_fsm, crisp_react, naked_react, hard_fsm_single).")
        # muss mit gates.VALID_METRICS übereinstimmen
        if self.eval_metric not in ("f1", "accuracy"):
            raise ValueError(
                f"eval_metric '{self.eval_metric}' unbekannt (erlaubt: f1, accuracy).")
