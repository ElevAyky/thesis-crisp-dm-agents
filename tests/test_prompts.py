# tests/test_prompts.py
import re

import pytest

from crispfsm.prompts import (
    CRISP_GUIDANCE,
    bu_initial_prompt,
    bu_resume_prompt,
    crisp_react_prompt,
    initial_prompt,
    react_prompt,
    resume_prompt,
)

PHASES = ["prep", "model", "eval"]


@pytest.mark.parametrize("phase", PHASES)
def test_initial_prompt_leaves_no_unsubstituted_placeholder(phase):
    """No raw {target} placeholder survives .format() for any phase."""
    out = initial_prompt(phase, "diagnosis")
    assert "{target}" not in out


@pytest.mark.parametrize("phase", ["prep", "model"])
def test_initial_prompt_substitutes_target_value(phase):
    """prep/model use {target} → the value is embedded in the rendered prompt."""
    out = initial_prompt(phase, "diagnosis")
    assert "diagnosis" in out


def test_eval_prompt_is_target_agnostic():
    """eval has no {target} placeholder by design (it scores a pickled model),
    so the target value is *not* present in its output."""
    out = initial_prompt("eval", "diagnosis")
    assert "diagnosis" not in out
    assert "{target}" not in out


@pytest.mark.parametrize("phase", ["prep", "model"])
def test_prep_and_model_forbid_test_parquet(phase):
    """N1 (23.09.): statt eines Verbots, das auf eine nicht existierende Datei zeigt,
    sagen prep/model positiv, woraus sie arbeiten — und nennen keine Testdatei."""
    out = initial_prompt(phase, "diagnosis")
    assert "the test labels are not available" in out.lower()
    assert "test.parquet" not in out
    if phase == "prep":
        assert "Use only 'data/train.parquet'" in out
    else:
        assert "build your test design from 'prep/prepared.parquet' only" in out


def test_eval_references_expected_paths():
    out = initial_prompt("eval", "diagnosis")
    assert "data/test_features.parquet" in out
    assert "model/model.pkl" in out
    assert "eval/predictions.parquet" in out
    assert "eval_metrics.json" not in out


def test_eval_does_not_forbid_test_parquet():
    """Sanity: eval is the one phase that is *supposed* to read the test split."""
    out = initial_prompt("eval", "diagnosis")
    assert "the test labels are not available" not in out.lower()
    assert "Use only 'data/train.parquet'" not in out


def test_model_prompt_renders_single_json_braces():
    """Escaped {{...}} must collapse to single braces — no doubling, no KeyError."""
    out = initial_prompt("model", "diagnosis")
    assert '{"metric"' in out
    assert '{{' not in out
    assert '}}' not in out


def test_eval_prompt_renders_single_json_braces():
    """EVAL enthält kein JSON-Beispiel mehr — nur: keine Brace-Reste."""
    out = initial_prompt("eval", "diagnosis")
    assert '{{' not in out
    assert '}}' not in out


def test_initial_prompt_unknown_phase_raises_keyerror():
    with pytest.raises(KeyError):
        initial_prompt("bogus", "t")


@pytest.mark.parametrize("phase", PHASES)
def test_resume_prompt_contains_gate_and_feedback(phase):
    feedback = "F1 zu niedrig"
    out = resume_prompt(phase, "diagnosis", feedback)
    assert "Quality Gate" in out
    assert "Gate" in out
    assert feedback in out
    # N3 (23.09.): kein interner Phasen-Code im Resume-Satz
    preamble = out.split("\n\n", 1)[0]
    assert f"'{phase}'" not in preamble
    assert "Address this and fulfill the original task again." in preamble


@pytest.mark.parametrize("phase", PHASES)
def test_resume_prompt_includes_substituted_template_body(phase):
    """resume_prompt appends the initial (substituted) template body verbatim."""
    body = initial_prompt(phase, "diagnosis")
    out = resume_prompt(phase, "diagnosis", "F1 zu niedrig")
    assert body in out
    assert "{target}" not in out


@pytest.mark.parametrize("phase", ["prep", "model"])
def test_resume_prompt_carries_target_value(phase):
    """For target-using phases the resume prompt embeds the substituted target."""
    out = resume_prompt(phase, "diagnosis", "F1 zu niedrig")
    assert "diagnosis" in out


def test_resume_prep_body_mentions_data_preparation():
    out = resume_prompt("prep", "diagnosis", "F1 zu niedrig")
    assert "Data Preparation" in out


def test_resume_prompt_unknown_phase_raises_keyerror():
    with pytest.raises(KeyError):
        resume_prompt("bogus", "t", "fb")


def test_bu_initial_prompt_contains_task_and_schema():
    p = bu_initial_prompt("Bestimme, ob ein Tumor bösartig ist.")
    assert "Business Understanding" in p
    assert "Bestimme, ob ein Tumor bösartig ist." in p
    assert "bu/task_spec.json" in p
    assert "cannot_determine" in p


def test_bu_resume_prompt_contains_feedback():
    p = bu_resume_prompt("Aufgabe X", "target_column 'ghost' existiert nicht")
    assert "ghost" in p and "Aufgabe X" in p


# --- Arm D: Naked-ReAct-Prompt (Spec 2026-07-18 §2.3) ------------------------

FORBIDDEN_IN_REACT = [
    "CRISP", "Phase", "Gate", "Quality",
    "Business Understanding", "Data Understanding", "Data Preparation",
    "Modeling", "Evaluation", "Deployment",
    "Rücksprung", "task_spec", "Retry", "Feedback",
]


def _has_word(word: str, text: str) -> bool:
    """Ganzwort-Match, case-insensitive: der woertliche CRISP-DM-Text enthaelt
    legitim z. B. 'aggregations' (Substring 'gate') und 'data quality'."""
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def test_react_prompt_contains_task_and_deliverables():
    p = react_prompt("Bestimme, ob ein Tumor bösartig ist.")
    assert "Bestimme, ob ein Tumor bösartig ist." in p
    assert "data/train.parquet" in p
    assert "data/test_features.parquet" in p
    assert "predictions.parquet" in p
    assert "result.json" in p
    assert "target_column" in p
    assert "cannot_determine" in p          # neutraler Abstain-Satz
    assert "{task_prompt}" not in p
    assert "{{" not in p and "}}" not in p


def test_react_prompt_is_process_pure():
    """Arm D darf NULL CRISP-/Prozess-Vokabular sehen — die Abwesenheit von
    Struktur IST das Treatment (Spec §2.3)."""
    p = react_prompt("Aufgabe X.")
    for word in FORBIDDEN_IN_REACT:
        assert not _has_word(word, p), f"Verbotswort im React-Prompt: {word!r}"


# --- Arm C: CRISP-ReAct-Prompt (roadmap §1) ----------------------------------

# Enforcement-Vokabular, das Arm C NICHT sehen darf: C bekommt Anleitung, aber
# keine Gates, kein Retry-/Feedback-Mechanik-Wissen und keine FSM-Artefakt-
# Kontrakte — die Abwesenheit von Enforcement IST das Treatment von C.
FORBIDDEN_IN_CRISP_REACT = [
    "Gate", "Quality Gate", "Retry", "Feedback", "Rücksprung",
    "task_spec", "data_report", "verdict", "Prüfer", "holdout",
]

CRISP_PHASE_NAMES = ["Business Understanding", "Data Understanding",
                     "Data Preparation", "Modeling", "Evaluation"]


def test_crisp_react_prompt_is_react_prompt_plus_guidance():
    """C↔D-Parität (experiment-design §1): Der C-Prompt ist byte-identisch der
    D-Prompt PLUS angehängter CRISP-Anleitung — der Diff ist genau ein Faktor."""
    t = "Bestimme, ob ein Tumor bösartig ist."
    assert crisp_react_prompt(t) == react_prompt(t) + CRISP_GUIDANCE


def test_crisp_react_prompt_contains_crisp_and_all_phases():
    p = crisp_react_prompt("Aufgabe X.")
    assert "CRISP-DM" in p
    for name in CRISP_PHASE_NAMES:
        assert name in p, f"Phasenname fehlt in der CRISP-Anleitung: {name!r}"
    # Reihenfolge-Empfehlung ist Teil des Treatments (Struktur per Anleitung)
    assert "in order" in p


def test_crisp_react_prompt_has_no_enforcement_vocabulary():
    p = crisp_react_prompt("Aufgabe X.")
    for word in FORBIDDEN_IN_CRISP_REACT:
        assert not _has_word(word, p), f"Enforcement-Wort im C-Prompt: {word!r}"


def test_crisp_react_prompt_keeps_deliverables_and_abstain():
    """Deliverable-Kontrakt und Abstain-Option müssen identisch zu D bleiben
    (gleiche Messung, gleiches Harness-Scoring)."""
    p = crisp_react_prompt("Aufgabe X.")
    assert "predictions.parquet" in p
    assert "result.json" in p
    assert "cannot_determine" in p
    assert "{task_prompt}" not in p
    assert "{{" not in p and "}}" not in p


def test_du_initial_prompt_contains_schema_and_target():
    from crispfsm import prompts
    p = prompts.du_initial_prompt("col_7")
    assert "Data Understanding" in p
    assert "du/data_report.json" in p
    assert "col_7" in p
    for field in ("n_rows", "columns", "class_balance", "quality_issues",
                  "excluded_columns", "spec_mismatch"):
        assert field in p
    # read-only-Auftrag ist Teil des Kontrakts
    assert "READ-ONLY" in p or "read-only" in p


def test_du_prompt_critical_columns_hint_only_when_present():
    from crispfsm import prompts
    assert "flagged" not in prompts.du_initial_prompt("t")
    p = prompts.du_initial_prompt("t", critical_columns=["patient_id", "col_9"])
    assert "patient_id" in p and "col_9" in p


def test_du_resume_prompt_contains_feedback():
    from crispfsm import prompts
    p = prompts.du_resume_prompt("t", "n_rows 999 stimmt nicht")
    assert "Quality Gate" in p
    assert "n_rows 999 stimmt nicht" in p
    assert "Data Understanding" in p


def test_prep_prompt_carries_du_findings():
    from crispfsm import prompts
    report = {"quality_issues": ["f1 hat 12% Missing Values"],
              "excluded_columns": [{"column": "col_9", "reason": "ID-Spalte"}]}
    p = prompts.initial_prompt("prep", "t", du_report=report)
    assert "f1 hat 12% Missing Values" in p
    assert "col_9" in p and "ID-Spalte" in p
    assert "NOT use these columns as features" in p


def test_prep_prompt_without_du_report_unchanged():
    from crispfsm import prompts
    assert prompts.initial_prompt("prep", "t") == prompts.initial_prompt("prep", "t", du_report=None)
    # leerer Report (keine Issues, keine Ausschlüsse) hängt nichts an
    assert prompts.initial_prompt("prep", "t", du_report={"quality_issues": [], "excluded_columns": []}) \
        == prompts.initial_prompt("prep", "t")


def test_du_findings_only_for_prep():
    from crispfsm import prompts
    report = {"quality_issues": ["Befund X"], "excluded_columns": []}
    assert "Befund X" not in prompts.initial_prompt("model", "t", du_report=report)


def test_prep_resume_prompt_carries_du_findings():
    from crispfsm import prompts
    report = {"quality_issues": ["Befund Y"], "excluded_columns": []}
    p = prompts.resume_prompt("prep", "t", "irgendein Feedback", du_report=report)
    assert "Befund Y" in p and "irgendein Feedback" in p


def test_bu_prompt_mentions_optional_critical_columns():
    from crispfsm import prompts
    p = prompts.bu_initial_prompt("Bestimme X.")
    assert "critical_columns" in p


def test_phase_keywords_are_unambiguous():
    """Fakes und Log-Analyse routen über Phasen-Keywords im Prompt — jedes
    Keyword darf NUR in den Prompts der eigenen Phase vorkommen (Regression:
    critical_columns-Hinweis und DU-Befunde hatten 'Data Understanding' in
    BU- bzw. Prep-Prompts eingeschleppt)."""
    from crispfsm import prompts
    report = {"quality_issues": ["x"], "excluded_columns": [{"column": "c", "reason": "r"}]}
    built = {
        "Business Understanding": prompts.bu_initial_prompt("Aufgabe."),
        "Data Understanding": prompts.du_initial_prompt("t", critical_columns=["c1"]),
        "Data Preparation": prompts.initial_prompt("prep", "t", du_report=report),
        "Modeling": prompts.initial_prompt("model", "t"),
        "Evaluation": prompts.initial_prompt("eval", "t"),
    }
    for kw in built:
        for phase, prompt in built.items():
            assert (kw in prompt) == (kw == phase), (kw, phase)


# --- CRISP-DM-Operationalisierung aus der Quelle (23.09.) ---------------------
# Phasen-Aufgaben stammen aus dem CRISP-DM-1.0-Referenzmodell (Chapman et al. 2000,
# Teil II) statt aus eigener Paraphrase. Derselbe Textblock steht im Phasen-Prompt von
# A/B und in der Anleitung von C -> Wissens-Paritaet A/C per Konstruktion.

from crispfsm import prompts as P


@pytest.mark.parametrize("phase", ["bu", "du", "prep", "model", "eval"])
def test_phase_prompt_contains_its_crisp_task_block(phase):
    block = P.CRISP_TASKS[phase]
    built = {"bu": P.bu_initial_prompt("Aufgabe."), "du": P.du_initial_prompt("t"),
             "prep": P.initial_prompt("prep", "t"), "model": P.initial_prompt("model", "t"),
             "eval": P.initial_prompt("eval", "t")}[phase]
    assert block in built


def test_crisp_guidance_carries_every_phase_task_block_in_order():
    """Wissens-Paritaet A/C: C bekommt exakt die Aufgabenbloecke, die A phasenweise bekommt."""
    positions = [P.CRISP_GUIDANCE.index(P.CRISP_TASKS[ph])
                 for ph in ("bu", "du", "prep", "model", "eval")]
    assert positions == sorted(positions)


def test_modeling_follows_crisp_not_single_split():
    """Regression 23.09.: 'splitte in train/val, trainiere einen Klassifikator' fuehrte zu
    einem Default-Modell ohne Tuning (CV/Tuning C 19 % vs. D 83 %). CRISP-DM 1.0 §4
    verlangt Testdesign, mehrere Modelle/Techniken und iteratives Bewerten + Tunen."""
    m = P.CRISP_TASKS["model"].lower()
    for concept in ("test design", "technique", "parameter", "compares", "best model"):
        assert concept in m, concept
    for old in ("train a classifier", "split internally into train/val"):
        assert old not in P.initial_prompt("model", "t").lower()
        assert old not in P.CRISP_GUIDANCE.lower()


# --- Prompt-Revision 23.09. (S2, S3, B1, B2, S4) -------------------------------

def _all_arm_a_prompts(task):
    return {"bu": P.bu_initial_prompt(task),
            "du": P.du_initial_prompt("t", task_prompt=task),
            "prep": P.initial_prompt("prep", "t", task_prompt=task),
            "model": P.initial_prompt("model", "t", task_prompt=task),
            "eval": P.initial_prompt("eval", "t", task_prompt=task)}


def test_every_arm_a_phase_prompt_contains_business_task():
    """S2: Die Geschaeftsaufgabe steht in allen fuenf A/B-Phasen-Prompts (initial + resume)."""
    task = "Bestimme, ob ein Tumor bösartig ist."
    for phase, p in _all_arm_a_prompts(task).items():
        assert task in p, phase
    for phase in ("du", "prep", "model", "eval"):
        built = _all_arm_a_prompts(task)[phase]
        assert f"Business task: \"{task}\"\n" in built, phase
        assert built.index("Business task:") == len(P._COMMON), phase   # direkt nach _COMMON
    assert f"Business task: \"{task}\"" in P.du_resume_prompt("t", "fb", task_prompt=task)
    for phase in ("prep", "model", "eval"):
        assert f"Business task: \"{task}\"" in P.resume_prompt(phase, "t", "fb", task_prompt=task)


def test_no_client_sentence_in_all_arms():
    """S3: _COMMON sagt allen Armen (auch D), dass es keinen Auftraggeber gibt."""
    s = "There is no client or domain expert to consult: make and state your own assumptions."
    assert s in P._COMMON
    for p in list(_all_arm_a_prompts("X.").values()) + [P.react_prompt("X."),
                                                        P.crisp_react_prompt("X.")]:
        assert s in p


def test_brief_reports_sentence_in_arm_a_and_arm_c():
    """S3: derselbe Kurz-Bericht-Satz in A/B (_CRISP_INTRO) und C (nach der Einleitung) —
    Paritaet A/C; D bekommt ihn nicht (er verweist auf das Referenzmodell)."""
    s = P._BRIEF_REPORTS
    assert "keep them brief" in s
    # Satz vor der Einleitung, damit die Einleitung direkt vor dem Zitat steht
    assert P._CRISP_INTRO.startswith(s)
    assert P._CRISP_INTRO.endswith("(quoted verbatim):\n")
    for phase, p in _all_arm_a_prompts("X.").items():
        assert p.count(s) == 1, phase
    g = P.CRISP_GUIDANCE
    assert g.count(s) == 1
    assert g.index(s) < g.index("1. Business Understanding:\n")
    assert s not in P.react_prompt("X.")


def test_crisp_guidance_opening_quotes_reference_model():
    """B1: Einleitung mit woertlichem Zitat aus CRISP-DM 1.0, S. 13."""
    assert P.CRISP_GUIDANCE.startswith(
        "APPROACH (CRISP-DM): Work on the task following the CRISP-DM process. Go through "
        "these phases in order and complete each one before you start the next, and return to "
        "an earlier phase when the results require it. As the "
        "CRISP-DM 1.0 reference model puts it: “The sequence of the phases is not rigid. "
        "Moving back and forth between different phases is always required. It depends on "
        "the outcome of each phase which phase or which particular task of a phase, has to be "
        "performed next.”\n")


def test_crisp_guidance_is_only_crisp_text():
    """B2: C und D unterscheiden sich NUR um CRISP-Text — die gemeinsamen ML-Hinweise
    stehen im D-Prompt, die Phasen-Ueberschriften in C sind nackt."""
    g = P.CRISP_GUIDANCE
    for heading in ("1. Business Understanding:\n", "2. Data Understanding:\n",
                    "3. Data Preparation:\n", "4. Modeling:\n", "5. Evaluation:\n"):
        assert heading in g, heading
    rest = g
    for block in P.CRISP_TASKS.values():
        rest = rest.replace(block, "")
    for hint in ("most frequent class", "suspiciously perfectly", "make all features numeric",
                 "same preprocessing", "Do not guess a target", "not encoded codes"):
        assert hint not in rest, hint
        assert hint in P.react_prompt("X."), hint


def test_model_prompt_asks_for_early_write():
    """S4: model.pkl/metrics.json frueh schreiben und bei Verbesserung ueberschreiben."""
    assert ("Write 'model/model.pkl' and 'model/metrics.json' as soon as you have a first "
            "working model and overwrite them whenever you find a better one.") \
        in P.initial_prompt("model", "t")


def test_crisp_tasks_cover_all_generic_tasks_of_phases_1_to_5():
    """S5: mechanische Regel — alle generischen Aufgaben der Phasen 1-5 in Guide-Reihenfolge,
    keine aus Phase 6 (Deployment)."""
    expected = {"bu": ["1.1", "1.2", "1.3", "1.4"], "du": ["2.1", "2.2", "2.3", "2.4"],
                "prep": ["3.1", "3.2", "3.3", "3.4", "3.5"],
                "model": ["4.1", "4.2", "4.3", "4.4"], "eval": ["5.1", "5.2", "5.3"]}
    for phase, nums in expected.items():
        assert re.findall(r"(?m)^(\d\.\d) ", P.CRISP_TASKS[phase]) == nums, phase
    assert not any(re.search(r"(?m)^6\.\d ", b) for b in P.CRISP_TASKS.values())


def test_abstain_wording_identical_in_all_arms():
    """R2-1: Enthaltungs-Praeferenz + Ausloeser stehen woertlich gleich in A/B (BU) und C/D."""
    for p in (P.bu_initial_prompt("X."), P.react_prompt("X."), P.crisp_react_prompt("X.")):
        assert P._NO_GUESS in p and P._NO_TARGET in p
    assert "IMPORTANT: Do not guess" not in P.bu_initial_prompt("X.")


def test_label_form_in_every_prediction_contract():
    """R2-4: Label-Form der Predictions in D/C und A/B-EVAL gleich festgelegt."""
    for p in (P.react_prompt("X."), P.initial_prompt("eval", "t")):
        assert "'prediction' " + P._LABEL_FORM in p


def test_react_requires_result_json_with_first_predictions():
    """R2-2: result.json entscheidet in C/D ueber target_ok -> frueh schreiben, Pflicht."""
    p = P.react_prompt("X.")
    assert "Together with your first 'predictions.parquet', write 'result.json'" in p
    assert "Both files are required." in p


def test_no_access_outside_cwd_in_all_arms():
    """R2-10: holdout/ liegt ein Verzeichnis ueber der cwd."""
    assert "do not access anything outside it (e.g. via '..')" in P._COMMON


@pytest.mark.parametrize("kind", ["later", "earlier"])
def test_resume_kinds_do_not_claim_own_failure(kind):
    for p in (P.resume_prompt("model", "t", "FB", kind=kind),
              P.bu_resume_prompt("X.", "FB", kind=kind),
              P.du_resume_prompt("t", "FB", kind=kind)):
        assert "did NOT pass" not in p
        assert ("FB" in p) == (kind == "later")
