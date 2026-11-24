#!/usr/bin/env python
"""Regelbasierte Prozess-Kodierung der Agenten-Laeufe (Kap. 7.2, Codebuch thesis/facts/codebook.md).

Kodiert jeden sauberen Run (`main-*`) deterministisch nach den Codes des Codebuchs:
  - Arme C/D (crisp_react, naked_react): alle Inhalts-Codes (BU, DU, PREP, MODEL, EVAL, DOC)
    plus Ablauf-Codes (Erstauftritte, kanonische Reihenfolge, Ruecksprunge, Struktur).
  - Arme A/B (hard_fsm, soft_fsm): nur die Inhalts-Codes, die die FSM nicht erzwingt
    (Modellfamilien, CV, Holdout, Tuning, Leakage erwogen, ...) -- als Referenz.
Die Aktionsfolge kommt aus `extract_actions.load_run` (dieselbe wie in den Markdown-Listen
unter thesis/facts/derived/actions/). Jeder Code traegt eine Belegstelle:
  "#7:L23" = Tool-Aktion 7, Zeile 23 des geschriebenen Inhalts bzw. Bash-Befehls
             (bei Edit: Zeile im neuen Text)
  "T>#5"   = sichtbarer Assistant-Text nach Aktion 5 ("T>#0" = vor der ersten Aktion)
  "N#9"    = Notizdatei (.md/.txt), geschrieben in Aktion 9

Ausgabe: thesis/facts/derived/process_coding.json (pro Run Codes + Belege; Zusammenfassung:
Raten je Arm und Umfang, Fisher exakt C vs. D je binaerem Code mit Holm, Zusammenhang mit
outcome und F1 minus Baseline innerhalb C und D). Alles explorativ und regelbasiert. Validiert
durch eine blinde Gegenkodierung von 24 C/D-Runs (validation_sample.py, coding_agreement.py,
Faktenblatt kap7.md F7-9xx); die A/B-Referenzcodes sind nicht gegenkodiert.

Aufruf aus dem Repo-Root:
    uv run python thesis/scripts/code_process.py [--runs-dir ../thesis-crisp-dm-runs]
    uv run python thesis/scripts/code_process.py --show main-vehicle-naked_react-r1   # Belege eines Runs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_actions as ea  # noqa: E402

REPO = ea.REPO
OUT = os.path.join(REPO, "thesis", "facts", "derived", "process_coding.json")
CODEBOOK_VERSION = "1.1 (2026-11-09)"  # 1.1: Codebuch-Listen an dieses Script angeglichen + Hilfscode bu_metric_rationale_unit

ARMS = ["hard_fsm", "soft_fsm", "crisp_react", "naked_react"]
REACT = ["crisp_react", "naked_react"]
FSM = ["hard_fsm", "soft_fsm"]
LETTER = {"hard_fsm": "A", "soft_fsm": "B", "crisp_react": "C", "naked_react": "D"}
PH = ["BU", "DU", "PREP", "MODEL", "EVAL"]
RANK = {p: i for i, p in enumerate(PH)}

# ============================================================================ Muster
# Modellfamilien (Konstruktor-Aufruf im Code, Kommentare ausgenommen). DummyClassifier ist
# Baseline, keine Familie.
FAMILIES = {
    "linear": r"\b(LogisticRegression(CV)?|SGDClassifier|RidgeClassifier(CV)?|Perceptron|PassiveAggressiveClassifier)\s*\(",
    "svm": r"(?<![A-Za-z])(SVC|LinearSVC|NuSVC)\s*\(",
    "random_forest": r"\bRandomForestClassifier\s*\(",
    "extra_trees": r"\bExtraTreesClassifier\s*\(",
    "hist_gb": r"\bHistGradientBoostingClassifier\s*\(",
    "gb": r"(?<!Hist)\bGradientBoostingClassifier\s*\(",
    "adaboost": r"\bAdaBoostClassifier\s*\(",
    "decision_tree": r"\bDecisionTreeClassifier\s*\(",
    "knn": r"\bKNeighborsClassifier\s*\(",
    "naive_bayes": r"\b(GaussianNB|MultinomialNB|BernoulliNB|CategoricalNB|ComplementNB)\s*\(",
    "mlp": r"\bMLPClassifier\s*\(",
    "discriminant": r"\b(LinearDiscriminantAnalysis|QuadraticDiscriminantAnalysis)\s*\(",
    "boosting_lib": r"\b(XGBClassifier|LGBMClassifier|CatBoostClassifier)\s*\(",
    "ensemble": r"\b(VotingClassifier|StackingClassifier|BaggingClassifier)\s*\(",
}
FAMILY_RE = {k: re.compile(v) for k, v in FAMILIES.items()}
# Konstruktor mit Argumenten (eine Klammer-Ebene verschachtelt, ueber Zeilen hinweg)
CTOR_ARGS = re.compile(r"\b([A-Z]\w*(?:Classifier|Regression|NB|Analysis)(?:CV)?|(?:Linear|Nu)?SVC)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)")


def _c(p):
    return re.compile(p)


def _ci(p):
    return re.compile(p, re.I)


RX = {
    # --- DU (Code) -- CRISP 2.2 Beschreiben, 2.3 Erkunden, 2.4 Qualitaet
    "du_schema": _c(r"\.shape\b|\.dtypes\b|\.head\(|\.info\(|\.tail\(|\.sample\(\d|print\([^)]*\.columns"),
    "du_missing": _c(r"\.(isna|isnull|notna|notnull)\(|\.info\(|hasnans"),
    "du_stats": _c(r"\.describe\(|\.quantile\(|\.skew\(|\.kurt(osis)?\(|\.agg\(\["),
    "du_quality_ext": _c(r"duplicated\(|drop_duplicates\(|\.nunique\(|is_unique"),
    "du_relations": _c(r"\.corr\(|corrwith\(|crosstab\(|mutual_info|\bchi2\(|f_classif|pointbiserial|\.groupby\("),
    "du_counts": _c(r"value_counts\(|bincount\(|\bCounter\(|return_counts"),
    # --- PREP (Code) -- CRISP 3.1 Auswahl, 3.2 Bereinigen, 3.3 Konstruieren, 3.5 Formatieren
    "prep_impute": _c(r"\b(SimpleImputer|KNNImputer|IterativeImputer)\b|\.fillna\("),
    "prep_encode": _c(r"\b(OneHotEncoder|OrdinalEncoder|LabelEncoder|TargetEncoder)\b|get_dummies\(|\.cat\.codes|factorize\("
                      r"|pd\.Categorical\(|astype\(\s*['\"]category['\"]\s*\)|\.map\(|\.replace\(\s*\{"),
    "prep_format": _c(r"pd\.to_numeric\(|astype\(\s*(np\.)?['\"]?(float|int)\w*['\"]?\s*\)"),
    "prep_scale": _c(r"\b(StandardScaler|MinMaxScaler|RobustScaler|MaxAbsScaler|Normalizer|QuantileTransformer|PowerTransformer)\b"
                     r"|/\s*255(\.0*)?\b"),
    "prep_select_tool": _c(r"\b(VarianceThreshold|SelectKBest|SelectFromModel|RFE|RFECV|SelectPercentile)\s*\("),
    "prep_construct": _c(r"\b(PolynomialFeatures|KBinsDiscretizer|SplineTransformer)\s*\("),
    "drop": _c(r"\.drop\((.*)\)"),
    # --- MODEL (Code) -- CRISP 4.2 Testdesign, 4.3 Bauen, 4.4 Bewerten/Parameter revidieren
    "model_cv": _c(r"cross_val_score\(|cross_val_predict\(|cross_validate\(|\b(Repeated)?(Stratified)?K[Ff]old\(|SearchCV\(|\bcv\s*=\s*\d"),
    "model_holdout": _c(r"train_test_split\(|\b(Stratified)?ShuffleSplit\("),
    "model_search": _c(r"SearchCV\(|ParameterGrid\(|ParameterSampler\(|\boptuna\b"),
    "model_class_weight": _c(r"class_weight\s*=|sample_weight|compute_sample_weight|compute_class_weight"),
    "model_assess_detail": _c(r"classification_report\(|confusion_matrix\(|ConfusionMatrixDisplay|precision_recall_fscore_support\(|average\s*=\s*None"),
    # --- EVAL (Code) -- Baseline (CRISP 5.1 gegen das Erfolgskriterium aus 1.3)
    "dummy": _c(r"\bDummyClassifier\s*\("),
    "baseline_word": _c(r"(?i:baseline|most[_ -]?frequent|majority|\bmode\(\))"),
    # --- Ausgabe / Verifikation / BU-Konstanten
    "pred_write": _c(r"to_parquet\(.*predictions|predictions\.parquet.*to_parquet|['\"]predictions\.parquet['\"]"),
    "reads_outputs": _c(r"read_parquet\(\s*['\"](\./)?predictions\.parquet|result\.json|cat\s+result"),
    "bu_target_const": _c(r"^\s*(TARGET|TARGET_COL(UMN)?|target|target_col(umn)?|LABEL|label_col)\s*=\s*['\"]"),
    "bu_metric_const": _c(r"^\s*(METRIC|SCORING|metric|scoring)\s*=\s*['\"]"),
}
# Text (sichtbarer Assistant-Text, Notizdateien, Code-Kommentare)
TX = {
    "target_word": _ci(r"\btarget\b|\blabel\b|\boutcome\b|\bpredict"),
    "metric": _ci(r"\b(macro[- ]?)?f1\b|\bf1[-_ ]?(score|macro)\b|\baccuracy\b"),
    "rationale": _ci(r"imbalanc|\bbalanced\b|minority|majority|skew|rare class|class distribution|dominat"),
    "success": _ci(r"success criteri|success (is|=|means|metric|would be)|\b(beat|exceed|outperform)s?\b[^.\n]{0,40}\bbaseline|must (beat|exceed|outperform)"),
    "business": _ci(r"\bbusiness\b|stakeholder|\bassum(e|es|ed|ing|ption)|\bcostly\b|cost of (a |an )?(error|mistake|false|miss)|false (negative|positive)s? (are|is|would)|project plan|\bobjectives?\b"),
    "leak": re.compile(r"(?i:leak)|\bIDs?\b|(?i:\bid[- ]?(like|column|col)\b|identifier|row[- ]?(id|index|number)|index column)"),
    # 5.1: Bewertung gegen Ziel/Erfolgskriterium -- Bezug UND Bewertungsverb im selben Satz
    "eval_business_ref": _ci(r"success criteri|business (objective|goal|success|need|question)|\bobjectives?\b|\bgoal\b"),
    "eval_business_verb": _ci(r"\b(is|are|was|were|has been|have been) (met|achieved|satisfied|fulfilled)\b|\bmeets?\b|\bsatisf(y|ies|ied)\b|\bachiev(es|ed)\b|\bfulfil(s|ls|led)?\b"),
    "eval_review_next": _ci(r"next steps?|further (iteration|improvement|work|tuning|gains?)|could (be )?(further )?improv|future (work|iteration|improvement)|\brecommend|process review|review of (the )?process|overlook|possible (improvement|action)s?|would (try|explore)|if (more )?time"),
    "baseline_num": _ci(r"(baseline|dummy|most[- ]frequent|majority)[^.\n]{0,60}\d\.\d|\d\.\d[^.\n]{0,40}(baseline|dummy|most[- ]frequent|majority)"),
    "crisp_phase": _ci(r"business understanding|data understanding|data preparation|\bmodel(l)?ing\b|\bevaluation\b"),
}
# Ziel-Referenz in einer Codezeile (du_class_dist, du_quality_ext); dazu der Zielspalten-Name
TGT_REF = r"\b(y|y_?train|y_?tr|TARGET|target|label)\b"
DOC_EXT = (".md", ".txt", ".rst")
HEREDOC_NOTES = re.compile(r"cat\s*>{1,2}\s*['\"]?[\w./\\-]+\.(md|txt)")
TRACEBACK = re.compile(r"Traceback \(most recent call last\)|^\w*Error:|Exit code [1-9]", re.M)
OUTPUT_REF = re.compile(r"\bpred(s|ictions?)?\b|result\.json|\bout\b|\bsub(mission)?\b", re.I)
# Zeilen, die aus einer Zaehlung nur einen Wert ziehen (Mehrheitsklasse fuer die Baseline,
# positives Label), sind keine Inspektion
DU_NOT = re.compile(r"idx(max|min)\(|\.mode\(\)")


def is_du_line(x: str, pat: str | None = None) -> bool:
    pats = (pat,) if pat else DU_PATS
    return any(RX[p].search(x) for p in pats) and not OUTPUT_REF.search(x) and not DU_NOT.search(x)
DU_PATS = ("du_schema", "du_missing", "du_stats", "du_quality_ext", "du_relations", "du_counts")
PREP_PATS = ("prep_impute", "prep_encode", "prep_format", "prep_scale", "prep_select_tool", "prep_construct")
IMPORT = re.compile(r"^\s*(from\s+[\w.]+\s+import\b|import\s+[\w.]+)")
# Aufbereitungs-Operatoren mit Argumenten (Token fuer die Ruecksprung-Erkennung)
PREP_OPS = re.compile(r"\b(SimpleImputer|KNNImputer|IterativeImputer|OneHotEncoder|OrdinalEncoder|LabelEncoder|"
                      r"TargetEncoder|StandardScaler|MinMaxScaler|RobustScaler|MaxAbsScaler|Normalizer|"
                      r"QuantileTransformer|PowerTransformer|VarianceThreshold|SelectKBest|SelectFromModel|RFE|"
                      r"RFECV|SelectPercentile|PolynomialFeatures|KBinsDiscretizer|SplineTransformer|fillna|"
                      r"get_dummies|factorize)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)")
SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n")


# ============================================================================ Zerlegung

def split_comment(line: str) -> tuple[str, str]:
    """Python-Zeile -> (Code, Kommentar). Einfacher Scanner (Anfuehrungszeichen je Zeile)."""
    q = None
    for i, ch in enumerate(line):
        if q:
            if ch == q and line[i - 1] != "\\":
                q = None
        elif ch in ("'", '"'):
            q = ch
        elif ch == "#":
            return line[:i], line[i + 1:]
    return line, ""


class Unit:
    """Eine Belegeinheit: Codezeile, Kommentar, Text oder Notizdatei."""
    __slots__ = ("pos", "ref", "kind", "s", "entry", "new", "final")

    def __init__(self, pos, ref, kind, s, entry=None, new=False, final=False):
        self.pos, self.ref, self.kind, self.s = pos, ref, kind, s
        self.entry, self.new, self.final = entry, new, final


def is_doc_action(e) -> bool:
    if e.tool in ("Write", "Edit") and e.path.lower().endswith(DOC_EXT):
        return True
    return e.tool == "Bash" and bool(HEREDOC_NOTES.search(e.code))


def is_code_action(e) -> bool:
    if e.tool == "Bash":
        return not HEREDOC_NOTES.search(e.code)
    if e.tool in ("Write", "Edit"):
        return e.path.lower().endswith(".py")
    return False


def runs_python(e) -> bool:
    return e.tool == "Bash" and bool(re.search(r"\bpython3?\b|\buv run\b", e.code))


def build_units(ra) -> list[Unit]:
    """Alle Belegeinheiten eines Runs in Reihenfolge. pos = (Eintrag-Index, Zeile)."""
    units, seen, last_action = [], set(), 0
    n_last = max((e.n for e in ra.actions), default=0)
    for k, e in enumerate(ra.entries):
        if e.kind == "prompt":
            continue
        if e.kind == "text":
            units.append(Unit((k, 0), f"T>#{last_action}", "text", e.text, e,
                              final=last_action == n_last))
            continue
        last_action = e.n
        if e.tool in ("Write", "Edit") and e.is_error:
            continue                     # fehlgeschlagene Aenderung (z. B. old_string nicht gefunden)
        if is_doc_action(e):
            units.append(Unit((k, 0), f"N#{e.n}", "notes", e.code, e))
            continue
        if not is_code_action(e):
            continue
        for li, line in enumerate(e.code.split("\n"), 1):
            code, com = split_comment(line)
            norm = " ".join(code.split())
            if IMPORT.match(code):
                continue                 # Importe sind keine Aktivitaet
            if norm:
                units.append(Unit((k, li), f"#{e.n}:L{li}", "code", code, e, new=norm not in seen))
                seen.add(norm)
            if com.strip():
                units.append(Unit((k, li), f"#{e.n}:L{li}", "comment", com, e))
    return units


def families_in(s: str) -> list[str]:
    return [f for f, r in FAMILY_RE.items() if r.search(s)]


def first(units, pred):
    for u in units:
        if pred(u):
            return u
    return None


def is_business_assessment(s: str) -> bool:
    """CRISP 5.1: ein Satz bewertet das Ergebnis gegen Ziel/Erfolgskriterium."""
    return any(TX["eval_business_ref"].search(x) and TX["eval_business_verb"].search(x)
               for x in SENT_SPLIT.split(s))


def is_baseline_line(s: str) -> bool:
    """Baseline-Beleg im Code: DummyClassifier oder eine als Baseline/Mehrheitsklasse
    bezeichnete Rechnung; Imputation mit 'most_frequent'/Modus zaehlt nicht."""
    if RX["dummy"].search(s):
        return True
    if not RX["baseline_word"].search(s):
        return False
    return not re.search(r"Imputer|fillna|import\b", s)


# ============================================================================ Hilfsregeln

def target_name(ra) -> str | None:
    p = os.path.join(ARGS.runs_dir, ra.run_id, "work", "result.json")
    try:
        with open(p, encoding="utf-8") as f:
            t = json.load(f).get("target_column")
            if t:
                return t
    except Exception:
        pass
    ts = ra.summary.get("task_spec") or {}
    return ts.get("target_column")


def drop_is_selection(code_line: str, all_code: str, tgt: str | None) -> str | None:
    """`.drop(...)` entfernt Feature-Spalten (nicht nur die Zielspalte)?
    -> "named" (benannte Spalten: Literal oder Liste aus Literalen), "rule" (datenabhaengige
    Regel, z. B. konstante Spalten ueber nunique), None."""
    m = RX["drop"].search(code_line)
    if not m or "print(" in code_line:
        return None
    if "columns" not in code_line and not re.search(r"axis\s*=\s*(1|['\"]columns['\"])", code_line):
        return None
    arg = re.sub(r"errors\s*=\s*['\"]\w+['\"]|axis\s*=\s*\S+|inplace\s*=\s*\w+", "", m.group(1))
    lits = re.findall(r"['\"]([^'\"]+)['\"]", arg)
    if any(lit != tgt for lit in lits):
        return "named"
    idents = set(re.findall(r"\b([A-Za-z_]\w*)\b", re.sub(r"['\"][^'\"]*['\"]", "", arg)))
    idents -= {"columns", "c", "col", "if", "in", "for", "not", "test", "te", "tr", "train",
               "TARGET", "target", "TARGET_COL", "target_col", "y", "X", "df", "list", "set"}
    kind = None
    for v in idents:
        assigns = [a.strip() for a in re.findall(rf"^\s*{re.escape(v)}\s*=\s*(.+)$", all_code, re.M)]
        if not assigns:
            if re.search(r"\.index|nunique|const", arg):     # z. B. nunq[nunq<=1].index
                kind = kind or "rule"
            continue
        for a in assigns:
            if a in ("[]", "list()", "set()"):
                continue
            lits_a = [x for x in re.findall(r"['\"]([^'\"]+)['\"]", a) if x != tgt]
            if lits_a and not re.search(r"\bfor\b|nunique|\.index", a):
                return "named"
            kind = kind or "rule"
    return kind


def derived_attr(code_line: str) -> bool:
    """Neue Spalte aus anderen Spalten berechnet (CRISP 3.3 derived attributes)."""
    m = re.match(r"\s*\w+\[['\"]([^'\"]+)['\"]\]\s*=(?!=)\s*(.+)", code_line)
    if not m:
        return False
    lhs, rhs = m.group(1), m.group(2)
    refs = re.findall(r"\[['\"]([^'\"]+)['\"]\]", rhs)
    return (bool(refs) and lhs not in refs and bool(re.search(r"[*/+\-]|np\.\w+\(", rhs))
            and not re.search(r"predict|astype|map\(|fillna", rhs))


def split_top(args: str) -> list[str]:
    """Argumentliste an Kommas der obersten Klammerebene trennen."""
    out, depth, cur = [], 0, ""
    for ch in args:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [re.sub(r"\s+", "", x).replace('"', "'") for x in out if x.strip()]


def norm_ctor_args(fam: str, args: str) -> str:
    """Normalisierte Konfiguration: ohne technische Argumente (random_state, n_jobs, verbose;
    bei linearen Modellen/MLP auch max_iter, solver), Positionsargument n_estimators benannt,
    Argumente sortiert, einheitliche Anfuehrungszeichen."""
    drop = {"random_state", "n_jobs", "verbose"} | ({"max_iter", "solver"} if fam in ("linear", "mlp") else set())
    parts = []
    for i, a in enumerate(split_top(args)):
        if "=" not in a or a.startswith("**"):
            if i == 0 and fam in ("random_forest", "extra_trees", "gb", "adaboost") and re.fullmatch(r"\d+", a):
                a = f"n_estimators={a}"
            parts.append(a)
            continue
        if a.split("=", 1)[0] not in drop:
            parts.append(a)
    return ",".join(sorted(parts))


# Technische Argumente von Aufbereitungs-Operatoren (aendern die Aufbereitung nicht inhaltlich)
PREP_TECH = r"handle_unknown|sparse_output|sparse|dtype|verbose_feature_names_out|n_jobs|copy|unknown_value|encoded_missing_value"


def norm_prep_args(args: str) -> str:
    a = re.sub(rf"\b({PREP_TECH})\s*=\s*[^,)]+,?", "", args)
    return re.sub(r"\s+", "", a).replace('"', "'").rstrip(",")


def repair_actions(ra) -> set:
    """Aktionsnummern, die einen Fehler reparieren: Write/Edit nach einer fehlgeschlagenen
    Python-Ausfuehrung (bis zur naechsten Ausfuehrung) sowie eine erneute Inline-Ausfuehrung
    mit geaendertem Code direkt nach einem Fehler."""
    out, failed_prev = set(), False
    for e in ra.actions:
        if not is_code_action(e):
            continue
        if failed_prev and (e.tool in ("Write", "Edit") or runs_python(e)):
            if e.tool in ("Write", "Edit") or re.search(r"python3?\s+(-c\b|-\s*<<)", e.code):
                out.add(e.n)
        if runs_python(e):
            failed_prev = bool(TRACEBACK.search(e.result or "")) or e.is_error
    return out


def tuning_by_loop(code: str) -> int | None:
    """Schleife ueber Parameterwerte, deren Variable in einem Konstruktor steckt -> Zeile."""
    lines = code.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"\s*for\s+(\w+)(\s*,\s*(\w+))?\s+in\s+(.+):", line)
        if not m:
            continue
        if not re.search(r"\[\s*[\d.'\"]|\(\s*[\d.'\"]|range\(|np\.(linspace|logspace|arange)|product\(|grid|param",
                         m.group(4)):
            continue
        vars_ = {m.group(1), m.group(3)} - {None}
        window = "\n".join(lines[i + 1:i + 12])
        for cm in CTOR_ARGS.finditer(window):
            if families_in(cm.group(0)) and any(re.search(rf"\b{re.escape(v)}\b", cm.group(2)) for v in vars_):
                return i + 1
        if re.search(r"set_params\(", window) and any(re.search(rf"\b{re.escape(v)}\b", window) for v in vars_):
            return i + 1
    return None


def tuning(ra, code_units, allowed_phases=None) -> tuple:
    """Hyperparameter-Suche/-Revision (CRISP 4.3/4.4). Reihenfolge der Regeln: Such-Werkzeug,
    gleiche Familie mit >= 2 Parametersaetzen (Reparaturen nach Fehlern ausgenommen),
    Parameter-Schleife, Schwellenwert-Optimierung auf Wahrscheinlichkeiten.
    Grenze: nur Konstruktor-Argumente zaehlen. Ein Vergleich ueber Fit-Argumente (z. B.
    fit(..., sample_weight=...) gegen fit(...) mit demselben Konstruktor) bleibt unerkannt,
    so in main-student_health-crisp_react-r4 (work/train.py:58-77)."""
    u = first(code_units, lambda u: RX["model_search"].search(u.s))
    if u:
        return u.ref, "search"
    repairs = repair_actions(ra)
    cfg = defaultdict(dict)
    for e in ra.actions:
        if (not is_code_action(e) or (allowed_phases and e.phase not in allowed_phases) or e.is_error
                or e.n in repairs):
            continue
        code = "\n".join(split_comment(line)[0] for line in e.code.split("\n"))
        for cm in CTOR_ARGS.finditer(code):
            fam = families_in(cm.group(0))
            if fam:
                line = code[:cm.start()].count("\n") + 1
                cfg[fam[0]].setdefault(norm_ctor_args(fam[0], cm.group(2)), (e.n, line))
    hits = []
    for fam, d in cfg.items():
        if len(d) >= 2:
            hits.append((sorted(d.values())[1], fam))
    if hits:
        (n, line), fam = min(hits)
        return f"#{n}:L{line}", f"configs:{fam}"
    for e in ra.actions:
        if not is_code_action(e) or (allowed_phases and e.phase not in allowed_phases):
            continue
        line = tuning_by_loop("\n".join(split_comment(x)[0] for x in e.code.split("\n")))
        if line:
            return f"#{e.n}:L{line}", "loop"
    u = first(code_units, lambda u: THRESH.search(u.s) and (allowed_phases is None
                                                           or u.entry.phase in allowed_phases))
    if u:
        return u.ref, "threshold"
    return None, None


# Schwellenwert-Optimierung: Wahrscheinlichkeit gegen eine variable Schwelle (Schleife/Suche)
THRESH = re.compile(r"(proba\w*|prob\w*|p_?\w*)\s*>=?\s*(t|th|thr|thresh\w*|best_t\w*|cut\w*)\b"
                    r"|\bbest_(th|thr|thresh\w*|t)\b\s*=")


# ============================================================================ Kodierung C/D

def code_react(ra) -> dict:
    units = build_units(ra)
    tgt = target_name(ra)
    code_units = [u for u in units if u.kind == "code"]
    all_code = "\n".join(u.s for u in code_units)
    nonfinal_txt = [u for u in units if u.kind in ("text", "notes", "comment") and not u.final]
    val, ev = {}, {}

    def mark(code, u, v=None):
        val[code] = (u is not None) if v is None else v
        ev[code] = u.ref if isinstance(u, Unit) else u

    # --- MODEL-Position: erster Konstruktor eines Klassifikators (keine Dummy-Baseline)
    m0 = first(code_units, lambda u: families_in(u.s))
    mpos = m0.pos if m0 else (10 ** 9, 0)

    def before(u):
        return u.pos < mpos

    def after(u):
        return u.pos >= mpos

    # --- BU (CRISP 1.1-1.4): vor der ersten Modellanpassung ausdruecklich festgehalten
    tname = re.compile(re.escape(tgt)) if tgt else re.compile(r"\bcol_\d+\b")

    def is_target_stmt(u):
        if u.kind in ("text", "notes", "comment"):
            return bool(tname.search(u.s) and TX["target_word"].search(u.s))
        return u.kind == "code" and bool(RX["bu_target_const"].search(u.s))

    def is_metric_stmt(u):
        if u.kind in ("text", "notes", "comment"):
            return bool(TX["metric"].search(u.s))
        return u.kind == "code" and bool(RX["bu_metric_const"].search(u.s))
    u_t_any = first(units, lambda u: not u.final and is_target_stmt(u))
    u_t = u_t_any if (u_t_any and before(u_t_any)) else None
    u_m = first(units, lambda u: before(u) and not u.final and is_metric_stmt(u))
    u_r = first(nonfinal_txt, lambda u: before(u) and u.kind in ("text", "notes") and any(
        TX["metric"].search(s) and TX["rationale"].search(s) for s in SENT_SPLIT.split(u.s)))
    # Hilfscode (Sensitivitaet, nicht in der Holm-Familie): Metrik- und Begruendungswort in derselben
    # Text-Einheit (Nachricht oder Notiz) statt im selben Satz -- erfasst knappe Begruendungen ueber
    # zwei aufeinanderfolgende Saetze ("imbalanced (~24%). I'll use F1."), kann aber zu weit greifen.
    u_r_unit = first(nonfinal_txt, lambda u: before(u) and u.kind in ("text", "notes")
                     and TX["metric"].search(u.s) and TX["rationale"].search(u.s))
    u_s = first(nonfinal_txt, lambda u: before(u) and TX["success"].search(u.s))
    u_b = first(nonfinal_txt, lambda u: before(u) and TX["business"].search(u.s))
    mark("bu_target", u_t)
    mark("bu_metric", u_m)
    later = max([u for u in (u_t, u_m) if u], key=lambda u: u.pos, default=None)
    mark("bu_explicit", later if (u_t and u_m) else None)
    mark("bu_metric_rationale", u_r)
    mark("bu_metric_rationale_unit", u_r_unit)
    mark("bu_success_criterion", u_s)
    mark("bu_business_context", u_b)

    # --- DU (CRISP 2.1-2.4): vor der ersten Modellanpassung, im Code
    tgt_ref = re.compile(TGT_REF + (rf"|{re.escape(tgt)}" if tgt else ""))

    def du_first(pat, extra=None):
        return first(code_units, lambda u: before(u) and is_du_line(u.s, pat) and (extra is None or extra(u)))
    u_insp = first(code_units, lambda u: before(u) and is_du_line(u.s))
    mark("du_inspect_before_model", u_insp)
    mark("du_class_dist", du_first("du_counts", lambda u: tgt_ref.search(u.s)))
    mark("du_missing", du_first("du_missing"))
    mark("du_quality_ext", du_first("du_quality_ext", lambda u: not (
        "nunique" in u.s and tgt_ref.search(u.s)
        and not re.search(r"\b(tr|train|df|X|data|te|test)\s*\.\s*nunique|\[c\]|\[col\]", u.s))))
    mark("du_stats", du_first("du_stats"))
    mark("du_relations", du_first("du_relations"))
    mark("du_leakage", first(nonfinal_txt, lambda u: before(u) and TX["leak"].search(u.s)))
    mark("leakage_mention_any", first([u for u in units if u.kind in ("text", "notes", "comment")],
                                      lambda u: TX["leak"].search(u.s)))

    # --- PREP (CRISP 3.1-3.5): irgendwo im Run
    found = {
        "prep_impute": first(code_units, lambda u: RX["prep_impute"].search(u.s) and not OUTPUT_REF.search(u.s)),
        "prep_encode": first(code_units, lambda u: RX["prep_encode"].search(u.s) and not OUTPUT_REF.search(u.s)),
        "prep_format": first(code_units, lambda u: RX["prep_format"].search(u.s) and not OUTPUT_REF.search(u.s)),
        "prep_scale": first(code_units, lambda u: RX["prep_scale"].search(u.s)),
        "prep_select": first(code_units, lambda u: drop_is_selection(u.s, all_code, tgt) == "named"),
        "prep_select_rule": first(code_units, lambda u: RX["prep_select_tool"].search(u.s)
                                  or drop_is_selection(u.s, all_code, tgt) == "rule"),
        "prep_construct": first(code_units, lambda u: RX["prep_construct"].search(u.s) or derived_attr(u.s)),
    }
    for c, u in found.items():
        mark(c, u)
    pu = [u for u in found.values() if u]
    u_prep = min(pu, key=lambda u: u.pos) if pu else None
    mark("prep_explicit", u_prep)

    # --- MODEL (CRISP 4.1-4.4)
    fams = {}
    for u in code_units:
        for f in families_in(u.s):
            fams.setdefault(f, u.ref)
    val["n_families"] = len(fams)
    ev["n_families"] = fams
    mark("model_multi_family", sorted(fams), len(fams) >= 2)
    u_cv = first(code_units, lambda u: RX["model_cv"].search(u.s))
    u_ho = first(code_units, lambda u: RX["model_holdout"].search(u.s))
    mark("model_cv", u_cv)
    mark("model_holdout", u_ho)
    td = [u for u in (u_cv, u_ho) if u]
    mark("model_test_design", min(td, key=lambda u: u.pos) if td else None)
    ref, how = tuning(ra, code_units)
    mark("model_tuning", ref, ref is not None)
    val["model_tuning_how"] = how
    mark("model_class_weight", first(code_units, lambda u: RX["model_class_weight"].search(u.s)))
    mark("model_assess_detail", first(code_units, lambda u: RX["model_assess_detail"].search(u.s)))

    # --- EVAL (CRISP 5.1-5.3)
    u_bc = first(code_units, lambda u: is_baseline_line(u.s))
    u_bt = first([u for u in units if u.kind in ("text", "notes")],
                 lambda u: after(u) and TX["baseline_num"].search(u.s))
    bl = [u for u in (u_bc, u_bt) if u]
    u_bl = min(bl, key=lambda u: u.pos) if bl else None
    mark("eval_baseline", u_bl)
    evtxt = [u for u in units if u.kind in ("text", "notes", "comment")]
    u_eb = first(evtxt, lambda u: after(u) and is_business_assessment(u.s))
    mark("eval_business", u_eb)
    mark("eval_review_next", first(evtxt, lambda u: after(u) and TX["eval_review_next"].search(u.s)))

    # --- DOC (CRISP: Berichte je Output; im Prompt: kurz halten)
    notes = [u for u in units if u.kind == "notes"]
    mark("doc_notes", notes[0] if notes else None)
    mark("doc_upfront", first(notes, before))
    mark("doc_crisp_sections", first(notes, lambda u: len({m.group(0).lower()
                                                            for m in TX["crisp_phase"].finditer(u.s)}) >= 3))
    # Hilfscode: eigene Bash-Aktion liest die geschriebenen Ausgaben (Kontrakt-Pruefung)
    ver = next((e for e in ra.actions if e.tool == "Bash" and RX["reads_outputs"].search(e.code)
                and not families_in(e.code)), None)
    mark("verify_outputs", f"#{ver.n}" if ver else None, ver is not None)

    # --- Ablauf: Erstauftritte der Phasen (Abschlussbericht ausgenommen)
    # EVAL: erster Baseline-/Bewertungsbeleg, fruehestens bei der ersten Modellanpassung
    # (Baseline im selben Skript wie das Modell zaehlt als gleichzeitig).
    e_cand = [u for u in (u_bc if (u_bc and after(u_bc)) or (u_bc and u_bc.entry is m0.entry) else None,
                          u_bt, u_eb) if u] if m0 else []
    if m0 and u_bc and not after(u_bc) and u_bc.entry is not m0.entry:
        val["baseline_before_model_action"] = u_bc.ref     # z. B. Mehrheitsquote in der Inspektion
    e0 = min(e_cand, key=lambda u: u.pos) if e_cand else None
    pos = {
        "BU": (u_t_any.pos, u_t_any.ref) if u_t_any else None,
        "DU": (u_insp.pos, u_insp.ref) if u_insp else None,
        "PREP": (u_prep.pos, u_prep.ref) if u_prep else None,
        "MODEL": (m0.pos, m0.ref) if m0 else None,
        "EVAL": (max(e0.pos, mpos), e0.ref) if e0 else None,
    }
    val["first_pos"] = {p: (v[1] if v else None) for p, v in pos.items()}
    present = [p for p in PH if pos[p]]
    # Effektive Position: PREP/MODEL/EVAL, die zuerst in DERSELBEN Aktion auftreten, gelten als
    # gleichzeitig (Pipeline-Definition: Skaler im selben Ausdruck wie das Modell) und werden
    # nach Phase geordnet; BU/DU behalten die Zeilenordnung (Inspektion oben im Skript).
    grp = [p for p in ("PREP", "MODEL", "EVAL") if pos[p]]
    minline = {}
    for p in grp:
        k, line = pos[p][0]
        minline[k] = min(minline.get(k, line), line)
    eff = {p: (pos[p][0][0], minline[pos[p][0][0]], RANK[p]) if p in grp else (*pos[p][0], RANK[p])
           for p in present}
    val["phase_order"] = " ".join(sorted(present, key=lambda p: eff[p]))
    cons = [("BU", "MODEL", "<"), ("DU", "PREP", "<"), ("DU", "MODEL", "<"),
            ("PREP", "MODEL", "<="), ("MODEL", "EVAL", "<=")]
    viol = [f"{a}{op}{b}" for a, b, op in cons if a in eff and b in eff and not (
        eff[a][:2] < eff[b][:2] if op == "<" else eff[a] <= eff[b])]
    val["order_violations"] = viol
    mark("canonical_order", viol or None, not viol)
    missing = [p for p in PH if not pos[p]] + ([] if val["bu_explicit"] else ["BU_explicit"])
    val["phases_missing"] = missing
    mark("implicit_crisp", missing + viol or None, not viol and not missing)

    # --- Ruecksprunge
    bj, seq = backjumps(ra, units, mpos, tgt)
    val["n_backjumps"] = len(bj)
    ev["n_backjumps"] = bj
    mark("backjump_any", [b["ref"] for b in bj], len(bj) > 0)
    val["phase_sequence"] = seq

    # --- Struktur
    val["n_actions"] = len(ra.actions)
    val["n_code_actions"] = sum(1 for e in ra.actions if is_code_action(e))
    scripts = sorted({ea._rel(e.path, ra.run_id) for e in ra.actions
                      if e.tool in ("Write", "Edit") and e.path.lower().endswith(".py")})
    val["n_scripts"] = len(scripts)
    ev["n_scripts"] = scripts
    val["n_inline_python"] = sum(1 for e in ra.actions if e.tool == "Bash" and not HEREDOC_NOTES.search(e.code)
                                 and re.search(r"python3?\s+(-c\b|-\s*<<)", e.code))
    mono, single = structure(ra)
    mark("monolith", mono, mono is not None)
    mark("single_pipeline_script", single, single is not None)
    return {"values": val, "evidence": ev}


def action_tokens(e, all_code: str, tgt) -> tuple[set, set]:
    """(PREP-Token, MODEL-Token) einer Code-Aktion: Aufbereitungs-Operatoren bzw.
    Modell-Konfigurationen mit normalisierten Argumenten, Such-Werkzeuge, Spalten-Auswahl,
    abgeleitete Attribute. Kommentare und Importe ausgenommen."""
    lines = [split_comment(x)[0] for x in e.code.split("\n")]
    lines = [x for x in lines if not IMPORT.match(x)]
    code = "\n".join(lines)
    prep, model = set(), set()
    for m in PREP_OPS.finditer(code):
        prep.add(m.group(1) + ":" + norm_prep_args(m.group(2)))
    for x in lines:
        if OUTPUT_REF.search(x):
            continue
        for kind, pat in (("cat.codes", r"\.cat\.codes"), ("map", r"\.map\("), ("categorical", r"pd\.Categorical\(|astype\(\s*['\"]category"),
                          ("replace", r"\.replace\(\s*\{"), ("format", RX["prep_format"].pattern),
                          ("scale255", r"/\s*255(\.0*)?\b")):
            if re.search(pat, x):
                prep.add(kind)
        if drop_is_selection(x, all_code, tgt):
            prep.add("select:" + re.sub(r"\s+", "", RX["drop"].search(x).group(1)))
        if RX["prep_select_tool"].search(x) or RX["prep_construct"].search(x) or derived_attr(x):
            prep.add("line:" + " ".join(x.split()))
        if RX["model_search"].search(x):
            model.add("search:" + " ".join(x.split()))
    for cm in CTOR_ARGS.finditer(code):
        fam = families_in(cm.group(0))
        if fam:
            model.add(fam[0] + ":" + norm_ctor_args(fam[0], cm.group(2)))
    return prep, model


def backjumps(ra, units, mpos, tgt=None) -> tuple[list, list]:
    """Phasenfolge und Ruecksprunge. Ereignisse (in Reihenfolge):
      - Code-Aktion (keine Reparatur nach einem Fehler, keine Verifikation der Ausgaben):
        PREP, wenn sie einen neuen Aufbereitungs-Token einfuehrt; MODEL, wenn sie eine neue
        Modell-Konfiguration oder Suche einfuehrt; EVAL, wenn sie eine neue Baseline-Zeile
        enthaelt; DU, wenn sie neue Inspektionszeilen enthaelt und gar keine PREP/MODEL-Token
        (reine Inspektion). Eine Aktion traegt ihre Phasen in kanonischer Reihenfolge bei.
      - Text nach der ersten Modellanpassung (nicht der Abschlussbericht) mit Baseline-Zahl
        oder Bewertung gegen Ziel/Kriterium: EVAL.
    Die Folge wird lauflaengen-komprimiert; Ruecksprung = Uebergang zu einer frueheren Phase
    (DU < PREP < MODEL < EVAL). BU ist in C/D eine Aussage, keine Aktivitaet, und fehlt hier."""
    all_code = "\n".join(u.s for u in units if u.kind == "code")
    new_by_entry = defaultdict(list)
    for u in units:
        if u.kind == "code" and u.new:
            new_by_entry[u.pos[0]].append(u.s)
    last_action_k = max((k for k, e in enumerate(ra.entries) if e.kind == "action"), default=-1)
    seen_prep, seen_model = set(), set()
    out, seq = [], []
    repairs = repair_actions(ra)

    def push(p, ref):
        if seq and RANK[p] < RANK[seq[-1]]:
            out.append({"ref": ref, "to": p, "from": seq[-1]})
        if not seq or seq[-1] != p:
            seq.append(p)
    for k, e in enumerate(ra.entries):
        if e.kind == "text":
            if (k, 0) > mpos and k < last_action_k and (
                    TX["baseline_num"].search(e.text) or is_business_assessment(e.text)):
                push("EVAL", f"T>#{max([x.n for x in ra.entries[:k] if x.kind == 'action'], default=0)}")
            continue
        if e.kind != "action" or not is_code_action(e) or (e.tool in ("Write", "Edit") and e.is_error):
            continue
        new = new_by_entry.get(k, [])
        repair = e.n in repairs
        verify = e.tool == "Bash" and RX["reads_outputs"].search(e.code) and not families_in(e.code)
        ptok, mtok = action_tokens(e, all_code, tgt)
        S = set()
        if ptok - seen_prep:
            S.add("PREP")
        if mtok - seen_model:
            S.add("MODEL")
        if k >= mpos[0] and any(is_baseline_line(x) and not OUTPUT_REF.search(x) for x in new):
            S.add("EVAL")      # Mehrheitsquote in einer Inspektion vor dem ersten Modell ist keine EVAL
        if not ptok and not mtok and any(is_du_line(x) for x in new):
            S.add("DU")
        seen_prep |= ptok
        seen_model |= mtok
        if not S or verify or repair:
            continue
        for p in sorted(S, key=RANK.get):
            push(p, f"#{e.n}")
    return out, seq


def structure(ra) -> tuple:
    """Monolith: eine Code-Einheit (Datei-Version oder Bash-Befehl) enthaelt DU + PREP +
    MODEL + Vorhersage-Ausgabe; Single-Pipeline-Skript: PREP + MODEL + Vorhersage-Ausgabe."""
    files, versions = {}, []
    for e in ra.actions:
        if not is_code_action(e):
            continue
        if e.tool == "Write":
            files[e.path] = e.code
            versions.append((f"#{e.n}", e.code))
        elif e.tool == "Edit":
            cur, old = files.get(e.path, ""), e.inp.get("old_string", "")
            if old and old in cur and not e.is_error:
                cur = cur.replace(old, e.code) if e.inp.get("replace_all") else cur.replace(old, e.code, 1)
                files[e.path] = cur
                versions.append((f"#{e.n}", cur))
        else:
            versions.append((f"#{e.n}", e.code))
    mono = single = None
    for ref, code in versions:
        lines = [split_comment(x)[0] for x in code.split("\n")]
        txt = "\n".join(lines)
        has_du = any(is_du_line(x) for x in lines)
        has_prep = any(RX[p].search(txt) for p in PREP_PATS)
        has_model = bool(families_in(txt))
        has_pred = bool(RX["pred_write"].search(txt))
        if has_prep and has_model and has_pred and single is None:
            single = ref
        if has_du and has_prep and has_model and has_pred and mono is None:
            mono = ref
    return mono, single


# ============================================================================ Kodierung A/B

def code_fsm(ra) -> dict:
    """Referenz-Codes fuer A/B (nur, was die FSM nicht erzwingt). MODEL-Codes nur aus den
    Sessions der Modeling-Phase (Pruefsonden in DU zaehlen nicht als Modellwahl)."""
    units = build_units(ra)
    phase_of = {k: e.phase for k, e in enumerate(ra.entries)}
    mcode = [u for u in units if u.kind == "code" and phase_of.get(u.pos[0]) == "model"]
    val, ev = {}, {}

    def mark(code, u, v=None):
        val[code] = (u is not None) if v is None else v
        ev[code] = u.ref if isinstance(u, Unit) else u
    fams = {}
    for u in mcode:
        for f in families_in(u.s):
            fams.setdefault(f, u.ref)
    val["n_families"] = len(fams)
    ev["n_families"] = fams
    mark("model_multi_family", sorted(fams), len(fams) >= 2)
    u_cv = first(mcode, lambda u: RX["model_cv"].search(u.s))
    u_ho = first(mcode, lambda u: RX["model_holdout"].search(u.s))
    mark("model_cv", u_cv)
    mark("model_holdout", u_ho)
    td = [u for u in (u_cv, u_ho) if u]
    mark("model_test_design", min(td, key=lambda u: u.pos) if td else None)
    ref, how = tuning(ra, mcode, allowed_phases=("model",))
    mark("model_tuning", ref, ref is not None)
    val["model_tuning_how"] = how
    mark("model_class_weight", first(mcode, lambda u: RX["model_class_weight"].search(u.s)))
    mark("model_assess_detail", first(mcode, lambda u: RX["model_assess_detail"].search(u.s)))
    pcode = [u for u in units if u.kind == "code" and phase_of.get(u.pos[0]) == "prep"]
    mark("prep_construct", first(pcode, lambda u: RX["prep_construct"].search(u.s) or derived_attr(u.s)))
    # Leakage erwogen VOR der Modeling-Phase: BU/DU/PREP-Sessions (Text, Notizen, Kommentare)
    # sowie Freitextfelder des DU-Reports (quality_issues, excluded_columns.reason).
    early = [u for u in units if u.kind in ("text", "notes", "comment")
             and phase_of.get(u.pos[0]) in ("bu", "du", "prep")]
    u_leak = first(early, lambda u: TX["leak"].search(u.s))
    rep = ra.summary.get("du_report") or {}
    rep_txt = " ".join([str(x) for x in rep.get("quality_issues") or []]
                       + [str(x.get("reason", "")) for x in rep.get("excluded_columns") or []])
    mark("du_leakage", u_leak.ref if u_leak else ("summary.du_report" if TX["leak"].search(rep_txt) else None),
         bool(u_leak) or bool(TX["leak"].search(rep_txt)))
    # Baseline-Vergleich ist in A/B Teil des Modeling-Kontrakts (metrics.json dummy_score;
    # gates.check_model: val_score > dummy_score) -> erzwungen, keine freie Entscheidung.
    mark("eval_baseline", "src/crispfsm/gates.py:68-85", "enforced")
    val["n_actions"] = len(ra.actions)
    return {"values": val, "evidence": ev}


# ============================================================================ Statistik

BIN_REACT = [  # Reihenfolge = Tabelle/Abbildung; (Code, Phase, CRISP-1.0-Bezug)
    ("bu_explicit", "BU", "1.3"), ("bu_metric_rationale", "BU", "1.3"),
    ("bu_success_criterion", "BU", "1.1/1.3"), ("bu_business_context", "BU", "1.1/1.2"),
    ("du_inspect_before_model", "DU", "2.2"), ("du_class_dist", "DU", "2.3"),
    ("du_missing", "DU", "2.4"), ("du_quality_ext", "DU", "2.4"), ("du_stats", "DU", "2.2/2.3"),
    ("du_relations", "DU", "2.3"), ("du_leakage", "DU", "2.4"),
    ("prep_explicit", "PREP", "3.1-3.5"), ("prep_select", "PREP", "3.1"), ("prep_construct", "PREP", "3.3"),
    ("model_multi_family", "MODEL", "4.1/4.4"), ("model_cv", "MODEL", "4.2"),
    ("model_test_design", "MODEL", "4.2"), ("model_tuning", "MODEL", "4.3/4.4"),
    ("model_assess_detail", "MODEL", "4.4"),
    ("eval_baseline", "EVAL", "5.1"), ("eval_business", "EVAL", "5.1"), ("eval_review_next", "EVAL", "5.2/5.3"),
    ("doc_notes", "DOC", "Outputs"), ("doc_crisp_sections", "DOC", "Outputs"),
    ("canonical_order", "FLOW", "-"), ("implicit_crisp", "FLOW", "-"),
    ("backjump_any", "FLOW", "-"), ("monolith", "FLOW", "-"),
]
AUX_REACT = ["bu_target", "bu_metric", "leakage_mention_any", "prep_impute", "prep_encode",
             "prep_scale", "prep_select_rule", "model_holdout", "model_class_weight", "doc_upfront",
             "single_pipeline_script", "verify_outputs", "bu_metric_rationale_unit"]
BIN_FSM = ["model_multi_family", "model_cv", "model_test_design", "model_tuning",
           "model_assess_detail", "model_class_weight", "model_holdout", "du_leakage", "prep_construct"]


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def rate(runs, code):
    return sum(1 for r in runs if r["values"].get(code) is True), len(runs)


def describe_num(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": round(statistics.mean(xs), 3), "median": statistics.median(xs),
            "min": min(xs), "max": max(xs)}


def summarize(coded: dict) -> dict:
    S = {"codebook_version": CODEBOOK_VERSION, "scopes": {}}
    scope_def = {"core": lambda r: not r["post_cutoff"], "post_cutoff": lambda r: r["post_cutoff"],
                 "all": lambda r: True}
    for sc, f in scope_def.items():
        by_arm = {a: [r for r in coded.values() if r["arm"] == a and f(r)] for a in ARMS}
        out = {"n": {LETTER[a]: len(v) for a, v in by_arm.items()}, "rates": {}, "fisher_C_vs_D": {}}
        for code in [c for c, _, _ in BIN_REACT] + AUX_REACT:
            out["rates"][code] = {}
            for a in ARMS:
                if a in FSM and code not in BIN_FSM:
                    continue
                out["rates"][code][LETTER[a]] = list(rate(by_arm[a], code))
        for a in FSM:
            out["rates"]["eval_baseline"][LETTER[a]] = "enforced"
        tests = []
        for code, _, _ in BIN_REACT:
            kc, nc = rate(by_arm["crisp_react"], code)
            kd, nd = rate(by_arm["naked_react"], code)
            odds, p = fisher_exact([[kc, nc - kc], [kd, nd - kd]])
            tests.append((code, kc, nc, kd, nd, float(odds), float(p)))
        adj = holm([t[-1] for t in tests])
        for (code, kc, nc, kd, nd, odds, p), pa in zip(tests, adj):
            out["fisher_C_vs_D"][code] = {
                "C": [kc, nc], "D": [kd, nd],
                "odds_ratio": odds if odds == odds and odds not in (float("inf"),) else str(odds),
                "p": round(p, 6), "p_holm": round(pa, 6)}
        out["fisher_family_size"] = len(tests)
        num = {}
        for a in ARMS:
            rs = by_arm[a]
            d = {"n_families": describe_num([r["values"].get("n_families") for r in rs]),
                 "n_families_dist": dict(sorted(Counter(r["values"].get("n_families") for r in rs).items())),
                 "n_actions": describe_num([r["values"].get("n_actions") for r in rs]),
                 "family_runs": dict(Counter(f for r in rs for f in r["evidence"]["model_multi_family"]).most_common()),
                 "tuning_how": dict(Counter(str(r["values"]["model_tuning_how"]) for r in rs).most_common())}
            if a in REACT:
                for k in ("n_backjumps", "n_scripts", "n_code_actions", "n_inline_python"):
                    d[k] = describe_num([r["values"].get(k) for r in rs])
                d["n_backjumps_dist"] = dict(sorted(Counter(r["values"]["n_backjumps"] for r in rs).items()))
                d["phase_order"] = dict(Counter(r["values"]["phase_order"] for r in rs).most_common())
                d["phase_sequence"] = dict(Counter(" ".join(r["values"]["phase_sequence"]) for r in rs).most_common())
                d["phases_missing"] = dict(Counter(p for r in rs for p in r["values"]["phases_missing"]).most_common())
                d["order_violations"] = dict(Counter(v for r in rs for v in r["values"]["order_violations"]).most_common())
                d["backjump_to"] = dict(Counter(b["to"] for r in rs for b in r["evidence"]["n_backjumps"]).most_common())
                d["baseline_before_model_action"] = sum(1 for r in rs if r["values"].get("baseline_before_model_action"))
            num[LETTER[a]] = d
        out["numeric"] = num
        out["outcome_link"] = outcome_link(by_arm)
        S["scopes"][sc] = out
    return S


def outcome_link(by_arm) -> dict:
    """Deskriptiv: pass-Rate und F1 minus Baseline (roh und set-zentriert) mit/ohne Code,
    getrennt innerhalb C und D. Set-zentriert = Differenz minus Mittel aller C+D-Runs des Sets."""
    rs_all = by_arm["crisp_react"] + by_arm["naked_react"]
    by_set = defaultdict(list)
    for r in rs_all:
        if r["diff"] is not None:
            by_set[r["dataset"]].append(r["diff"])
    set_mean = {k: statistics.mean(v) for k, v in by_set.items()}
    out = {}
    for a in REACT:
        rs = by_arm[a]
        out[LETTER[a]] = {}
        for code in [c for c, _, _ in BIN_REACT] + ["model_class_weight"]:
            row = {}
            for flag in (True, False):
                g = [r for r in rs if (r["values"].get(code) is True) == flag]
                dd = [r["diff"] for r in g if r["diff"] is not None]
                dc = [r["diff"] - set_mean[r["dataset"]] for r in g if r["diff"] is not None]
                row["with" if flag else "without"] = {
                    "n": len(g), "pass": sum(1 for r in g if r["success"]),
                    "mean_diff": round(statistics.mean(dd), 4) if dd else None,
                    "mean_diff_set_centered": round(statistics.mean(dc), 4) if dc else None}
            out[LETTER[a]][code] = row
    return out


# ============================================================================ main

def code_run(ra) -> dict:
    return code_react(ra) if ra.arm in REACT else code_fsm(ra)


def run_all(runs_dir: str) -> dict:
    coded = {}
    for rid in ea.list_runs(runs_dir):
        ra = ea.load_run(runs_dir, rid)
        bc = ra.summary.get("baseline_comparison") or {}
        coded[rid] = {"arm": ra.arm, "letter": LETTER[ra.arm], "dataset": ra.dataset, "rep": ra.rep,
                      "post_cutoff": ra.post_cutoff, "outcome": ra.summary.get("outcome"),
                      "success": bool(ra.summary.get("success")),
                      "f1": ra.summary.get("best_f1_measured"), "diff": bc.get("diff"), **code_run(ra)}
    return coded


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default=ea.DEFAULT_RUNS)
    ap.add_argument("--show", nargs="*", default=None, help="Codes + Belege dieser Runs ausgeben, nichts schreiben")
    ap.add_argument("--out", default=OUT)
    ARGS = ap.parse_args()
    if ARGS.show:
        for rid in ARGS.show:
            res = code_run(ea.load_run(ARGS.runs_dir, rid))
            print(f"== {rid}")
            for k, v in res["values"].items():
                print(f"  {k:28s} {str(v)[:60]:60s} {str(res['evidence'].get(k, ''))[:90]}")
        return
    coded = run_all(ARGS.runs_dir)
    summ = summarize(coded)
    os.makedirs(os.path.dirname(ARGS.out), exist_ok=True)
    with open(ARGS.out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"codebook": "thesis/facts/codebook.md", "version": CODEBOOK_VERSION,
                            "runs_dir": "../thesis-crisp-dm-runs", "n_runs": len(coded),
                            "note": "regelbasiert, explorativ; blind gegenkodiert an 24 C/D-Runs "
                                    "(coding_agreement.json, kap7.md F7-9xx), A/B-Referenzcodes nicht"},
                   "summary": summ, "runs": coded}, f, indent=1, ensure_ascii=False)
    print(f"{len(coded)} Runs kodiert -> {os.path.relpath(ARGS.out, REPO)}")
    for sc in ("core", "post_cutoff"):
        S = summ["scopes"][sc]
        print(f"\n[{sc}] {'code':26s} {'A':>7s} {'B':>7s} {'C':>7s} {'D':>7s}  p(C,D)   p_holm")
        for code, _, _ in BIN_REACT:
            r = S["rates"][code]
            cells = [("enf" if r.get(x) == "enforced" else f"{r[x][0]}/{r[x][1]}") if x in r else "-" for x in "ABCD"]
            fz = S["fisher_C_vs_D"][code]
            print(f"   {code:26s} {cells[0]:>7s} {cells[1]:>7s} {cells[2]:>7s} {cells[3]:>7s}  {fz['p']:.4f}  {fz['p_holm']:.4f}")


ARGS = argparse.Namespace(runs_dir=ea.DEFAULT_RUNS)

if __name__ == "__main__":
    main()
