"""Inter-coder agreement: rule-based coder (process_coding.json) vs. blind second coding
(process_coding_manual.json) on the validation sample (validation_sample.py, 24 runs C/D).

Per binary code: percent agreement, Cohen's kappa (undefined when the expected agreement is 1,
i.e. both coders assign the same single value to every run), prevalence per coder, list of
disagreements with both pieces of evidence. Per count code: exact agreement and mean absolute
difference. Sensitivity: the same statistics with the second coder's `alt_intent` values
(code definition instead of the literal decision rule, where the two differ). Full-set checks:
how often the error patterns seen in the sample occur in all 180 C/D runs (text-search
heuristics on thesis/facts/derived/actions/, candidates only).

Output: thesis/facts/derived/coding_agreement.json (+ table on stdout).
Run: uv run python thesis/scripts/coding_agreement.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "thesis" / "facts" / "derived"
AUTO = DER / "process_coding.json"
MAN = DER / "process_coding_manual.json"
OUT = DER / "coding_agreement.json"

# main codes = BIN_REACT in code_process.py (28), auxiliary codes as listed in the codebook
MAIN = ["bu_explicit", "bu_metric_rationale", "bu_success_criterion", "bu_business_context",
        "du_inspect_before_model", "du_class_dist", "du_missing", "du_quality_ext", "du_stats",
        "du_relations", "du_leakage", "prep_explicit", "prep_select", "prep_construct",
        "model_multi_family", "model_cv", "model_test_design", "model_tuning", "model_assess_detail",
        "eval_baseline", "eval_business", "eval_review_next", "doc_notes", "doc_crisp_sections",
        "canonical_order", "implicit_crisp", "backjump_any", "monolith"]
AUX = ["bu_target", "bu_metric", "leakage_mention_any", "prep_select_rule", "model_class_weight",
       "doc_upfront", "verify_outputs", "single_pipeline_script"]
COUNTS = ["n_families", "n_backjumps", "n_actions", "n_code_actions", "n_scripts", "n_inline_python"]
THRESH_KAPPA, THRESH_AGREE = 0.6, 0.9
# Correction of a clerical slip in the second coder's file (reviews K7-04c and R7-01, 2026-11-09):
# meta.policy puts the literal-rule value into codes.<code>.value and the value by the code's
# definition into alt_intent. In this run the coder swapped the two fields: value = 1 with the
# definition reason ("target referenced via loop variable"), alt_intent = 0 with the reason
# "literal: loop variable c is none of name/y/TARGET/target/label". The swap is undone in both
# readings: rule value 0 (as coder A, no disagreement), definition value 1 (F7-914).
SWAP_FIX = {("main-vehicle-naked_react-r4", "du_class_dist"): {"rule": 0, "alt": 1}}


def kappa(a: list[int], b: list[int]) -> tuple[float, float | None, dict]:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    k = None if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)
    cells = {"both_1": sum(1 for x, y in zip(a, b) if x and y),
             "both_0": sum(1 for x, y in zip(a, b) if not x and not y),
             "auto1_manual0": sum(1 for x, y in zip(a, b) if x and not y),
             "auto0_manual1": sum(1 for x, y in zip(a, b) if not x and y)}
    return po, k, cells


def verdict(po: float, k: float | None, var_auto: bool, var_man: bool) -> str:
    if k is None or not (var_auto and var_man):
        # no variance in at least one marginal: kappa undefined or uninformative
        return "reliable (agreement, no variance)" if po >= THRESH_AGREE else "not reliable"
    return "reliable" if k >= THRESH_KAPPA else "not reliable"


ACT = DER / "actions"      # extract_actions.py output (gitignored, reproducible)
BUS = re.compile(r"\bbusiness\b|stakeholder|\bassum(e|es|ed|ing|ption)|\bcostly\b|cost of (a |an )?"
                 r"(error|mistake|false|miss)|false (negative|positive)s? (are|is|would)|project plan|\bobjectives?\b", re.I)
PHASE_HEAD = re.compile(r"business|data (mining )?understanding|prep|model|evaluat|result", re.I)
EVAL_VERB_FORMS = re.compile(r"(meeting|satisfying|achieving|fulfilling)[^.\n]{0,30}(success criteri|objective|goal)", re.I)


def _section(run: str, n: str) -> list[str]:
    """Content lines of tool action #n in the action list of a run."""
    out, inside = [], False
    for ln in (ACT / f"{run}.md").read_text(encoding="utf-8").splitlines():
        if ln.startswith(f"### #{n} "):
            inside = True
            continue
        if inside and (ln.startswith("### #") or ln.startswith("**[Text]")):
            break
        if inside:
            m = re.match(r"\s*\d+\| (.*)$", ln)
            if m:
                out.append(m.group(1))
    return out


def _line(run: str, ref: str | None) -> str:
    m = re.match(r"#(\d+):L(\d+)", ref or "")
    if not m:
        return ""
    lines = _section(run, m.group(1))
    k = int(m.group(2))
    return lines[k - 1] if 0 < k <= len(lines) else ""


def full_set_checks(auto: dict, fisher_core: dict) -> dict:
    """How often do the error patterns seen in the sample occur in all 180 C/D runs?
    Text-search heuristics on the action lists; the hits are candidates, the sample cases are verified."""
    cd = {k: v for k, v in auto.items() if v["arm"] in ("crisp_react", "naked_react")}
    txt = {k: (ACT / f"{k}.md").read_text(encoding="utf-8") for k in cd}
    core = {k: not v["post_cutoff"] for k, v in cd.items()}
    out: dict = {"n_runs": len(cd)}

    out["prep_explicit_false"] = [{"run": k, "core": core[k], "tfidf": "TfidfTransformer(" in txt[k],
                                   "phases_missing": v["values"]["phases_missing"],
                                   "implicit_crisp": v["values"]["implicit_crisp"]}
                                  for k, v in cd.items() if not v["values"]["prep_explicit"]]
    out["eval_first_from_final_report"] = [
        k for k, v in cd.items()
        if (v["values"]["first_pos"].get("EVAL") or "").startswith("T>#")
        and int(v["values"]["first_pos"]["EVAL"][3:]) == v["values"]["n_actions"]]
    sel = []
    for k, v in cd.items():
        if v["values"]["prep_select_rule"]:
            ln = _line(k, v["evidence"]["prep_select_rule"])
            var = re.search(r"drop\(\s*columns\s*=\s*([A-Za-z_]\w*)\s*\)", ln)
            assign = ""
            if var:
                ma = re.search(rf"^\s*\d+\| \s*{re.escape(var.group(1))}\s*=\s*(.+)$", txt[k], re.M)
                assign = ma.group(1) if ma else ""
            data_rule = bool(re.search(r"\bfor\b|nunique|\.index|Threshold|SelectKBest|const", ln + " " + assign))
            sel.append({"run": k, "line": ln.strip()[:100], "assignment": assign.strip()[:80], "data_rule": data_rule})
    out["prep_select_rule_hits"] = sel
    out["baseline_word_as_string_marker"] = [k for k in cd if re.search(r"split\(\s*['\"]#\s*[Bb]aseline", txt[k])]
    out["du_class_dist_false"] = [{"run": k, "loop_variable_value_counts": bool(
        re.search(r"\[(c|col)\]\.value_counts", txt[k]))} for k, v in cd.items() if not v["values"]["du_class_dist"]]

    bus = {}
    for k, v in cd.items():
        if not v["values"]["bu_business_context"]:
            continue
        ref = v["evidence"]["bu_business_context"]
        kind = "text_or_comment"
        if ref.startswith("N#"):
            hits = [ln for ln in _section(k, ref[2:]) if BUS.search(ln)]
            kind = "note_heading_only" if hits and all(h.lstrip().startswith("#") for h in hits) else "note_body"
        bus.setdefault(kind, []).append(k)
    c_core = sum(1 for k, v in cd.items() if v["letter"] == "C" and core[k] and v["values"]["bu_business_context"])
    d_core = sum(1 for k, v in cd.items() if v["letter"] == "D" and core[k] and v["values"]["bu_business_context"])
    c_core_wo = c_core - sum(1 for k in bus.get("note_heading_only", []) if core[k] and cd[k]["letter"] == "C")
    p_wo = fisher_exact([[c_core_wo, 78 - c_core_wo], [d_core, 78 - d_core]])[1]
    # Holm over the same 28 main codes, with the p-value of bu_business_context replaced
    ps = {c: (p_wo if c == "bu_business_context" else fisher_core[c]["p"]) for c in MAIN}
    run_max, holm_wo = 0.0, None
    for rank, c in enumerate(sorted(MAIN, key=lambda c: ps[c])):
        run_max = max(run_max, min(1.0, (len(MAIN) - rank) * ps[c]))
        if c == "bu_business_context":
            holm_wo = run_max
    out["bu_business_context"] = {
        "by_evidence_kind": bus, "core_C": c_core, "core_D": d_core,
        "core_C_without_heading_only": c_core_wo,
        "fisher_p_core_without_heading_only": round(p_wo, 4),
        "holm_p_core_without_heading_only": round(holm_wo, 4)}
    var = []
    for k, v in cd.items():
        if v["values"]["doc_notes"] and not v["values"]["doc_crisp_sections"]:
            heads = [ln for ln in _section(k, v["evidence"]["doc_notes"][2:])
                     if ln.lstrip().startswith("#") and PHASE_HEAD.search(ln)]
            if len(heads) >= 3:
                var.append({"run": k, "core": core[k], "headings": [h.strip()[:40] for h in heads[:6]]})
    out["doc_crisp_sections_variant_headings"] = var
    out["eval_business_verb_form_misses"] = [k for k, v in cd.items()
                                             if not v["values"]["eval_business"] and EVAL_VERB_FORMS.search(txt[k])]
    return out


def main() -> None:
    auto_all = json.loads(AUTO.read_text(encoding="utf-8"))
    auto = auto_all["runs"]
    fisher_core = auto_all["summary"]["scopes"]["core"]["fisher_C_vs_D"]
    man = json.loads(MAN.read_text(encoding="utf-8"))
    runs = list(man["runs"])

    def a_val(run, code):
        v = auto[run]["values"][code]
        return int(v) if isinstance(v, bool) else v

    def a_ev(run, code):
        return auto[run]["evidence"].get(code)

    def m_val(run, code, use_alt=False):
        r = man["runs"][run]
        if (run, code) in SWAP_FIX:
            return SWAP_FIX[(run, code)]["alt" if use_alt else "rule"]
        if use_alt and code in r["alt_intent"]:
            return r["alt_intent"][code]["value"]
        return r["codes"][code]["value"]

    res = {"meta": {
        "n_runs": len(runs), "runs": runs,
        "coder_a": "thesis/scripts/code_process.py (rule-based) -> process_coding.json",
        "coder_b": "LLM agent (Claude Opus 5.5), blind to coder A -> process_coding_manual.json",
        "kappa_undefined_when": "expected agreement p_e = 1 (both coders constant on the same value)",
        "reliability_rule": f"kappa >= {THRESH_KAPPA}; if at least one coder is constant over the sample "
                            f"(kappa then undefined or 0 for any disagreement): agreement >= {THRESH_AGREE:.0%}; "
                            "same rule for the rule and the alt reading",
        "alt": "sensitivity: coder B with alt_intent values (definition instead of literal rule)",
        "swap_fix": {f"{r}:{c}": v for (r, c), v in SWAP_FIX.items()},
    }, "binary": {}, "counts": {}}

    for group, codes in (("main", MAIN), ("aux", AUX)):
        for code in codes:
            entry = {"group": group}
            for tag, use_alt in (("rule", False), ("alt", True)):
                a = [a_val(r, code) for r in runs]
                b = [m_val(r, code, use_alt) for r in runs]
                po, k, cells = kappa(a, b)
                var_a, var_b = len(set(a)) > 1, len(set(b)) > 1
                d = {"agreement": round(po, 4), "kappa": None if k is None else round(k, 4),
                     "prevalence_auto": sum(a), "prevalence_manual": sum(b), "cells": cells,
                     "verdict": verdict(po, k, var_a, var_b)}
                if tag == "rule":
                    d["disagreements"] = [{
                        "run": r, "auto": x, "auto_evidence": a_ev(r, code),
                        "manual": y, "manual_evidence": man["runs"][r]["codes"][code]["evidence"],
                        "manual_alt": man["runs"][r]["alt_intent"].get(code, {}).get("value"),
                    } for r, x, y in zip(runs, a, b) if x != y]
                entry[tag] = d
            res["binary"][code] = entry

    for code in COUNTS:
        a = [a_val(r, code) for r in runs]
        b = [m_val(r, code) for r in runs]
        balt = [m_val(r, code, True) for r in runs]
        exact = sum(x == y for x, y in zip(a, b)) / len(runs)
        mad = sum(abs(x - y) for x, y in zip(a, b)) / len(runs)
        res["counts"][code] = {
            "exact_agreement": round(exact, 4), "mean_abs_diff": round(mad, 4),
            "exact_agreement_alt": round(sum(x == y for x, y in zip(a, balt)) / len(runs), 4),
            "mean_auto": round(sum(a) / len(a), 3), "mean_manual": round(sum(b) / len(b), 3),
            "disagreements": [{"run": r, "auto": x, "manual": y} for r, x, y in zip(runs, a, b) if x != y],
        }

    # summary lists
    rel = [c for c in MAIN + AUX if res["binary"][c]["rule"]["verdict"].startswith("reliable")]
    res["summary"] = {
        "reliable_rule": rel,
        "not_reliable_rule": [c for c in MAIN + AUX if c not in rel],
        "reliable_alt": [c for c in MAIN + AUX if res["binary"][c]["alt"]["verdict"].startswith("reliable")],
        "main_mean_agreement": round(sum(res["binary"][c]["rule"]["agreement"] for c in MAIN) / len(MAIN), 4),
        "main_kappas_defined": sum(1 for c in MAIN if res["binary"][c]["rule"]["kappa"] is not None),
        "n_disagreement_cells_main": sum(len(res["binary"][c]["rule"]["disagreements"]) for c in MAIN),
        "n_disagreement_cells_aux": sum(len(res["binary"][c]["rule"]["disagreements"]) for c in AUX),
        "n_cells_main": len(MAIN) * len(runs), "n_cells_aux": len(AUX) * len(runs),
    }
    res["full_set_checks"] = (full_set_checks(auto, fisher_core) if ACT.exists()
                              else "actions/ missing: run extract_actions.py")
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"{'code':26s} {'grp':4s} {'agree':>6s} {'kappa':>7s} {'A':>3s} {'B':>3s}  {'agree_alt':>9s} {'k_alt':>7s}  verdict")
    for c in MAIN + AUX:
        e = res["binary"][c]
        r, a = e["rule"], e["alt"]
        ks = "  n.d." if r["kappa"] is None else f"{r['kappa']:7.3f}"
        ka = "  n.d." if a["kappa"] is None else f"{a['kappa']:7.3f}"
        print(f"{c:26s} {e['group']:4s} {r['agreement']:6.1%} {ks:>7s} {r['prevalence_auto']:3d} {r['prevalence_manual']:3d}"
              f"  {a['agreement']:9.1%} {ka:>7s}  {r['verdict']}")
    print()
    for c in COUNTS:
        e = res["counts"][c]
        print(f"{c:18s} exact {e['exact_agreement']:6.1%}  MAD {e['mean_abs_diff']:.3f}  "
              f"mean A {e['mean_auto']}  mean B {e['mean_manual']}")
    print()
    print(json.dumps(res["summary"], indent=1))
    print()
    print(json.dumps(res["full_set_checks"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
