#!/usr/bin/env python
"""Abbildungen fuer Kap. 5 (Experiment Execution) und Kap. 6 (Results).

Liest nur Rohdaten (Quellen-Rang 1 nach docs/schreib-workflow.md):
  - <runs>/main-*/summary.json, events.jsonl, transcripts/*.jsonl (Zeitstempel)
  - <runs>/leak-*, spur-*: nur die Transkript-Zeitstempel (Zeitleiste)
  - <runs>/_main/analysis/analysis.json, <runs>/_post_cutoff/analysis/analysis.json
  - <runs>/_main/main_runs.json (Run-SD je Zelle, Abgleich der Zellmittel)
  - git log des Runs-Repos (Zeitleiste, erster Durchlauf)
dazu die Manifeste (Rang 2; baseline_f1_cv -> Schwere-Terzile). Ereignisse vor dem
Runs-Repo (Kalibrierung 08.08., Varianz-Scan 08.-09.08.) und die Meeting-Termine
stammen aus docs/roadmap.md (Rang 3) und stehen unten als Konstanten mit Quelle.

Archivierte Runs (automem_*, gitctx_*, promptv1_*) werden nie gelesen. Der erste
(kontaminierte) Durchlauf erscheint in fig5_timeline nur als Zeitspanne, ohne Ergebniswerte:
Enden und Zwischenpunkte aus den Commit-Zeitpunkten des Runs-Repos, Beginn (23.09. 18:12) und
Beginn der Leakage-Batches (24.10. 12:00) als Konstanten aus den summary.json-Zeitstempeln der
archivierten Runs (F5-431, F5-433), die Design-Revision aus dem Harness-Commit e0e190a.

Aufruf aus dem Repo-Root:
    uv run python thesis/scripts/make_figures.py
        [--runs-dir ../thesis-crisp-dm-runs] [--out-dir thesis/figs]
        [--preview-dir <Ordner fuer PNG-Vorschau>] [--only fig6_cd fig6_heatmap ...]

Farben: Die vier Arme bilden den Enforcement-Gradienten A > B > C > D und sind deshalb
ordinal kodiert -- ein Farbton (Blau), Helligkeit monoton von A (dunkel) nach D (hell),
geprueft mit dem dataviz-Validator (--ordinal, Flaeche #ffffff: monoton, Abstand
>= 0.06 L, hellste Stufe 2.11:1). Jeder Arm hat zusaetzlich eine eigene Markerform,
damit die Arme in Graustufen und fuer Farbsehschwache unterscheidbar bleiben.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import (Formatter, FuncFormatter, MaxNLocator,  # noqa: E402
                               MultipleLocator)

REPO = Path(__file__).resolve().parents[2]
CM = 1 / 2.54
FULL = 15.0 * CM
HALF = 7.3 * CM

# --- Arme (Benennung exakt wie Kap. 3, tab:3:arms) ------------------------------------
ARMS = ["hard_fsm", "soft_fsm", "crisp_react", "naked_react"]
LETTER = {"hard_fsm": "A", "soft_fsm": "B", "crisp_react": "C", "naked_react": "D"}
NAME = {"hard_fsm": "Hard FSM", "soft_fsm": "Soft FSM",
        "crisp_react": "CRISP-ReAct", "naked_react": "Naked ReAct"}
# Blau-Rampe der dataviz-Referenzpalette, Stufen 700 / 550 / 400 / 250.
COLOR = {"hard_fsm": "#0d366b", "soft_fsm": "#1c5cab",
         "crisp_react": "#3987e5", "naked_react": "#86b6ef"}
MARKER = {"hard_fsm": "s", "soft_fsm": "D", "crisp_react": "^", "naked_react": "o"}
# Flaechenausgleich: Raute und Quadrat wirken bei gleicher markersize groesser als Kreis/Dreieck.
MSCALE = {"hard_fsm": 0.88, "soft_fsm": 0.80, "crisp_react": 1.05, "naked_react": 1.0}

# --- Tinte / Flaechen (dataviz-Referenz, helle Variante; Druck = weisse Flaeche) ------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
MID = "#f0efec"
CRITICAL = "#d03b3b"

# --- Datensaetze (Anzeigenamen wie tab:3:datasets) ------------------------------------
DS_LABEL = {
    "wine": "wine", "breast_cancer": "breast cancer", "cnae_9": "cnae-9",
    "covertype": "covertype", "phoneme": "phoneme", "fashion_mnist": "fashion-mnist",
    "adult": "adult", "vehicle": "vehicle", "bank_marketing": "bank-marketing",
    "credit_g": "credit-g", "connect_4": "connect-4", "eucalyptus": "eucalyptus",
    "cmc": "cmc", "student_health": "student health", "ev_purchase": "EV purchase",
}
POST_CUTOFF = ("ev_purchase", "student_health")
TERCILE_SIZES = (("easy", 4), ("medium", 4), ("hard", 5))  # Decker 14.09., Protokoll Punkt 7a

# --- Ereignisse ausserhalb der Run-Daten (Quelle: docs/roadmap.md, Rang 3) --------------
CALIBRATION = (datetime(2026, 8, 8), datetime(2026, 8, 9))        # roadmap §2 "Kalibrierungslauf (08.08.)"
VARIANCE_SCAN = (datetime(2026, 8, 8), datetime(2026, 8, 10))     # roadmap §2 "Varianz-Scan (08.-09.08.)"
EVENTS = [  # (Datum, Kurzlabel) -- Langform steht in der Caption
    (datetime(2026, 9, 14, 12), "design\nfixed"),                 # meetings/2026-09-14-entscheidungen.md
    (datetime(2026, 10, 9, 12), "leakage\nadded"),               # meetings/2026-10-09-kern-ergebnisse.md
]
# Design-Revision 23.09. (decisions-2026-09-23.md): Zeitpunkt = Commit im Harness-Repo (git log, Rang 2).
DESIGN_COMMIT = "e0e190a"  # "feat: Kern-Neustart-Design (S1 Split 70/10/20, B3, S4 Kontingent, S8)", 23.09. 18:11
# Commits des Runs-Repos fuer den ersten Durchlauf (Quelle: git log, Rang 1).
PASS1_COMMITS = {  # Kurz-Hash -> Label (was mit dem Commit fertig war)
    "f69d795": "D", "f0d2c8d": "C", "9b91bae": "A", "15df189": "B", "3d32ff4": "PC",
}
# Beginn des ersten Durchlaufs und seine Leakage-Batches: Zeitstempel der summary.json-Dateien der
# archivierten Runs (F5-431, F5-433; nur Zeitpunkte, keine Ergebniswerte -- die archivierten Ordner
# werden hier nicht gelesen). Erster Kern-Run D endete 23.09. 18:12, eine Minute nach DESIGN_COMMIT.
PASS1_START = datetime(2026, 9, 23, 18, 12)
PASS1_LEAK_START = datetime(2026, 10, 24, 12, 0)   # erster automem_leak-Run; Ende = Commit PASS1_LEAK_END
PASS1_LEAK_END = "3d979f6"  # "Leakage-Batch 2 (spurious) Arm D: 39/39 ..." 24.10. 15:20
ARCHIVE_COMMIT = "4a17bf4"  # "Archiv: alle Runs mit geteiltem Auto-Memory -> automem_*"


# ======================================================================================
# Daten
# ======================================================================================

@dataclass
class Run:
    run_id: str
    arm: str
    dataset: str
    rep: int
    outcome: str
    fail_reason: str | None
    f1: float | None          # best_f1_measured
    baseline: float           # baseline_comparison.baseline_f1 (Test-Anteil, dieselben Zeilen)
    cost: float               # total_cost_usd (inkl. Prueferkosten bei B)
    gate_fails: Counter = field(default_factory=Counter)  # Phase -> Anzahl gescheiterter Gates
    start: datetime | None = None
    end: datetime | None = None

    @property
    def post_cutoff(self) -> bool:
        return self.dataset in POST_CUTOFF

    @property
    def delta(self) -> float | None:
        return None if self.f1 is None else self.f1 - self.baseline


def to_local(iso: str) -> datetime:
    """UTC-Zeitstempel -> Ortszeit Europe/Berlin (naiv). 2026 endet die Sommerzeit am
    25.10. um 01:00 UTC; davor +2 h, danach +1 h (alle Zeitstempel liegen Aug.-Okt.)."""
    t = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    off = 2 if t < datetime(2026, 10, 25, 1, 0, tzinfo=timezone.utc) else 1
    return (t + timedelta(hours=off)).replace(tzinfo=None)


def _ts_of(line: str) -> str | None:
    try:
        return json.loads(line).get("timestamp")
    except (json.JSONDecodeError, AttributeError):
        return None


def transcript_span(run_dir: Path) -> tuple[datetime | None, datetime | None]:
    """Erster und letzter Zeitstempel ueber alle Transkripte eines Runs."""
    first, last = [], []
    for f in sorted((run_dir / "transcripts").glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                ts = _ts_of(line)
                if ts:
                    first.append(to_local(ts))
                    break
        with open(f, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 262144))
            tail = fh.read().decode("utf-8", errors="ignore").splitlines()
        for line in reversed(tail):
            ts = _ts_of(line)
            if ts:
                last.append(to_local(ts))
                break
    return (min(first) if first else None, max(last) if last else None)


def parse_run_id(run_id: str) -> tuple[str, str, int]:
    _, ds, arm, rep = run_id.split("-")
    return ds, arm, int(rep.lstrip("r"))


def load_runs(runs: Path, with_time: bool) -> list[Run]:
    out = []
    for d in sorted(runs.glob("main-*")):
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        ds, arm, rep = parse_run_id(s["run_id"])
        assert arm == s["arm"], s["run_id"]
        # Plausibilitaet des sauberen Stands (roadmap "sauber (26.10.)"/"(27.10.)")
        assert s["harness"]["commit"].startswith("14e99e4"), s["run_id"]
        assert s["harness"]["dirty"] is False and s["holdout_intact"] is True, s["run_id"]
        assert s["access_audit"]["flagged"] is False, s["run_id"]
        r = Run(run_id=s["run_id"], arm=arm, dataset=ds, rep=rep, outcome=s["outcome"],
                fail_reason=s["fail_reason"], f1=s["best_f1_measured"],
                baseline=s["baseline_comparison"]["baseline_f1"],
                cost=s["total_cost_usd"])
        for line in (d / "events.jsonl").read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e.get("gate_passed") is False:
                r.gate_fails[e["phase"]] += 1
        if with_time:
            r.start, r.end = transcript_span(d)
        out.append(r)
    n_core = sum(not r.post_cutoff for r in out)
    n_pc = sum(r.post_cutoff for r in out)
    assert (n_core, n_pc) == (312, 48), (n_core, n_pc)
    return out


def load_spans(runs: Path, prefix: str) -> list[tuple[str, datetime, datetime]]:
    out = []
    for d in sorted(runs.glob(prefix + "-*")):
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        a, b = transcript_span(d)
        if a is not None:
            out.append((s["arm"], a, b))
    return out


GIT_CACHE = REPO / "thesis" / "facts" / "derived" / "git_commits.json"


def git_commits(runs: Path, rev: str | None = None, need: tuple[str, ...] = ()) -> dict[str, tuple[datetime, str]]:
    """Commits (Kurz-Hash -> Ortszeit, Betreff). Im Arbeits-Repo aus git log; das Ergebnis wird in
    derived/git_commits.json gespeichert. Die Pruefer-Repos sind Schnappschuesse ohne diese Historie
    (Anhang B); fehlt ein benoetigter Commit (`rev` bzw. alle aus `need`), werden die gespeicherten Werte
    gelesen. Schluessel nach Rolle, nicht nach Ordnername (ZIP-Downloads heissen anders)."""
    key = f"harness@{rev}" if rev else "runs"
    fmt = "%h%x1f%cI%x1f%s"
    p = subprocess.run(["git", "-C", str(runs), "log", f"--format={fmt}", "--abbrev=7"] + ([rev] if rev else []),
                       capture_output=True, text=True, encoding="utf-8")
    out = {}
    for line in p.stdout.splitlines() if p.returncode == 0 else []:
        h, ts, subj = line.split("\x1f")
        t = datetime.fromisoformat(ts)  # Ortszeit mit Offset -> naiv Ortszeit
        out[h[:7]] = (t.replace(tzinfo=None), subj)
    cache = json.loads(GIT_CACHE.read_text(encoding="utf-8")) if GIT_CACHE.exists() else {}
    if all(h in out for h in ([rev[:7]] if rev else list(need))):
        stored = {h: [t.isoformat(), s] for h, (t, s) in out.items()}
        if cache.get(key) != stored:
            cache[key] = stored
            GIT_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        return out
    if key not in cache:
        raise SystemExit(f"Git-Historie fuer {key} fehlt, und {GIT_CACHE} enthaelt sie nicht")
    print(f"  Hinweis: keine Git-Historie fuer {key} -> gespeicherte Werte aus {GIT_CACHE.name}")
    return {h: (datetime.fromisoformat(t), s) for h, (t, s) in cache[key].items()}


def terciles(manifests: Path, datasets: list[str]) -> tuple[list[str], dict[str, str]]:
    """Reihenfolge leicht -> schwer nach baseline_f1_cv (absteigend), Terzile 4/4/5."""
    cv = {}
    for ds in datasets:
        m = yaml.safe_load((manifests / f"{ds}.yaml").read_text(encoding="utf-8"))
        cv[ds] = float(m["baseline_f1_cv"])
    order = sorted(datasets, key=lambda d: -cv[d])
    terc, i = {}, 0
    for name, size in TERCILE_SIZES:
        for ds in order[i:i + size]:
            terc[ds] = name
        i += size
    assert i == len(order)
    return order, terc


@dataclass
class Data:
    runs: list[Run]
    an_core: dict
    an_pc: dict
    main_runs: dict
    order: list[str]
    tercile: dict[str, str]
    runs_dir: Path

    def core(self) -> list[Run]:
        return [r for r in self.runs if not r.post_cutoff]

    def pc(self) -> list[Run]:
        return [r for r in self.runs if r.post_cutoff]

    def cell(self, ds: str, arm: str) -> list[Run]:
        rs = sorted((r for r in self.runs if r.dataset == ds and r.arm == arm), key=lambda r: r.rep)
        assert len(rs) == 6, (ds, arm, len(rs))
        return rs

    def baseline(self, ds: str) -> float:
        vals = {round(r.baseline, 9) for r in self.runs if r.dataset == ds}
        assert len(vals) == 1, (ds, vals)
        return vals.pop()


def load_data(runs_dir: Path, with_time: bool) -> Data:
    runs = load_runs(runs_dir, with_time)
    an_core = json.loads((runs_dir / "_main/analysis/analysis.json").read_text(encoding="utf-8"))
    an_pc = json.loads((runs_dir / "_post_cutoff/analysis/analysis.json").read_text(encoding="utf-8"))
    main_runs = json.loads((runs_dir / "_main/main_runs.json").read_text(encoding="utf-8"))
    core_ds = sorted({r.dataset for r in runs if not r.post_cutoff})
    order, terc = terciles(REPO / "manifests", core_ds)
    d = Data(runs, an_core, an_pc, main_runs, order, terc, runs_dir)
    # Abgleich: Zellmittel aus den Summaries == Matrix in analysis.json (f1_measured)
    met = an_core["metrics"]["f1_measured"]
    for i, ds in enumerate(met["datasets"]):
        for j, arm in enumerate(an_core["arms"]):
            m = statistics.fmean(r.f1 for r in d.cell(ds, arm))
            assert abs(m - met["matrix"][i][j]) < 1e-9, (ds, arm)
    return d


# ======================================================================================
# Stil
# ======================================================================================

def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 3,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "lines.solid_capstyle": "round",
        "hatch.linewidth": 0.5,
        "axes.unicode_minus": True,
    })


def ms(arm: str, base: float = 4.6) -> float:
    return base * MSCALE[arm]


def arm_label(arm: str) -> str:
    return f"{LETTER[arm]} – {NAME[arm]}"


def arm_handles(arms=ARMS, base: float = 4.6) -> list[Line2D]:
    return [Line2D([], [], ls="none", marker=MARKER[a], ms=ms(a, base), mfc=COLOR[a],
                   mec=SURFACE, mew=0.6, label=arm_label(a)) for a in arms]


def mark(ax, x, y, arm, base=4.6, hollow=False, zorder=3, **kw):
    """Arm-Marker mit 2-px-Flaechenring (dataviz: surface ring); hollow = FAIL."""
    if hollow:
        return ax.plot(x, y, ls="none", marker=MARKER[arm], ms=ms(arm, base), mfc=SURFACE,
                       mec=COLOR[arm], mew=0.9, zorder=zorder, **kw)
    return ax.plot(x, y, ls="none", marker=MARKER[arm], ms=ms(arm, base), mfc=COLOR[arm],
                   mec=SURFACE, mew=0.6, zorder=zorder, **kw)


def fmt_signed(v: float, nd: int = 3) -> str:
    if round(v, nd) == 0:  # kein Vorzeichen fuer gerundete Null (wine: alle Arme = Baseline)
        return f"{0:.{nd}f}"
    s = f"{v:+.{nd}f}"
    return s.replace("-", "−")


class PlainFormatter(Formatter):
    """So wenige Nachkommastellen wie die Ticks exakt brauchen, mindestens zwei (0.70 statt
    0.7); kein Offset. signed=True setzt '+' vor positive Werte, die Null bleibt '0'."""

    def __init__(self, signed: bool = False, min_decimals: int = 2):
        self.signed = signed
        self.min_decimals = min_decimals
        self.nd = min_decimals

    def set_locs(self, locs):
        super().set_locs(locs)
        self.nd = self.min_decimals
        while self.nd < 6 and any(abs(round(v, self.nd) - v) > 1e-9 for v in locs):
            self.nd += 1

    def __call__(self, x, pos=None):
        if abs(x) < 1e-12:
            return "0"
        s = f"{x:.{self.nd}f}".replace("-", "−")
        return "+" + s if self.signed and x > 0 else s


def nice_y(ax, n: int = 3, signed: bool = False) -> None:
    ax.yaxis.set_major_locator(MaxNLocator(n, steps=[1, 2, 5, 10], min_n_ticks=2))
    ax.yaxis.set_major_formatter(PlainFormatter(signed))


def save(fig, name: str, out_dir: Path, preview_dir: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None,
                                                    "Creator": "thesis/scripts/make_figures.py"})
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  -> {name}.pdf")


def tercile_bounds(order: list[str], terc: dict[str, str]) -> list[tuple[str, int, int]]:
    out, start = [], 0
    for i in range(1, len(order) + 1):
        if i == len(order) or terc[order[i]] != terc[order[start]]:
            out.append((terc[order[start]], start, i - 1))
            start = i
    return out


# ======================================================================================
# Abbildungen
# ======================================================================================

def fig5_timeline(D: Data, out: Path, prev: Path | None) -> None:
    commits = git_commits(D.runs_dir, need=tuple(PASS1_COMMITS) + (ARCHIVE_COMMIT, PASS1_LEAK_END, "3d32ff4"))
    for h in list(PASS1_COMMITS) + [ARCHIVE_COMMIT, PASS1_LEAK_END]:
        assert h in commits, f"Commit {h} fehlt im Runs-Repo"
    t_design = git_commits(REPO, DESIGN_COMMIT)[DESIGN_COMMIT][0]
    assert t_design <= PASS1_START, (t_design, PASS1_START)
    leak = load_spans(D.runs_dir, "leak")
    spur = load_spans(D.runs_dir, "spur")
    core = [r for r in D.core()]
    pc = [r for r in D.pc()]
    assert all(r.start for r in D.runs) and len(leak) == 39 and len(spur) == 39

    clean_start = min([a for _, a, _ in leak + spur] + [r.start for r in D.runs])
    clean_end = max([b for _, _, b in leak + spur] + [r.end for r in D.runs])
    p1_start = PASS1_START                                  # erster Kern-Run D nach der Design-Revision
    p1_end = commits["3d32ff4"][0]                          # Post-Cutoff 1. Durchlauf fertig
    p1_leak = (PASS1_LEAK_START, commits[PASS1_LEAK_END][0])  # Leakage-Batches des 1. Durchlaufs, 24.10.
    t_archive = commits[ARCHIVE_COMMIT][0]
    assert p1_leak[1] <= t_archive, (p1_leak, t_archive)

    fig = plt.figure(figsize=(FULL, 8.4 * CM), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.06, hspace=0.06, wspace=0.0)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 48], height_ratios=[1.0, 1.75])
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axZ = fig.add_subplot(gs[1, :])
    num = mdates.date2num

    # --- (a) Uebersicht mit unterbrochener Achse ------------------------------------
    lanes = {"cal": 2, "p1": 1, "p2": 0}
    h = 0.46
    for ax in (axL, axR):
        ax.set_ylim(-0.6, 3.55)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axL.set_xlim(num(datetime(2026, 8, 7)), num(datetime(2026, 8, 11)))
    axR.set_xlim(num(datetime(2026, 9, 11)), num(datetime(2026, 10, 29)))
    axL.set_yticks([2, 1, 0], ["Calibration, variance scan", "Main runs, 1st pass (archived)",
                               "Main runs, 2nd pass (clean)"])
    axR.set_yticks([])
    # Kalibrierung + Scan (ein Balken: 08.08. Kalibrierung, 08.-09.08. Scan)
    axL.barh(2, num(VARIANCE_SCAN[1]) - num(VARIANCE_SCAN[0]), left=num(VARIANCE_SCAN[0]),
             height=h, color=MUTED, lw=0)
    # 1. Durchlauf: archiviert (schraffiert) -- Kern + Post-Cutoff 23.09.-04.10., Leakage 24.10.
    p1_kw = dict(height=h, color=GRID, edgecolor=MUTED, hatch="//////", lw=0)
    axR.barh(1, num(p1_end) - num(p1_start), left=num(p1_start), **p1_kw)
    # Leakage-Segment: real ~3,3 h, auf der Tagesachse unsichtbar -> Mindestbreite 0,8 Tage, rechtsbuendig
    # an seinem echten Ende (vor der Archivierung); die Caption in 05-execution.tex nennt die Verbreiterung.
    lw_ = max(num(p1_leak[1]) - num(p1_leak[0]), 0.8)
    axR.barh(1, lw_, left=num(p1_leak[1]) - lw_, **p1_kw)
    axR.text(num(p1_leak[1]) - 0.15, 1 + h / 2 + 0.08, "leakage", ha="right", va="bottom", fontsize=6.5,
             color=INK2, zorder=3)
    seen = {}
    for hsh, lab in PASS1_COMMITS.items():
        t = commits[hsh][0]
        axR.vlines(num(t), 1 - h / 2, 1 + h / 2, color=INK2, lw=0.8)
        seen.setdefault(t.date(), []).append(lab)
    # Labels je Tag: das mittlere Paket linksbuendig, die aeusseren rechtsbuendig an ihren Strich
    items = list(seen.items())
    for n, (day, labs) in enumerate(items):
        ts = [commits[hh][0] for hh, ll in PASS1_COMMITS.items() if ll in labs]
        left_aligned = 0 < n < len(items) - 1
        t = min(ts) if left_aligned else max(ts)
        txt = ", ".join(sorted(labs, key=lambda s: "DCABP".index(s[0]))).replace("PC", "post-cutoff")
        axR.text(num(t) + (0.15 if left_aligned else 0.1), 1 + h / 2 + 0.08, txt,
                 ha="left" if left_aligned else "right", va="bottom", fontsize=6.5, color=INK2,
                 bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6), zorder=3)
    # 2. Durchlauf: sauber
    axR.barh(0, num(clean_end) - num(clean_start), left=num(clean_start), height=h, color=INK2, lw=0)
    axR.text(num(clean_start) - 0.4, 0, "detail below", ha="right", va="center", fontsize=6.5,
             color=INK2)
    # Ereignisse
    for t, lab in sorted(EVENTS + [(t_design, "design\nrevised")]):
        axR.vlines(num(t), -0.5, 2.55, color=MUTED, lw=0.6, zorder=0)
        axR.text(num(t), 2.62, lab, ha="center", va="bottom", fontsize=6.5, color=INK, linespacing=0.95)
    axR.vlines(num(t_archive), -0.5, 2.55, color=CRITICAL, lw=0.8, zorder=0)
    axR.text(num(t_archive), 2.62, "contamination\nfound", ha="center", va="bottom", fontsize=6.5,
             color=INK, linespacing=0.95)
    # Achsen: Tagesmarken
    day_fmt = FuncFormatter(lambda x, _: f"{mdates.num2date(x).day} {mdates.num2date(x):%b}")
    axL.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[8]))
    axL.xaxis.set_major_formatter(day_fmt)
    axR.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axR.xaxis.set_minor_locator(mdates.DayLocator())
    axR.xaxis.set_major_formatter(day_fmt)
    axR.tick_params(axis="x", which="minor", length=1.2, color=MUTED)
    # Bruchstriche auf der x-Achse
    kw = dict(color=INK2, lw=0.6, clip_on=False, transform=None)
    for ax, x in ((axL, 1.0), (axR, 0.0)):
        tr = ax.transAxes
        ax.plot([x - 0.012 * (12 if ax is axL else 1), x + 0.012 * (12 if ax is axL else 1)],
                [-0.06, 0.06], **{**kw, "transform": tr})
    axL.set_title("(a) Overview", pad=2)

    # --- (b) Detail der sauberen Batches 24.-27.10. ----------------------------------
    rows = [("Leakage, perfect (D)", [("naked_react", a, b) for _, a, b in leak])]
    rows.append(("Leakage, spurious (D)", [("naked_react", a, b) for _, a, b in spur]))
    for arm in ARMS:
        rows.append((f"Core suite, {LETTER[arm]}", [(arm, r.start, r.end) for r in core if r.arm == arm]))
    for arm in ARMS:
        rows.append((f"Post-cutoff, {LETTER[arm]}", [(arm, r.start, r.end) for r in pc if r.arm == arm]))
    z0, z1 = datetime(2026, 10, 24, 12), datetime(2026, 10, 27, 12)
    axZ.set_xlim(num(z0), num(z1))
    min_w = 6 / (24 * 60)  # 6 min Mindestbreite, damit kurze D-Runs sichtbar bleiben
    ys = []
    for i, (lab, segs) in enumerate(rows):
        y = len(rows) - 1 - i + (0.6 if i < 2 else (0.3 if i < 6 else 0.0))
        ys.append(y)
        for arm, a, b in segs:
            w = max(num(b) - num(a), min_w)
            axZ.barh(y, w, left=num(a), height=0.62, color=COLOR[arm], lw=0)
    axZ.set_yticks(ys, [lab for lab, _ in rows])
    axZ.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    axZ.spines["left"].set_visible(False)
    axZ.tick_params(axis="y", length=0)
    axZ.vlines(num(t_archive), min(ys) - 0.6, max(ys) + 0.6, color=CRITICAL, lw=0.8, zorder=0)
    axZ.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axZ.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 18]))

    axZ.text(num(t_archive) + 0.04, min(ys), "contamination found,\naffected runs archived",
             ha="left", va="bottom", fontsize=6.5, color=INK, linespacing=0.95)

    def zfmt(x, _):
        t = mdates.num2date(x)
        return f"{t.day} {t:%b}" if t.hour == 0 else f"{t:%H:%M}"
    axZ.xaxis.set_major_formatter(FuncFormatter(zfmt))
    axZ.tick_params(axis="x", which="minor", length=1.5, color=MUTED)
    axZ.set_title("(b) Clean batches, 24–27 Oct (one bar per run, local time)", pad=2)
    # Balken ohne Marker -> Legende als Farbflaechen, nicht als Markerformen
    axZ.legend(handles=[Patch(facecolor=COLOR[a], edgecolor="none", label=arm_label(a)) for a in ARMS],
               loc="upper right", ncol=1, fontsize=6.5, bbox_to_anchor=(1.0, 1.0), handlelength=1.2,
               handleheight=0.8, handletextpad=0.4, labelspacing=0.25)

    save(fig, "fig5_timeline", out, prev)
    span_core = (min(r.start for r in core), max(r.end for r in core))
    span_pc = (min(r.start for r in pc), max(r.end for r in pc))
    print(f"     clean leakage {min(a for _, a, _ in leak)} .. {max(b for _, _, b in spur)}")
    print(f"     clean core    {span_core[0]} .. {span_core[1]}")
    print(f"     clean pc      {span_pc[0]} .. {span_pc[1]}")
    seq = sorted([(a, b) for _, a, b in leak + spur] + [(r.start, r.end) for r in D.runs])

    def elapsed_h(t0: datetime, t1: datetime) -> float:
        """Verstrichene Stunden zwischen zwei naiven Ortszeiten (Sommerzeit endet 25.10. 03:00 CEST)."""
        utc = [t - timedelta(hours=2 if t < datetime(2026, 10, 25, 3, 0) else 1) for t in (t0, t1)]
        return (utc[1] - utc[0]).total_seconds() / 3600
    for (a0, b0), (a1, _) in zip(seq, seq[1:]):
        if a1 - b0 > timedelta(minutes=45):
            print(f"     pause {b0:%d.%m. %H:%M} -> {a1:%d.%m. %H:%M} (Ortszeit; verstrichen {elapsed_h(b0, a1):.1f} h)")
    print(f"     archive commit {t_archive}; pass-1 commits: "
          + ", ".join(f"{l}={commits[h][0]}" for h, l in PASS1_COMMITS.items()))


def fig6_pass_rates(D: Data, out: Path, prev: Path | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 5.2 * CM), sharey=True, layout="constrained")
    for ax, an, title in ((axes[0], D.an_core, "(a) Core suite (13 datasets)"),
                          (axes[1], D.an_pc, "(b) Post-cutoff control (2 datasets)")):
        labels = []
        for i, arm in enumerate(ARMS):
            r = an["rates"]["per_arm"][arm]
            lo, hi = r["wilson"]
            ax.vlines(i, lo, hi, color=COLOR[arm], lw=1.6, zorder=2)
            for y in (lo, hi):
                ax.hlines(y, i - 0.09, i + 0.09, color=COLOR[arm], lw=1.0, zorder=2)
            mark(ax, i, r["rate"], arm, base=5.2)
            labels.append(f"{LETTER[arm]}\n{r['success']}/{r['n']}")
            print(f"     {title[:6]} {LETTER[arm]} {r['success']}/{r['n']} = {r['rate']:.3f} "
                  f"[{lo:.3f}, {hi:.3f}]")
        ax.set_xticks(range(4), labels)
        ax.set_xlim(-0.6, 3.6)
        ax.set_title(title)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylim(0, 1.04)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.25))
    axes[0].set_ylabel("Pass rate (95% Wilson interval)")
    fig.legend(handles=arm_handles(base=5.2), loc="outside upper center", ncol=4)
    save(fig, "fig6_pass_rates", out, prev)


def fig6_cd(D: Data, out: Path, prev: Path | None) -> None:
    met = D.an_core["metrics"]["f1_measured"]
    ranks = met["avg_ranks"]
    cd = met["cd"]
    cliques = met["cliques"]
    k = len(ARMS)
    order = sorted(ARMS, key=lambda a: (ranks[a], ARMS.index(a)))
    half = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(FULL, 3.3 * CM), layout="constrained")
    ax.axis("off")
    ax.set_xlim(-0.35, k + 1.35)
    lev = [-0.62 - 0.42 * i for i in range(half)]
    ax.set_ylim(lev[-1] - 0.3, 0.95)
    ax.hlines(0, 1, k, color=INK, lw=0.8)
    for r in range(1, k + 1):
        ax.vlines(r, 0, 0.09, color=INK, lw=0.8)
        ax.text(r, 0.13, str(r), ha="center", va="bottom", fontsize=8)
    for x in np.arange(1.5, k, 1.0):
        ax.vlines(x, 0, 0.05, color=INK, lw=0.6)
    ax.text((1 + k) / 2, 0.42, "mean rank (lower is better)", ha="center", va="bottom", fontsize=7.5,
            color=INK2)
    # CD-Balken
    ax.hlines(0.72, 1, 1 + cd, color=INK, lw=1.0)
    ax.vlines([1, 1 + cd], 0.66, 0.78, color=INK, lw=1.0)
    ax.text(1 + cd / 2, 0.80, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7.5)
    # Arme
    for i, arm in enumerate(order):
        r = ranks[arm]
        left = i < half
        y = lev[i] if left else lev[k - 1 - i]
        xt = 0.85 if left else k + 0.15
        ax.plot([r, r, xt], [0, y, y], color=INK2, lw=0.7, zorder=1, solid_joinstyle="miter")
        # Marker am Linienende statt auf der Achse: gebundene Raenge (B = C) verdecken sich sonst
        mark(ax, xt, y, arm, base=5.0, zorder=4)
        ax.text(xt + (-0.09 if left else 0.09), y, f"{arm_label(arm)} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=7.5)
    # Cliquen (nicht signifikant verschiedene Arme)
    for j, g in enumerate(cliques):
        lo, hi = min(ranks[a] for a in g), max(ranks[a] for a in g)
        y = -0.24 - 0.12 * j
        ax.hlines(y, lo - 0.04, hi + 0.04, color=INK, lw=2.4)
    save(fig, "fig6_cd", out, prev)
    print("     ranks " + ", ".join(f"{LETTER[a]} {ranks[a]:.2f}" for a in ARMS)
          + f"; CD {cd:.2f}; Friedman p_F {met['friedman']['p_F']:.3f}, chi2 p {met['friedman']['p_chi2']:.3f}")


def _diverging_cmap() -> LinearSegmentedColormap:
    # dataviz: zwei Pole + neutraler Grau-Mittelpunkt (Blau 600 <-> Rot critical).
    return LinearSegmentedColormap.from_list("div", ["#d03b3b", MID, "#184f95"])


def _rel_lum(rgb) -> float:
    def ch(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ink_for(rgb) -> str:
    """Weiss oder Fast-Schwarz -- was den hoeheren WCAG-Kontrast zur Zellfarbe hat."""
    lc = _rel_lum(rgb)
    c_white = 1.05 / (lc + 0.05)
    c_ink = (lc + 0.05) / (_rel_lum(to_rgb(INK)) + 0.05)
    return SURFACE if c_white > c_ink else INK


def fig6_heatmap(D: Data, out: Path, prev: Path | None) -> None:
    M = np.array([[statistics.fmean(r.delta for r in D.cell(ds, arm)) for arm in ARMS]
                  for ds in D.order])
    vmax = 0.12  # symmetrisch; groesster Betrag im Kern: connect-4 x A (+0.110)
    assert np.abs(M).max() <= vmax, np.abs(M).max()
    cmap = _diverging_cmap()
    fig, ax = plt.subplots(figsize=(HALF, 9.2 * CM), layout="constrained")
    # Zellen einzeln (2-px-Flaechenluecke zwischen den Zellen, groessere Luecke zwischen Terzilen)
    bounds = tercile_bounds(D.order, D.tercile)
    gap = {b[2] for b in bounds[:-1]}
    ytop, ypos = 0.0, []
    for i in range(len(D.order)):
        ypos.append(ytop)
        ytop += 1.0 + (0.35 if i in gap else 0.0)
    for i, ds in enumerate(D.order):
        for j, arm in enumerate(ARMS):
            v = M[i, j]
            c = cmap((v + vmax) / (2 * vmax))
            ax.add_patch(plt.Rectangle((j + 0.03, ypos[i] + 0.04), 0.94, 0.92, color=c, lw=0))
            ax.text(j + 0.5, ypos[i] + 0.5, fmt_signed(v), ha="center", va="center", fontsize=6.6,
                    color=_ink_for(c))
    ax.set_xlim(0, 4)
    ax.set_ylim(ytop, 0)
    ax.set_xticks(np.arange(4) + 0.5, [LETTER[a] for a in ARMS])
    ax.xaxis.tick_top()
    ax.set_yticks(np.array(ypos) + 0.5, [DS_LABEL[d] for d in D.order])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for name, a, b in bounds:
        ax.text(4.12, (ypos[a] + ypos[b] + 1) / 2, name, rotation=270, ha="left", va="center",
                fontsize=7.5, color=INK2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, aspect=30)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=7)
    cb.set_ticks([-0.10, -0.05, 0, 0.05, 0.10])
    cb.ax.set_xlim(-vmax, vmax)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_signed(x, 2) if x else "0"))
    cb.set_label("Mean F1 − baseline F1 (test rows)", fontsize=7.5)
    save(fig, "fig6_heatmap", out, prev)
    print(f"     range {M.min():+.3f} .. {M.max():+.3f}; cells < 0: "
          + ", ".join(f"{DS_LABEL[D.order[i]]}x{LETTER[ARMS[j]]} {M[i, j]:+.3f}"
                      for i, j in zip(*np.where(M < 0))))
    print("     per-arm median of cell deltas: "
          + ", ".join(f"{LETTER[a]} {np.median(M[:, j]):+.4f}" for j, a in enumerate(ARMS)))


JIT = np.linspace(-0.22, 0.22, 6)


def _f1_panel(ax, D: Data, ds: str, show_xlabels: bool = True) -> None:
    base = D.baseline(ds)
    vals = []
    for j, arm in enumerate(ARMS):
        cell = D.cell(ds, arm)
        for r, dx in zip(cell, JIT):
            if r.f1 is None:
                continue
            vals.append(r.f1)
            mark(ax, j + dx, r.f1, arm, base=3.6, hollow=r.outcome != "SUCCESS")
        m = statistics.fmean(r.f1 for r in cell if r.f1 is not None)
        ax.hlines(m, j - 0.32, j + 0.32, color=INK, lw=0.9, zorder=4)
    ax.axhline(base, color=INK2, lw=0.8, zorder=1)
    lo, hi = min(vals + [base]), max(vals + [base])
    span = max(hi - lo, 0.012)
    ax.set_ylim(lo - 0.12 * span, hi + 0.18 * span)
    ax.set_xlim(-0.55, 3.55)
    ax.set_xticks(range(4), [LETTER[a] for a in ARMS] if show_xlabels else [])
    ax.tick_params(axis="x", length=0)
    nice_y(ax, 3)
    ax.set_title(DS_LABEL[ds], pad=2)


def _f1_legend_handles() -> list:
    return arm_handles(base=4.0) + [
        Line2D([], [], color=INK2, lw=0.8, label="baseline (test rows)"),
        Line2D([], [], color=INK, lw=0.9, label="cell mean"),
        Line2D([], [], ls="none", marker="o", ms=4.0, mfc=SURFACE, mec=INK2, mew=0.9,
               label="run not passed"),
    ]


def fig6_f1_runs(D: Data, out: Path, prev: Path | None) -> None:
    bounds = tercile_bounds(D.order, D.tercile)
    ncol = max(b - a + 1 for _, a, b in bounds)
    fig, axes = plt.subplots(len(bounds), ncol, figsize=(FULL, 12.6 * CM), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.05, w_pad=0.03, hspace=0.06, wspace=0.04)
    for row, (name, a, b) in enumerate(bounds):
        for col in range(ncol):
            ax = axes[row, col]
            idx = a + col
            if idx > b:
                ax.axis("off")
                continue
            _f1_panel(ax, D, D.order[idx])
            if col == 0:
                ax.set_ylabel(f"{name}\nmacro-F1", fontsize=8)
    # Legende in die erste freie Zelle
    free = [axes[r, c] for r, (_, a, b) in enumerate(bounds) for c in range(ncol) if a + c > b]
    free[0].legend(handles=_f1_legend_handles(), loc="center left", fontsize=6.8,
                   labelspacing=0.45, borderaxespad=0.0)
    save(fig, "fig6_f1_runs", out, prev)
    fails = [r.run_id for r in D.core() if r.outcome != "SUCCESS"]
    print(f"     core runs not passed: {fails}")


def fig6_run_sd(D: Data, out: Path, prev: Path | None) -> None:
    sd = {(c["dataset"], c["arm"]): c["f1_measured_sd"] for c in D.main_runs["cells"]}
    for (ds, arm), v in sd.items():  # Abgleich mit den Summaries (Stichproben-SD, ddof = 1)
        assert abs(v - statistics.stdev(r.f1 for r in D.cell(ds, arm))) < 1e-9, (ds, arm)
    bounds = tercile_bounds(D.order, D.tercile)
    gap = {b[2] for b in bounds}
    fig, ax = plt.subplots(figsize=(HALF, 9.0 * CM), layout="constrained")
    y, ypos = 0.0, []
    for i in range(len(D.order)):
        ypos.append(y)
        y += 1.0 + (0.45 if i in gap else 0.0)
    y_med = y + 0.1
    dy = {a: off for a, off in zip(ARMS, (-0.27, -0.09, 0.09, 0.27))}
    for i, ds in enumerate(D.order):
        ax.hlines(ypos[i], 0, 0.05, color=GRID, lw=0.5, zorder=0)
        for arm in ARMS:
            mark(ax, sd[(ds, arm)], ypos[i] + dy[arm], arm, base=4.2)
    ax.hlines(y_med, 0, 0.05, color=GRID, lw=0.5, zorder=0)
    meds = {}
    for arm in ARMS:
        meds[arm] = statistics.median(sd[(ds, arm)] for ds in D.order)
        mark(ax, meds[arm], y_med + dy[arm], arm, base=4.2)
    ax.set_yticks(ypos + [y_med], [DS_LABEL[d] for d in D.order] + ["median"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(y_med + 0.7, -0.7)
    ax.set_xlim(-0.001, 0.051)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}" if x else "0"))
    ax.set_xlabel("Run-to-run SD of F1 (n = 6)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for name, a, b in bounds:
        ax.text(0.0525, (ypos[a] + ypos[b]) / 2, name, rotation=270, ha="left", va="center",
                fontsize=7.5, color=INK2, clip_on=False)
    fig.legend(handles=arm_handles(base=4.2), loc="outside upper center", ncol=2, fontsize=7)
    save(fig, "fig6_run_sd", out, prev)
    print("     median SD " + ", ".join(f"{LETTER[a]} {meds[a]:.4f}" for a in ARMS)
          + "; max SD " + ", ".join(f"{LETTER[a]} {max(sd[(d, a)] for d in D.order):.4f}" for a in ARMS))
    for a in ARMS:
        top = max(D.order, key=lambda d: sd[(d, a)])
        print(f"     max cell {LETTER[a]}: {top} {sd[(top, a)]:.4f}")


def _ranks_per_dataset(D: Data) -> dict[str, dict[str, float]]:
    """Rang 1 = hoechstes Zellmittel (best_f1_measured), Bindungen gemittelt (wie analyze_main)."""
    out = {}
    for ds in D.order:
        means = {a: statistics.fmean(r.f1 for r in D.cell(ds, a)) for a in ARMS}
        vals = sorted(means.values(), reverse=True)
        out[ds] = {}
        for a in ARMS:
            pos = [i + 1 for i, v in enumerate(vals) if v == means[a]]
            out[ds][a] = statistics.fmean(pos)
    return out


def fig6_terciles(D: Data, out: Path, prev: Path | None) -> None:
    rk = _ranks_per_dataset(D)
    ar = D.an_core["metrics"]["f1_measured"]["avg_ranks"]
    for a in ARMS:  # Abgleich mit analysis.json
        assert abs(statistics.fmean(rk[d][a] for d in D.order) - ar[a]) < 1e-9, a
    names = [n for n, _ in TERCILE_SIZES]
    by_t = {n: [d for d in D.order if D.tercile[d] == n] for n in names}
    off = {a: o for a, o in zip(ARMS, (-0.24, -0.08, 0.08, 0.24))}
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 5.4 * CM), layout="constrained")
    ax = axes[0]
    ax.axhline(2.5, color=MUTED, lw=0.6, zorder=0)
    ax.text(-0.46, 2.5 - 0.05, "no arm effect (2.5)", ha="left", va="bottom", fontsize=6.5, color=INK2)
    for ti, n in enumerate(names):
        for a in ARMS:
            m = statistics.fmean(rk[d][a] for d in by_t[n])
            mark(ax, ti + off[a], m, arm=a, base=5.0)
            print(f"     rank {n:6s} {LETTER[a]} {m:.2f}")
    ax.set_ylim(4.0, 1.0)
    ax.set_ylabel("Mean rank within tercile\n(1 = best)")
    ax.set_title("(a) Mean rank of macro-F1")
    ax2 = axes[1]
    ax2.axhline(0, color=INK2, lw=0.8, zorder=1)
    for ti, n in enumerate(names):
        for a in ARMS:
            ds_vals = [statistics.fmean(r.delta for r in D.cell(d, a)) for d in by_t[n]]
            ax2.plot([ti + off[a]] * len(ds_vals), ds_vals, ls="none", marker=MARKER[a],
                     ms=2.2 * MSCALE[a], mfc=COLOR[a], mec="none", zorder=2)
            m = statistics.fmean(ds_vals)
            mark(ax2, ti + off[a], m, a, base=5.0, zorder=3)
            print(f"     delta {n:6s} {LETTER[a]} mean {m:+.4f} (min {min(ds_vals):+.3f}, max {max(ds_vals):+.3f})")
    ax2.set_ylabel("F1 − baseline F1")
    ax2.set_title("(b) Gain over the default baseline")
    nice_y(ax2, 4, signed=True)
    for a_ in axes:
        a_.set_xticks(range(3), [f"{n} ({len(by_t[n])})" for n in names])
        a_.set_xlim(-0.5, 2.5)
        a_.tick_params(axis="x", length=0)
    fig.legend(handles=arm_handles(base=5.0), loc="outside upper center", ncol=4)
    save(fig, "fig6_terciles", out, prev)


def fig6_post_cutoff(D: Data, out: Path, prev: Path | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 5.4 * CM), layout="constrained")
    for ax, ds in zip(axes, POST_CUTOFF):
        _f1_panel(ax, D, ds)
        ax.set_title(f"{DS_LABEL[ds]} (baseline {D.baseline(ds):.3f})", pad=2)
        for arm in ARMS:
            cell = D.cell(ds, arm)
            k = sum(r.outcome == "SUCCESS" for r in cell)
            print(f"     {ds} {LETTER[arm]} {k}/6 passed; F1 " +
                  " ".join(f"{r.f1:.3f}" for r in cell) + f"; mean {statistics.fmean(r.f1 for r in cell):.3f}")
        ax.set_xticks(range(4), [LETTER[a] for a in ARMS])
    axes[0].set_ylabel("Macro-F1 (test rows)")
    fig.legend(handles=_f1_legend_handles(), loc="outside right center", fontsize=7, labelspacing=0.5)
    save(fig, "fig6_post_cutoff", out, prev)


PHASES = [("bu", "Business understanding"), ("du", "Data understanding"),
          ("prep", "Data preparation"), ("model", "Modeling"),
          ("eval", "Evaluation (→ back to modeling)")]


def fig6_phases(D: Data, out: Path, prev: Path | None) -> None:
    arms = ["hard_fsm", "soft_fsm"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 5.0 * CM), sharey=True, layout="constrained")
    for ax, (runs, title) in zip(axes, ((D.core(), "(a) Core suite (78 runs per arm)"),
                                         (D.pc(), "(b) Post-cutoff control (12 runs per arm)"))):
        for pi, (ph, _) in enumerate(PHASES):
            for arm, dy in zip(arms, (-0.17, 0.17)):
                rs = [r for r in runs if r.arm == arm]
                k = sum(r.gate_fails[ph] > 0 for r in rs)
                share = k / len(rs)
                y = pi + dy
                ax.hlines(y, 0, share, color=COLOR[arm], lw=1.6, zorder=2)
                mark(ax, share, y, arm, base=4.6)
                ax.text(share + 0.025, y, f"{k}/{len(rs)}", va="center", ha="left", fontsize=6.6,
                        color=INK2)
                print(f"     {title[:6]} {LETTER[arm]} {ph:5s} {k}/{len(rs)}")
        ax.set_xlim(-0.012, 0.40)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}".replace("%", " %")))
        ax.set_xlabel("Runs with ≥ 1 failed gate in the phase")
        ax.set_title(title)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(PHASES)), [p for _, p in PHASES])
    axes[0].set_ylim(len(PHASES) - 0.5, -0.5)
    fig.legend(handles=arm_handles(arms, base=4.6), loc="outside upper center", ncol=2)
    save(fig, "fig6_phases", out, prev)
    # DU-Gruende (Kurzauswertung fuer die Caption)
    reasons = Counter()
    for r in D.runs:
        if r.arm in arms and r.gate_fails["du"]:
            reasons[(r.arm, "pc" if r.post_cutoff else "core")] += 1
    print(f"     DU-fail runs: {dict(reasons)}")


def fig6_cost(D: Data, out: Path, prev: Path | None) -> None:
    core = D.core()
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 5.6 * CM), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    rng = np.random.default_rng(7)  # nur Jitter der Darstellung
    for i, arm in enumerate(ARMS):
        c = np.array([r.cost for r in core if r.arm == arm])
        jit = rng.uniform(-0.2, 0.2, len(c))
        ax.plot(c, i + jit, ls="none", marker=MARKER[arm], ms=2.4 * MSCALE[arm], mfc=COLOR[arm],
                mec="none", zorder=2)
        med = float(np.median(c))
        ax.vlines(med, i - 0.34, i + 0.34, color=INK, lw=1.1, zorder=3)
        print(f"     cost {LETTER[arm]}: total {c.sum():.2f} $, mean {c.mean():.3f}, median {med:.3f}, "
              f"min {c.min():.3f}, max {c.max():.3f}")
    ax.set_xscale("log")
    ax.set_xlim(0.03, 6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_yticks(range(4), [LETTER[a] for a in ARMS])
    ax.set_ylim(3.6, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Cost per run (US$, log scale)")
    ax.set_title("(a) All 78 runs per arm; bar = median")
    ax2 = axes[1]
    ax2.axhline(0, color=INK2, lw=0.8, zorder=1)
    for arm in ARMS:
        xs, ys = [], []
        for ds in D.order:
            cell = D.cell(ds, arm)
            xs.append(statistics.fmean(r.cost for r in cell))
            ys.append(statistics.fmean(r.delta for r in cell))
        mark(ax2, xs, ys, arm, base=4.0)
    ax2.set_xscale("log")
    ax2.set_xlim(0.03, 6)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    nice_y(ax2, 4, signed=True)
    ax2.set_xlabel("Mean cost per run in the cell (US$, log scale)")
    ax2.set_ylabel("Mean F1 − baseline F1")
    ax2.set_title("(b) Cost versus gain, one mark per dataset")
    fig.legend(handles=arm_handles(base=4.6), loc="outside upper center", ncol=4)
    save(fig, "fig6_cost", out, prev)


FIGS = {
    "fig5_timeline": fig5_timeline,
    "fig6_pass_rates": fig6_pass_rates,
    "fig6_cd": fig6_cd,
    "fig6_heatmap": fig6_heatmap,
    "fig6_f1_runs": fig6_f1_runs,
    "fig6_run_sd": fig6_run_sd,
    "fig6_terciles": fig6_terciles,
    "fig6_post_cutoff": fig6_post_cutoff,
    "fig6_phases": fig6_phases,
    "fig6_cost": fig6_cost,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", type=Path, default=REPO.parent / "thesis-crisp-dm-runs")
    ap.add_argument("--out-dir", type=Path, default=REPO / "thesis" / "figs")
    ap.add_argument("--preview-dir", type=Path, default=None,
                    help="zusaetzlich PNG-Vorschau (300 dpi) hierhin; nicht ins Repo legen")
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS), default=None)
    args = ap.parse_args()
    names = args.only or list(FIGS)
    setup_style()
    data = load_data(args.runs_dir.resolve(), with_time="fig5_timeline" in names)
    for n in names:
        print(n)
        FIGS[n](data, args.out_dir, args.preview_dir)


if __name__ == "__main__":
    main()
