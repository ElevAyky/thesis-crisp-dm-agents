# Codebuch Prozess-Kodierung (Kap. 7.2)

> Version 1.1, 09.11.2026 (1.0 → 1.1: Muster- und Wortlisten an `code_process.py` angeglichen,
> Hilfscode `bu_metric_rationale_unit`, Grenzen ergänzt; § 8 „Versionen“). Kodierer:
> `thesis/scripts/code_process.py` (regelbasiert, deterministisch),
> Aktionslisten: `thesis/scripts/extract_actions.py` → `thesis/facts/derived/actions/<run>.md`
> (gitignored, reproduzierbar). Ergebnis: `thesis/facts/derived/process_coding.json`.
> Anlass: Decker 09.10. Punkt 3 („Warum ist Naked ReAct genauso gut? Folgt er einem Prozess — welchem?“,
> `docs/meetings/2026-10-09-kern-ergebnisse.md`). **Explorativ.** Validiert durch eine blinde
> Gegenkodierung von 24 C/D-Runs durch einen zweiten Kodierer (LLM-Agent; Faktenblatt `kap7.md`,
> IDs F7-9xx); die A/B-Referenzcodes sind nicht gegenkodiert.

## 1. Material und Einheiten

- **Runs:** alle sauberen `main-*` (Harness `14e99e4`): C (crisp_react) und D (naked_react) je 78 Kern + 12
  Post-Cutoff, voll kodiert; A (hard_fsm) und B (soft_fsm) je 90, nur Referenz-Codes (§ 7). Archivierte
  Ordner (`automem_*`, `gitctx_*`, `promptv1_*`) werden nie gelesen.
- **Quelle je Run:** `transcripts/*.jsonl` (Claude-Code-Session). Bei B werden die Prüfer-Sessions
  (erste Nachricht „You are the independent reviewer …“) weggelassen; die Phase einer A/B-Session kommt aus
  `events.jsonl` (session_id → phase). `summary.json` liefert outcome und F1-Differenz zur Baseline,
  `work/result.json` die vom Agenten erklärte Zielspalte.
- **Sichtbar ist nur, was der Agent nach außen schreibt.** Thinking-Blöcke sind in den Transkripten leer
  (nur Signatur). Kodiert wird also beobachtbares Verhalten, nicht Absicht.
- **Belegeinheiten** (jede Kodierung trägt eine Belegstelle):
  - *Codezeile* `#n:Lk` — Zeile k des Inhalts von Tool-Aktion n (Bash-Befehl inkl. `python -c`/Heredoc,
    Write-Inhalt einer `.py`-Datei, bei Edit der neue Text). Kommentarteil abgetrennt.
  - *Kommentar* — der `#`-Teil einer Codezeile (gleiche Belegstelle).
  - *Text* `T>#n` — sichtbarer Assistant-Text nach Aktion n (`T>#0` = vor der ersten Aktion).
  - *Notiz* `N#n` — `.md`/`.txt`, geschrieben per Write/Edit oder per `cat > x.md` in Aktion n.
- **Ausgeschlossen:** Import-Zeilen (keine Aktivität); fehlgeschlagene Write/Edit (`is_error`, z. B.
  `old_string` nicht gefunden — die Datei blieb unverändert); Read-Aktionen; `result.json` (Kontrakt-Ausgabe).
- **Abschlussbericht** = alle Texte nach der letzten Tool-Aktion. Zählt nicht für BU, DU-Leakage und
  Ablauf (er fasst nachträglich zusammen), wohl aber für die EVAL-Codes (er ist die Bewertung).

## 2. Ableitung aus CRISP-DM 1.0

Grundlage ist der Text, den A, B und C wörtlich bekommen (`src/crispfsm/prompts.py`, `CRISP_TASKS`,
Chapman et al. 2000, generische Aufgaben 1.1–5.3). D bekommt ihn nicht, wohl aber die gemeinsamen
ML-Hinweise des REACT-Prompts (Zielspalte ableiten, ungeeignete Spalten wie IDs/Konstanten/Leakage
erkennen, Fehlwerte behandeln, numerisch machen, **gegen eine einfache Baseline vergleichen**, gleiche
Vorverarbeitung für Test). **Folge:** Codes, die diese Hinweise abdecken (Baseline-Vergleich,
Leakage/ID erwogen, Fehlwerte), messen bei D Prompt-Befolgung, nicht spontanes Prozesswissen.

| CRISP-Aufgabe | Codes |
|---|---|
| 1.1 Business objectives / 1.2 Assess situation | `bu_business_context`, `bu_success_criterion` |
| 1.3 Data mining goals + success criteria | `bu_explicit` (= `bu_target` + `bu_metric`), `bu_metric_rationale`, `bu_success_criterion` |
| 1.4 Project plan | `doc_upfront` (Hilfscode) |
| 2.2 Describe data | `du_inspect_before_model`, `du_stats` |
| 2.3 Explore data | `du_class_dist`, `du_stats`, `du_relations` |
| 2.4 Verify data quality | `du_missing`, `du_quality_ext`, `du_leakage` |
| 3.1 Select data | `prep_select` (+ Hilfscode `prep_select_rule`) |
| 3.2 Clean / 3.5 Format | `prep_explicit` (Imputation, Kodierung, Typumwandlung, Skalierung) |
| 3.3 Construct data | `prep_construct` |
| 4.1 Select modeling technique / 4.4 Assess model | `model_multi_family`, `model_assess_detail` |
| 4.2 Generate test design | `model_cv`, `model_test_design` |
| 4.3 Build model (Parameter) / 4.4 Revised parameter settings | `model_tuning` |
| 5.1 Evaluate results | `eval_baseline`, `eval_business` |
| 5.2 Review process / 5.3 Determine next steps | `eval_review_next` |
| Outputs je Aufgabe („reports … keep them brief“) | `doc_notes`, `doc_crisp_sections` |

## 3. Zentrale Hilfsbegriffe

- **Modellierungsbeginn (MODEL-Position):** erste Codezeile mit einem Klassifikator-Konstruktor
  (Familienliste § 4, `DummyClassifier` ausgenommen). „Vor der Modellanpassung“ = Belegstelle davor
  (Aktion früher, oder gleiche Aktion und frühere Zeile). Die Position der Konstruktor-Zeile steht für
  den ersten Fit, weil Konstruktor und Fit in den Skripten fast immer im selben Codeblock stehen.
- **Zielspalten-Name:** `target_column` aus `work/result.json`; Fallback Regex `col_\d+` (die Zielspalten
  sind in allen Sets anonymisiert, Target-Neutralisierung).
- **Reparatur:** Write/Edit nach einer fehlgeschlagenen Python-Ausführung (Traceback/Exit-Code ≠ 0), bis
  zur nächsten Ausführung, sowie eine erneute Inline-Ausführung (`python -c`) direkt nach einem Fehler.
- **Verifikation:** Bash-Aktion, die `predictions.parquet`/`result.json` liest und kein Modell baut.

## 4. Codes C/D — Definition, Entscheidungsregel, Beispiel

Alle binär (ja/nein) außer wo angegeben. „Text“ = Assistant-Text, Notiz oder Kommentar (ohne
Abschlussbericht, außer bei EVAL). Regex-Details: `code_process.py`, Dicts `RX`/`TX`.

### BU — Business Understanding (CRISP 1.x)

| Code | Definition | Entscheidungsregel | Beispiel |
|---|---|---|---|
| `bu_target` (Hilfscode) | Zielspalte ausdrücklich als Ziel benannt, vor der Modellanpassung | Text nennt den Zielspalten-Namen **und** ein Zielwort (target/label/outcome/predict…) **oder** Codezeile `TARGET = "…"` (bzw. target/label_col) | `main-adult-naked_react-r1 #4:L10` `TARGET = "col_14"` |
| `bu_metric` (Hilfscode) | Metrik ausdrücklich gewählt, vor der Modellanpassung | Text nennt f1/macro-F1/accuracy **oder** Codezeile `METRIC = "…"`/`SCORING = "…"`. Ein `scoring=`-Argument im Fit-Aufruf selbst zählt nicht (gleichzeitig, nicht vorher) | `main-adult-naked_react-r2 T>#3` „I'll use **f1**“ |
| `bu_explicit` | Ziel **und** Metrik vor der Modellanpassung festgehalten (Startvorschlag BU) | `bu_target` ∧ `bu_metric`; Beleg = der spätere der beiden | `main-adult-naked_react-r2 T>#3` „Target `col_14`, imbalanced (76% majority) → I'll use **f1**“ |
| `bu_metric_rationale` | Metrikwahl begründet (Klassenverteilung) | Text/Notiz (kein Kommentar) vor der Modellanpassung; **ein Satz** enthält Metrikwort und Begründungswort (imbalanc/balanced/minority/majority/skew/rare class/class distribution/dominat) | wie oben |
| `bu_metric_rationale_unit` (Hilfscode, v1.1) | wie `bu_metric_rationale`, Sensitivität für knappe Begründungen über zwei Sätze | Metrik- und Begründungswort in derselben Text-Einheit (eine Assistant-Nachricht oder Notiz) vor der Modellanpassung | `main-adult-naked_react-r6 T>#3` „Target is `col_14`, imbalanced (~24% `>50K`). I'll use F1.“ |
| `bu_success_criterion` | Erfolgskriterium vor der Modellanpassung formuliert (1.1/1.3) | Text/Notiz vor der Modellanpassung: „success criteri…“, „success is/=/means/metric/would be …“, „beat/exceed/outperform … baseline“, „must beat/exceed/outperform …“ | `main-bank_marketing-crisp_react-r1 N#5` (Notiz mit Abschnitt Business Understanding, vor dem Modell) |
| `bu_business_context` | Geschäftliche Rahmung über Ziel/Metrik hinaus (1.1/1.2) | Text/Notiz/Kommentar vor der Modellanpassung: business, stakeholder, assume/assumption, costly, cost of errors, false negatives/positives are …, project plan, objective | `main-credit_g-crisp_react-r1 T>#3` „catching "bad" risks is the business-critical minority class“ |

### DU — Data Understanding (CRISP 2.x), jeweils **vor der Modellanpassung**, im Code

| Code | Definition | Entscheidungsregel | Beispiel |
|---|---|---|---|
| `du_inspect_before_model` | Dateninspektion vor dem ersten Modell (Startvorschlag) | irgendeine DU-Zeile: `.shape`, `.dtypes`, `.head(`, `.tail(`, `.sample(<Zahl>`, `.info(`, `print(… .columns)`, Fehlwert-, Statistik-, Qualitäts-, Beziehungs- oder Zählzeile | `main-adult-naked_react-r1 #2:L1` |
| `du_class_dist` | Klassenverteilung des Ziels geprüft | `value_counts(`/`bincount(`/`Counter(`/`return_counts` in einer Zeile, die das Ziel referenziert (Name, `y`, `y_train`/`ytrain`, `y_tr`/`ytr`, `TARGET`, `target`, `label`) | `main-adult-naked_react-r1 #3:L5` `print(tr['col_14'].value_counts())` |
| `du_missing` | Fehlwerte geprüft | `.isna(`/`.isnull(`/`.notna(`/`.notnull(`/`.info(`/`hasnans` | `main-adult-naked_react-r1 #3:L6` |
| `du_quality_ext` | weitere Qualitätsprüfung: Duplikate, Konstanten, ID-artige Spalten | `duplicated(`/`drop_duplicates(`/`.nunique(`/`is_unique`; `nunique` nur auf dem Ziel zählt nicht | `main-adult-naked_react-r1 #2:L1` `print(df.nunique())` |
| `du_stats` | Deskriptive Statistik | `.describe(`/`.quantile(`/`.skew(`/`.kurt(`/`.agg([` | `main-bank_marketing-naked_react-r4 #3:L10` |
| `du_relations` | Beziehungen zwischen Attributen bzw. zum Ziel (2.3) | `.corr(`/`corrwith(`/`crosstab(`/`mutual_info`/`chi2(`/`f_classif`/`pointbiserial`/`.groupby(` | `main-bank_marketing-crisp_react-r1 #4:L7` `df.groupby('col_16')['V12'].mean()` |
| `du_leakage` | Leakage- oder ID-Spalten ausdrücklich erwogen (Startvorschlag) | Text/Notiz/Kommentar vor der Modellanpassung: leak…, „ID(s)“ (Großschreibung), id-like/id column/id col, identifier, row id/index/number, index column | `main-adult-naked_react-r2 #4:L23` Kommentar „No ID/constant columns present.“ |
| `leakage_mention_any` (Hilfscode) | dasselbe irgendwo im Run, inkl. Abschlussbericht | wie oben, ohne Zeitbedingung | — |

Ausgenommen sind DU-Zeilen, die Ausgaben betreffen (`pred`, `prediction(s)`, `result.json`, `out`,
`sub(mission)`) oder aus einer Zählung nur einen Wert ziehen (`idxmax(`/`idxmin(`/`.mode()` —
Mehrheitsklasse für die Baseline, positives Label).

### PREP — Data Preparation (CRISP 3.x), irgendwo im Run, im Code

| Code | Definition | Entscheidungsregel | Beispiel |
|---|---|---|---|
| `prep_explicit` | explizite Aufbereitung (Startvorschlag) | irgendeiner von: Imputation (`SimpleImputer`/`KNNImputer`/`IterativeImputer`/`.fillna(`), Kodierung (`OneHotEncoder`/`OrdinalEncoder`/`LabelEncoder`/`TargetEncoder`/`get_dummies(`/`.cat.codes`/`factorize(`/`pd.Categorical(`/`astype("category")`/`.map(`/`.replace({`), Typumwandlung (`pd.to_numeric(`/`astype(float…|int…)`), Skalierung (`StandardScaler` u. a., `/ 255`), Auswahl (`prep_select`, `prep_select_rule`), Konstruktion (`prep_construct`); Zeilen mit Ausgabe-Bezug ausgenommen | `main-adult-naked_react-r1 #4:L26` `….fillna("Missing")` |
| `prep_select` | benannte Feature-Spalten ausgeschlossen (3.1 „rationale for inclusion/exclusion“) | `.drop(` mit Spalten-Achse, das ein Literal ≠ Zielspalte enthält oder eine Variable, der eine Liste mit solchen Literalen zugewiesen ist | `main-bank_marketing-naked_react-r5 #4:L20` `drop(columns=[TARGET] + LEAK)` mit `LEAK = ["V12"]` |
| `prep_select_rule` (Hilfscode) | datenabhängige Ausschlussregel (z. B. konstante Spalten), ggf. ohne Wirkung | `.drop(` mit Variable aus Comprehension/`nunique`/`.index`, oder `VarianceThreshold`/`SelectKBest`/`SelectPercentile`/`SelectFromModel`/`RFE`/`RFECV` | `main-bank_marketing-naked_react-r4 #4:L17` |
| `prep_construct` | abgeleitete Attribute (3.3) | `PolynomialFeatures`/`KBinsDiscretizer`/`SplineTransformer` oder Zuweisung `df["neu"] = …` mit Arithmetik aus anderen Spalten (linke Spalte ≠ rechte) | — (in keinem Run) |

### MODEL — Modeling (CRISP 4.x), im Code

| Code | Definition | Entscheidungsregel | Beispiel |
|---|---|---|---|
| `n_families` (Zahl) / `model_multi_family` | Zahl der verglichenen Modellfamilien (Startvorschlag); binär ≥ 2 | Konstruktor-Aufrufe (`Name(`), Familien: linear (LogisticRegression[CV], SGD, Ridge, Perceptron, PassiveAggressive), svm (SVC, LinearSVC, NuSVC), random_forest, extra_trees, hist_gb, gb, adaboost, decision_tree, knn, naive_bayes, mlp, discriminant, boosting_lib (XGB/LGBM/CatBoost), ensemble (Voting/Stacking/Bagging); Dummy zählt nicht | `main-breast_cancer-naked_react-r6`: linear, random_forest, gb |
| `model_cv` | Kreuzvalidierung (4.2) | `cross_val_score(`/`cross_val_predict(`/`cross_validate(`/`(Repeated)(Stratified)KFold(`/`…SearchCV(`/`cv=<Zahl>` | `main-adult-naked_react-r2 #4:L50` |
| `model_test_design` | interne Validierung vor der finalen Vorhersage: CV **oder** Holdout (Startvorschlag „Bewertung auf interner Validierung/CV“, hier unter 4.2 einsortiert) | `model_cv` ∨ `train_test_split(`/`(Stratified)ShuffleSplit(` | `main-adult-naked_react-r1 #4:L38` |
| `model_tuning` | Hyperparameter-Suche oder -Revision (4.3 Parameter settings, 4.4 Revised parameter settings) | in dieser Reihenfolge: (a) `…SearchCV(`/`ParameterGrid(`/`ParameterSampler(`/`optuna`; (b) dieselbe Familie mit ≥ 2 verschiedenen Parametersätzen im Run (normalisiert: ohne `random_state`, `n_jobs`, `verbose`, bei linear/MLP ohne `max_iter`/`solver`; Positionsargument = `n_estimators`; Reihenfolge und Anführungszeichen egal; **Reparaturen ausgenommen**); (c) Schleife über Parameterwerte, deren Variable in einem Konstruktor steckt; (d) Schwellenwert-Optimierung (`proba >= t`, `best_thr = …`) | `main-bank_marketing-naked_react-r1 #6:L14` (Schleife `class_weight` ∈ {None, balanced} + Schwelle) |
| `model_assess_detail` | Modellbewertung über eine Kennzahl hinaus (4.4) | `classification_report(`/`confusion_matrix(`/`ConfusionMatrixDisplay`/`precision_recall_fscore_support(`/`average=None` | `main-adult-naked_react-r2 #4:L53` |
| `model_class_weight` (Hilfscode) | Klassen-Gewichtung (Mechanismus Post-Cutoff, F6-526/532) | `class_weight=`/`sample_weight`/`compute_sample_weight`/`compute_class_weight` | `main-bank_marketing-naked_react-r1 #4:L37` |

### EVAL — Evaluation (CRISP 5.x), nach Modellierungsbeginn, inkl. Abschlussbericht

| Code | Definition | Entscheidungsregel | Beispiel |
|---|---|---|---|
| `eval_baseline` | Ergebnis gegen eine Baseline verglichen (Startvorschlag) | Code: `DummyClassifier(` oder Zeile mit baseline/most_frequent/majority/`mode()` (Imputer-, `fillna`- und Import-Zeilen ausgenommen); oder Text/Notiz nach Modellierungsbeginn mit Baseline-Wort (baseline/dummy/most frequent/majority) und Zahl | `main-adult-naked_react-r1 #4:L42` |
| `eval_business` | Ergebnis ausdrücklich gegen Ziel/Erfolgskriterium bewertet (5.1, „Abgleich mit Ziel/Metrik“ des Startvorschlags) | **ein Satz** enthält Bezug (success criteri…, business objective/goal/success/need/question, objective, goal) **und** Bewertungsverb (is/are/was/were/has been/have been met/achieved/satisfied/fulfilled, meet(s), satisfy/satisfies/satisfied, achieves/achieved, fulfil/fulfils/fulfilled) | `main-covertype-crisp_react-r3 N#8` „Model vastly exceeds baseline and meets the DM success criterion.“ |
| `eval_review_next` | Prozess-Review oder nächste Schritte (5.2/5.3) | Text/Notiz/Kommentar: next step(s), further iteration/improvement/work/tuning/gain(s), could (be) (further) improve(d), future work/iteration/improvement, recommend, process review, review of (the) process, overlook, possible improvements/actions, would try/explore, if (more) time | `main-wine-crisp_react-r4 T>#6` |

### DOC — Berichte (CRISP-Outputs, im Prompt: „keep them brief“)

| Code | Definition | Entscheidungsregel |
|---|---|---|
| `doc_notes` | Notizdatei geschrieben | Write/Edit einer `.md`/`.txt`/`.rst` oder `cat > *.md` per Bash |
| `doc_crisp_sections` | Notiz nach CRISP-Phasen gegliedert | eine Notiz nennt ≥ 3 der Begriffe business understanding, data understanding, data preparation, modeling/modelling, evaluation |
| `doc_upfront` (Hilfscode) | Notiz vor der Modellanpassung angelegt (Plan, 1.4) | erste Notiz liegt vor der MODEL-Position |
| `verify_outputs` (Hilfscode) | Ausgaben nach dem Schreiben geprüft (Qualitätssicherung, Nähe zu 5.2 „did we correctly build the model?“) | eigene Bash-Aktion liest `predictions.parquet` oder `result.json` und baut kein Modell; eine Prüfung im Modell-Skript selbst (z. B. `assert len(out) == len(test)`) zählt nicht |

## 5. Ablauf-Codes (nur C/D)

### 5.1 Erstauftritte und Reihenfolge

Erstauftritt je Phase (Abschlussbericht ausgenommen):
- **BU:** erste ausdrückliche Zielspalten-Aussage (`bu_target`-Regel, ohne Zeitbedingung).
- **DU:** erste DU-Zeile (`du_inspect_before_model`-Regel).
- **PREP:** erste PREP-Zeile (`prep_explicit`-Regel).
- **MODEL:** Modellierungsbeginn (§ 3).
- **EVAL:** erster Baseline-Code (`eval_baseline`) in der Aktion des ersten Modells oder später, oder
  erster Baseline-/Bewertungstext nach Modellierungsbeginn; Position = max(Beleg, Modellierungsbeginn).
  Eine Mehrheitsquote, die schon in einer Inspektion **vor** der ersten Modell-Aktion ausgegeben wird,
  ist kein EVAL-Erstauftritt (Hilfswert `baseline_before_model_action`).
- **Gleichzeitigkeit:** PREP, MODEL und EVAL, die zuerst in **derselben** Aktion auftreten, gelten als
  gleichzeitig und werden nach Phase geordnet (Pipeline-Definition: `('sc', StandardScaler()), ('svc',
  SVC())` in einer Zeile nach dem ersten RF-Konstruktor ist kein „PREP nach MODEL“). BU und DU behalten
  die Zeilenordnung.

**`canonical_order`** = für alle vorhandenen Phasen gelten BU < MODEL, DU < PREP, DU < MODEL,
PREP ≤ MODEL, MODEL ≤ EVAL. Fehlende Phasen werden übersprungen.

**`implicit_crisp`** („folgt implizit CRISP-DM“) = **alle fünf Phasen vorhanden** — BU in der strengen
Form `bu_explicit` (Ziel und Metrik vor der Modellanpassung), DU, PREP, MODEL, EVAL mit Erstauftritt —
**und** `canonical_order`. Die fehlenden Teile stehen pro Run in `phases_missing`, die Verletzungen in
`order_violations`.

**Bewusste Abweichung vom Startvorschlag (BU ≤ DU entfällt):** Die Zielspalte ist in allen Sets
anonymisiert (`col_N`); um sie zu bestimmen, muss jeder Agent zuerst die Daten ansehen (auch der
BU-Prompt von A/B verlangt das: „Inspect … columns, dtypes, example values“). Eine Bedingung BU ≤ DU
wäre damit für fast alle Runs aller Arme per Versuchsaufbau verletzt und würde nichts über den Prozess
aussagen. Stattdessen muss BU vor dem Modell liegen. Die Folge DU→BU ist in der Phasenfolge
(`phase_order`) trotzdem sichtbar.

### 5.2 Phasenfolge und Rücksprünge

`phase_sequence` = lauflängen-komprimierte Folge der Phasen über die Ereignisse eines Runs:
- **Code-Aktion** (keine Reparatur, keine Verifikation): **PREP**, wenn sie einen neuen
  Aufbereitungs-Token einführt (Operator + normalisierte Argumente ohne technische wie
  `handle_unknown`/`sparse_output`; Spalten-Auswahl; Kodierungsart); **MODEL**, wenn sie eine neue
  Modell-Konfiguration (Familie + normalisierte Argumente) oder eine Suche einführt; **EVAL**, wenn sie eine
  neue Baseline-Zeile enthält (ab der Aktion des ersten Modells); **DU**, wenn sie neue Inspektionszeilen
  und **keinerlei** PREP/MODEL-Token enthält (reine Inspektion). Eine Aktion trägt ihre Phasen in
  kanonischer Reihenfolge bei. Neu = in keiner früheren Aktion des Runs vorhanden.
- **Text** nach Modellierungsbeginn (nicht der Abschlussbericht) mit Baseline-Zahl oder Bewertung
  gegen Ziel/Kriterium: **EVAL**.
- BU fehlt in der Folge (in C/D eine Aussage, keine Aktivität).

**Rücksprung** = Übergang in der komprimierten Folge zu einer früheren Phase (DU < PREP < MODEL < EVAL),
z. B. EVAL → MODEL (neue Konfiguration nach dem Baseline-Vergleich) oder EVAL → DU (Inspektion nach dem
ersten Ergebnis). `n_backjumps` (Zahl), `backjump_any` (binär), Ziel in `evidence.n_backjumps`.
Wiederholtes Ausführen desselben Skripts erzeugt nichts; Reparaturen nach Fehlern zählen nicht.

### 5.3 Struktur

- `n_actions` (Tool-Aktionen), `n_code_actions`, `n_scripts` (verschiedene `.py`-Dateien),
  `n_inline_python` (Bash mit `python -c`/Heredoc).
- **`monolith`** = eine Code-Einheit (Datei-Version nach Write/Edit oder ein Bash-Befehl) enthält
  Inspektion (DU) **und** Aufbereitung (PREP) **und** Modell (MODEL) **und** schreibt
  `predictions.parquet` — „ein Skript erledigt Inspektion bis Vorhersage“.
- `single_pipeline_script` (Hilfscode) = dasselbe ohne DU.

## 6. Statistik (in `code_process.py`)

- Raten je Arm getrennt für Kern (78/Arm), Post-Cutoff (12/Arm) und gesamt.
- **Fisher exakt, zweiseitig, C vs. D je binärem Hauptcode** (28 Codes, Liste `BIN_REACT`), Holm über
  diese 28 Codes, getrennt je Umfang. **Explorativ** — nicht präregistriert, Codebuch während der
  Kodierer-Entwicklung festgelegt (§ 8).
- Zusammenhang mit dem Ergebnis, **deskriptiv**, getrennt innerhalb C und D: pass-Rate und mittlere
  F1-Differenz zur Baseline (`summary.baseline_comparison.diff`) mit/ohne Code, roh und set-zentriert
  (Differenz minus Mittel aller C+D-Runs desselben Sets).

## 7. Referenz A/B

Nur Codes, die die FSM nicht erzwingt, aus den Worker-Sessions:
- MODEL-Codes (`n_families`, `model_multi_family`, `model_cv`, `model_holdout`, `model_test_design`,
  `model_tuning`, `model_assess_detail`, `model_class_weight`) **nur aus den Sessions der Phase `model`**
  (Prüfsonden in DU zählen nicht als Modellwahl). `model_test_design` ist in A/B durch den Kontrakt
  vorgegeben (`metrics.json` verlangt `val_score` auf Holdout oder CV), CV vs. Holdout ist frei.
- `prep_construct` aus den Sessions der Phase `prep`.
- `du_leakage` = Leakage/ID erwogen in BU-, DU- oder PREP-Sessions (Text, Notiz, Kommentar) oder in den
  Freitextfeldern des DU-Reports (`quality_issues`, `excluded_columns[].reason`). Beide Prompts fragen
  danach (BU: `critical_columns` „suspected ID or leakage columns“, DU: `excluded_columns`).
- **`eval_baseline` ist in A/B erzwungen**, nicht frei: Der Modeling-Kontrakt verlangt `dummy_score`
  (`prompts.MODEL`), das Model-Gate verlangt `val_score > dummy_score` (`src/crispfsm/gates.py:68-85`).
  Der Startvorschlag zählte den Baseline-Vergleich zu den freien A/B-Codes; er wird deshalb als
  „erzwungen“ berichtet, nicht als Rate.

## 8. Entstehung, Abweichungen, Grenzen

**Entstehung.** Startvorschlag aus dem Auftrag (Paket 2) → Codes an CRISP 1.0 ausgerichtet (§ 2) →
Kodierer an den Aktionslisten kalibriert. Während der Kalibrierung waren die Gesamtraten sichtbar; die
Regeländerungen wurden an Einzelfällen begründet (Artefakt der Regel, nicht Richtung des Ergebnisses):
1. Import-Zeilen und fehlgeschlagene Edits ausgeschlossen (Import `StandardScaler` war als „PREP“ gezählt).
2. Rücksprünge über neue Tokens statt neue Zeilen (Pipeline im Tuning-Skript neu deklariert ≠ Rückkehr
   zu PREP); technische Argumente und Anführungszeichen normalisiert.
3. Reparaturen nach Fehlern zählen weder als Rücksprung noch als Tuning (`class_weight` nach Absturz
   entfernt ≠ Parameter-Revision).
4. Schwellenwert-Optimierung als Tuning ergänzt (A-Runs bauen dafür eigene Wrapper, F6-637).
5. Konstruktor-Regex für `SVC(` ohne Präfix korrigiert; `/ 255`, `astype(np.float32)`, `.map(`-Kodierung,
   `pd.to_numeric` als Aufbereitung ergänzt (vorher fehlende PREP-Phase bei fashion-mnist/student health).
6. `prep_select` auf benannte Spalten eingeengt; generische Konstanten-Filter als `prep_select_rule`.
7. `eval_business` auf Bezug + Bewertungsverb im selben Satz eingeengt (vorher zählte ein nachträglich
   notiertes BU-Erfolgskriterium; „Goal: … income exceeds $50K“ zählte wegen „exceeds“).
8. Mehrheitsquote aus der Inspektion vor der ersten Modell-Aktion ist kein EVAL-Erstauftritt.
9. BU ≤ DU entfällt (§ 5.1); PREP/MODEL/EVAL derselben Aktion gelten als gleichzeitig.

**Gegenüber dem Startvorschlag:** „Bewertung auf interner Validierung/CV“ steht unter MODEL 4.2
(`model_test_design`), weil CRISP 1.0 das Testdesign der Modeling-Phase zuordnet; „Ausdrücklicher Abgleich
mit Ziel/Metrik“ = `eval_business`; Prozess-Review/nächste Schritte (5.2/5.3) und Dokumentation sind
ergänzt; BU in Teil-Codes zerlegt (Ziel, Metrik, Begründung, Erfolgskriterium, Geschäftsrahmung).

**Grenzen des Kodierers.**
- Regelbasiert auf Code-Mustern und Schlüsselwörtern: erkennt, *dass* etwas im Code/Text steht, nicht ob
  es inhaltlich richtig oder wirksam ist (eine Konstanten-Regel, die nichts entfernt, zählt als
  `prep_select_rule`; eine `.map(`-Zeile auf dem Ziel zählt als Kodierung).
- Nur Sichtbares: leere Thinking-Blöcke; was der Agent denkt, aber nicht schreibt, fehlt.
- Zeilenordnung als Zeitordnung innerhalb eines Skripts; Funktionsdefinitionen, die erst später
  aufgerufen werden, verzerren das (selten).
- Text-Codes hängen an englischen Schlüsselwörtern; C-Notizen mit CRISP-Überschriften erzeugen leichter
  Treffer (z. B. „Objective“), deshalb Satzregeln bei `bu_metric_rationale` und `eval_business`.
- `du_leakage` und `eval_baseline` messen bei C und D auch Prompt-Befolgung (REACT-Prompt, § 2).
- Rücksprünge auf Aktionsebene; ein Monolith, der alles auf einmal umschreibt, zählt als ein Ereignis.
- A/B nur als Referenz; die Phasenstruktur ist dort erzwungen, ihre Ablauf-Codes sind nicht vergleichbar.
  In A/B sind `model_test_design` und `eval_baseline` vom Modeling-Kontrakt verlangt, `du_leakage` von den
  BU-/DU-Kontrakten erfragt (§ 7); Tabelle und Abbildung kennzeichnen das („enf.“, kursiv bzw. „prompted“).
- **Untergrenzen** (Prüfbefund K7-05, Faktenblatt F7-927/928): Die Ein-Satz-Regel von `bu_metric_rationale`
  übersieht knappe Begründungen über zwei Sätze („imbalanced (~24%). I'll use F1.“); der Hilfscode
  `bu_metric_rationale_unit` zeigt das Ausmaß (Kern C 60 statt 57, D 40 statt 33). `du_quality_ext`
  erkennt eine Unique-Zählung je Spalte über `len(x.unique())` nicht. Dazu die Untergrenzen aus der
  Validierung (`doc_crisp_sections`, `eval_business`, F7-917/918) und, um höchstens einen Run je Arm,
  `du_class_dist` (Ziel über Schleifenvariable, F7-914) und `prep_explicit` (TF-IDF, F7-912).
- **`model_tuning` nur über Konstruktor-Argumente** (Prüfbefund K7-06, F7-929): ein Vergleich über
  Fit-Argumente (`fit(…, sample_weight=…)` gegen `fit(…)`) zählt nicht als Tuning
  (`main-student_health-crisp_react-r4`, work/train.py:58-77, bestanden).
- **Ziel über Schleifenvariable** (`for c in cand: tr[c].value_counts()`) ist keine Ziel-Referenz von
  `du_class_dist` (F7-914).
- Die Definitions-Sensitivität (zweiter Kodierer, `alt_intent`) ist nur an der Stichprobe von 24 Runs
  geprüft; Lücken, die dort nicht vorkommen, bleiben unentdeckt.

**Versionen.** 1.0 (09.11.2026) war die Fassung, nach der der zweite Kodierer gearbeitet hat; er las die
Muster- und Wortlisten als abgeschlossen. `code_process.py` enthielt schon damals einige Muster mehr
(Faktenblatt F7-930). 1.1 gleicht die Listen in § 4 an das Script an und ergänzt
`bu_metric_rationale_unit` und die Grenzen oben. Eine Neukodierung aller 180 C/D-Runs mit den Listen von
1.0 ändert keinen Wert der 28 Haupt- und vier Hilfscodes (`thesis/scripts/codebook_v10_check.py`); die
Werte beider Fassungen sind also identisch, und beide Kodierer wenden für jeden kodierten Wert dieselben
Regeln an.
