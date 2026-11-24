# src/crispfsm/prompts.py
"""Feste Phasen-Prompts (Kontrollvariable: identischer Wortlaut in allen Laeufen).

Sprache (seit 23.09.): Alle Strings, die ein LLM sieht, sind englisch; Kommentare und
Docstrings bleiben deutsch.

CRISP-DM-Inhalt (seit 23.09.): Die Aufgaben jeder Phase stammen aus dem Referenzmodell
von CRISP-DM 1.0 (Chapman et al. 2000, "CRISP-DM 1.0: Step-by-step data mining guide",
Teil II, S. 16-31), woertlich zitiert (Task- + Output-Absaetze). Mechanische Regel
(S5, 23.09.): ALLE generischen Aufgaben der Phasen 1-5, vollstaendig; nur Phase 6
(Deployment) ist ausgelassen. Keine inhaltliche Auswahl mehr nach "im Setting
anwendbar" — das war eine Forscher-Entscheidung, die der Worker selbst treffen soll
(Kontext: docs/crisp-operationalization.md).
Derselbe Block steht im Phasen-Prompt von Arm A/B und in der Anleitung von Arm C
(Wissens-Paritaet A/C per Konstruktion, tests/test_prompts.py).
Vorher (bis 23.09.): eigene Paraphrase ("splitte in train/val, trainiere einen
Klassifikator") — liess Testdesign, Technikvergleich und Tuning aus CRISP-DM §4 weg."""
from __future__ import annotations

_COMMON = (
    "You are an ML engineer working in the current directory. "
    "Write Python code to files and run it with Bash. "
    "Use pandas and scikit-learn. Work exclusively in the current directory; do not access "
    "anything outside it (e.g. via '..').\n"
    # S3 (23.09.): CRISP-DM setzt Stakeholder voraus (1.1 "the client", 4.4 "domain
    # experts") — im Setting gibt es keine. Gilt fuer alle Arme (auch D), damit der
    # Satz kein Unterschied zwischen den Armen ist.
    "There is no client or domain expert to consult: make and state your own assumptions.\n"
)

# S3 (23.09.): CRISP-DM verlangt je Output Berichte/Dokumentation/Plots. Einmal definiert
# und in A/B (_CRISP_INTRO) UND C (CRISP_GUIDANCE) verwendet -> Paritaet A/C.
_BRIEF_REPORTS = ("Where the reference model asks for reports, documentation or plots, keep "
                  "them brief (e.g. a short notes file); they do not replace the required "
                  "output files.\n")

# S2 (23.09.): Die Geschaeftsaufgabe steht in JEDEM A/B-Phasen-Prompt (BU hat sie schon im
# Auftrag), direkt nach _COMMON — sonst sehen DU..EVAL die Aufgabe nie, obwohl CRISP-DM
# laufend auf "business objectives" verweist.
_BUSINESS_TASK = "Business task: \"{task_prompt}\"\n"

# R2-4 (Review 2, 23.09.): Form der Vorhersagen war nirgends festgelegt — kodierte Labels
# sind ein Formfehler (scoring), den A/B per Gate-Feedback lernen, C/D nie. Ein Satz fuer
# alle Arme (REACT + EVAL).
_LABEL_FORM = ("containing class labels in the same form as the values of the target column "
               "in 'data/train.parquet' (not encoded codes or probabilities)")

# R2-1: Enthaltungs-Satz identisch in A/B (BU) und C/D (REACT) — vorher bekamen nur A/B
# die Praeferenz fuer ein ehrliches cannot_determine (verzerrt die vergifteten Varianten).
_NO_GUESS = ("Do not guess a target column that does not fit the task — an honest "
             "cannot_determine is better than an invented target.")
_NO_TARGET = ("If no target column can be derived from the task and the data (the target "
              "variable does not exist in the data or is ambiguous)")

# Generische CRISP-DM-1.0-Aufgaben je Phase (Chapman et al. 2000, Teil II), woertlich:
# je uebernommener Aufgabe der Task-Absatz + alle Output-Absaetze. Bereinigt nur
# PDF-Extraktionsartefakte (Ligaturen, Silbentrennung am Zeilenende, Zeilenumbrueche,
# Seitenfusszeilen, Abbildungsunterschriften); mehrere Absaetze einer Aufgabe bzw. eines
# Outputs sind mit Leerzeichen zu einer Zeile verbunden. Keine Phasennamen anderer Phasen
# in Grossschreibung (Phasen-Keywords muessen eindeutig bleiben) und keine geschweiften
# Klammern (die Bloecke laufen durch str.format).
# Umfang (S5, 23.09.): ALLE generischen Aufgaben der Phasen 1-5 in Guide-Reihenfolge,
# je vollstaendig mit allen Outputs; nur Phase 6 (Deployment, S. 32-34) ist ausgelassen.
# Geprueft gegen das PDF in beide Richtungen (jeder Prompt-Satz steht im Guide; jeder Satz
# eines uebernommenen Aufgaben-Abschnitts steht im Prompt).
CRISP_TASKS = {
    # §1 Business Understanding, S. 16-19, vollstaendig, woertlich: 1.1-1.4.
    "bu": (
        "1.1 Determine business objectives\n"
        "The first objective of the data analyst is to thoroughly understand, from a business "
        "perspective, what the client really wants to accomplish. Often the client has many "
        "competing objectives and constraints that must be properly balanced. The analyst’s goal "
        "is to uncover important factors, at the beginning, that can influence the outcome of the "
        "project. A possible consequence of neglecting this step is to expend a great deal of "
        "effort producing the right answers to the wrong questions.\n"
        "Background: Record the information that is known about the organization’s business "
        "situation at the beginning of the project.\n"
        "Business objectives: Describe the customer’s primary objective, from a business "
        "perspective. In addition to the primary business objective, there are typically other "
        "related business questions that the customer would like to address. For example, the "
        "primary business goal might be to keep current customers by predicting when they are "
        "prone to move to a competitor. Examples of related business questions are “How does the "
        "primary channel (e.g., ATM, visit branch, internet) a bank customer uses affect whether "
        "they stay or go?” or “Will lower ATM fees significantly reduce the number of high-value "
        "customers who leave?”\n"
        "Business success criteria: Describe the criteria for a successful or useful outcome to "
        "the project from the business point of view. This might be quite specific and able to "
        "be measured objectively, such as reduction of customer churn to a certain level or "
        "general and subjective such as “give useful insights into the relationships.” In the "
        "latter case it should be indicated who makes the subjective judgment.\n"
        "1.2 Assess situation\n"
        "This task involves more detailed fact-finding about all of the resources, "
        "constraints, assumptions and other factors that should be considered in determining "
        "the data analysis goal and project plan. In the previous task, your objective is to "
        "quickly get to the crux of the situation. Here, you want to flesh out the details.\n"
        "Inventory of resources: List the resources available to the project, including: "
        "personnel (business experts, data experts, technical support, data mining personnel), "
        "data (fixed extracts, access to live warehoused or operational data), computing "
        "resources (hardware platforms) and software (data mining tools, other relevant "
        "software).\n"
        "Requirements, assumptions and constraints: List all requirements of the project "
        "including schedule of completion, comprehensibility and quality of results and "
        "security as well as legal issues. As part of this output, make sure that you are "
        "allowed to use the data. List the assumptions made by the project. These may be "
        "assumptions about the data that can be checked during data mining, but may also "
        "include non-checkable assumptions about the business upon which the project rests. "
        "It is particularly important to list the latter if they form conditions on the "
        "validity of the results. List the constraints on the project. These may be "
        "constraints on the availability of resources, but may also include technological "
        "constraints such as the size of data that it is practical to use for modeling.\n"
        "Risks and contingencies: List the risks or events that might occur to delay the "
        "project or cause it to fail. List the corresponding contingency plans; what action "
        "will be taken if the risks happen.\n"
        "Terminology: Compile a glossary of terminology relevant to the project. This may "
        "include two components: (1) A glossary of relevant business terminology, which forms "
        "part of the business understanding available to the project. Constructing this "
        "glossary is a useful “knowledge elicitation” and education exercise. (2) A glossary "
        "of data mining terminology, illustrated with examples relevant to the business "
        "problem in question.\n"
        "Costs and benefits: Construct a cost-benefit analysis for the project, which "
        "compares the costs of the project with the potential benefit to the business if it "
        "is successful. The comparison should be as specific as possible, for example using "
        "monetary measures in a commercial situation.\n"
        "1.3 Determine data mining goals\n"
        "A business goal states objectives in business terminology. A data mining goal states "
        "project objectives in technical terms. For example, the business goal might be "
        "“Increase catalog sales to existing customers.” A data mining goal might be “Predict "
        "how many widgets a customer will buy, given their purchases over the past three years, "
        "demographic information (age, salary, city, etc.) and the price of the item.”\n"
        "Data mining goals: Describe the intended outputs of the project that enables the "
        "achievement of the business objectives.\n"
        "Data mining success criteria: Define the criteria for a successful outcome to the "
        "project in technical terms, for example a certain level of predictive accuracy or a "
        "propensity to purchase profile with a given degree of “lift.” As with business success "
        "criteria, it may be necessary to describe these in subjective terms, in which case the "
        "person or persons making the subjective judgment should be identified.\n"
        "1.4 Produce project plan\n"
        "Describe the intended plan for achieving the data mining goals and thereby achieving "
        "the business goals. The plan should specify the anticipated set of steps to be "
        "performed during the rest of the project including an initial selection of tools and "
        "techniques.\n"
        "Project plan: List the stages to be executed in the project, together with duration, "
        "resources required, inputs, outputs and dependencies. Where possible make explicit "
        "the large-scale iterations in the data mining process, for example repetitions of the "
        "modeling and evaluation phases. As part of the project plan, it is also important to "
        "analyze dependencies between time schedule and risks. Mark results of these analyses "
        "explicitly in the project plan, ideally with actions and recommendations if the risks "
        "appear. Note: the project plan contains detailed plans for each phase. For example, "
        "decide at this point which evaluation strategy will be used in the evaluation phase. "
        "The project plan is a dynamic document in the sense that at the end of each phase a "
        "review of progress and achievements is necessary and an update of the project plan "
        "accordingly is recommended. Specific review points for these reviews are part of the "
        "project plan, too.\n"
        "Initial assessment of tools and techniques: At the end of the first phase, the "
        "project also performs an initial assessment of tools and techniques. Here, you "
        "select a data mining tool that supports various methods for different stages of the "
        "process, for example. It is important to assess tools and techniques early in the "
        "process since the selection of tools and techniques possibly influences the entire "
        "project.\n"
    ),
    # §2 Data Understanding, S. 20-22, vollstaendig, woertlich: 2.1-2.4.
    "du": (
        "2.1 Collect initial data\n"
        "Acquire within the project the data (or access to the data) listed in the project "
        "resources. This initial collection includes data loading if necessary for data "
        "understanding. For example, if you apply a specific tool for data understanding, it "
        "makes perfect sense to load your data into this tool. This effort possibly leads to "
        "initial data preparation steps. Note: if you acquire multiple data sources, "
        "integration is an additional issue, either here or in the later data preparation "
        "phase.\n"
        "Initial data collection report: List the dataset (or datasets) acquired, together "
        "with their locations within the project, the methods used to acquire them and any "
        "problems encountered. Record problems encountered and any solutions achieved to aid "
        "with future replication of this project or with the execution of similar future "
        "projects.\n"
        "2.2 Describe data\n"
        "Examine the “gross” or “surface” properties of the acquired data and report on the "
        "results.\n"
        "Data description report: Describe the data which has been acquired, including: the "
        "format of the data, the quantity of data, for example number of records and fields in "
        "each table, the identities of the fields and any other surface features of the data "
        "which have been discovered. Does the data acquired satisfy the relevant requirements?\n"
        "2.3 Explore data\n"
        "This task tackles the data mining questions, which can be addressed using querying, "
        "visualization and reporting. These include: distribution of key attributes, for example "
        "the target attribute of a prediction task; relations between pairs or small numbers of "
        "attributes; results of simple aggregations; properties of significant sub-populations; "
        "simple statistical analyses. These analyses may address directly the data mining goals; "
        "they may also contribute to or refine the data description and quality reports and feed "
        "into the transformation and other data preparation needed for further analysis.\n"
        "Data exploration report: Describe results of this task including first findings or "
        "initial hypothesis and their impact on the remainder of the project. If appropriate, "
        "include graphs and plots, which indicate data characteristics or lead to interesting "
        "data subsets for further examination.\n"
        "2.4 Verify data quality\n"
        "Examine the quality of the data, addressing questions such as: is the data complete "
        "(does it cover all the cases required)? Is it correct or does it contain errors and if "
        "there are errors how common are they? Are there missing values in the data? If so how "
        "are they represented, where do they occur and how common are they?\n"
        "Data quality report: List the results of the data quality verification; if quality "
        "problems exist, list possible solutions. Solutions to data quality problems generally "
        "depend heavily on both data and business knowledge.\n"
    ),
    # §3 Data Preparation, S. 24-26, vollstaendig, woertlich: 3.1-3.5. Nicht enthalten: die
    # Phasen-Outputs auf S. 23 (Dataset, Dataset description) — sie gehoeren zu keiner
    # generischen Aufgabe.
    "prep": (
        "3.1 Select data\n"
        "Decide on the data to be used for analysis. Criteria include relevance to the data "
        "mining goals, quality and technical constraints such as limits on data volume or data "
        "types. Note that data selection covers selection of attributes (columns) as well as "
        "selection of records (rows) in a table.\n"
        "Rationale for inclusion/exclusion: List the data to be included/excluded and the "
        "reasons for these decisions.\n"
        "3.2 Clean data\n"
        "Raise the data quality to the level required by the selected analysis techniques. This "
        "may involve selection of clean subsets of the data, the insertion of suitable defaults "
        "or more ambitious techniques such as the estimation of missing data by modeling.\n"
        "Data cleaning report: Describe what decisions and actions were taken to address the "
        "data quality problems reported during the verify data quality task of the data "
        "understanding phase. Transformations of the data for cleaning purposes and the possible "
        "impact on the analysis results should be considered.\n"
        "3.3 Construct data\n"
        "This task includes constructive data preparation operations such as the production of "
        "derived attributes, entire new records or transformed values for existing attributes.\n"
        "Derived attributes: Derived attributes are new attributes that are constructed from one "
        "or more existing attributes in the same record. Examples: area = length * width.\n"
        "Generated records: Describe the creation of completely new records. Example: create "
        "records for customers who made no purchase during the past year. There was no reason to "
        "have such records in the raw data, but for modeling purposes it might make sense to "
        "explicitly represent the fact that certain customers made zero purchases.\n"
        "3.4 Integrate data\n"
        "These are methods whereby information is combined from multiple tables or records to "
        "create new records or values.\n"
        "Merged data: Merging tables refers to joining together two or more tables that have "
        "different information about the same objects. Example: a retail chain has one table "
        "with information about each store’s general characteristics (e.g., floor space, type "
        "of mall), another table with summarized sales data (e.g., profit, percent change in "
        "sales from previous year) and another with information about the demographics of the "
        "surrounding area. Each of these tables contains one record for each store. These "
        "tables can be merged together into a new table with one record for each store, "
        "combining fields from the source tables. Merged data also covers aggregations. "
        "Aggregation refers to operations where new values are computed by summarizing "
        "together information from multiple records and/or tables. For example, converting a "
        "table of customer purchases where there is one record for each purchase into a new "
        "table where there is one record for each customer, with fields such as number of "
        "purchases, average purchase amount, percent of orders charged to credit card, percent "
        "of items under promotion, etc.\n"
        "3.5 Format data\n"
        "Formatting transformations refer to primarily syntactic modifications made to the data "
        "that do not change its meaning, but might be required by the modeling tool.\n"
        "Reformatted data: Some tools have requirements on the order of the attributes, such as "
        "the first field being a unique identifier for each record or the last field being the "
        "outcome field the model is to predict. It might be important to change the order of the "
        "records in the dataset. Perhaps the modeling tool requires that the records be sorted "
        "according to the value of the outcome attribute. A common situation is that the records "
        "of the dataset are initially ordered in some way but the modeling algorithm needs them "
        "to be in a fairly random order. For example, when using neural networks it is generally "
        "best for the records to be presented in a random order although some tools handle this "
        "automatically without explicit user intervention. Additionally, there are purely "
        "syntactic changes made to satisfy the requirements of the specific modeling tool. "
        "Examples: removing commas from within text fields in comma-delimited data files, "
        "trimming all values to a maximum of 32 characters.\n"
    ),
    # §4 Modeling, S. 27-29, vollstaendig, woertlich: 4.1-4.4 (4.1 inkl. Output "Modeling
    # assumptions" auf S. 28); 4.4 fordert ausdruecklich mehrfache Anwendung/alternative
    # Techniken, Vergleich und iteratives Tuning.
    "model": (
        "4.1 Select modeling technique\n"
        "As the first step in modeling, select the actual modeling technique that is to be used. "
        "Whereas you possibly already selected a tool in business understanding, this task "
        "refers to the specific modeling technique, e.g., decision tree building with C4.5 or "
        "neural network generation with back propagation. If multiple techniques are applied, "
        "perform this task for each technique separately.\n"
        "Modeling technique: Document the actual modeling technique that is to be used.\n"
        "Modeling assumptions: Many modeling techniques make specific assumptions on the data, "
        "e.g., all attributes have uniform distributions, no missing values allowed, class "
        "attribute must be symbolic etc. Record any such assumptions made.\n"
        "4.2 Generate test design\n"
        "Before we actually build a model, we need to generate a procedure or mechanism to test "
        "the model’s quality and validity. For example, in supervised data mining tasks such as "
        "classification, it is common to use error rates as quality measures for data mining "
        "models. Therefore, we typically separate the dataset into train and test set, build the "
        "model on the train set and estimate its quality on the separate test set.\n"
        "Test design: Describe the intended plan for training, testing and evaluating the "
        "models. A primary component of the plan is to decide how to divide the available "
        "dataset into training data, test data and validation datasets.\n"
        "4.3 Build model\n"
        "Run the modeling tool on the prepared dataset to create one or more models.\n"
        "Parameter settings: With any modeling tool, there are often a large number of "
        "parameters that can be adjusted. List the parameters and their chosen value, along with "
        "the rationale for the choice of parameter settings.\n"
        "Models: These are the actual models produced by the modeling tool, not a report.\n"
        "Model description: Describe the resultant model. Report on the interpretation of the "
        "models and document any difficulties encountered with their meanings.\n"
        "4.4 Assess model\n"
        "The data mining engineer interprets the models according to his domain knowledge, the "
        "data mining success criteria and the desired test design. This task interferes with the "
        "subsequent evaluation phase. Whereas the data mining engineer judges the success of the "
        "application of modeling and discovery techniques more technically, he contacts business "
        "analysts and domain experts later in order to discuss the data mining results in the "
        "business context. Moreover, this task only considers models whereas the evaluation "
        "phase also takes into account all other results that were produced in the course of "
        "the project. The data mining engineer tries to rank the models. He assesses the models "
        "according to the evaluation criteria. As far as possible he also takes into account "
        "business objectives and business success criteria. In most data mining projects, the "
        "data mining engineer applies a single technique more than once or generates data mining "
        "results with different alternative techniques. In this task, he also compares all "
        "results according to the evaluation criteria.\n"
        "Model assessment: Summarize results of this task, list qualities of generated models "
        "(e.g., in terms of accuracy) and rank their quality in relation to each other.\n"
        "Revised parameter settings: According to the model assessment, revise parameter "
        "settings and tune them for the next run in the Build Model task. Iterate model building "
        "and assessment until you strongly believe that you found the best model(s). Document "
        "all such revisions and assessments.\n"
    ),
    # §5 Evaluation, S. 30-31, vollstaendig, woertlich: 5.1-5.3. (Phase 6 Deployment,
    # S. 32-34, ist als einzige Phase ausgelassen.)
    "eval": (
        "5.1 Evaluate results\n"
        "Previous evaluation steps dealt with factors such as the accuracy and generality of "
        "the model. This step assesses the degree to which the model meets the business "
        "objectives and seeks to determine if there is some business reason why this model is "
        "deficient. Another option of evaluation is to test the model(s) on test applications "
        "in the real application if time and budget constraints permit. Moreover, evaluation "
        "also assesses other data mining results generated. Data mining results cover models "
        "which are necessarily related to the original business objectives and all other "
        "findings which are not necessarily related to the original business objectives but "
        "might also unveil additional challenges, information or hints for future directions.\n"
        "Assessment of data mining results with respect to business success criteria: "
        "Summarize assessment results in terms of business success criteria including a final "
        "statement whether the project already meets the initial business objectives.\n"
        "Approved models: After model assessment with respect to business success criteria, "
        "the generated models that meet the selected criteria become approved models.\n"
        "5.2 Review process\n"
        "At this point the resultant model hopefully appears to be satisfactory and to satisfy "
        "business needs. It is now appropriate to do a more thorough review of the data mining "
        "engagement in order to determine if there is any important factor or task that has "
        "somehow been overlooked. This review also covers quality assurance issues, e.g., did we "
        "correctly build the model? Did we only use attributes that we are allowed to use and "
        "that are available for future analyses?\n"
        "Review of process: Summarize the process review and highlight activities that have "
        "been missed and/or should be repeated.\n"
        "5.3 Determine next steps\n"
        "According to the assessment results and the process review, the project decides how "
        "to proceed at this stage. The project needs to decide whether to finish this project "
        "and move on to deployment if appropriate or whether to initiate further iterations or "
        "set up new data mining projects. This task includes analyses of remaining resources "
        "and budget that influences the decisions.\n"
        "List of possible actions: List the potential further actions along with the reasons "
        "for and against each option.\n"
        "Decision: Describe the decision as to how to proceed along with the rationale.\n"
    ),
}

# Reihenfolge: erst der Hinweis zu Berichten, dann die Einleitung direkt vor dem Zitat.
_CRISP_INTRO = _BRIEF_REPORTS + ("Follow the generic tasks of this phase from the CRISP-DM 1.0 "
                                 "reference model (quoted verbatim):\n")

PREP = _COMMON + _BUSINESS_TASK + (
    "TASK (Data Preparation): Read 'data/train.parquet' (feature columns + column "
    "'{target}'). Prepare the data (handle missing values, make all features "
    "numeric — int/float; convert booleans to 0/1). Write the result to "
    "'prep/prepared.parquet' — feature columns + column '{target}' with its original values, "
    "no NaNs. Save your preprocessing code in 'prep/' so that it can be applied identically "
    "to new data. Use only 'data/train.parquet'; the test labels are not available.\n"
) + _CRISP_INTRO + CRISP_TASKS["prep"]

MODEL = _COMMON + _BUSINESS_TASK + (
    "TASK (Modeling): Read 'prep/prepared.parquet' (feature columns + column '{target}'). "
    "Save the best trained model according to the metric in 'bu/task_spec.json' with pickle to 'model/model.pkl' and write "
    "'model/metrics.json' with "
    "{{\"metric\": \"<the metric from 'bu/task_spec.json'>\", \"val_score\": <float>, "
    "\"dummy_score\": <float>}} — val_score: this metric for the saved model on the "
    "validation data of your test design (a held-out validation split or the "
    "cross-validation mean); dummy_score: the same metric for "
    "DummyClassifier(strategy='most_frequent'), measured the same way. Write 'model/model.pkl' and 'model/metrics.json' as soon as you have "
    "a first working model and overwrite them whenever you find a better one. "
    "The test labels are not available; build your test design from "
    "'prep/prepared.parquet' only.\n"
) + _CRISP_INTRO + CRISP_TASKS["model"]

EVAL = _COMMON + _BUSINESS_TASK + (
    "TASK (Evaluation): Load the model from 'model/model.pkl' and the test features from "
    "'data/test_features.parquet' (contains NO target column). Apply exactly the same "
    "preprocessing as in training (reuse the code in 'prep/') and produce a prediction for "
    "every row. Write the predictions to 'eval/predictions.parquet' — exactly one column "
    "'prediction' " + _LABEL_FORM + ", same row count and row order as test_features. "
    "Do not retrain or replace 'model/model.pkl' in this phase; note any recommended "
    "further iterations in your notes.\n"
) + _CRISP_INTRO + CRISP_TASKS["eval"]

BU = _COMMON + (
    "TASK (Business Understanding): Your business task is:\n"
    "  \"{task_prompt}\"\n"
    "Inspect 'data/train.parquet' with pandas (columns, dtypes, example values) and derive "
    "an ML task specification from it. Write EXACTLY ONE of the two variants as "
    "JSON to 'bu/task_spec.json':\n"
    "1) If the task can be solved with this data:\n"
    "   {{\"task_type\": \"classification\", \"target_column\": \"<column>\", "
    "\"metric\": \"f1\"|\"accuracy\", \"feature_columns\": [<all remaining columns>], "
    "\"critical_columns\": [<optional: columns that should be checked especially carefully "
    "in the later data understanding (DU) phase, e.g. suspected ID or leakage columns>]}}\n"
    "2) " + _NO_TARGET + ":\n"
    "   {{\"cannot_determine\": true, \"reason\": \"<justification>\", \"missing_info\": [<what is missing>]}}\n"
    + _NO_GUESS + " Train NOTHING, only analyze.\n"
) + _CRISP_INTRO + CRISP_TASKS["bu"]

DU = _COMMON + _BUSINESS_TASK + (
    "TASK (Data Understanding): Analyze 'data/train.parquet' READ-ONLY — "
    "do not modify any data and do not create any new data files. Write a "
    "diagnostic report as JSON to 'du/data_report.json' with EXACTLY these fields:\n"
    "- \"n_rows\": <int> row count of train.parquet\n"
    "- \"columns\": {{\"<column>\": {{\"dtype\": \"numeric\"|\"bool\"|\"categorical\", "
    "\"n_missing\": <int>}}}} for EVERY column. dtype rule: boolean pandas dtype -> \"bool\", "
    "otherwise numeric dtype -> \"numeric\", everything else -> \"categorical\".\n"
    "- \"class_balance\": {{\"<str(class value)>\": <int>}} for the target column '{target}'\n"
    "- \"quality_issues\": [<string>] — data quality findings in your own words "
    "(e.g. missing values, suspicious columns); empty list if nothing stands out\n"
    "- \"excluded_columns\": [{{\"column\": \"<column>\", \"reason\": \"<why>\"}}] — columns "
    "that are NOT suitable as features (e.g. IDs, constants, suspected leakage columns "
    "that separate the target suspiciously perfectly); empty list if none\n"
    "{critical_hint}"
    "If the data fundamentally contradicts the task (e.g. the column '{target}' does not "
    "match the business task or is not a sensible target for it), instead write ONLY "
    "{{\"spec_mismatch\": true, \"spec_mismatch_reason\": \"<why>\"}}.\n"
    "IMPORTANT: Compute all numbers exactly with pandas — no estimates. "
    "Only analyze, do not train anything.\n"
) + _CRISP_INTRO + CRISP_TASKS["du"]

_TEMPLATES = {"prep": PREP, "model": MODEL, "eval": EVAL}


def _critical_hint(critical_columns: list[str] | None) -> str:
    if not critical_columns:
        return ""
    # „BU phase" statt „Business Understanding": Phasen-Keywords müssen
    # prompt-übergreifend eindeutig bleiben (Fakes/Log-Analyse routen darüber).
    return ("Columns flagged in the business understanding (BU) phase — check them especially carefully "
            f"for reasons to exclude them: {critical_columns}.\n")


def du_initial_prompt(target: str, critical_columns: list[str] | None = None, *,
                      task_prompt: str = "") -> str:
    return DU.format(target=target, critical_hint=_critical_hint(critical_columns),
                     task_prompt=task_prompt)


def du_resume_prompt(target: str, feedback: str,
                     critical_columns: list[str] | None = None, *,
                     task_prompt: str = "", kind: str = "own") -> str:
    return (_resume_head(kind, feedback,
                         f"Your last attempt at the phase 'Data Understanding' did NOT pass the "
                         f"Quality Gate.\nReason: {feedback}\n"
                         f"Correct 'du/data_report.json' accordingly.\n\n")
            + du_initial_prompt(target, critical_columns, task_prompt=task_prompt))


def bu_initial_prompt(task_prompt: str) -> str:
    return BU.format(task_prompt=task_prompt)


def bu_resume_prompt(task_prompt: str, feedback: str, *, kind: str = "own") -> str:
    return (_resume_head(kind, feedback,
                         f"Your last attempt at the phase 'Business Understanding' did NOT pass "
                         f"the Quality Gate.\nReason: {feedback}\n"
                         f"Correct the specification in 'bu/task_spec.json' accordingly.\n\n")
            + BU.format(task_prompt=task_prompt))

# Arm D (Naked ReAct). B2 (23.09.): Die gemeinsamen ML-Hinweise (Ziel ableiten statt raten,
# ungeeignete Feature-Spalten, Missing Values/numerisch, Baseline-Vergleich, gleiche
# Vorverarbeitung fuer Test) stehen HIER statt als Paritaetssaetze in CRISP_GUIDANCE —
# D und C bekommen sie damit beide, A/B ueber ihre Artefakt-Kontrakte. Frueh-Schreiben der
# Predictions: Gegenstueck zu S4 (MODEL), damit ein abgebrochener Lauf trotzdem liefert.
# Kein Prozess-Vokabular (Purity-Test, tests/test_prompts.py).
REACT = _COMMON + (
    "TASK: {task_prompt}\n"
    "The directory contains 'data/train.parquet' and 'data/test_features.parquet' "
    "(contains NO target column). This is a classification task. Derive the target column "
    "and choose a metric (f1 or accuracy). " + _NO_GUESS + " Identify columns that are not suitable as features (e.g. IDs, constants or "
    "suspected leakage columns that separate the target suspiciously perfectly). Handle "
    "missing values and make all features numeric. Compare your model against a simple "
    "baseline (e.g. the most frequent class). Apply exactly the same preprocessing to the "
    "test features as in training.\n"
    "Produce a prediction for every row of test_features and write them to "
    "'predictions.parquet' — exactly one column 'prediction' " + _LABEL_FORM + ", same row "
    "count and row order as test_features. Write 'predictions.parquet' as soon as you have a "
    "first working model and overwrite it whenever you find a better one. Together with your "
    "first 'predictions.parquet', write 'result.json' with "
    "{{\"target_column\": \"<column>\", \"task_type\": \"classification\", "
    "\"metric\": \"f1\"|\"accuracy\"}} and update it if your choice changes. Both files are "
    "required.\n"
    + _NO_TARGET + ", instead write 'result.json' with "
    "{{\"cannot_determine\": true, \"reason\": \"<justification>\"}}.\n"
)


def react_prompt(task_prompt: str) -> str:
    """Arm D (Naked ReAct): Task + Deliverables + gemeinsame ML-Hinweise + neutraler
    Abstain-Satz — sonst nichts. Kein CRISP-Vokabular, keine Phasen (Purity-Test)."""
    return REACT.format(task_prompt=task_prompt)


# Arm C (CRISP-ReAct) = Arm-D-Prompt + CRISP_GUIDANCE. Reine Methoden-Anleitung, KEIN
# Enforcement — keine Gates, kein Retry-/Feedback-Vokabular, keine Artefakt-Kontrakte der
# FSM-Arme. B2 (23.09.): Der Diff C<->D ist jetzt exakt der CRISP-Text — Einleitung mit dem
# woertlichen Zitat zur Phasenfolge (CRISP-DM 1.0, S. 13), der Kurz-Bericht-Satz (_BRIEF_
# REPORTS, identisch zu A/B), fuenf Phasen-Ueberschriften und die CRISP_TASKS-Bloecke. Die
# frueheren Paritaetssaetze je Phase stehen jetzt im REACT-Prompt (gilt fuer C und D).
# Erzwungen wird nichts: Ob der Worker der Reihenfolge folgt, ist die Messung.
CRISP_GUIDANCE = (
    "APPROACH (CRISP-DM): Work on the task following the CRISP-DM process. "
    "Go through these phases in order and complete each one before you start the next, and "
    "return to an earlier phase when the results require it. As the CRISP-DM 1.0 reference model puts it: “The sequence of the phases is not rigid. "
    "Moving back and forth between different phases is always required. It depends on the "
    "outcome of each phase which phase or which particular task of a phase, has to be "
    "performed next.”\n"
    + _BRIEF_REPORTS +
    "1. Business Understanding:\n"
    + CRISP_TASKS["bu"] +
    "2. Data Understanding:\n"
    + CRISP_TASKS["du"] +
    "3. Data Preparation:\n"
    + CRISP_TASKS["prep"] +
    "4. Modeling:\n"
    + CRISP_TASKS["model"] +
    "5. Evaluation:\n"
    + CRISP_TASKS["eval"]
)


def crisp_react_prompt(task_prompt: str) -> str:
    """Arm C (CRISP-ReAct): byte-identisch zum Arm-D-Prompt PLUS angehängte
    CRISP-Anleitung — der Diff C↔D ist damit exakt ein Faktor (Domänenwissen,
    experiment-design.md §1). Paritäts-Test: test_prompts.py."""
    return react_prompt(task_prompt) + CRISP_GUIDANCE


def _du_findings(phase: str, du_report: dict | None) -> str:
    """Feed-Forward (roadmap §1): quality_issues + excluded_columns aus dem
    DU-Report gehen in den Prep-Prompt — die Phasen-Kette ist inhaltlich
    verbunden, nicht nur sequenziell."""
    if phase != "prep" or not du_report:
        return ""
    lines = []
    issues = du_report.get("quality_issues") or []
    excluded = du_report.get("excluded_columns") or []
    if issues:
        lines.append("FINDINGS from the data understanding (DU) phase (take them into account):")
        lines += [f"- {i}" for i in issues]
    if excluded:
        lines.append("Do NOT use these columns as features (excluded in the data "
                     "understanding (DU) phase):")
        lines += [f"- {e['column']}: {e.get('reason', '')}" for e in excluded]
    return "\n".join(lines) + "\n" if lines else ""


def initial_prompt(phase: str, target: str, du_report: dict | None = None, *,
                   task_prompt: str = "") -> str:
    return (_TEMPLATES[phase].format(target=target, task_prompt=task_prompt)
            + _du_findings(phase, du_report))


def resume_prompt(phase: str, target: str, feedback: str,
                  du_report: dict | None = None, *, task_prompt: str = "",
                  kind: str = "own") -> str:
    # N3 (23.09.): kein interner Phasen-Code ('prep', 'model', ...) im Resume-Satz.
    return (_resume_head(kind, feedback,
                         "Your last attempt at this phase did NOT pass the Quality Gate.\n"
                         f"Reason: {feedback}\n"
                         "Address this and fulfill the original task again.\n\n")
            + initial_prompt(phase, target, du_report, task_prompt=task_prompt))


# R2-3 (Review 2, 23.09.): Warum eine Phase erneut laeuft, haengt davon ab, woher der
# Sprung kommt (phases.resume_kind): "own" = eigenes Gate fehlgeschlagen; "later" = eine
# spaetere Phase ist gescheitert und springt zurueck (eval -> model, du -> bu); "earlier" =
# eine fruehere Phase wurde ueberarbeitet und die Kette laeuft erneut (model -> eval). Vorher
# bekamen alle drei den "did NOT pass"-Satz, "earlier" sogar mit dem Feedback eines
# BESTANDENEN Gates.
RESUME_KINDS = ("own", "later", "earlier")


def _resume_head(kind: str, feedback: str, own: str) -> str:
    if kind == "own":
        return own
    if kind == "later":
        return ("A later quality check found a problem with the pipeline result.\n"
                f"Reason: {feedback}\n"
                "Revise your work accordingly and complete the task below again.\n\n")
    if kind == "earlier":
        return ("An earlier step of the pipeline has been revised since your last attempt. "
                "Complete the task below again based on the current files.\n\n")
    raise ValueError(f"unknown resume kind {kind!r}")
