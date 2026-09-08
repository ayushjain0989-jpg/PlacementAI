"""
Train and select the placement-readiness model.

Two things here differ from a textbook train/test script, both because the
dataset is small (401 rows):

  * Model selection uses stratified 5-fold cross-validation rather than a single
    80/20 split. On 401 rows one split puts ~80 students in the test set, and
    which 80 you happen to draw moves F1 by several points -- more than the gap
    between the candidate models. Picking a winner on that basis is picking a
    winner on noise.

  * The raw CSV is cleaned first (see data_prep.py). Casing variants and typos
    were previously being encoded as distinct categories.

Artifacts written to models/:

    best_model.pkl        the fitted pipeline the Flask app serves
    features.pkl          ordered feature list
    model_results.pkl     per-model metrics (DataFrame) for the web UI
    feature_options.pkl   the exact input domain, used to render the web form
"""

import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from data_prep import (
    build_feature_options,
    clean_dataset
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "dataset/Sample.csv"

MODEL_DIR = "models"

TARGET = "Placement(Y/N)?"

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5

# Model selection metric. F1 balances the cost of telling an unprepared student
# they are ready against discouraging one who is.
SELECTION_METRIC = "F1 Score"


FEATURES = [

    "Gender",

    "10th board",

    "10th marks",

    "12th board",

    "12th marks",

    "Stream",

    "Cgpa",

    "Internships(Y/N)",

    "Training(Y/N)",

    "Backlog in 5th sem",

    "Innovative Project(Y/N)",

    "Communication level",

    "Technical Course(Y/N)"

]


CATEGORICAL_FEATURES = [

    "Gender",

    "10th board",

    "12th board",

    "Stream",

    "Internships(Y/N)",

    "Training(Y/N)",

    "Backlog in 5th sem",

    "Innovative Project(Y/N)",

    "Technical Course(Y/N)"

]


NUMERIC_FEATURES = [

    "10th marks",

    "12th marks",

    "Cgpa",

    "Communication level"

]


TARGET_LABELS = {

    "Placed": 1,

    "Not Placed": 0

}


def heading(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 2. LOAD DATASET
# ============================================================

heading("AI-BASED STUDENT PLACEMENT READINESS PREDICTOR")

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Loaded {df.shape[0]} rows "
    f"and {df.shape[1]} columns."
)


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [

    column

    for column in FEATURES + [TARGET]

    if column not in df.columns

]


if missing_columns:

    raise ValueError(

        "Missing required columns:\n"

        + "\n".join(
            f" - {column}"
            for column in missing_columns
        )

    )


print("All required columns are present.")


# ============================================================
# 4. REMOVE IDENTIFYING COLUMNS
# ============================================================
#
# Name and Email identify the student and carry no predictive signal, so they
# are dropped before anything else touches the frame.

df = df.drop(

    columns=[
        "Name",
        "Email"
    ],

    errors="ignore"

)


# ============================================================
# 5. CLEAN THE DATA
# ============================================================

heading("DATA CLEANING")

df, cleaning_report = clean_dataset(df)

for note in cleaning_report:

    print(" -", note)


# ============================================================
# 6. CLEAN TARGET VARIABLE
# ============================================================

df[TARGET] = (

    df[TARGET]
    .astype(str)
    .str.strip()
    .map(TARGET_LABELS)

)


before_rows = len(df)

df = df.dropna(
    subset=[TARGET]
)

dropped_rows = before_rows - len(df)


if dropped_rows:

    print(
        f"\nRemoved {dropped_rows} row(s) "
        f"with an unrecognised target value."
    )


df[TARGET] = df[TARGET].astype(int)


print("\nTarget distribution:")

print(
    df[TARGET]
    .value_counts()
    .sort_index()
    .rename(
        index={
            0: "Not Placed",
            1: "Placed"
        }
    )
    .to_string()
)


# ============================================================
# 7. FEATURES AND TARGET
# ============================================================

X = df[FEATURES]

y = df[TARGET]


heading("FEATURE SUMMARY")

print(
    f"{len(NUMERIC_FEATURES)} numeric, "
    f"{len(CATEGORICAL_FEATURES)} categorical, "
    f"{len(FEATURES)} total."
)


print("\nCategories per categorical feature (post-cleaning):")

for column in CATEGORICAL_FEATURES:

    values = sorted(
        df[column].dropna().unique().tolist()
    )

    print(
        f" - {column}: "
        f"{len(values)} -> {values}"
    )


# ============================================================
# 8. PIPELINE FACTORIES
# ============================================================
#
# Each candidate needs its own preprocessor and its own estimator instance.
# Sharing one ColumnTransformer across pipelines means they share fitted state,
# and reusing an estimator across cross-validation folds risks carrying fitted
# state between them. Building fresh objects from a factory removes both
# concerns, and keeps each model's hyperparameters written down exactly once.

def build_preprocessor():

    numeric_transformer = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )

        ]

    )

    categorical_transformer = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )

        ]

    )

    return ColumnTransformer(

        transformers=[

            (
                "num",
                numeric_transformer,
                NUMERIC_FEATURES
            ),

            (
                "cat",
                categorical_transformer,
                CATEGORICAL_FEATURES
            )

        ]

    )


MODEL_FACTORIES = {

    "Random Forest":
        lambda: RandomForestClassifier(

            n_estimators=300,

            max_depth=None,

            random_state=RANDOM_STATE,

            class_weight="balanced"

        ),

    "XGBoost":
        lambda: XGBClassifier(

            n_estimators=300,

            max_depth=4,

            learning_rate=0.05,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=RANDOM_STATE,

            eval_metric="logloss"

        ),

    "CatBoost":
        lambda: CatBoostClassifier(

            iterations=300,

            depth=5,

            learning_rate=0.05,

            loss_function="Logloss",

            verbose=False,

            random_seed=RANDOM_STATE

        )

}


def build_pipeline(model_name):

    return Pipeline(

        steps=[

            (
                "preprocessor",
                build_preprocessor()
            ),

            (
                "model",
                MODEL_FACTORIES[model_name]()
            )

        ]

    )


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================
#
# The holdout set is kept for reporting a metric on data no fold ever scored,
# but selection itself is done on the cross-validated scores below.

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y

)


heading("DATA SPLIT")

print(
    f"Training: {len(X_train)} | "
    f"Holdout: {len(X_test)} | "
    f"Cross-validation: {CV_FOLDS}-fold stratified"
)


# ============================================================
# 10. EVALUATE CANDIDATES
# ============================================================

CV_SCORING = {

    "accuracy":
        "accuracy",

    "precision":
        "precision",

    "recall":
        "recall",

    "f1":
        "f1",

    "roc_auc":
        "roc_auc"

}


cv_splitter = StratifiedKFold(

    n_splits=CV_FOLDS,

    shuffle=True,

    random_state=RANDOM_STATE

)


results = {}

trained_models = {}


for model_name in MODEL_FACTORIES:

    heading(f"Evaluating: {model_name}")

    # ---------------------------------------------------------
    # Cross-validation on the training portion
    # ---------------------------------------------------------

    cv_scores = cross_validate(

        build_pipeline(model_name),

        X_train,

        y_train,

        cv=cv_splitter,

        scoring=CV_SCORING,

        n_jobs=1

    )

    cv_f1_mean = float(
        np.mean(cv_scores["test_f1"])
    )

    cv_f1_std = float(
        np.std(cv_scores["test_f1"])
    )

    print(
        f"{CV_FOLDS}-fold CV F1 : "
        f"{cv_f1_mean:.4f} +/- {cv_f1_std:.4f}"
    )

    print(
        f"{CV_FOLDS}-fold CV AUC: "
        f"{np.mean(cv_scores['test_roc_auc']):.4f} +/- "
        f"{np.std(cv_scores['test_roc_auc']):.4f}"
    )

    # ---------------------------------------------------------
    # Holdout metrics, for display in the web UI
    # ---------------------------------------------------------

    pipeline = build_pipeline(model_name)

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    results[model_name] = {

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "F1 Score":
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "ROC-AUC":
            roc_auc_score(
                y_test,
                probabilities
            ),

        "CV F1 Mean":
            cv_f1_mean,

        "CV F1 Std":
            cv_f1_std

    }

    trained_models[model_name] = pipeline

    print(
        "Holdout   F1 : "
        f"{results[model_name]['F1 Score']:.4f}"
    )


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
).T


heading("MODEL PERFORMANCE COMPARISON")

print(
    results_df.round(4).to_string()
)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================
#
# Selection uses the cross-validated F1 rather than the holdout F1: the holdout
# is a single 81-student draw, and choosing on it would be choosing on noise.

best_model_name = results_df["CV F1 Mean"].idxmax()


heading("BEST MODEL")

print(
    f"Selected      : {best_model_name}"
)

print(
    "Criterion     : "
    f"highest {CV_FOLDS}-fold cross-validated F1"
)

print(
    "CV F1         : "
    f"{results_df.loc[best_model_name, 'CV F1 Mean']:.4f} "
    f"+/- {results_df.loc[best_model_name, 'CV F1 Std']:.4f}"
)

print(
    "Holdout F1    : "
    f"{results_df.loc[best_model_name, SELECTION_METRIC]:.4f}"
)


# ============================================================
# 13. RETRAIN BEST MODEL ON THE FULL DATASET
# ============================================================
#
# Now that the choice is made, refit on every row so the deployed model has seen
# as much of the (small) dataset as possible.

heading("FINAL MODEL TRAINING")

print(
    f"Retraining {best_model_name} "
    f"on all {len(X)} rows..."
)

final_model = build_pipeline(best_model_name)

final_model.fit(
    X,
    y
)

print("Final model trained.")


# ============================================================
# 14. SAVE ARTIFACTS
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


feature_options = build_feature_options(
    df,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES
)


artifacts = {

    "best_model.pkl":
        final_model,

    "features.pkl":
        FEATURES,

    "model_results.pkl":
        results_df,

    "feature_options.pkl":
        feature_options

}


heading("SAVED ARTIFACTS")

for filename, payload in artifacts.items():

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    joblib.dump(
        payload,
        path
    )

    print(" -", path)


# ============================================================
# 15. VERIFY THE SAVED INPUT DOMAIN
# ============================================================
#
# The web form is rendered from feature_options.pkl. Confirming here that every
# recorded option is a category the fitted encoder actually knows is what keeps
# the form and the model from drifting apart: previously the form offered board
# and stream values the model had never seen, and OneHotEncoder's
# handle_unknown="ignore" silently turned each one into an all-zero row.

heading("INPUT DOMAIN CHECK")

fitted_preprocessor = final_model.named_steps["preprocessor"]

fitted_encoder = (
    fitted_preprocessor
    .named_transformers_["cat"]
    .named_steps["onehot"]
)

known_categories = dict(
    zip(
        CATEGORICAL_FEATURES,
        fitted_encoder.categories_
    )
)


unknown_options = []

for column, values in feature_options["categorical"].items():

    known = set(
        known_categories[column].tolist()
    )

    for value in values:

        if value not in known:

            unknown_options.append(
                f"{column}={value!r}"
            )


if unknown_options:

    raise SystemExit(

        "Recorded form options the model does not know: "

        + ", ".join(unknown_options)

    )


total_options = sum(
    len(values)
    for values in feature_options["categorical"].values()
)

print(
    f"All {total_options} categorical option(s) across "
    f"{len(feature_options['categorical'])} field(s) are "
    f"categories the model was trained on."
)

print(
    "Communication level accepted range: "
    f"{feature_options['numeric']['Communication level']['min']:.0f}"
    "-"
    f"{feature_options['numeric']['Communication level']['max']:.0f}"
)


# ============================================================
# 16. SAMPLE PREDICTION
# ============================================================
#
# Built from the recorded options so it cannot go stale as categories change.

def most_common(column):

    return feature_options["categorical"][column][0]


sample_student = pd.DataFrame([{

    "Gender":
        most_common("Gender"),

    "10th board":
        most_common("10th board"),

    "10th marks":
        90.0,

    "12th board":
        most_common("12th board"),

    "12th marks":
        85.0,

    "Stream":
        most_common("Stream"),

    "Cgpa":
        8.5,

    "Internships(Y/N)":
        "Yes",

    "Training(Y/N)":
        "Yes",

    "Backlog in 5th sem":
        "No",

    "Innovative Project(Y/N)":
        "Yes",

    "Communication level":
        4,

    "Technical Course(Y/N)":
        "Yes"

}])


prediction = final_model.predict(
    sample_student
)[0]

probability = final_model.predict_proba(
    sample_student
)[0][1] * 100


heading("TEST PREDICTION")

print(
    "Profile   : strong "
    f"({most_common('Stream')}, CGPA 8.5)"
)

print(
    "Prediction: "
    + (
        "PLACED"
        if prediction == 1
        else "NOT PLACED"
    )
)

print(
    f"Probability: {probability:.2f}%"
)


heading("TRAINING COMPLETED SUCCESSFULLY")

print(
    f"Rows: {len(df)} | "
    f"Features: {len(FEATURES)} | "
    f"Best model: {best_model_name}"
)

print("\nReady for the Flask application.")
