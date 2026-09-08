"""
The trained pipeline, loaded exactly once.

Previously both app.py and explain.py called joblib.load on best_model.pkl at import
time, so a 300-tree forest was held in memory twice. Everything that needs the model
now imports it from here.

A load failure is recorded rather than raised: the app should still serve its
informational pages and show a clear banner on the assessment form, which is more
useful than refusing to start.
"""

import logging

import joblib
import pandas as pd

from config import MODEL_PATH, MODEL_RESULTS_PATH


logger = logging.getLogger(__name__)


# =========================================================
# LOAD
# =========================================================

pipeline = None

load_error = None


try:

    pipeline = joblib.load(
        MODEL_PATH
    )

    logger.info(
        "Loaded model from %s",
        MODEL_PATH
    )

except Exception as error:

    load_error = str(error)

    logger.error(
        "Could not load the model from %s",
        MODEL_PATH,
        exc_info=True
    )


def is_available():

    return pipeline is not None


# =========================================================
# PREDICT
# =========================================================

def predict(input_frame):
    """
    Return (label, probability_percent) for a one-row DataFrame.

    The probability is the model's confidence in the positive ("Placed") class,
    already scaled to 0-100 and rounded, because every consumer wants it that way.
    """

    if pipeline is None:

        raise RuntimeError(
            "The prediction model is not loaded."
        )

    label = pipeline.predict(
        input_frame
    )[0]

    probability = pipeline.predict_proba(
        input_frame
    )[0][1] * 100

    return label, round(
        float(probability),
        2
    )


# =========================================================
# PIPELINE INTERNALS
# =========================================================
#
# Exposed for explain.py (which needs the preprocessor and the bare estimator to
# build a TreeExplainer) and for options.py, which uses the fitted encoder as a
# fallback source of valid form choices.

def get_preprocessor():

    if pipeline is None:
        return None

    return pipeline.named_steps.get(
        "preprocessor"
    )


def get_estimator():

    if pipeline is None:
        return None

    return pipeline.named_steps.get(
        "model"
    )


def known_categories():
    """
    Map each categorical column to the categories the fitted encoder learned.

    This is the ground truth for what the model can actually respond to. Any form
    option outside these sets is silently converted to an all-zero one-hot row by
    OneHotEncoder(handle_unknown="ignore") -- it does not error, it just stops
    mattering.
    """

    preprocessor = get_preprocessor()

    if preprocessor is None:
        return {}

    try:

        columns = dict(
            (name, cols)
            for name, _, cols in preprocessor.transformers_
        )["cat"]

        encoder = (
            preprocessor
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )

        return {

            column: [
                str(value)
                for value in categories
            ]

            for column, categories in zip(
                columns,
                encoder.categories_
            )

        }

    except Exception:

        logger.warning(
            "Could not read categories from the fitted preprocessor",
            exc_info=True
        )

        return {}


# =========================================================
# TRAINING RESULTS
# =========================================================
#
# train_model.py always writes a DataFrame indexed by model name. The view that
# consumed this used to carry ~250 lines branching over four hypothetical layouts
# (metric-keyed dict, model-keyed dict, DataFrame, with "F1" or "F1 Score" and
# "ROC-AUC" or "ROC AUC") -- none of which the training script has ever produced.

RESULT_COLUMNS = {

    "accuracy":
        "Accuracy",

    "precision":
        "Precision",

    "recall":
        "Recall",

    "f1":
        "F1 Score",

    "roc_auc":
        "ROC-AUC",

    "cv_f1":
        "CV F1 Mean",

    "cv_f1_std":
        "CV F1 Std"

}


def load_results():
    """
    Metrics per candidate model, as percentages, for the performance page.

    Raises on a missing or malformed artifact so the caller can show its banner --
    silently returning an empty list would render an empty page with no explanation.
    """

    results = joblib.load(
        MODEL_RESULTS_PATH
    )

    if not isinstance(results, pd.DataFrame):

        raise TypeError(
            "model_results.pkl should hold a DataFrame, found "
            + type(results).__name__
        )

    table = []

    for name, row in results.iterrows():

        entry = {
            "name": str(name)
        }

        for key, column in RESULT_COLUMNS.items():

            value = row.get(column)

            entry[key] = (
                round(
                    float(value) * 100,
                    2
                )
                if value is not None
                and not pd.isna(value)
                else None
            )

        table.append(entry)

    return table


def best_result(table):
    """
    The winning model, chosen the same way train_model.py chose it.

    Selection is on cross-validated F1 where available, falling back to the holdout
    F1 for artifacts written before CV was added -- otherwise this page would crown a
    different model than the one actually saved in best_model.pkl.
    """

    if not table:
        return None

    def key(entry):

        return (

            entry["cv_f1"]
            if entry.get("cv_f1") is not None
            else entry.get("f1") or 0

        )

    return max(
        table,
        key=key
    )
