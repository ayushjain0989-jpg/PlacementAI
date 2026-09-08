"""
SHAP explanations for a single prediction.

Two changes from the previous version:

  * The model is imported from model.py instead of being joblib.load-ed again here.
    Loading it in both modules kept two copies of a 300-tree forest in memory.

  * Feature names are matched by column prefix rather than by substring. The old
    `if old_name in feature_name` test meant "10th marks" and "10th board" were
    distinguished only by dictionary ordering, which is a fragile thing to rely on
    when a new column could be added at any time.
"""

import logging

import numpy as np
import pandas as pd
import shap

import model
from config import SHAP_TOP_FEATURES


logger = logging.getLogger(__name__)


# =========================================================
# FEATURE NAMES
# =========================================================
#
# Dataset column -> the label a student should see. Ordered longest-first so that a
# column which is a prefix of another can never shadow it.

FRIENDLY_NAMES = [

    (
        "Communication level",
        "Communication Skills"
    ),

    (
        "Innovative Project(Y/N)",
        "Innovative Project"
    ),

    (
        "Technical Course(Y/N)",
        "Technical Course"
    ),

    (
        "Internships(Y/N)",
        "Internship Experience"
    ),

    (
        "Backlog in 5th sem",
        "Academic Backlog"
    ),

    (
        "Training(Y/N)",
        "Technical Training"
    ),

    (
        "10th marks",
        "10th Marks"
    ),

    (
        "12th marks",
        "12th Marks"
    ),

    (
        "10th board",
        "10th Board"
    ),

    (
        "12th board",
        "12th Board"
    ),

    (
        "Stream",
        "Engineering Stream"
    ),

    (
        "Gender",
        "Gender"
    ),

    (
        "Cgpa",
        "CGPA"
    )

]


def clean_feature_name(feature_name):
    """Strip the ColumnTransformer prefix and the one-hot suffix."""

    for prefix in (
        "num__",
        "cat__"
    ):

        if feature_name.startswith(prefix):

            feature_name = feature_name[len(prefix):]

    for column, friendly in FRIENDLY_NAMES:

        if (
            feature_name == column
            or feature_name.startswith(column + "_")
        ):

            return friendly

    return feature_name


# =========================================================
# STUDENT-FRIENDLY EXPLANATION
# =========================================================

POSITIVE_EXPLANATIONS = {

    "CGPA":
        "Your CGPA positively influenced your placement readiness prediction.",

    "10th Marks":
        "Your 10th standard academic performance positively influenced the prediction.",

    "12th Marks":
        "Your 12th standard academic performance positively influenced the prediction.",

    "Communication Skills":
        "Your communication level positively influenced your placement readiness.",

    "Internship Experience":
        "Your internship experience contributed positively to your placement readiness.",

    "Technical Training":
        "Your technical training contributed positively to the prediction.",

    "Innovative Project":
        "Having an innovative project strengthened your placement readiness profile.",

    "Technical Course":
        "Your technical course experience positively influenced the prediction.",

    "Academic Backlog":
        "Your academic record contributed positively to the prediction.",

    "10th Board":
        "Your 10th board information contributed positively to the model prediction.",

    "12th Board":
        "Your 12th board information contributed positively to the model prediction.",

    "Engineering Stream":
        "Your engineering stream contributed positively to the model prediction.",

    "Gender":
        "This feature contributed positively to the model prediction."

}


NEGATIVE_EXPLANATIONS = {

    "CGPA":
        "Your CGPA reduced the predicted placement readiness. "
        "Improving academic performance may help.",

    "10th Marks":
        "Your 10th standard marks reduced the predicted placement readiness.",

    "12th Marks":
        "Your 12th standard marks reduced the predicted placement readiness.",

    "Communication Skills":
        "Your communication level reduced the predicted placement readiness. "
        "More communication practice may help.",

    "Internship Experience":
        "The absence or level of internship experience reduced the "
        "predicted placement readiness.",

    "Technical Training":
        "Your technical training profile reduced the predicted placement readiness.",

    "Innovative Project":
        "Your project profile reduced the predicted placement readiness. "
        "Building a stronger project may help.",

    "Technical Course":
        "Your technical course profile reduced the predicted placement readiness. "
        "Consider completing relevant courses.",

    "Academic Backlog":
        "Your academic backlog information reduced the predicted placement readiness.",

    "10th Board":
        "This academic-board feature had a negative influence on the model prediction.",

    "12th Board":
        "This academic-board feature had a negative influence on the model prediction.",

    "Engineering Stream":
        "Your engineering stream had a negative influence on the model prediction.",

    "Gender":
        "This feature had a negative influence on the model prediction."

}


def get_feature_explanation(
    feature,
    value,
    student_data=None
):

    if value >= 0:

        return POSITIVE_EXPLANATIONS.get(
            feature,
            "This factor positively influenced "
            "your placement readiness prediction."
        )

    return NEGATIVE_EXPLANATIONS.get(
        feature,
        "This factor negatively influenced "
        "your placement readiness prediction."
    )


# =========================================================
# SHAP VALUES
# =========================================================

def _positive_class_values(shap_values):
    """
    Pull out the per-feature contributions for the positive class.

    TreeExplainer's return shape varies by model family and SHAP version -- a list
    per class, a 3-D array, or a plain 2-D array -- so all three are handled.
    """

    if isinstance(shap_values, list):

        return np.asarray(
            shap_values[1][0]
        )

    values = np.asarray(
        shap_values
    )

    if values.ndim == 3:

        return values[0, :, 1]

    return values[0]


def get_shap_explanation(input_data):
    """Per-feature SHAP contributions, one row per original column, most important first."""

    preprocessor = model.get_preprocessor()

    estimator = model.get_estimator()

    if preprocessor is None or estimator is None:

        raise RuntimeError(
            "The model pipeline is not available for explanation."
        )

    transformed = np.asarray(
        preprocessor.transform(
            input_data
        )
    )

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(
        estimator
    )

    values = _positive_class_values(
        explainer.shap_values(
            transformed
        )
    )

    explanation = pd.DataFrame({

        "feature":
            [
                clean_feature_name(name)
                for name in feature_names
            ],

        "shap_value":
            values

    })

    # One-hot columns are contributions to the *same* underlying question ("which
    # stream?"), so they are summed back into a single row per original column.
    explanation = (
        explanation
        .groupby(
            "feature",
            as_index=False
        )["shap_value"]
        .sum()
    )

    explanation["importance"] = (
        explanation["shap_value"].abs()
    )

    return explanation.sort_values(
        "importance",
        ascending=False
    )


# =========================================================
# DISPLAY PAYLOAD
# =========================================================

def build_shap_data(
    input_data,
    student_data,
    top_n=SHAP_TOP_FEATURES
):
    """
    The list the templates render: feature, signed value, bar width, direction, prose.

    Widths are relative to the largest absolute contribution, so the top bar is always
    100% and the rest are read against it. Returns [] rather than raising if SHAP
    fails -- an explanation is worth having but not worth losing the prediction over.
    """

    try:

        explanation = get_shap_explanation(
            input_data
        ).head(top_n)

    except Exception:

        logger.error(
            "Could not compute the SHAP explanation",
            exc_info=True
        )

        return []

    largest = (
        explanation["shap_value"]
        .abs()
        .max()
    )

    if not largest:

        largest = 1

    payload = []

    for _, row in explanation.iterrows():

        value = float(
            row["shap_value"]
        )

        payload.append({

            "feature":
                row["feature"],

            "value":
                round(value, 3),

            "width":
                round(
                    abs(value) / largest * 100,
                    1
                ),

            "direction":
                "positive"
                if value >= 0
                else "negative",

            "explanation":
                get_feature_explanation(
                    row["feature"],
                    value,
                    student_data
                )

        })

    return payload
