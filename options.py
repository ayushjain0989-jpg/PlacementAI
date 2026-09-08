"""
The input domain of the assessment form.

Every dropdown on the form is built from here, and "here" is derived from the
trained model rather than typed by hand. That is the whole point of the module: the
form previously offered `State` and `Other` for the boards and `Electronics` /
`Mechanical` / `Civil` for the stream, none of which appear in the training data.
OneHotEncoder(handle_unknown="ignore") does not complain about an unseen category --
it emits an all-zero row -- so those fields looked meaningful and influenced nothing.

Preferred source is models/feature_options.pkl, written by train_model.py. If that is
missing we fall back to interrogating the fitted encoder inside best_model.pkl, which
is the same information by a slower route. We deliberately do not fall back to a
hardcoded list, because a hardcoded list is exactly how the drift happened.
"""

import logging

import joblib

import model
from config import FEATURE_OPTIONS_PATH


logger = logging.getLogger(__name__)


# =========================================================
# FORM FIELD -> MODEL COLUMN
# =========================================================
#
# The form uses snake_case names; the dataset uses its own column headings. This is
# the only place the two vocabularies meet.

CHOICE_FIELDS = {

    "gender":
        "Gender",

    "tenth_board":
        "10th board",

    "twelfth_board":
        "12th board",

    "stream":
        "Stream",

    "internship":
        "Internships(Y/N)",

    "training":
        "Training(Y/N)",

    "backlog":
        "Backlog in 5th sem",

    "project":
        "Innovative Project(Y/N)",

    "technical_course":
        "Technical Course(Y/N)"

}


NUMBER_FIELDS = {

    "tenth_marks":
        "10th marks",

    "twelfth_marks":
        "12th marks",

    "cgpa":
        "Cgpa",

    "communication":
        "Communication level"

}


# Human labels, used in validation messages so an error names the field the way the
# form does.
FIELD_LABELS = {

    "name":
        "Student Name",

    "gender":
        "Gender",

    "tenth_board":
        "10th Board",

    "tenth_marks":
        "10th Marks",

    "twelfth_board":
        "12th Board",

    "twelfth_marks":
        "12th Marks",

    "stream":
        "Engineering Stream",

    "cgpa":
        "CGPA",

    "internship":
        "Internship Experience",

    "training":
        "Technical Training",

    "backlog":
        "5th Semester Backlog",

    "project":
        "Innovative Project",

    "communication":
        "Communication Level",

    "technical_course":
        "Technical Course"

}


# The four yes/no questions read better in a fixed order than in frequency order.
YES_NO_ORDER = [
    "Yes",
    "No"
]


# Descriptions for the communication scale. The dataset uses 1-5; the form used to
# stop at 4, which made 70 of 401 training rows unexpressible.
COMMUNICATION_LABELS = {

    1: "Poor",

    2: "Below Average",

    3: "Average",

    4: "Good",

    5: "Excellent"

}


NAME_MAX_LENGTH = 100


# =========================================================
# LOAD
# =========================================================

_options = None

load_error = None


def _from_pickle():

    payload = joblib.load(
        FEATURE_OPTIONS_PATH
    )

    if (
        not isinstance(payload, dict)
        or "categorical" not in payload
    ):

        raise ValueError(
            "feature_options.pkl has an unexpected shape."
        )

    return payload


def _from_fitted_model():
    """
    Reconstruct the domain from the fitted pipeline.

    Numeric bounds are not recoverable this way -- an imputer does not record the
    valid range -- so they come from data_prep, which is where they are declared.
    """

    categories = model.known_categories()

    if not categories:

        raise RuntimeError(
            "The fitted model exposed no categories."
        )

    from data_prep import NUMERIC_RANGES

    numeric = {}

    for column in NUMBER_FIELDS.values():

        low, high = NUMERIC_RANGES.get(
            column,
            (None, None)
        )

        numeric[column] = {

            "min": low,

            "max": high,

            "observed_min": None,

            "observed_max": None,

            "integer": column == "Communication level"

        }

    return {

        "categorical": categories,

        "numeric": numeric

    }


def _load():

    global _options, load_error

    if _options is not None:
        return _options

    try:

        _options = _from_pickle()

        logger.info(
            "Loaded form options from %s",
            FEATURE_OPTIONS_PATH
        )

        return _options

    except Exception:

        logger.warning(
            "Could not read %s; falling back to the fitted model",
            FEATURE_OPTIONS_PATH,
            exc_info=True
        )

    try:

        _options = _from_fitted_model()

        logger.info(
            "Derived form options from the fitted model."
        )

        return _options

    except Exception as error:

        load_error = str(error)

        logger.error(
            "No source of form options is available",
            exc_info=True
        )

        _options = {

            "categorical": {},

            "numeric": {}

        }

        return _options


def is_available():

    loaded = _load()

    return bool(
        loaded["categorical"]
    )


# =========================================================
# LOOKUPS
# =========================================================

def allowed_values(field):
    """The exact set of values the model will recognise for a choice field."""

    column = CHOICE_FIELDS.get(field)

    if column is None:
        return []

    values = _load()["categorical"].get(
        column,
        []
    )

    if set(values) == set(YES_NO_ORDER):

        return list(YES_NO_ORDER)

    return list(values)


def numeric_bounds(field):
    """Return (low, high, is_integer) for a numeric field."""

    column = NUMBER_FIELDS.get(field)

    spec = _load()["numeric"].get(
        column,
        {}
    )

    return (

        spec.get("min"),

        spec.get("max"),

        bool(
            spec.get("integer")
        )

    )


def communication_levels():
    """[(value, "3 - Average"), ...] across whatever range the data supports."""

    low, high, _ = numeric_bounds(
        "communication"
    )

    if low is None or high is None:

        low, high = 1, 5

    return [

        (
            level,
            f"{level} - "
            + COMMUNICATION_LABELS.get(
                level,
                "Level " + str(level)
            )
        )

        for level in range(
            int(low),
            int(high) + 1
        )

    ]


def for_template():
    """Everything assessment.html needs to render its inputs."""

    numeric = {}

    for field in NUMBER_FIELDS:

        low, high, is_integer = numeric_bounds(
            field
        )

        numeric[field] = {

            "min": low,

            "max": high,

            "step":
                1
                if is_integer
                else 0.01

        }

    return {

        "choices": {

            field: allowed_values(field)

            for field in CHOICE_FIELDS

        },

        "numeric": numeric,

        "communication_levels":
            communication_levels()

    }
