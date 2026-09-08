"""
Dataset cleaning for the placement dataset.

The raw CSV carries the usual hand-entry damage: casing variants ("yes" next to
"Yes"), typos ("Yess", "Engineeing"), several spellings of the same board, and
values outside the possible range for their column (a CGPA of 90 on a 10-point
scale).

Left alone, none of this raises an error. OneHotEncoder simply treats "yes" and
"Yes" as two unrelated categories, and the tree models happily split on a CGPA of
90. The result is a model that looks fine and quietly wastes a chunk of a
401-row dataset.

Everything here is deliberately explicit rather than clever: the label maps are
written out so that a reader can see exactly which spellings were folded
together, and every repair is counted and reported so nothing is silently
rewritten.
"""

import re


# =========================================================
# COLUMN GROUPS
# =========================================================

YES_NO_COLUMNS = [

    "Internships(Y/N)",

    "Training(Y/N)",

    "Backlog in 5th sem",

    "Innovative Project(Y/N)",

    "Technical Course(Y/N)"

]


TEXT_COLUMNS = [

    "Gender",

    "10th board",

    "12th board",

    "Stream"

]


# =========================================================
# VALID NUMERIC RANGES
# =========================================================
#
# Values outside these bounds are impossible rather than merely unusual, so they
# are treated as missing and left to the pipeline's median imputer. We do not
# guess at the intended value: the one CGPA of 90.0 in the dataset could be 9.0
# or a percentage, and inventing either would be putting words in the data's
# mouth.

NUMERIC_RANGES = {

    "10th marks":
        (0.0, 100.0),

    "12th marks":
        (0.0, 100.0),

    "Cgpa":
        (0.0, 10.0),

    "Communication level":
        (1.0, 5.0)

}


# =========================================================
# LABEL CANONICALISATION
# =========================================================
#
# Only spellings that genuinely denote the same thing are folded together. Boards
# that merely resemble each other (WBBSE vs WBCHSE) are left distinct.

CANONICAL_LABELS = {

    "10th board": {
        # Already consistent in the current dataset; kept for future imports.
    },

    "12th board": {

        # MSBTE is a diploma-awarding board, written three ways.
        "MSBTE":
            "Diploma",

        "Diploma board - MSBTE":
            "Diploma",

        # ISE / ISC / CISCE all refer to the same council's examination.
        "ISC":
            "ISE",

        "CISCE":
            "ISE",

        # Capitalisation only, so the dropdown reads consistently.
        "Other state Board":
            "Other State Board"

    },

    "Stream": {

        # "Engineeing" / duplicated "and" are typos of the ECE stream.
        "Electronics and Communication and Engineeing":
            "Electronics and Communication Engineering",

        # Singular/plural variants of the same degree.
        "Electronic Engineering":
            "Electronics Engineering"

    }

}


# =========================================================
# RARE CATEGORY BUCKETING
# =========================================================
#
# A category seen once or twice in 401 rows cannot be learned from -- it becomes
# a one-hot column that memorises a single student. Folding these into a shared
# bucket both removes that noise and turns the "Other" dropdown option into a
# category the model has actually been trained on.
#
# For 12th board the leftovers (BSEB, WBBSE) really are other state boards, so
# they fold into that existing label instead of a generic one.

RARE_CATEGORY_MIN_COUNT = 3

RARE_CATEGORY_LABEL = {

    "10th board":
        "Other",

    "12th board":
        "Other State Board",

    "Stream":
        "Other",

    "Gender":
        "Other"

}


# =========================================================
# HELPERS
# =========================================================

def collapse_whitespace(value):
    """Trim and squeeze runs of internal whitespace down to single spaces."""

    if not isinstance(value, str):
        return value

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def normalise_yes_no(value):
    """
    Map a free-text yes/no answer onto exactly "Yes" or "No".

    Matching on the first letter is what rescues "Yess" and "yes" without
    needing an entry for every typo. Anything that starts with neither letter is
    returned as None so it is imputed rather than guessed at.
    """

    if not isinstance(value, str):
        return None

    text = collapse_whitespace(value).casefold()

    if text.startswith("y"):
        return "Yes"

    if text.startswith("n"):
        return "No"

    return None


# =========================================================
# CLEANING STEPS
# =========================================================

def clean_text_columns(df, report):
    """Trim whitespace and apply the canonical label maps."""

    for column in TEXT_COLUMNS:

        if column not in df.columns:
            continue

        original = df[column].copy()

        df[column] = df[column].apply(
            collapse_whitespace
        )

        mapping = CANONICAL_LABELS.get(
            column,
            {}
        )

        if mapping:

            df[column] = df[column].replace(
                mapping
            )

        changed = int(
            (
                original.fillna("")
                != df[column].fillna("")
            ).sum()
        )

        if changed:

            report.append(
                f"{column}: relabelled {changed} value(s)"
            )

    return df


def clean_yes_no_columns(df, report):
    """Force the Y/N columns into exactly two categories."""

    for column in YES_NO_COLUMNS:

        if column not in df.columns:
            continue

        original = df[column].copy()

        df[column] = df[column].apply(
            normalise_yes_no
        )

        changed = int(
            (
                original.fillna("")
                != df[column].fillna("")
            ).sum()
        )

        if changed:

            report.append(
                f"{column}: normalised {changed} value(s) to Yes/No"
            )

    return df


def clean_numeric_columns(df, report):
    """Blank out values that fall outside their column's possible range."""

    for column, (low, high) in NUMERIC_RANGES.items():

        if column not in df.columns:
            continue

        numeric = df[column].astype(float)

        out_of_range = (
            numeric.notna()
            & (
                (numeric < low)
                | (numeric > high)
            )
        )

        count = int(
            out_of_range.sum()
        )

        if count:

            offending = sorted(
                numeric[out_of_range].unique().tolist()
            )

            report.append(
                f"{column}: dropped {count} out-of-range "
                f"value(s) {offending} "
                f"(valid range {low}-{high}); "
                f"these will be median-imputed"
            )

            numeric = numeric.mask(
                out_of_range
            )

        df[column] = numeric

    return df


def bucket_rare_categories(df, report):
    """Fold categories below the count threshold into a shared bucket."""

    for column, fallback in RARE_CATEGORY_LABEL.items():

        if column not in df.columns:
            continue

        counts = df[column].value_counts()

        rare = [
            label
            for label, count in counts.items()
            if count < RARE_CATEGORY_MIN_COUNT
            and label != fallback
        ]

        if not rare:
            continue

        affected = int(
            df[column].isin(rare).sum()
        )

        df[column] = df[column].replace(
            dict.fromkeys(
                rare,
                fallback
            )
        )

        report.append(
            f"{column}: folded {len(rare)} rare "
            f"category/categories {rare} "
            f"into '{fallback}' ({affected} row(s))"
        )

    return df


# =========================================================
# ENTRY POINT
# =========================================================

def clean_dataset(df):
    """
    Clean a raw placement dataframe.

    Returns the cleaned copy plus a list of human-readable notes describing
    every change made, so the training run can print exactly what it repaired.
    """

    report = []

    df = df.copy()

    df = clean_text_columns(df, report)

    df = clean_yes_no_columns(df, report)

    df = clean_numeric_columns(df, report)

    df = bucket_rare_categories(df, report)

    if not report:

        report.append(
            "No corrections were necessary."
        )

    return df, report


# =========================================================
# FEATURE DOMAINS
# =========================================================

def build_feature_options(
    df,
    categorical_features,
    numeric_features
):
    """
    Record the exact input domain the model was trained on.

    The web form is rendered from this, which is what stops the dropdowns from
    drifting away from the training data again -- previously the form offered
    board and stream values the model had never seen, and OneHotEncoder's
    handle_unknown="ignore" turned every one of them into an all-zero row.
    """

    options = {

        "categorical": {},

        "numeric": {}

    }

    for column in categorical_features:

        values = (
            df[column]
            .dropna()
            .unique()
            .tolist()
        )

        # Most frequent first, so the common choices sit at the top of the
        # dropdown rather than in whatever order pandas happened to see them.
        counts = df[column].value_counts()

        options["categorical"][column] = sorted(
            values,
            key=lambda value: (
                -counts.get(value, 0),
                str(value)
            )
        )

    for column in numeric_features:

        low, high = NUMERIC_RANGES.get(
            column,
            (None, None)
        )

        series = df[column].dropna()

        options["numeric"][column] = {

            "min":
                low,

            "max":
                high,

            "observed_min":
                float(series.min())
                if not series.empty
                else None,

            "observed_max":
                float(series.max())
                if not series.empty
                else None,

            "integer":
                column == "Communication level"

        }

    return options
