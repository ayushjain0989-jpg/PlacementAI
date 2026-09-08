"""
Parsing and validation for the assessment form.

The validation here is stricter than what it replaces in one way that matters:
choice fields are checked against the categories the model was actually trained on.
Before, any string at all was accepted for board and stream, and an unrecognised one
became an all-zero one-hot row -- a silent no-op rather than an error.

Returns errors as a list and echoes the submitted values back, so a rejected form can
be re-rendered with the student's answers still in it instead of blanked.
"""

import options


def _clean(form, field):

    return form.get(
        field,
        ""
    ).strip()


def collect_submitted(form):
    """The raw strings, kept so a failed submission can be re-displayed."""

    fields = (
        ["name"]
        + list(options.CHOICE_FIELDS)
        + list(options.NUMBER_FIELDS)
    )

    return {

        field: _clean(form, field)

        for field in fields

    }


def _format_bound(value):
    """Render a bound as 5 rather than 5.0, which reads oddly in an error message."""

    if value is None:
        return "?"

    if float(value).is_integer():

        return str(
            int(value)
        )

    return str(value)


# =========================================================
# VALIDATION STEPS
# =========================================================

def _check_required(submitted, errors):

    missing = [

        options.FIELD_LABELS.get(field, field)

        for field, value in submitted.items()

        if value == ""

    ]

    if missing:

        errors.append(

            "Please fill in every field. Missing: "

            + ", ".join(missing)

            + "."

        )

    return not missing


def _check_name(submitted, errors):

    name = submitted["name"]

    if len(name) > options.NAME_MAX_LENGTH:

        errors.append(

            "Student name must be "

            f"{options.NAME_MAX_LENGTH} characters or fewer."

        )

        return False

    return True


def _check_choices(submitted, errors):
    """Reject any choice the model has never seen."""

    valid = True

    for field in options.CHOICE_FIELDS:

        allowed = options.allowed_values(
            field
        )

        if not allowed:

            # No known domain at all means the model or its options failed to
            # load; the caller reports that separately rather than blaming the
            # student's input.
            continue

        if submitted[field] not in allowed:

            errors.append(

                "Please choose a valid option for "

                f"{options.FIELD_LABELS.get(field, field)}."

            )

            valid = False

    return valid


def _parse_numbers(submitted, errors):
    """Convert the numeric fields and range-check them. Returns a dict or None."""

    values = {}

    valid = True

    for field in options.NUMBER_FIELDS:

        low, high, is_integer = options.numeric_bounds(
            field
        )

        label = options.FIELD_LABELS.get(
            field,
            field
        )

        try:

            number = (

                int(submitted[field])

                if is_integer

                else float(submitted[field])

            )

        except ValueError:

            errors.append(
                f"{label} must be a number."
            )

            valid = False

            continue

        if (
            low is not None
            and high is not None
            and not low <= number <= high
        ):

            errors.append(

                f"{label} must be between "

                f"{_format_bound(low)} and "

                f"{_format_bound(high)}."

            )

            valid = False

            continue

        values[field] = number

    return values if valid else None


# =========================================================
# ENTRY POINT
# =========================================================

def parse_assessment(form):
    """
    Validate a submitted assessment.

    Returns (name, student_data, submitted, errors). student_data is None whenever
    errors is non-empty; it is keyed by dataset column name and ready to become a
    one-row DataFrame.
    """

    submitted = collect_submitted(
        form
    )

    errors = []

    if not _check_required(submitted, errors):

        return None, None, submitted, errors

    _check_name(submitted, errors)

    _check_choices(submitted, errors)

    numbers = _parse_numbers(
        submitted,
        errors
    )

    if errors:

        return None, None, submitted, errors

    student_data = {}

    for field, column in options.CHOICE_FIELDS.items():

        student_data[column] = submitted[field]

    for field, column in options.NUMBER_FIELDS.items():

        student_data[column] = numbers[field]

    return (
        submitted["name"],
        student_data,
        submitted,
        errors
    )
