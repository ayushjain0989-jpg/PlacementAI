"""
The advice layer: readiness bands, skill gaps and recommendations.

These rules used to live inline in the predict view, with the skill list and the
recommendation list built by two separate runs of nearly identical `if` statements.
Keeping one table means a threshold cannot be changed in one list and forgotten in the
other.

Note that low 10th/12th marks produce a skill-gap entry but no recommendation. That is
intentional and matches the original behaviour -- there is no useful advice to give a
final-year student about an exam they sat six years ago, whereas the gap is still worth
naming because it affects the prediction.
"""

from config import (
    READINESS_HIGH_MIN,
    READINESS_MODERATE_MIN,
    STATUS_HIGH,
    STATUS_LOW,
    STATUS_MODERATE
)


# =========================================================
# THRESHOLDS
# =========================================================

CGPA_MIN = 7.5

SCHOOL_MARKS_MIN = 70

COMMUNICATION_MIN = 3


# =========================================================
# PREDICATES
# =========================================================

def _below(limit):

    return lambda value: float(value) < limit


def _is_no(value):

    return str(value).strip().casefold() == "no"


def _is_yes(value):

    return str(value).strip().casefold() == "yes"


# =========================================================
# RULE TABLE
# =========================================================
#
# Order here is the order the student sees.

RULES = [

    {
        "column":
            "Cgpa",

        "applies":
            _below(CGPA_MIN),

        "skill":
            "CGPA",

        "recommendation":
            "Focus on improving your academic performance and CGPA."
    },

    {
        "column":
            "10th marks",

        "applies":
            _below(SCHOOL_MARKS_MIN),

        "skill":
            "10th Academic Performance",

        "recommendation":
            None
    },

    {
        "column":
            "12th marks",

        "applies":
            _below(SCHOOL_MARKS_MIN),

        "skill":
            "12th Academic Performance",

        "recommendation":
            None
    },

    {
        "column":
            "Communication level",

        "applies":
            _below(COMMUNICATION_MIN),

        "skill":
            "Communication Skills",

        "recommendation":
            "Practice communication, group discussions and HR interviews."
    },

    {
        "column":
            "Internships(Y/N)",

        "applies":
            _is_no,

        "skill":
            "Internship Experience",

        "recommendation":
            "Try to gain internship or industry experience."
    },

    {
        "column":
            "Innovative Project(Y/N)",

        "applies":
            _is_no,

        "skill":
            "Projects",

        "recommendation":
            "Build an industry-oriented project and add it to your resume."
    },

    {
        "column":
            "Technical Course(Y/N)",

        "applies":
            _is_no,

        "skill":
            "Technical Courses",

        "recommendation":
            "Complete relevant technical courses or certifications."
    },

    {
        "column":
            "Training(Y/N)",

        "applies":
            _is_no,

        "skill":
            "Training",

        "recommendation":
            "Participate in technical training programs."
    },

    {
        "column":
            "Backlog in 5th sem",

        "applies":
            _is_yes,

        "skill":
            "Academic Backlogs",

        "recommendation":
            "Clear academic backlogs and maintain consistent performance."
    }

]


NO_GAPS_RECOMMENDATION = (
    "Excellent profile! Continue developing your "
    "technical and communication skills."
)


# =========================================================
# READINESS BANDS
# =========================================================

def readiness_status(probability):

    if probability >= READINESS_HIGH_MIN:
        return STATUS_HIGH

    if probability >= READINESS_MODERATE_MIN:
        return STATUS_MODERATE

    return STATUS_LOW


STATUS_DESCRIPTIONS = {

    STATUS_HIGH:
        "Your profile shows strong placement readiness. Keep developing "
        "your technical and communication skills.",

    STATUS_MODERATE:
        "Your profile shows moderate readiness. Improving the identified "
        "skill gaps can strengthen your placement profile.",

    STATUS_LOW:
        "Your profile has several areas that should be improved before "
        "participating in placement drives."

}


def status_description(status):

    return STATUS_DESCRIPTIONS.get(
        status,
        ""
    )


# =========================================================
# EVALUATION
# =========================================================

def _triggered(student_data):
    """The subset of rules that fire for this student, in table order."""

    fired = []

    for rule in RULES:

        value = student_data.get(
            rule["column"]
        )

        if value is None:
            continue

        try:

            if rule["applies"](value):

                fired.append(rule)

        except (TypeError, ValueError):

            # A malformed stored value should not take the whole report down.
            continue

    return fired


def evaluate(student_data):
    """Return (skills, recommendations) for one student."""

    fired = _triggered(
        student_data
    )

    skills = [
        rule["skill"]
        for rule in fired
    ]

    recommendations = [

        rule["recommendation"]

        for rule in fired

        if rule["recommendation"]

    ]

    if not recommendations:

        recommendations.append(
            NO_GAPS_RECOMMENDATION
        )

    return skills, recommendations
