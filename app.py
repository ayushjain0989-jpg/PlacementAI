"""
PlacementAI web application.

Routing only. Anything that is not "read the request, call a module, pick a template"
lives elsewhere:

    config.py     paths, readiness bands, page size, secret key
    model.py      the trained pipeline (loaded once) and the metrics artifact
    options.py    the form's valid inputs, derived from the trained model
    forms.py      request parsing and validation
    rules.py      skill gaps and recommendations
    db.py         schema and every query
    security.py   CSRF token
    explain.py    SHAP

`app` stays a module-level name so `gunicorn app:app` keeps working.
"""

import csv
import io
import logging
import re
import os

import pandas as pd
from flask import (
    Flask,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for
)

import config
import db
import explain
import forms
import model
import options
import rules
import security


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# APPLICATION
# =========================================================

app = Flask(__name__)

app.secret_key = config.SECRET_KEY

security.register(app)

db.init_database()


if config.SECRET_KEY_IS_DEFAULT:

    logger.warning(
        "SECRET_KEY is not set; using the built-in development key. "
        "Set SECRET_KEY in the environment before deploying."
    )


MODEL_UNAVAILABLE_MESSAGE = (
    "The prediction model could not be loaded, so new assessments "
    "cannot be scored right now. Please contact the administrator."
)


# =========================================================
# STATIC PAGES
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        active_page="home"
    )


@app.route("/about")
def about():

    return render_template(
        "about.html",
        active_page="about"
    )


# =========================================================
# ASSESSMENT FORM
# =========================================================

def render_assessment(
    submitted=None,
    errors=None
):
    """
    Render the form, optionally with validation errors and the student's answers.

    Re-passing `submitted` is what stops a rejected submission from wiping fourteen
    fields the student just filled in.
    """

    errors = list(errors or [])

    if not model.is_available():

        errors.insert(
            0,
            MODEL_UNAVAILABLE_MESSAGE
        )

    elif not options.is_available():

        errors.insert(
            0,
            "The list of valid assessment options could not be loaded. "
            "Please run train_model.py."
        )

    return render_template(

        "assessment.html",

        active_page="assessment",

        options=options.for_template(),

        submitted=submitted or {},

        errors=errors

    )


@app.route("/assessment")
def assessment():

    return render_assessment()


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():
    """
    Score an assessment, store it, and redirect to its report.

    POST-redirect-GET rather than rendering the result directly: the old version
    answered POST /predict with the report itself, so refreshing the page re-submitted
    the form (or produced a 405 once the browser turned it into a GET), and the report
    had no address of its own.
    """

    if not model.is_available():

        return render_assessment(
            submitted=forms.collect_submitted(request.form)
        )

    name, student_data, submitted, errors = forms.parse_assessment(
        request.form
    )

    if errors:

        return render_assessment(
            submitted=submitted,
            errors=errors
        )

    try:

        input_frame = pd.DataFrame(
            [student_data]
        )

        prediction, probability = model.predict(
            input_frame
        )

        status = rules.readiness_status(
            probability
        )

        skills, recommendations = rules.evaluate(
            student_data
        )

        shap_data = explain.build_shap_data(
            input_frame,
            student_data
        )

        prediction_id = db.insert_prediction(
            name=name,
            probability=probability,
            status=status,
            prediction=prediction,
            student_data=student_data,
            skills=skills,
            recommendations=recommendations,
            shap_data=shap_data
        )

    except Exception:

        logger.error(
            "Failed to score an assessment",
            exc_info=True
        )

        return render_assessment(

            submitted=submitted,

            errors=[
                "Unable to process the assessment. "
                "Please check your inputs and try again."
            ]

        )

    return redirect(
        url_for(
            "report",
            prediction_id=prediction_id
        )
    )


@app.route("/report/<int:prediction_id>")
def report(prediction_id):
    """The result page for one stored assessment, now with a shareable URL."""

    record = db.get_prediction(
        prediction_id
    )

    if record is None:

        abort(404)

    return render_template(

        "result.html",

        active_page="assessment",

        record=record,

        name=record["student_name"],

        probability=record["probability"],

        status=record["status"],

        status_description=rules.status_description(
            record["status"]
        ),

        prediction=record["prediction"],

        data=record["student_data"],

        skills=record["skills"],

        recommendations=record["recommendations"],

        shap_data=record["shap_data"],

        communication_max=options.numeric_bounds(
            "communication"
        )[1]

    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

@app.route("/model-performance")
def model_performance():

    try:

        table = model.load_results()

    except Exception:

        logger.error(
            "Could not load model performance data",
            exc_info=True
        )

        return render_template(

            "model_performance.html",

            active_page="model-performance",

            models=[],

            best_model=None,

            errors=[
                "Unable to load model performance data. "
                "Run train_model.py to generate it."
            ]

        )

    return render_template(

        "model_performance.html",

        active_page="model-performance",

        models=table,

        best_model=model.best_result(table)

    )


# =========================================================
# HISTORY
# =========================================================

def _history_filters():
    """Read and normalise the history query string."""

    sort = request.args.get(
        "sort",
        db.DEFAULT_SORT
    )

    if sort not in db.SORT_COLUMNS:

        sort = db.DEFAULT_SORT

    order = request.args.get(
        "order",
        db.DEFAULT_ORDER
    ).lower()

    if order not in ("asc", "desc"):

        order = db.DEFAULT_ORDER

    status = request.args.get(
        "status",
        ""
    ).strip()

    if status not in config.STATUS_CHOICES:

        status = ""

    try:

        page = max(
            1,
            int(
                request.args.get(
                    "page",
                    1
                )
            )
        )

    except ValueError:

        page = 1

    return {

        "q":
            request.args.get(
                "q",
                ""
            ).strip(),

        "status":
            status,

        "sort":
            sort,

        "order":
            order,

        "page":
            page

    }


@app.route("/history")
def history():

    filters = _history_filters()

    try:

        total = db.count_predictions(
            query=filters["q"],
            status=filters["status"]
        )

        page_size = config.HISTORY_PAGE_SIZE

        total_pages = max(
            1,
            -(-total // page_size)
        )

        # A stale ?page=9 after a filter narrows the results should show the last
        # page, not an empty one.
        page = min(
            filters["page"],
            total_pages
        )

        filters["page"] = page

        predictions = db.list_predictions(
            query=filters["q"],
            status=filters["status"],
            sort=filters["sort"],
            order=filters["order"],
            page=page,
            page_size=page_size
        )

    except Exception:

        logger.error(
            "Could not load prediction history",
            exc_info=True
        )

        return render_template(

            "history.html",

            active_page="history",

            predictions=[],

            filters=filters,

            total=0,

            total_pages=1,

            status_choices=config.STATUS_CHOICES,

            errors=[
                "Unable to load prediction history."
            ]

        )

    return render_template(

        "history.html",

        active_page="history",

        predictions=predictions,

        filters=filters,

        total=total,

        total_pages=total_pages,

        status_choices=config.STATUS_CHOICES

    )


@app.route("/history/<int:prediction_id>")
def prediction_details(prediction_id):

    record = db.get_prediction(
        prediction_id
    )

    if record is None:

        abort(404)

    return render_template(

        "prediction_details.html",

        active_page="history",

        record=record,

        student_data=record["student_data"],

        skills=record["skills"],

        recommendations=record["recommendations"],

        shap_data=record["shap_data"],

        communication_max=options.numeric_bounds(
            "communication"
        )[1]

    )


@app.route(
    "/history/<int:prediction_id>/delete",
    methods=["POST"]
)
@security.csrf_protect
def delete_prediction(prediction_id):

    try:

        removed = db.delete_prediction(
            prediction_id
        )

    except Exception:

        logger.error(
            "Could not delete assessment %s",
            prediction_id,
            exc_info=True
        )

        flash(
            "Unable to delete that assessment.",
            "error"
        )

        return redirect(
            url_for("history")
        )

    flash(
        "Assessment deleted."
        if removed
        else "That assessment no longer exists.",
        "success" if removed else "error"
    )

    return redirect(
        url_for("history")
    )


@app.route(
    "/history/clear",
    methods=["POST"]
)
@security.csrf_protect
def clear_history():

    #
    # The template asks the user to type DELETE and marks the field `required`, but
    # `required` is enforced by the browser and nothing else: a form posted with
    # curl, or from a page with novalidate, reached this route and emptied the table
    # without the word ever being typed. The check has to live here.
    #
    if request.form.get("confirm", "").strip().upper() != "DELETE":

        flash(
            "Type DELETE in the confirmation box to clear the history.",
            "error"
        )

        return redirect(
            url_for("history")
        )

    try:

        removed = db.clear_predictions()

    except Exception:

        logger.error(
            "Could not clear prediction history",
            exc_info=True
        )

        flash(
            "Unable to clear the history.",
            "error"
        )

        return redirect(
            url_for("history")
        )

    flash(
        f"Cleared {removed} assessment(s)."
        if removed
        else "There was nothing to clear.",
        "success" if removed else "error"
    )

    return redirect(
        url_for("history")
    )


# =========================================================
# EXPORT
# =========================================================

def _download_filename(record, extension):
    """
    A safe filename built from the student's name.

    Names come from user input and end up in a Content-Disposition header, so
    everything outside a small alphabet is dropped rather than escaped.
    """

    slug = re.sub(
        r"[^A-Za-z0-9]+",
        "-",
        record["student_name"] or ""
    ).strip("-").lower()

    return (
        f"assessment-{record['id']}"
        + (f"-{slug}" if slug else "")
        + f".{extension}"
    )


@app.route("/history/<int:prediction_id>/export.csv")
def export_prediction(prediction_id):

    record = db.get_prediction(
        prediction_id
    )

    if record is None:

        abort(404)

    buffer = io.StringIO()

    writer = csv.writer(buffer)

    writer.writerow(
        [
            "Field",
            "Value"
        ]
    )

    writer.writerows([

        [
            "Assessment ID",
            record["id"]
        ],

        [
            "Student Name",
            record["student_name"]
        ],

        [
            "Assessment Date",
            record["assessment_date"]
        ],

        [
            "Placement Probability (%)",
            record["probability"]
        ],

        [
            "Readiness Status",
            record["status"]
        ],

        [
            "Model Prediction",
            "Placed"
            if str(record["prediction"]).lower() in ("1", "placed")
            else "Not Placed"
        ]

    ])

    writer.writerow([])

    writer.writerow(
        [
            "Assessment Input",
            "Value"
        ]
    )

    for key, value in record["student_data"].items():

        writer.writerow(
            [
                key,
                value
            ]
        )

    writer.writerow([])

    writer.writerow(
        [
            "Skill Gap"
        ]
    )

    for skill in record["skills"]:

        writer.writerow(
            [
                skill
            ]
        )

    writer.writerow([])

    writer.writerow(
        [
            "Recommendation"
        ]
    )

    for recommendation in record["recommendations"]:

        writer.writerow(
            [
                recommendation
            ]
        )

    writer.writerow([])

    writer.writerow(
        [
            "Factor",
            "SHAP Value",
            "Direction"
        ]
    )

    for item in record["shap_data"]:

        writer.writerow([

            item.get("feature"),

            item.get("value"),

            item.get("direction")

        ])

    return Response(

        buffer.getvalue(),

        mimetype="text/csv",

        headers={

            "Content-Disposition":
                "attachment; filename="
                + _download_filename(record, "csv")

        }

    )


@app.route("/history/<int:prediction_id>/print")
def print_prediction(prediction_id):
    """A print-optimised view; the browser turns it into a PDF."""

    record = db.get_prediction(
        prediction_id
    )

    if record is None:

        abort(404)

    return render_template(

        "report_print.html",

        record=record,

        student_data=record["student_data"],

        skills=record["skills"],

        recommendations=record["recommendations"],

        shap_data=record["shap_data"],

        status_description=rules.status_description(
            record["status"]
        ),

        communication_max=options.numeric_bounds(
            "communication"
        )[1]

    )


# =========================================================
# ANALYTICS
# =========================================================

def _percentage(count, total):

    if not total:
        return 0

    return round(
        count / total * 100,
        2
    )


@app.route("/analytics")
def analytics():

    try:

        stats = db.summary()

        recent = db.recent_predictions(
            config.RECENT_ASSESSMENTS_LIMIT
        )

    except Exception:

        logger.error(
            "Could not load analytics data",
            exc_info=True
        )

        stats = {

            "total": 0,

            "average_probability": 0,

            "by_status": {},

            "predicted_placed": 0,

            "predicted_not_placed": 0

        }

        recent = []

        error = "Unable to load analytics data."

    else:

        error = None

    total = stats["total"]

    counts = {

        status: stats["by_status"].get(status, 0)

        for status in config.STATUS_CHOICES

    }

    return render_template(

        "analytics.html",

        active_page="analytics",

        total_assessments=total,

        average_probability=stats["average_probability"],

        predicted_placed=stats["predicted_placed"],

        predicted_not_placed=stats["predicted_not_placed"],

        placed_percentage=_percentage(
            stats["predicted_placed"],
            total
        ),

        high_readiness=counts[config.STATUS_HIGH],

        moderate_readiness=counts[config.STATUS_MODERATE],

        needs_improvement=counts[config.STATUS_LOW],

        high_percentage=_percentage(
            counts[config.STATUS_HIGH],
            total
        ),

        moderate_percentage=_percentage(
            counts[config.STATUS_MODERATE],
            total
        ),

        improvement_percentage=_percentage(
            counts[config.STATUS_LOW],
            total
        ),

        recent_predictions=recent,

        errors=[error] if error else []

    )


@app.route("/analytics/data.json")
def analytics_data():
    """Chart data. Kept separate so the page renders without waiting on it."""

    try:

        stats = db.summary()

        histogram = db.probability_histogram()

        per_day = db.assessments_per_day()

    except Exception:

        logger.error(
            "Could not build analytics chart data",
            exc_info=True
        )

        return jsonify(
            {
                "error":
                    "Unable to load chart data."
            }
        ), 500

    return jsonify({

        "total":
            stats["total"],

        "readiness": {

            "labels":
                config.STATUS_CHOICES,

            "counts": [
                stats["by_status"].get(status, 0)
                for status in config.STATUS_CHOICES
            ]

        },

        "histogram": {

            "labels": [
                bucket["label"]
                for bucket in histogram
            ],

            "counts": [
                bucket["count"]
                for bucket in histogram
            ]

        },

        "over_time": {

            "labels": [
                entry["day"]
                for entry in per_day
            ],

            "counts": [
                entry["count"]
                for entry in per_day
            ]

        }

    })


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.errorhandler(400)
def bad_request(error):

    return render_template(

        "error.html",

        active_page=None,

        code=400,

        title="Bad request",

        message=getattr(
            error,
            "description",
            "That request could not be processed."
        )

    ), 400


@app.errorhandler(404)
def not_found(error):

    return render_template(

        "error.html",

        active_page=None,

        code=404,

        title="Page not found",

        message=(
            "We could not find that page. "
            "It may have been deleted, or the link may be wrong."
        )

    ), 404


@app.errorhandler(500)
def server_error(error):

    logger.error(
        "Unhandled server error",
        exc_info=True
    )

    return render_template(

        "error.html",

        active_page=None,

        code=500,

        title="Something went wrong",

        message=(
            "An unexpected error occurred. "
            "Please try again in a moment."
        )

    ), 500


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
