"""
Application configuration.

Paths are absolute, derived from this file's location. Previously they were
relative ("models/best_model.pkl"), which only worked when the process happened to
be started from the project root -- fine under `flask run`, fragile under anything
else.
"""

import os


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "features.pkl"
)

MODEL_RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "model_results.pkl"
)

FEATURE_OPTIONS_PATH = os.path.join(
    MODEL_DIR,
    "feature_options.pkl"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "placement_history.db"
)


# =========================================================
# READINESS BANDS
# =========================================================
#
# Single source of truth. These thresholds used to live in the predict view and
# were then re-derived twice inside result.html, so changing a band meant editing
# three places and the template could disagree with the stored status.

READINESS_HIGH_MIN = 75.0

READINESS_MODERATE_MIN = 50.0

STATUS_HIGH = "High Readiness"

STATUS_MODERATE = "Moderate Readiness"

STATUS_LOW = "Needs Improvement"

# Order matters: the history filter dropdown and the analytics chart both use it.
STATUS_CHOICES = [

    STATUS_HIGH,

    STATUS_MODERATE,

    STATUS_LOW

]


# =========================================================
# DISPLAY
# =========================================================

HISTORY_PAGE_SIZE = 10

SHAP_TOP_FEATURES = 8

RECENT_ASSESSMENTS_LIMIT = 5

DATE_DISPLAY_FORMAT = "%d %b %Y, %I:%M %p"


# =========================================================
# SECRET KEY
# =========================================================
#
# Needed for the session, which is where the CSRF token lives. A fixed
# development fallback keeps `python app.py` working out of the box; on Render,
# set SECRET_KEY in the environment so sessions survive restarts and are not
# guessable.

SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "dev-only-insecure-key-set-SECRET_KEY-in-production"
)

SECRET_KEY_IS_DEFAULT = (
    "SECRET_KEY" not in os.environ
)
