"""
All SQLite access for the app.

Two things worth knowing:

  * `assessment_date` stores a *display* string ("05 Sep 2026, 03:14 PM"). You cannot
    sort or group by that in SQL -- "05 Sep" and "05 Oct" compare as text. A
    `created_at` column holding ISO-8601 was added alongside it, backfilled by parsing
    the old strings. History sorting and the over-time chart both need it.

  * Every query is parameterized. The one place a value reaches SQL by string
    substitution is the ORDER BY column, which cannot be a bound parameter, so it is
    resolved through a whitelist and never taken from the request directly.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from config import (
    DATABASE_PATH,
    DATE_DISPLAY_FORMAT,
    HISTORY_PAGE_SIZE
)


logger = logging.getLogger(__name__)


# Columns holding JSON documents; parsed on the way out, dumped on the way in.
JSON_COLUMNS = [

    "student_data",

    "skills",

    "recommendations",

    "shap_data"

]


# ORDER BY targets. Request values are keys of this map, never SQL fragments.
SORT_COLUMNS = {

    "date":
        "created_at",

    "name":
        "student_name",

    "probability":
        "probability",

    "status":
        "status"

}

DEFAULT_SORT = "date"

DEFAULT_ORDER = "desc"


# =========================================================
# CONNECTION
# =========================================================

@contextmanager
def connect():
    """Yield a Row-returning connection, committing on success."""

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    try:

        yield connection

        connection.commit()

    finally:

        connection.close()


# =========================================================
# SCHEMA
# =========================================================

def init_database():

    with connect() as connection:

        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                student_name TEXT NOT NULL,

                probability REAL NOT NULL,

                status TEXT NOT NULL,

                prediction TEXT NOT NULL,

                assessment_date TEXT NOT NULL,

                created_at TEXT,

                student_data TEXT,

                skills TEXT,

                recommendations TEXT,

                shap_data TEXT

            )
        """)

        cursor.execute(
            "PRAGMA table_info(predictions)"
        )

        existing = [
            row[1]
            for row in cursor.fetchall()
        ]

        # Same additive-migration pattern the table already used, extended to
        # created_at. SQLite has no ADD COLUMN IF NOT EXISTS.
        for column in (
            JSON_COLUMNS
            + ["created_at"]
        ):

            if column not in existing:

                cursor.execute(
                    "ALTER TABLE predictions "
                    f"ADD COLUMN {column} TEXT"
                )

                logger.info(
                    "Added missing column %s to predictions",
                    column
                )

    _backfill_created_at()


def _backfill_created_at():
    """
    Give pre-existing rows a sortable timestamp.

    Rows whose display string cannot be parsed keep a NULL created_at; queries fall
    back to id ordering, which is the same chronological order for an
    autoincrementing key.
    """

    with connect() as connection:

        cursor = connection.cursor()

        cursor.execute("""
            SELECT id, assessment_date
            FROM predictions
            WHERE created_at IS NULL
        """)

        pending = cursor.fetchall()

        if not pending:
            return

        repaired = 0

        for row in pending:

            try:

                parsed = datetime.strptime(
                    row["assessment_date"],
                    DATE_DISPLAY_FORMAT
                )

            except (ValueError, TypeError):

                continue

            cursor.execute(
                """
                UPDATE predictions
                SET created_at = ?
                WHERE id = ?
                """,
                (
                    parsed.isoformat(
                        timespec="seconds"
                    ),
                    row["id"]
                )
            )

            repaired += 1

        logger.info(
            "Backfilled created_at for %d of %d row(s)",
            repaired,
            len(pending)
        )


# =========================================================
# SERIALISATION
# =========================================================

def _hydrate(row):
    """Turn a Row into a dict, decoding the JSON columns."""

    if row is None:
        return None

    record = dict(row)

    for column in JSON_COLUMNS:

        raw = record.get(column)

        if not raw:

            record[column] = (
                {}
                if column == "student_data"
                else []
            )

            continue

        try:

            record[column] = json.loads(raw)

        except (ValueError, TypeError):

            logger.warning(
                "Row %s has unreadable JSON in %s",
                record.get("id"),
                column
            )

            record[column] = (
                {}
                if column == "student_data"
                else []
            )

    return record


# =========================================================
# WRITES
# =========================================================

def insert_prediction(
    name,
    probability,
    status,
    prediction,
    student_data,
    skills,
    recommendations,
    shap_data,
    now=None
):
    """Store one assessment and return its new id."""

    now = now or datetime.now()

    with connect() as connection:

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO predictions (
                student_name,
                probability,
                status,
                prediction,
                assessment_date,
                created_at,
                student_data,
                skills,
                recommendations,
                shap_data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                probability,
                status,
                str(prediction),
                now.strftime(DATE_DISPLAY_FORMAT),
                now.isoformat(timespec="seconds"),
                json.dumps(student_data),
                json.dumps(skills),
                json.dumps(recommendations),
                json.dumps(shap_data)
            )
        )

        return cursor.lastrowid


def delete_prediction(prediction_id):
    """Delete one row. Returns True if a row was actually removed."""

    with connect() as connection:

        cursor = connection.cursor()

        cursor.execute(
            "DELETE FROM predictions WHERE id = ?",
            (
                prediction_id,
            )
        )

        return cursor.rowcount > 0


def clear_predictions():
    """Delete every row. Returns how many were removed."""

    with connect() as connection:

        cursor = connection.cursor()

        cursor.execute(
            "DELETE FROM predictions"
        )

        return cursor.rowcount


# =========================================================
# READS
# =========================================================

def get_prediction(prediction_id):

    with connect() as connection:

        cursor = connection.cursor()

        cursor.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (
                prediction_id,
            )
        )

        return _hydrate(
            cursor.fetchone()
        )


def _build_filter(query, status):
    """Return (where_clause, params) for the history filters."""

    clauses = []

    params = []

    if query:

        clauses.append(
            "student_name LIKE ? ESCAPE '\\'"
        )

        escaped = (
            query
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

        params.append(
            f"%{escaped}%"
        )

    if status:

        clauses.append(
            "status = ?"
        )

        params.append(status)

    if not clauses:

        return "", params

    return (
        "WHERE " + " AND ".join(clauses),
        params
    )


def count_predictions(
    query=None,
    status=None
):

    where, params = _build_filter(
        query,
        status
    )

    with connect() as connection:

        cursor = connection.cursor()

        cursor.execute(
            "SELECT COUNT(*) AS total FROM predictions "
            + where,
            params
        )

        return cursor.fetchone()["total"]


def list_predictions(
    query=None,
    status=None,
    sort=DEFAULT_SORT,
    order=DEFAULT_ORDER,
    page=1,
    page_size=HISTORY_PAGE_SIZE
):
    """
    One page of history rows.

    `sort` and `order` come from the query string, so both are resolved through
    whitelists; an unknown value falls back to the default rather than being
    interpolated into SQL.
    """

    column = SORT_COLUMNS.get(
        sort,
        SORT_COLUMNS[DEFAULT_SORT]
    )

    direction = (
        "ASC"
        if str(order).lower() == "asc"
        else "DESC"
    )

    where, params = _build_filter(
        query,
        status
    )

    page = max(1, int(page))

    # id is the tiebreaker so that equal keys (two assessments in the same minute,
    # or two students with the same name) still come back in a stable order.
    sql = (
        "SELECT * FROM predictions "
        + where
        + f" ORDER BY {column} {direction}, id {direction}"
        + " LIMIT ? OFFSET ?"
    )

    with connect() as connection:

        cursor = connection.cursor()

        cursor.execute(
            sql,
            params
            + [
                page_size,
                (page - 1) * page_size
            ]
        )

        return [
            _hydrate(row)
            for row in cursor.fetchall()
        ]


def recent_predictions(limit):

    with connect() as connection:

        cursor = connection.cursor()

        cursor.execute(
            "SELECT * FROM predictions "
            "ORDER BY id DESC LIMIT ?",
            (
                limit,
            )
        )

        return [
            _hydrate(row)
            for row in cursor.fetchall()
        ]


# =========================================================
# AGGREGATES
# =========================================================

def summary():
    """Headline numbers for the analytics page."""

    with connect() as connection:

        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                COUNT(*)         AS total,
                AVG(probability) AS average
            FROM predictions
        """)

        totals = cursor.fetchone()

        cursor.execute("""
            SELECT status, COUNT(*) AS count
            FROM predictions
            GROUP BY status
        """)

        by_status = {
            row["status"]: row["count"]
            for row in cursor.fetchall()
        }

        # prediction is a TEXT column and older rows may hold "Placed" rather than
        # "1", so both spellings are counted.
        cursor.execute("""
            SELECT COUNT(*) AS count
            FROM predictions
            WHERE prediction = '1'
               OR LOWER(prediction) = 'placed'
        """)

        placed = cursor.fetchone()["count"]

    total = totals["total"] or 0

    return {

        "total":
            total,

        "average_probability":
            round(
                float(totals["average"]),
                2
            )
            if totals["average"] is not None
            else 0,

        "by_status":
            by_status,

        "predicted_placed":
            placed,

        "predicted_not_placed":
            total - placed

    }


def probability_histogram(bucket_size=10):
    """
    Counts per probability band.

    The top band is inclusive of 100, so a perfect score lands in 90-100 rather than
    opening a 100-109 bucket of its own.
    """

    bucket_count = 100 // bucket_size

    buckets = [0] * bucket_count

    with connect() as connection:

        cursor = connection.cursor()

        cursor.execute(
            "SELECT probability FROM predictions"
        )

        for row in cursor.fetchall():

            index = int(
                float(row["probability"]) // bucket_size
            )

            index = min(
                max(index, 0),
                bucket_count - 1
            )

            buckets[index] += 1

    return [

        {
            "label":
                f"{index * bucket_size}-"
                + str(
                    (index + 1) * bucket_size
                    - (0 if index == bucket_count - 1 else 1)
                ),

            "count":
                count
        }

        for index, count in enumerate(buckets)

    ]


def assessments_per_day(limit=30):
    """Assessment counts by calendar day, oldest first."""

    with connect() as connection:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                DATE(created_at) AS day,
                COUNT(*)         AS count
            FROM predictions
            WHERE created_at IS NOT NULL
            GROUP BY day
            ORDER BY day DESC
            LIMIT ?
            """,
            (
                limit,
            )
        )

        rows = [
            {
                "day": row["day"],
                "count": row["count"]
            }
            for row in cursor.fetchall()
        ]

    return list(
        reversed(rows)
    )
