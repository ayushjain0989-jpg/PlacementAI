"""
CSRF protection for the destructive POST routes.

Deleting an assessment and clearing the history are state-changing requests, so
without a token any other site could trigger them from a logged-in browser with a
hidden auto-submitting form. That is the whole attack, and a per-session token in a
hidden field is the whole defence.

Written out here rather than pulled in via Flask-WTF: it is about thirty lines, and the
project ships an unpinned-turned-pinned requirements file that already caused one
version-drift problem. One fewer dependency is worth more than the convenience.
"""

import hmac
import logging
import secrets
from functools import wraps

from flask import abort, request, session


logger = logging.getLogger(__name__)


SESSION_KEY = "csrf_token"

FORM_FIELD = "csrf_token"


def issue_token():
    """Return this session's token, creating one on first use."""

    if SESSION_KEY not in session:

        session[SESSION_KEY] = secrets.token_urlsafe(32)

    return session[SESSION_KEY]


def token_is_valid():

    expected = session.get(
        SESSION_KEY
    )

    supplied = request.form.get(
        FORM_FIELD,
        ""
    )

    if not expected or not supplied:
        return False

    # Constant-time comparison: a plain == leaks how much of the token matched
    # through timing, which is enough to reconstruct it given enough attempts.
    return hmac.compare_digest(
        str(expected),
        str(supplied)
    )


def csrf_protect(view):
    """Reject a POST that arrives without this session's token."""

    @wraps(view)
    def wrapper(*args, **kwargs):

        if request.method == "POST" and not token_is_valid():

            logger.warning(
                "Rejected %s: missing or invalid CSRF token",
                request.path
            )

            abort(
                400,
                description=(
                    "This form has expired or was not submitted from this site. "
                    "Please go back and try again."
                )
            )

        return view(*args, **kwargs)

    return wrapper


def register(app):
    """Make `csrf_token` available to every template."""

    @app.context_processor
    def inject_token():

        return {

            "csrf_token":
                issue_token()

        }
