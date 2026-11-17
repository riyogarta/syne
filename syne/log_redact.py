"""Logging filter that scrubs secrets before a record reaches any handler.

Why: httpx logs every request at INFO as "HTTP Request: GET <full URL>".
python-telegram-bot calls https://api.telegram.org/bot<TOKEN>/... through
httpx, and providers that take their key as a query parameter (?key=AIza...)
put it in the URL too, so syne.log (and the systemd journal, via the console
handler) collected the bot token and API keys on every call.

main.py silences httpx/httpcore to WARNING (first layer). This filter is the
second layer: it also covers warnings/errors and tracebacks (exc_info) that
embed a URL, whichever logger emits them.
"""

import logging

from .communication.errors import redact_secrets


class RedactSecretsFilter(logging.Filter):
    """Rewrite record message and traceback text through redact_secrets().

    Attach to HANDLERS (not loggers): handler filters see records propagated
    from every logger, logger filters only see records created on that logger.
    Never drops a record; on any internal error the record passes unchanged.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            clean = redact_secrets(msg)
            if clean != msg:
                record.msg = clean
                record.args = None
            if record.exc_info and not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = redact_secrets(record.exc_text)
            if record.stack_info:
                record.stack_info = redact_secrets(record.stack_info)
        except Exception:
            pass
        return True


def install(handlers) -> None:
    """Add one RedactSecretsFilter to each handler (idempotent)."""
    for h in handlers:
        if not any(isinstance(f, RedactSecretsFilter) for f in h.filters):
            h.addFilter(RedactSecretsFilter())
