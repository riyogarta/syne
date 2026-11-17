"""1.22.88: secrets never reach syne.log / the journal.

httpx logged every request at INFO with the full URL, so the Telegram bot
token (/bot<TOKEN>/...) and ?key= API keys landed in syne.log thousands of
times a day. main.py now silences httpx/httpcore and installs
RedactSecretsFilter on its handlers.
"""

import io
import logging

from syne.communication.errors import redact_secrets
from syne.log_redact import RedactSecretsFilter, install

TG_SECRET = "AAH" + "x" * 32
TG_TOKEN = "1234567890:" + TG_SECRET
GKEY = "AIza" + "B" * 35
TG_URL = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates"


def test_redact_telegram_token_in_url_path():
    out = redact_secrets(f'HTTP Request: POST {TG_URL} "HTTP/1.1 200 OK"')
    assert TG_SECRET not in out
    assert "bot1234567890:***/getUpdates" in out


def test_redact_bare_telegram_token():
    assert TG_SECRET not in redact_secrets(f"token={TG_TOKEN} end")
    assert TG_SECRET not in redact_secrets(f"using {TG_TOKEN}")


def test_redact_leaves_ordinary_text_alone():
    for s in ("12:30 meeting", "ratio 16:9", "id 1234567890:short", "HTTP/1.1 200 OK"):
        assert redact_secrets(s) == s


def _logger_with_buffer(name):
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    install([h])
    lg = logging.getLogger(name)
    lg.handlers = [h]
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    return lg, h, buf


def test_filter_scrubs_message_with_args():
    lg, _, buf = _logger_with_buffer("t.redact.args")
    lg.info("HTTP Request: %s %s", "GET", f"https://maps.googleapis.com/x?key={GKEY}&q=a")
    lg.info("HTTP Request: %s %s", "POST", TG_URL)
    out = buf.getvalue()
    assert GKEY not in out and TG_SECRET not in out
    assert "?key=***&q=a" in out and "bot1234567890:***" in out


def test_filter_scrubs_traceback():
    lg, _, buf = _logger_with_buffer("t.redact.exc")
    try:
        raise RuntimeError(f"Client error for url '{TG_URL}'")
    except RuntimeError:
        lg.error("send failed", exc_info=True)
    out = buf.getvalue()
    assert "Traceback" in out and "RuntimeError" in out
    assert TG_SECRET not in out


def test_filter_never_drops_records():
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "plain %s", ("ok",), None)
    assert RedactSecretsFilter().filter(rec) is True
    assert rec.getMessage() == "plain ok"


def test_install_is_idempotent():
    h = logging.StreamHandler(io.StringIO())
    install([h])
    install([h])
    assert sum(isinstance(f, RedactSecretsFilter) for f in h.filters) == 1


def test_main_silences_httpx_and_installs_filter():
    # Read main.py instead of importing it: importing would attach a file
    # handler to the real ~/.log-syne/syne.log from inside the test run.
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "syne" / "main.py").read_text()
    bc = src.index("logging.basicConfig(")
    assert src.index('for _noisy in ("httpx", "httpcore")') > bc
    assert "logging.getLogger(_noisy).setLevel(logging.WARNING)" in src
    assert src.index("_install_log_redact(logging.getLogger().handlers)") > bc
