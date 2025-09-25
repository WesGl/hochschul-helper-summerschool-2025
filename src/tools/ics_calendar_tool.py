# src/tools/calendar.py
from __future__ import annotations

import re
from datetime import datetime, timedelta


def make_ics(summary: str, start: datetime, end: datetime, location: str | None = None, description: str | None = None):
    dtfmt = "%Y%m%dT%H%M%S"
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//HKA Helper//DE",
        "BEGIN:VEVENT",
        f"UID:{int(datetime.now().timestamp())}@hka-helper",
        f"DTSTAMP:{datetime.utcnow().strftime(dtfmt)}Z",
        f"DTSTART:{start.strftime(dtfmt)}",
        f"DTEND:{end.strftime(dtfmt)}",
        f"SUMMARY:{summary}",
    ]
    if location:
        lines.append(f"LOCATION:{location}")
    if description:
        lines.append(f"DESCRIPTION:{description}")
    lines += ["END:VEVENT", "END:VCALENDAR", ""]
    return "\n".join(lines).encode("utf-8")


# simple NLP heuristic as an example
_DEF_DURATION = timedelta(hours=1)


def make_ics_from_text(text: str):
    summary = text.strip()[:80]
    start = datetime.now() + timedelta(days=1)
    end = start + _DEF_DURATION
    ics = make_ics(summary=summary, start=start, end=end)
    filename = "hka-event.ics"
    return ics, filename
