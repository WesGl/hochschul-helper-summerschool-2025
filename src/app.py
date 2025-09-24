# --- sys.path-Bootstrap, damit "from src.*" immer funktioniert ---
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../hka-helper
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os

import chainlit as cl

from router import guard_check, supervise


@cl.on_chat_start
async def start():
    await cl.Message(content="👋 Willkommen beim HKA Hochschul‑Helper. Stelle deine Frage!").send()


@cl.on_message
async def main(message: cl.Message):
    user_msg = message.content.strip()

    with cl.Step(name="Guard"):
        g = guard_check(user_msg)
        if not g.valid:
            await cl.Message(content=f"❌ Anfrage abgelehnt: {g.reason or 'Policy'}").send()
            return

    with cl.Step(name="Supervisor & Tools"):
        result = supervise(user_msg)

    # ICS optional delivery
    ics_tuple = result.get("ics") if isinstance(result, dict) else None
    if ics_tuple:
        filename, ics_bytes = ics_tuple[0], ics_tuple[1]
        await cl.Message(content=result.get("answer", "Kalender erstellt.")).send()
        await cl.File(content=ics_bytes, name=filename, mime="text/calendar").send()
        return

    # Answer text and citations
    content = result.get("answer") if isinstance(result, dict) else str(result)
    msg = content if content is not None else ""
    cites = result.get("citations") if isinstance(result, dict) else None
    if cites:
        msg += "\n\nQuellen:\n\n" + "\n".join(cites)

    await cl.Message(content=msg).send()
