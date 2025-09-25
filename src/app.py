# --- sys.path-Bootstrap, damit "from src.*" immer funktioniert ---
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../hka-helper
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import chainlit as cl

# Wichtig: Der Router exportiert jetzt nur noch `supervise`, der Guard steckt im Agent.
from router import supervise

WELCOME_TEXT = "👋 Willkommen beim HKA Hochschul-Helper. Stelle deine Frage!"


@cl.on_chat_start
async def start():
    await cl.Message(content=WELCOME_TEXT).send()


@cl.on_message
async def main(message: cl.Message):
    user_msg = (message.content or "").strip()
    if not user_msg:
        await cl.Message(content="Bitte formuliere deine Frage zur HKA.").send()
        return

    try:
        with cl.Step(name="Agent (Plan & Tools)"):
            result = supervise(user_msg)

        # Debug-Ausgabe
        print(f"APP DEBUG - Received result: {result}")

        # Falls der Agent nichts zurückgibt
        if result is None:
            await cl.Message(content="Leider keine Antwort erzeugt. Bitte versuche es erneut.").send()
            return

        # Kalender-spezifische Behandlung
        if isinstance(result, dict):
            answer = result.get("answer", "")

            # Prüfe auf Kalender-Operationen
            if "📅" in answer or "Kalenderergebnis" in answer or result.get("calendar_events"):
                await cl.Message(content=f"📅 Calendar tool finished task").send()
                return

        # Standard Textantwort (bestehender Code)
        if isinstance(result, dict):
            answer = result.get("answer")
            citations = result.get("citations") or []
            confidence = result.get("confidence")
        else:
            answer = str(result)
            citations, confidence = [], None

        if not answer:
            answer = "Ich habe leider keine passende Antwort gefunden."

        msg_lines = [answer]
        if confidence is not None:
            try:
                conf_pct = float(confidence) * 100.0 if confidence <= 1.0 else float(confidence)
                msg_lines.append(f"\n(Vertrauen: {conf_pct:.0f}%)")
            except Exception:
                pass

        if citations:
            msg_lines.append("\nQuellen:\n" + "\n".join(citations))

        await cl.Message(content="\n".join(msg_lines)).send()

    except Exception as e:
        await cl.Message(content=f"❌ Unerwarteter Fehler: {type(e).__name__}: {e}").send()
