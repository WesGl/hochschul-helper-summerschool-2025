# src/router.py – minimal angepasst
from src.agent import GuardResult, run_agent  # Agent wie oben


# guard_check/supervise entfallen – der Graph macht das Routing.
def supervise(user_msg: str):
    return run_agent(user_msg)
