import logging
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("fraud_agent")

load_dotenv(".env.local")

# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fraud_cases.json")
)


def _read_db() -> List[Dict[str, Any]]:
    """Read DB safely and always return a list."""
    if not os.path.exists(DB_PATH):
        return []

    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Error reading database: {e}")
        return []


def _write_db(data: List[Dict[str, Any]]):
    """Atomic write to prevent corruption during concurrent updates."""
    tmp_path = DB_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, DB_PATH)
    except Exception as e:
        logger.error(f"Failed to write DB: {e}")


# ---------------------------------------------------------------------------
# Tools exposed to LLM
# ---------------------------------------------------------------------------

@function_tool
async def load_case(ctx: RunContext, userName: str) -> Optional[dict]:
    """Retrieve a fraud case for a given userName."""
    userName = (userName or "").strip().lower()
    if not userName:
        return None

    for case in _read_db():
        if case.get("userName", "").lower() == userName:
            return case

    return None


@function_tool
async def verify_answer(ctx: RunContext, userName: str, answer: str) -> bool:
    """Verify the stored non-sensitive security question answer."""
    case = await load_case(ctx, userName)
    if not case:
        return False

    expected = (case.get("securityAnswer") or "").strip().lower()
    answer = (answer or "").strip().lower()

    return answer == expected


@function_tool
async def update_case(ctx: RunContext, userName: str, status: str, outcomeNote: str) -> str:
    """Update status + append history entry."""
    userName = (userName or "").strip().lower()
    cases = _read_db()

    for c in cases:
        if c.get("userName", "").lower() == userName:

            c["status"] = status
            c.setdefault("history", [])
            c["history"].append({
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "note": outcomeNote,
            })

            _write_db(cases)
            return f"saved:{DB_PATH}"

    return "error:not_found"


# ---------------------------------------------------------------------------
# Agent Behaviour
# ---------------------------------------------------------------------------

class FraudAgent(Agent):
    def __init__(self) -> None:
        instructions = (
            "You are a calm, clear, highly professional fraud specialist from the fictional bank 'Acme Trust'. "
            "Your job is to assist customers with verifying a suspicious transaction on their account.\n\n"

            "Conversation Flow:\n"
            "1. Greet warmly and introduce yourself as Acme Trust's Fraud Prevention Department.\n"
            "2. Explain that you're calling regarding a potentially suspicious transaction.\n"
            "3. Ask for the customer’s FIRST NAME only.\n"
            "4. Use `load_case(userName)` to retrieve their case.\n"
            "   - If no case exists, politely say you cannot find their profile and end the call.\n\n"

            "Verification Step:\n"
            "5. Once case is loaded, ask ONLY the stored non-sensitive verification question.\n"
            "6. After customer answers, call `verify_answer(userName, answer)`.\n"
            "   - If verification fails, update with `verification_failed` and end the call.\n\n"

            "Transaction Review:\n"
            "7. If verification succeeds, read the suspicious transaction details clearly:\n"
            "   - Merchant Name\n"
            "   - Amount\n"
            "   - Masked card ending\n"
            "   - Timestamp\n"
            "   - Category\n"
            "   - Source website/app\n"
            "8. Ask: “Did you make this transaction?”\n\n"

            "Outcome Handling:\n"
            "- If customer says YES: call\n"
            "  `update_case(userName, \"confirmed_safe\", \"Customer confirmed legitimate transaction.\")`\n"
            "  and explain the case is closed.\n"
            "- If customer says NO: call\n"
            "  `update_case(userName, \"confirmed_fraud\", \"Customer reports fraudulent transaction; card blocked & dispute opened.\")`\n"
            "  and reassure them their card is blocked and a dispute has been opened.\n\n"

            "Behavior Rules:\n"
            "- NEVER request sensitive info: no full card numbers, PINs, OTPs, SSNs, passwords, or addresses.\n"
            "- Maintain a calm, supportive, trustworthy tone.\n"
            "- Keep responses concise, friendly, and easy to understand.\n"
            "- If the user seems confused, gently restate the question.\n"
            "- Do not break character.\n"
        )

        super().__init__(
            instructions=instructions,
            tools=[load_case, verify_answer, update_case]
        )


# ---------------------------------------------------------------------------
# Worker & Session Setup
# ---------------------------------------------------------------------------

def prewarm(proc: JobProcess):
    """Load heavy models once at worker startup."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Collect LLM/STT/TTS usage for analytics
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage Summary: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )
