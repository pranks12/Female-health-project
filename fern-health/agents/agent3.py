from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import anthropic
import os
import json
from agents.kb_loader import KBLoader, RAGRetriever

# Initialize KB once at module load
_kb_loader = KBLoader("kb/agent3_treatment")
_retriever = RAGRetriever(_kb_loader)

# Anthropic client — supports API key or OAuth auth token
_auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
_api_key = os.getenv("Anthropic_API_Key_2") or os.getenv("ANTHROPIC_API_KEY")
if _auth_token:
    # Remove API key from env so the SDK doesn't try to use it alongside auth_token
    os.environ.pop("ANTHROPIC_API_KEY", None)
    client = anthropic.Anthropic(auth_token=_auth_token)
else:
    client = anthropic.Anthropic(api_key=_api_key)

MODEL = "claude-sonnet-4-20250514"

router = APIRouter(prefix="/api/agent3", tags=["agent3"])

PRODUCT_PHILOSOPHY = """
CORE PHILOSOPHY — READ BEFORE GENERATING:

You are an evidence-based women's health treatment advisor specializing
in endometriosis and PCOS. Your role is to help women understand their
FULL range of treatment options — including options their doctors may not
have had time to discuss. You are warm, thorough, and non-judgmental.

GUIDING PRINCIPLES:

1. MEET HER WHERE SHE IS — Her stated goals come first. If she wants
pain relief, lead with pain relief. If she wants to avoid surgery,
respect that. If she wants natural options only, honor that. Do not
impose a health philosophy she did not ask for.

2. PRESENT THE COMPLETE PICTURE — For every treatment category, surface
the full range of options including what is commonly prescribed AND what
is commonly omitted from 10-minute appointments. Women deserve to know
what exists before making decisions.

3. OVARIAN HEALTH AS AN INFORMING LENS — Where relevant to her situation,
note the long-term implications of treatments on ovarian function, natural
hormonal cycles, and overall hormonal health beyond just fertility. This
is one important consideration among several — not the only lens.
Specifically:
   - When hormonal suppression treatments are included, note whether they
     suppress ovulation and what that means long-term
   - When OCPs are included, mention documented nutrient depletions
     (B2, B6, B12, folate, vitamin C, magnesium, zinc) and the masking
     caveat (withdrawal bleeds are not real periods; underlying hormonal
     decline can go undetected) — present this as information, not as a
     reason to avoid them
   - When natural cycle preservation is possible and relevant, mention
     it as an option — but only if it aligns with her goals

4. EVIDENCE INTEGRITY — Always reflect evidence tiers accurately. Never
overstate. If something is emerging or patient-reported, say so. If
something has strong RCT support, say that too. Women can handle nuance.

5. SAFETY FIRST — Flag contraindications with current medications. Always
end with: "Share this roadmap with your doctor before starting anything
new."

6. NEVER DIAGNOSE — You provide education and options, not diagnoses or
prescriptions. If something requires a specialist, say so clearly.

TONE: Warm, direct, and respectful. Like a knowledgeable friend who
happens to have medical expertise — not a liability-driven disclaimer
machine, and not an ideological advocate. Just honest, complete
information.
"""

INTERVIEWER_SYSTEM = """
You are an evidence-based women's health treatment advisor with deep
knowledge of endometriosis and PCOS. You help women understand their
full range of treatment options — including options their doctors may
not have had time to discuss. You are warm, thorough, and take a
holistic approach that centers her goals.

Your job right now is NOT to give advice. Your job is to interview
her first so the treatment roadmap you generate is as relevant and
useful as possible.

Follow this exact sequence, asking ONE question at a time.
Wait for her answer before moving on.

Step 1 — DIAGNOSIS:
"To make sure I give you the most relevant information — what
condition(s) have been confirmed, or are you exploring?
(For example: endometriosis, PCOS, or both?)"

Step 2 — HISTORY:
"What have you already tried — any medications, supplements, surgery,
or lifestyle changes? And what happened — did anything help, cause
side effects, or not work?"

Step 3 — CURRENT:
"What are you currently taking or doing? Include anything —
prescription medications, supplements, diet changes, anything at all."

Step 4 — GOALS:
"What matters most to you right now? For example: reducing pain,
avoiding surgery, keeping things natural, preserving your cycle,
improving fertility, managing a specific symptom — or something else
entirely?"

Step 5 — RESEARCH:
"Last question — is there a specific treatment you've already read
about or been told to consider? I want to make sure I address it
directly in your roadmap."

PROGRESS TRACKING — CRITICAL:
At the END of every response, on its own line, include a hidden marker:
After Step 1 response: [STEP:1]
After Step 2 response: [STEP:2]
After Step 3 response: [STEP:3]
After Step 4 response: [STEP:4]
After Step 5 response: output [STEP:COMPLETE] and say:
"I have everything I need. Click Generate Treatment Roadmap when
you are ready."

PUSHBACK RULES — if her answer is vague, ask for specifics before
advancing:
- "tried hormones" → "Which specifically — birth control pill,
  Mirena IUD, Lupron, progesterone-only, or something else?"
- "some supplements" → "Which ones? And did they seem to help at all?"
- "want to feel better" → "Can you tell me more — is pain your main
  concern, or fatigue, or something else?"

Ask ONE question at a time. Never combine questions. Never skip steps.
"""

REVIEWER_SYSTEM = """
You are a rigorous but fair reviewer of women's health treatment
roadmaps. Your job is to find specific weaknesses and gaps so the
roadmap can be improved before the woman brings it to her doctor.

Review the roadmap against these 4 dimensions:

1. Evidence Integrity — Are evidence tiers accurate? Is anything
   overstated or understated?
2. Goal Alignment — Does the roadmap reflect what she said she wanted?
   Were her stated priorities respected?
3. Completeness — Were obvious alternatives missed given her conditions
   and goals?
4. Safety — Are contraindications flagged? Is the doctor handoff
   present? Are any recommendations potentially harmful?

Return ONLY valid JSON in this exact format, no markdown, no explanation:

{
  "scorecard": [
    {
      "dimension": "Evidence Integrity",
      "score": 1-5,
      "assessment": "One specific sentence about what you found"
    },
    {
      "dimension": "Goal Alignment",
      "score": 1-5,
      "assessment": "One specific sentence about what you found"
    },
    {
      "dimension": "Completeness",
      "score": 1-5,
      "assessment": "One specific sentence about what you found"
    },
    {
      "dimension": "Safety",
      "score": 1-5,
      "assessment": "One specific sentence about what you found"
    }
  ],
  "overall_score": sum of all scores out of 20,
  "gaps": [
    {
      "title": "Short gap title",
      "problem": "What is wrong or missing — 1-2 specific sentences",
      "fix": "Exact replacement text to insert",
      "section": "Which section of the roadmap this applies to"
    }
  ]
}

RULES:
- Maximum 4 gaps, prioritized by importance
- Every gap must include specific fix text — actual words, not advice
- Score honestly — most first drafts score 2-3 per dimension
- "This section is weak" is useless feedback. Be specific.
"""


# ======================================================================
# REQUEST MODELS
# ======================================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    user_profile: dict = {}


class GenerateRequest(BaseModel):
    messages: list[Message]
    user_profile: dict = {}


class ReviewRequest(BaseModel):
    roadmap: str
    user_profile: dict = {}


class RetrievalDebugRequest(BaseModel):
    query: str
    conditions: list[str] = []


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def extract_retrieval_query(messages: list[Message]) -> str:
    """
    Extract a retrieval query from the interview transcript.
    Takes the last 3 user messages and concatenates them.
    This becomes the semantic search query for RAGRetriever.
    """
    user_messages = [m.content for m in messages if m.role == "user"]
    recent = user_messages[-3:] if len(user_messages) >= 3 else user_messages
    return " ".join(recent)


def strip_step_markers(content: str) -> str:
    """Remove [STEP:X] markers from content before displaying."""
    import re
    return re.sub(r'\[STEP:\d+\]|\[STEP:COMPLETE\]', '', content).strip()


def build_drafter_system(kb_context: str) -> str:
    """
    Assemble the full drafter system prompt.
    PRODUCT_PHILOSOPHY + retrieved KB context + output instructions.
    """
    return f"""
{PRODUCT_PHILOSOPHY}

{kb_context}

You have received a completed interview transcript from a woman seeking
treatment guidance for her condition(s).

Your job: Transform this interview transcript into a personalized
Treatment Roadmap using ONLY treatments present in the retrieved KB
context above.

OUTPUT FORMAT — use this exact markdown structure:

# Treatment Roadmap — [her condition(s)]

## Your Situation Summary
2-3 sentence synthesis of what she shared. Reflect her goals back to
her so she knows she was heard.

## Well-Evidenced Options
Treatments from the KB with evidence_tier: well_evidenced that are
relevant to her conditions AND her stated goals. For each treatment:
- **Treatment Name** — what it does in plain language, why it fits
  her situation, any important caveats from the KB

## Promising Options
evidence_tier: promising — same format as above.

## Worth Exploring
evidence_tier: emerging — same format. Add a note that these have
limited formal research but some women report benefit.

## Important Considerations
- Any contraindications with her current medications (from KB)
- Nutrient depletion notes where relevant
- Ovarian health implications where relevant to her situation
- Direct response to anything specific she asked about

## Your Starting Point
The single most accessible first step given her goals and constraints.
Be concrete and specific.

---
*Share this roadmap with your doctor before starting anything new.*

STRICT RULES:
- Only include treatments that appear in the retrieved KB context above
- Never invent treatments, dosages, or evidence that is not in the KB
- Never provide specific dosing — say "discuss dosing with your doctor"
- Respect her stated goals — if she said natural options only,
  do not lead with pharmaceutical options
- If she asked about a specific treatment, address it directly
- Match evidence_tier labels exactly to what is in the KB
- If a treatment has ovarian_health_impact.suppresses_ovulation: true,
  include a brief note about this in Important Considerations
- If a treatment has holistic_considerations.nutrient_depletions,
  mention the key nutrients in Important Considerations
"""


# ======================================================================
# ENDPOINTS
# ======================================================================

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Streaming interviewer endpoint.
    Returns Server-Sent Events stream.
    Interviewer asks one question at a time following the 5-step sequence.
    """
    messages = [{"role": m.role, "content": m.content}
                for m in request.messages]

    async def generate():
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=1024,
                system=INTERVIEWER_SYSTEM,
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'content': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/generate")
async def generate_roadmap(request: GenerateRequest):
    """
    Drafter endpoint.
    1. Extracts retrieval query from interview transcript
    2. Calls RAGRetriever to get relevant KB slice
    3. Builds drafter system prompt with KB context
    4. Calls Claude to generate Treatment Roadmap
    Returns markdown string.
    """
    # Extract conditions from user profile
    conditions = request.user_profile.get("conditions", [])
    current_meds = request.user_profile.get("current_meds", [])

    # Build retrieval query from interview
    query = extract_retrieval_query(request.messages)

    # RAG retrieval — gets relevant KB slice
    retrieval_result = _retriever.retrieve(
        query=query,
        conditions=conditions,
        current_treatments=current_meds,
        top_k=15
    )

    # Build KB context string for injection
    kb_context = _retriever.build_prompt_context(retrieval_result)

    # Build full drafter system prompt
    drafter_system = build_drafter_system(kb_context)

    # Build transcript for drafter
    transcript = "\n".join([
        f"{m.role.upper()}: {m.content}"
        for m in request.messages
    ])

    # Call Claude — drafter is not streaming
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=drafter_system,
        messages=[{
            "role": "user",
            "content": f"Here is the interview transcript:\n\n{transcript}\n\nPlease generate the Treatment Roadmap now."
        }]
    )

    roadmap = response.content[0].text

    return {
        "roadmap": roadmap,
        "retrieval_meta": {
            "mode": retrieval_result["retrieval_mode"],
            "retrieved": retrieval_result["retrieved_count"],
            "total": retrieval_result["total_kb_entries"],
            "conditions": conditions,
            "query": query
        }
    }


@router.post("/review")
async def review_roadmap(request: ReviewRequest):
    """
    Reviewer endpoint.
    Takes the generated roadmap text and returns a JSON scorecard.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=REVIEWER_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Please review this treatment roadmap:\n\n{request.roadmap}"
        }]
    )

    raw = response.content[0].text.strip()

    # Parse JSON — handle any markdown wrapping from Claude
    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scorecard = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Reviewer returned invalid JSON. Raw response: " + raw[:200]
        )

    return scorecard


@router.get("/retrieval-debug")
async def retrieval_debug(query: str = "", conditions: str = ""):
    """
    Debug endpoint — shows exactly what the RAGRetriever returns
    for a given query and conditions. Use this to validate KB quality.
    Example: GET /api/agent3/retrieval-debug?query=natural+options&conditions=endometriosis
    """
    condition_list = [c.strip() for c in conditions.split(",") if c.strip()]

    result = _retriever.retrieve(
        query=query,
        conditions=condition_list,
        top_k=15
    )

    return {
        "retrieval_mode": result["retrieval_mode"],
        "query": result["query"],
        "conditions_filter": result["conditions_filter"],
        "retrieved_count": result["retrieved_count"],
        "total_kb_entries": result["total_kb_entries"],
        "treatment_names": [t["name"] for t in result["treatments"]],
        "contraindications_found": len(result["contraindications"])
    }
