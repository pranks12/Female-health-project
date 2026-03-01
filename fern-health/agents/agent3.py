"""
Agent 3 — Fern Health Women's Wellness Agent
Handles chat with preview (KB-only) and live (Claude AI) modes.
"""
import os
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["Agent 3"])

TOPICS = [
    "menstrual cycle",
    "hormonal health",
    "fertility",
    "PCOS",
    "menopause",
    "nutrition",
    "mental wellness",
]

DISCLAIMER = "This information is for educational purposes only and does not constitute medical advice. Please consult a healthcare provider for personal guidance."

# ── Preview knowledge base ────────────────────────────────────────────────────
PREVIEW_KB: dict[str, str] = {
    "menstrual cycle": (
        "A typical menstrual cycle lasts 21–35 days, with menstruation itself lasting 2–7 days. "
        "The cycle has four phases: menstruation, the follicular phase, ovulation, and the luteal phase.\n\n"
        "• **Menstruation** (days 1–5): The uterine lining sheds. Estrogen and progesterone are at their lowest.\n"
        "• **Follicular phase** (days 1–13): FSH prompts follicle growth; estrogen rises.\n"
        "• **Ovulation** (around day 14): A surge in LH releases a mature egg.\n"
        "• **Luteal phase** (days 15–28): Progesterone peaks to prepare the uterus; if no fertilisation, levels drop and the cycle restarts."
    ),
    "hormonal health": (
        "Key reproductive hormones include estrogen, progesterone, FSH, LH, and testosterone.\n\n"
        "• Balanced hormones support regular cycles, stable mood, bone density, and metabolic health.\n"
        "• Common disruptors: chronic stress, poor sleep, ultra-processed foods, endocrine-disrupting chemicals (plastics, pesticides).\n"
        "• Lifestyle pillars: 7–9 hours of sleep, regular moderate exercise, stress management (meditation, breathwork), and a fibre-rich whole-food diet."
    ),
    "fertility": (
        "Fertility peaks in the mid-20s and gradually declines after 35. Key facts:\n\n"
        "• The fertile window is roughly 5 days before ovulation plus ovulation day itself.\n"
        "• Basal body temperature (BBT) tracking and ovulation predictor kits (OPKs) can identify ovulation.\n"
        "• Factors supporting fertility: healthy weight, folate intake (400–800 µg/day), limiting alcohol and caffeine, avoiding smoking.\n"
        "• Common causes of subfertility: ovulatory disorders (e.g., PCOS), tubal factors, endometriosis, and male-factor infertility."
    ),
    "PCOS": (
        "Polycystic ovary syndrome (PCOS) affects ~1 in 10 women of reproductive age.\n\n"
        "**Common symptoms:** irregular or absent periods, excess androgens (acne, hirsutism), polycystic ovaries on ultrasound, difficulty losing weight, and insulin resistance.\n\n"
        "**Evidence-based management:**\n"
        "• Low-GI diet and regular aerobic + resistance exercise to improve insulin sensitivity.\n"
        "• Even a 5–10% weight loss can restore ovulation in overweight individuals.\n"
        "• Medical options include metformin, combined oral contraceptives, and letrozole for ovulation induction (discuss with your doctor).\n"
        "• Regular screening for type 2 diabetes and cardiovascular risk."
    ),
    "menopause": (
        "Menopause is confirmed after 12 consecutive months without a period, typically between ages 45–55.\n\n"
        "**Perimenopause** can begin 4–10 years earlier with irregular cycles, hot flashes, night sweats, sleep disturbances, and mood changes.\n\n"
        "**Management options:**\n"
        "• Hormone replacement therapy (HRT/MHT) is the most effective treatment for vasomotor symptoms — discuss benefits vs. risks with your GP.\n"
        "• Non-hormonal: cognitive behavioural therapy (CBT) for hot flashes, SSRIs/SNRIs, gabapentin.\n"
        "• Lifestyle: weight-bearing exercise (bone health), calcium + vitamin D, limiting alcohol and caffeine."
    ),
    "nutrition": (
        "Nutrition tailored to the menstrual cycle can support energy, mood, and hormonal balance:\n\n"
        "• **Follicular phase:** Focus on fresh vegetables, lean protein, and fermented foods to boost estrogen metabolism.\n"
        "• **Ovulatory phase:** Antioxidant-rich foods (berries, leafy greens, seeds) support egg quality.\n"
        "• **Luteal phase:** Complex carbs and magnesium-rich foods (dark chocolate, pumpkin seeds) ease PMS.\n"
        "• **Menstruation:** Iron-rich foods (red meat, legumes, spinach) and vitamin C to enhance absorption.\n\n"
        "Key nutrients: omega-3s (anti-inflammatory), B vitamins (energy, mood), zinc (hormone synthesis), and vitamin D (immune and reproductive health)."
    ),
    "mental wellness": (
        "Hormonal fluctuations across the cycle significantly affect mood and mental health:\n\n"
        "• Low estrogen before menstruation can reduce serotonin, contributing to irritability, anxiety, or low mood (PMS/PMDD).\n"
        "• PMDD affects ~3–8% of women and may require medical support (SSRIs, hormonal treatment).\n\n"
        "**Evidence-based strategies:**\n"
        "• Regular aerobic exercise (≥150 min/week) has antidepressant effects.\n"
        "• Mindfulness-based stress reduction (MBSR) reduces anxiety and improves sleep.\n"
        "• Prioritise social connection and therapeutic support when needed.\n"
        "• Track your cycle to anticipate and prepare for mood shifts."
    ),
}


def _preview_response(message: str, topic: str) -> str:
    """Return a KB-based preview response matching the topic or message keywords."""
    msg_lower = message.lower()

    # Keyword matching across KB entries
    keyword_map = {
        "pcos": "PCOS",
        "polycystic": "PCOS",
        "period": "menstrual cycle",
        "menstrual": "menstrual cycle",
        "cycle": "menstrual cycle",
        "ovulat": "menstrual cycle",
        "luteal": "menstrual cycle",
        "follicular": "menstrual cycle",
        "fertilit": "fertility",
        "pregnant": "fertility",
        "conceiv": "fertility",
        "menopaus": "menopause",
        "perimenopaus": "menopause",
        "hot flash": "menopause",
        "hormone": "hormonal health",
        "estrogen": "hormonal health",
        "progesterone": "hormonal health",
        "insulin": "hormonal health",
        "nutrition": "nutrition",
        "diet": "nutrition",
        "food": "nutrition",
        "eat": "nutrition",
        "mental": "mental wellness",
        "mood": "mental wellness",
        "anxiety": "mental wellness",
        "stress": "mental wellness",
        "pms": "mental wellness",
        "pmdd": "mental wellness",
    }

    matched_topic = topic  # default to sidebar selection
    for kw, mapped in keyword_map.items():
        if kw in msg_lower:
            matched_topic = mapped
            break

    return PREVIEW_KB.get(matched_topic, PREVIEW_KB["menstrual cycle"])


# ── Models ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    topic: str = "menstrual cycle"


class ChatResponse(BaseModel):
    response: str
    mode: str
    topic: str
    disclaimer: str


# ── Routes ─────────────────────────────────────────────────────────────────────
@router.get("/topics")
def list_topics():
    return {"topics": TOPICS}


@router.post("/chat/preview", response_model=ChatResponse)
def chat_preview(req: ChatRequest):
    """Preview mode: responds from built-in knowledge base, no API key required."""
    response_text = _preview_response(req.message, req.topic)
    return ChatResponse(
        response=response_text,
        mode="preview",
        topic=req.topic,
        disclaimer=DISCLAIMER,
    )


@router.post("/chat", response_model=ChatResponse)
def chat_live(req: ChatRequest):
    """Live mode: uses Claude AI when ANTHROPIC_API_KEY is set, else falls back to preview."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Graceful fallback to preview when no key is configured
        response_text = _preview_response(req.message, req.topic)
        return ChatResponse(
            response=f"[Falling back to preview — no API key configured]\n\n{response_text}",
            mode="preview-fallback",
            topic=req.topic,
            disclaimer=DISCLAIMER,
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = (
            "You are Fern, a compassionate and knowledgeable women's health assistant. "
            "You provide evidence-based, empathetic responses about women's health topics "
            f"including {', '.join(TOPICS)}. "
            "Always remind users that your responses are educational and not a substitute for "
            "professional medical advice. Keep responses concise and well-structured."
        )

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": req.message}],
        )
        response_text = message.content[0].text
    except Exception as exc:
        response_text = (
            f"AI response unavailable ({type(exc).__name__}). "
            f"Showing preview response instead.\n\n"
            + _preview_response(req.message, req.topic)
        )

    return ChatResponse(
        response=response_text,
        mode="live",
        topic=req.topic,
        disclaimer=DISCLAIMER,
    )
