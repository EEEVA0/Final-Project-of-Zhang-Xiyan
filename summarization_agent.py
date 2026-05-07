from openai import OpenAI

client = OpenAI(
    api_key="OPENAI_API_KEY",
    base_url="https://one.ocoolai.com/v1"
)
def summarization_agent(reviewed_text):
    prompt = f"""
    You are a Patent Abstract Specialist.

    TASK:
    From the following reviewed patent:
    {reviewed_text}

    Generate:
    1. A **concise abstract** (100–150 words)
    2. A **highlight summary** in bullet points (key technical advantages)
    3. A **one-sentence summary** suitable for comparison dashboards.

    Use clear and formal patent English.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
