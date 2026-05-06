# Part 2: Mini-Project — Job Application Helper

from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()


# --- Task 1: Setup and System Prompt ---

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_prompt = """
You are an AI job application coach helping career changers improve their job materials.

Your role:
- Help users rewrite resume bullet points so they are clear, results-oriented, and tailored to a new field
- Help draft and refine cover letters
- Ask thoughtful follow-up questions when needed to better understand the user’s experience and goals

Your audience:
- People transitioning into a new industry who may struggle to translate their past experience into relevant language

Guidelines:
- Stay focused only on job application materials (resume, cover letter, professional summaries, interview prep)
- Use clear, professional, and concise language
- When rewriting, preserve the user’s original meaning but improve clarity, impact, and relevance
- Ask clarifying questions if the input is vague or incomplete

Important constraints:
- Always remind the user to review and edit your output before submitting it anywhere
- Acknowledge that you may not fully understand specific industry expectations and encourage the user to apply their own judgment
- Do not fabricate experience or qualifications for the user

Tone:
- Supportive, practical, and constructive (not overly enthusiastic or vague)
"""

# I instructed the model not to add experience or skills that are not true, because it may sometimes generate incorrect or exaggerated information. This helps ensure the resume stays honest and appropriate for real job applications.

# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
  
    response = get_completion(messages)
    
    # Clean the response (remove ```json ... ``` if present)
    cleaned = response.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]  # remove first ```
    if cleaned.startswith("json"):
        cleaned = cleaned[4:]  # remove 'json'
    cleaned = cleaned.strip()
    
    # Parse JSON
    try:
        results = json.loads(cleaned)
    except json.JSONDecodeError:
        print("Error: Could not parse JSON response.")
        print(response)
        return []

    # Print side-by-side comparison
    for item in results:
        print(f"Original: {item['original']}")
        print(f"Improved: {item['improved']}")
        print("-" * 40)

    return results

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)

# These bullet points are weak because they are vague and do not show clear results or impact. They use basic language and do not explain what was achieved.
# The model improves them by adding stronger action verbs, more detail, and clearer descriptions of the outcome or value of the work.

# Did json.loads() succeed?
# Yes, after adjusting the prompt to say "Respond ONLY with valid JSON, no other text" (and removing code blocks if needed), the JSON was parsed successfully without errors.

# Are both versions printing clearly?
# Yes, both the original and improved bullet points are printed clearly side by side, making it easy to compare them.

# Are the improvements meaningful?
# Yes, the improved bullets are stronger. They use better action verbs, add more detail, and explain the impact of the work. They are not just rearranged, they are clearer and more professional.