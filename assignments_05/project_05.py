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

# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:

    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    
    response = get_completion(messages)
    return response.strip()

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed a Python course and built data pipelines using Prefect and Pandas."

result = generate_cover_letter(job_title, background)
print("Cover letter for ", job_title)
print(result)
print("-" * 40)

# I chose these examples because they show strong, specific openings from people changing careers. They clearly connect past experience to new skills and avoid generic language.
# The few-shot examples help guide the model’s tone, structure, and level of detail, so the output is more consistent and tailored instead of vague or repetitive.

job_title_2 = "Marketing Coordinator"
background_2 = "Three years as a barista and social media hobbyist"
print("Cover letter for ", job_title_2)
result = generate_cover_letter(job_title_2, background_2)
print(result)