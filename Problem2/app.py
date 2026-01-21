import streamlit as st
import json
import os
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(page_title="Clinical Note Extractor", layout="centered")

st.title("🩺 Clinical Note Extractor")
st.write("Extract structured medical information using a free LLM")

# Input clinical note
clinical_note = st.text_area(
    "Enter Clinical Note",
    height=220,
    placeholder="Paste the clinical note here..."
)

def extract_medical_info(note):
    prompt = f"""
You are a medical information extraction system.

Extract ONLY the following fields and return VALID JSON:
- patient_name (string)
- age (number)
- gender (string)
- symptoms (list)
- diagnosis (string)
- medications (list)

Rules:
- Return ONLY raw JSON
- No markdown
- No explanation
- If missing, use null

Clinical Note:
\"\"\"
{note}
\"\"\"
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    raw_output = response.choices[0].message.content.strip()

    # Cleanup markdown if present
    if raw_output.startswith("```"):
        raw_output = raw_output.replace("```json", "").replace("```", "").strip()

    return json.loads(raw_output)

# Button
if st.button("🔍 Extract Information"):
    if clinical_note.strip() == "":
        st.warning("Please enter a clinical note.")
    else:
        with st.spinner("Extracting medical information..."):
            try:
                result = extract_medical_info(clinical_note)
                st.success("Extraction successful!")
                st.json(result)
            except Exception as e:
                st.error("Failed to extract information.")
                st.text(str(e))

st.markdown("---")
st.caption("LLM-powered Clinical Note Extraction using Groq (Free)")
