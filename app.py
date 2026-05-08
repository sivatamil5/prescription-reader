import streamlit as st
from groq import Groq
import PyPDF2
import io
import base64

# ── Setup ──────────────────────────────────────────────────────
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── Helper: Extract text from PDF ──────────────────────────────
def extract_from_pdf(uploaded_file) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# ── Helper: Convert image to base64 ────────────────────────────
def image_to_base64(uploaded_file) -> str:
    image_bytes = uploaded_file.read()
    return base64.b64encode(image_bytes).decode("utf-8")

# ── Agent: Analyze prescription text ───────────────────────────
def analyze_prescription_text(text: str, language: str) -> str:
    prompt = f"""
    You are a helpful medical assistant.
    Explain this prescription in simple {language} language.

    For each medicine explain:
    1. 💊 Medicine Name
    2. 🎯 What it is used for
    3. ⏰ How many times per day
    4. ⚠️ Side effects
    5. 🍽️ Food to avoid
    6. 📝 Special instructions

    End with:
    "⚕️ Always follow your doctor's instructions."

    Prescription:
    {text}
    """
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# ── Agent: Analyze prescription image ──────────────────────────
def analyze_prescription_image(base64_image: str, language: str) -> str:
    prompt = f"""
    You are a helpful medical assistant.
    Read this prescription image carefully.
    Explain in simple {language} language.

    For each medicine explain:
    1. 💊 Medicine Name
    2. 🎯 What it is used for
    3. ⏰ How many times per day
    4. ⚠️ Side effects
    5. 🍽️ Food to avoid
    6. 📝 Special instructions

    End with:
    "⚕️ Always follow your doctor's instructions."
    """
    response = client.chat.completions.create(
        model="llava-v1.5-7b-4096-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# ── Streamlit UI ────────────────────────────────────────────────
st.set_page_config(
    page_title="Prescription Reader",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 AI Prescription Reader")
st.caption("Upload prescription → Get simple explanation in Tamil or English")

# Language
st.markdown("### 🌐 Select Language")
language = st.radio(
    "Choose language:",
    ["English", "Tamil", "Hindi"],
    horizontal=True
)

st.markdown("---")

# Upload
st.markdown("### 📁 Upload Prescription")
upload_type = st.radio(
    "Choose format:",
    ["📷 Photo (JPG/PNG)", "📄 PDF"],
    horizontal=True
)

uploaded_file = None

if upload_type == "📷 Photo (JPG/PNG)":
    uploaded_file = st.file_uploader(
        "Upload photo", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        st.image(uploaded_file, caption="Your Prescription", width=300)
else:
    uploaded_file = st.file_uploader(
        "Upload PDF", type=["pdf"]
    )
    if uploaded_file:
        st.success("✅ PDF uploaded!")

st.markdown("---")

# Analyze
if st.button("🔍 Explain My Prescription", type="primary", use_container_width=True):
    if not uploaded_file:
        st.error("❌ Please upload your prescription first!")
    else:
        with st.spinner("🤖 Reading prescription... please wait..."):
            try:
                if upload_type == "📷 Photo (JPG/PNG)":
                    base64_img = image_to_base64(uploaded_file)
                    result = analyze_prescription_image(base64_img, language)
                else:
                    text = extract_from_pdf(uploaded_file)
                    if not text:
                        st.error("❌ Cannot read PDF. Try photo instead.")
                        st.stop()
                    result = analyze_prescription_text(text, language)

                st.success("✅ Done!")
                st.markdown("---")
                st.markdown("## 📋 Explanation")
                st.markdown(result)
                st.markdown("---")

                st.download_button(
                    label="⬇️ Download Explanation",
                    data=result,
                    file_name="prescription_explanation.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("⚕️ For information only. Always follow your doctor's instructions. Built for Chennai patients ❤️")
