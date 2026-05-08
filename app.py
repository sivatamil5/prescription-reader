import streamlit as st
from groq import Groq
import PyPDF2
import io
import base64

# ── Setup ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prescription Reader",
    page_icon="🏥",
    layout="centered"
)

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── Helper Functions ───────────────────────────────────────────

def extract_from_pdf(uploaded_file) -> str:
    try:
        bytes_data = uploaded_file.getvalue()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(bytes_data))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"PDF reading error: {str(e)}")
        return ""

def image_to_base64(uploaded_file) -> str:
    try:
        bytes_data = uploaded_file.getvalue()
        return base64.b64encode(bytes_data).decode("utf-8")
    except Exception:
        return ""

def analyze_prescription_text(text: str, language: str) -> str:
    prompt = f"""
You are a helpful medical assistant. Explain this prescription in simple, easy-to-understand {language} language.

For each medicine, explain:
1. 💊 Medicine Name
2. 🎯 What it is used for
3. ⏰ How many times per day & dosage
4. ⚠️ Common side effects
5. 🍽️ Food/drink to avoid
6. 📝 Special instructions

End with: "⚕️ Always follow your doctor's instructions. This is only for information."

Prescription:
{text}
"""
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


def analyze_prescription_image(base64_image: str, language: str) -> str:
    prompt = f"""
You are a helpful medical assistant. Carefully read this handwritten/printed prescription image and explain it in simple, easy-to-understand {language} language.

For each medicine, explain:
1. 💊 Medicine Name
2. 🎯 What it is used for
3. ⏰ How many times per day & dosage
4. ⚠️ Common side effects
5. 🍽️ Food/drink to avoid
6. 📝 Special instructions

End with: "⚕️ Always follow your doctor's instructions. This is only for information."
"""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=1200,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


# ── Streamlit UI ────────────────────────────────────────────────

st.title("🏥 AI Prescription Reader")
st.caption("Upload your prescription → Get simple explanation in your language")

# Language Selection
st.markdown("### 🌐 Select Language")
language = st.radio(
    "Choose explanation language:",
    ["English", "Tamil", "Hindi"],
    horizontal=True
)

st.markdown("---")

# Upload Section
st.markdown("### 📁 Upload Prescription")
upload_type = st.radio(
    "Choose format:",
    ["📷 Photo (JPG/PNG)", "📄 PDF"],
    horizontal=True
)

uploaded_file = st.file_uploader(
    "Upload your prescription",
    type=["jpg", "jpeg", "png"] if upload_type == "📷 Photo (JPG/PNG)" else ["pdf"]
)

if uploaded_file:
    if upload_type == "📷 Photo (JPG/PNG)":
        st.image(uploaded_file, caption="Uploaded Prescription", use_column_width=True)
    else:
        st.success(f"✅ PDF uploaded: {uploaded_file.name}")

st.markdown("---")

# Analyze Button
if st.button("🔍 Explain My Prescription", type="primary", use_container_width=True):
    if not uploaded_file:
        st.error("❌ Please upload a prescription first!")
    else:
        with st.spinner("🤖 AI is reading your prescription..."):
            try:
                if upload_type == "📷 Photo (JPG/PNG)":
                    base64_img = image_to_base64(uploaded_file)
                    if not base64_img:
                        st.error("❌ Failed to process image. Please try again.")
                        st.stop()
                    result = analyze_prescription_image(base64_img, language)
                else:
                    text = extract_from_pdf(uploaded_file)
                    if not text or len(text) < 20:
                        st.error("❌ Could not extract text from PDF. Please try uploading as photo instead.")
                        st.stop()
                    result = analyze_prescription_text(text, language)

                st.success("✅ Analysis Complete!")
                st.markdown("---")
                st.markdown("## 📋 Your Prescription Explanation")
                st.markdown(result)

                st.download_button(
                    label="⬇️ Download Explanation",
                    data=result,
                    file_name="prescription_explanation.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Something went wrong: {str(e)}")

# Footer
st.markdown("---")
st.caption("⚕️ **For informational purposes only.** Always consult your doctor. Built with ❤️ for patients")
