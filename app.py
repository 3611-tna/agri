import streamlit as st
import pandas as pd
import concurrent.futures
from io import BytesIO
from fpdf import FPDF
from google import genai
from google.genai.errors import APIError

# ============= OPENAI Import Compatibility ============
try:
    from openai import OpenAI
except ImportError:
    import openai
    OpenAI = None

# ============= PAGE CONFIGURATION =====================
st.set_page_config(page_title="🤖 AgriAI CRM Pro", layout="wide", page_icon="🤖")

# ============= STYLE ==================================
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; background-color:#fafafa;}
h1,h2,h3 {color:#AE1C3F;}
.analysis-card {
  background-color: #fff;
  border-left: 5px solid #AE1C3F;
  border-radius: 10px;
  padding: 1rem;
  margin-bottom: 1rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.chat-box {
  background: #f8f8f8;
  border-radius: 10px;
  padding: 0.6rem;
  margin: 0.4rem 0;
}
.chat-user {color:#AE1C3F;font-weight:bold;}
.chat-ai {color:#333;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🤖 AGRIAI CRM PRO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Phân tích & tư vấn khách hàng Agribank bằng AI có trí nhớ & chat tương tác</p><hr>", unsafe_allow_html=True)

# ============= CONFIG PANEL ===========================
with st.expander("⚙️ Cấu hình AI & API"):
    c1, c2 = st.columns(2)
    with c1:
        openai_key = st.text_input("🔹 OpenAI API Key:", type="password")
        openai_model = st.selectbox("🔹 Model OpenAI:", ["gpt-5", "gpt-4o-mini", "gpt-4-turbo"])
    with c2:
        gemini_key = st.text_input("🔸 Gemini API Key:", type="password")
        gemini_model = st.selectbox("🔸 Model Gemini:", ["gemini-2.0-flash", "gemini-1.5-flash"])
    creativity = st.slider("🎨 Mức sáng tạo (0–2)", 0.0, 2.0, 1.0, 0.1)
    ai_mode = st.radio("🤝 Chế độ AI:", ["Gemini", "OpenAI", "Hybrid"], horizontal=True)

# ============= UPLOAD FILE ============================
uploaded = st.file_uploader("📥 Tải file Excel dữ liệu khách hàng (sheet: KhachHang)", type=["xlsx", "xls"])

# ============= AI CALL FUNCTIONS ======================
def call_gemini(prompt, key, model_name):
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini lỗi: {e}"

def call_openai(prompt, key, model_name, creativity):
    try:
        if OpenAI:
            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=creativity
            )
            return resp.choices[0].message.content.strip()
        else:
            openai.api_key = key
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=creativity
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI lỗi: {e}"

# ============= PDF EXPORT =============================
def export_pdf(customer_name, sections):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"BÁO CÁO PHÂN TÍCH KHÁCH HÀNG: {customer_name}", ln=True)
    pdf.set_font("Arial", "", 12)
    for title, content in sections.items():
        pdf.multi_cell(0, 10, f"\n[{title}]\n{content}")
    output = BytesIO()
    pdf.output(output)
    return output.getvalue()

# ============= CONTEXT MEMORY =========================
if "memory" not in st.session_state:
    st.session_state["memory"] = {}

if uploaded:
    df = pd.read_excel(uploaded, sheet_name="KhachHang")
    st.success(f"✅ Đã tải {len(df)} khách hàng.")
    selected = st.selectbox("👥 Chọn khách hàng cần phân tích", df["Họ tên"].tolist())

    if selected not in st.session_state["memory"]:
        st.session_state["memory"][selected] = {"chat": [], "missing": []}

    cust_data = df[df["Họ tên"] == selected].to_dict(orient="records")[0]
    st.dataframe(pd.DataFrame([cust_data]), use_container_width=True)

    # ============= PHÂN TÍCH CHÍNH =====================
    if st.button("🚀 Phân tích khách hàng"):
        missing = [col for col, val in cust_data.items() if pd.isna(val) or val == ""]
        st.session_state["memory"][selected]["missing"] = missing
        st.write(f"🧩 Dữ liệu thiếu: {', '.join(missing) if missing else 'Không có'}")

        prompt = f"""
        Bạn là chuyên gia Agribank với 15 năm kinh nghiệm.
        Hãy phân tích khách hàng dưới góc độ chuyên gia các mảng sau:
        1️⃣ Tài chính & năng lực thanh khoản
        2️⃣ Hành vi & tâm lý tiêu dùng
        3️⃣ Sản phẩm tiềm năng phù hợp
        4️⃣ Định hướng chăm sóc & giữ chân
        Dữ liệu khách hàng:
        {cust_data}
        """

        gemini_text, openai_text = None, None
        if ai_mode == "Hybrid":
            with concurrent.futures.ThreadPoolExecutor() as pool:
                futures = []
                if gemini_key: futures.append(pool.submit(call_gemini, prompt, gemini_key, gemini_model))
                if openai_key: futures.append(pool.submit(call_openai, prompt, openai_key, openai_model, creativity))
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
                texts = [r for r in results if "lỗi" not in r]
                summary = "\n\n---\n\n".join(texts)
        elif ai_mode == "Gemini":
            summary = call_gemini(prompt, gemini_key, gemini_model)
        else:
            summary = call_openai(prompt, openai_key, openai_model, creativity)

        st.session_state["memory"][selected]["analysis"] = summary
        st.markdown(f"<div class='analysis-card'>{summary}</div>", unsafe_allow_html=True)

        # EXPORT
        pdf_data = export_pdf(selected, {"Phân tích & Tư vấn": summary})
        st.download_button("📄 Xuất báo cáo PDF", pdf_data, file_name=f"{selected}_report.pdf")

    # ============= CHAT KHUNG ==========================
    st.markdown("### 💬 Trợ lý AI - Tư vấn & bổ sung dữ liệu")

    for msg in st.session_state["memory"][selected]["chat"]:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-box chat-user'>🧑‍💼 Bạn: {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-box chat-ai'>🤖 AI: {msg['content']}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Nhập câu hỏi hoặc thông tin bổ sung...")
    if user_input:
        st.session_state["memory"][selected]["chat"].append({"role": "user", "content": user_input})

        # Xác định nếu thiếu dữ liệu
        missing = st.session_state["memory"][selected]["missing"]
        if missing:
            prompt = f"""
            Bạn đang tư vấn cho nhân viên Agribank.
            Thông tin khách hàng: {cust_data}
            Lịch sử hội thoại: {st.session_state["memory"][selected]['chat']}
            Người dùng vừa nói: "{user_input}"
            Hãy hỏi thêm thông tin cần thiết hoặc tiếp tục phân tích, 
            nhưng chỉ hỏi những gì còn thiếu: {missing}.
            """
        else:
            prompt = f"""
            Bạn đang nói chuyện về khách hàng: {selected}.
            Dữ liệu: {cust_data}
            Nhân viên vừa nói: {user_input}
            Hãy phản hồi ngắn gọn, cụ thể và thực tế.
            """

        response = call_openai(prompt, openai_key, openai_model, creativity)
        st.session_state["memory"][selected]["chat"].append({"role": "assistant", "content": response})
        st.experimental_rerun()

else:
    st.info("⬆️ Hãy tải file Excel (sheet: KhachHang) để bắt đầu.")

# ============= FOOTER ==========================
st.markdown("""
<hr><div style='text-align:center;background:#AE1C3F;color:white;padding:10px;border-radius:12px;'>
© 2025 Agribank Training & Development — AgriAI CRM Pro<br>
Tư vấn khách hàng thông minh, hiểu ngữ cảnh và có trách nhiệm.
</div>
""", unsafe_allow_html=True)
