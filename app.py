# ==========================================================
# 🌾 AGRIAI CRM PRO v3.1
# AI Phân tích & Tư vấn Khách hàng Agribank
# Tác giả: Shine | Agribank Digital Transformation
# ==========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import concurrent.futures
from openai import OpenAI
from google import genai

# ========== CẤU HÌNH TRANG ==========
st.set_page_config(page_title="AgriAI CRM Pro 3.1", layout="wide", page_icon="🤖")

# ========== STYLE ==========
st.markdown("""
<style>
h1,h2,h3,h4 {color:#AE1C3F;}
.stButton>button {
    background: linear-gradient(90deg,#AE1C3F,#D72638);
    color:white;font-weight:bold;border:none;
    padding:0.6em 1.2em;border-radius:8px;
}
.stButton>button:hover {opacity:0.9;transform:scale(1.02);}
.analysis-box {
    background:white;padding:18px;border-radius:10px;
    border-left:6px solid #AE1C3F;
    box-shadow:0 3px 10px rgba(0,0,0,0.1);
}
.chat-box {
    background:#fff;border-left:4px solid #AE1C3F;
    padding:10px;margin:8px 0;border-radius:6px;
}
.footer {
    text-align:center;color:#777;margin-top:40px;font-size:0.9em;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AgriAI CRM Pro 3.1")
st.caption("Ứng dụng AI hỗ trợ phân tích, tư vấn và chăm sóc khách hàng Agribank – kết hợp OpenAI & Gemini")

# ========== KHU CẤU HÌNH ==========
with st.expander("⚙️ Cấu hình AI"):
    c1, c2 = st.columns(2)
    with c1:
        openai_key = st.text_input("🔹 OpenAI API Key", type="password")
        openai_model = st.selectbox("🔹 Model OpenAI", ["gpt-4o-mini", "gpt-4-turbo", "gpt-5"])
    with c2:
        gemini_key = st.text_input("🔸 Gemini API Key", type="password")
        gemini_model = st.selectbox("🔸 Model Gemini", ["gemini-2.0-flash", "gemini-1.5-flash"])
    creativity = st.slider("🎨 Mức sáng tạo (0–2)", 0.0, 2.0, 1.0, 0.1)
    ai_mode = st.radio("🤝 Chế độ AI", ["OpenAI", "Gemini", "Hybrid"], horizontal=True)

# ========== UPLOAD FILE ==========
uploaded = st.file_uploader("📥 Tải file Excel khách hàng (sheet: KhachHang)", type=["xlsx"])

# ========== HÀM GỌI AI ==========
def call_openai(prompt, key, model_name, creativity):
    try:
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia phân tích khách hàng Agribank, có 15 năm kinh nghiệm. Tư vấn thực tế, ngắn gọn, có cơ sở."},
                {"role": "user", "content": prompt},
            ],
            temperature=creativity,
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI lỗi: {e}"

def call_gemini(prompt, key, model_name):
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini lỗi: {e}"

def export_excel(customer_name, summary):
    df = pd.DataFrame({"Khách hàng": [customer_name], "Phân tích & Tư vấn": [summary]})
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="PhanTich")
    return buf.getvalue()

# ========== MAIN ==========
if uploaded:
    df = pd.read_excel(uploaded, sheet_name="KhachHang")
    st.success(f"✅ Đã tải dữ liệu gồm {len(df)} khách hàng.")

    search = st.text_input("🔍 Nhập tên khách hàng để lọc")
    filtered = df[df["Họ tên"].str.contains(search, case=False, na=False)] if search else df
    st.dataframe(filtered, use_container_width=True)

    selected = st.selectbox("👤 Chọn khách hàng cần phân tích", filtered["Họ tên"].tolist())
    cust_data = df[df["Họ tên"] == selected].iloc[0].to_dict()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        if st.button("🚀 Phân tích khách hàng"):
            prompt = f"""
            Dữ liệu khách hàng: {cust_data}
            Hãy phân tích khách hàng này theo 4 khía cạnh:
            1️⃣ Năng lực tài chính và mức độ rủi ro.
            2️⃣ Tâm lý, độ tuổi, sở thích, khu vực sinh sống.
            3️⃣ Gợi ý sản phẩm phù hợp (tiết kiệm, vay, bảo hiểm...).
            4️⃣ Đề xuất chiến lược chăm sóc và giữ chân khách hàng.
            Trình bày có lý do, có dẫn chứng, theo phong cách chuyên gia Agribank.
            """

            if ai_mode == "Hybrid":
                with concurrent.futures.ThreadPoolExecutor() as ex:
                    futures = []
                    if openai_key:
                        futures.append(ex.submit(call_openai, prompt, openai_key, openai_model, creativity))
                    if gemini_key:
                        futures.append(ex.submit(call_gemini, prompt, gemini_key, gemini_model))
                    results = [f.result() for f in concurrent.futures.as_completed(futures)]
                    summary = "\n\n---\n\n".join(results)
            elif ai_mode == "OpenAI":
                summary = call_openai(prompt, openai_key, openai_model, creativity)
            else:
                summary = call_gemini(prompt, gemini_key, gemini_model)

            st.session_state["analysis"] = summary
            st.markdown(f"<div class='analysis-box'>{summary}</div>", unsafe_allow_html=True)

            excel_data = export_excel(selected, summary)
            st.download_button("📊 Tải kết quả (Excel)", excel_data, file_name=f"{selected}_AI.xlsx")

    with col2:
        try:
            income = float(str(cust_data.get("Thu nhập", "0")).replace(",", "").replace(".", ""))
            deposit = float(str(cust_data.get("Số dư tiền gửi", "0")).replace(",", "").replace(".", ""))
            loan = float(str(cust_data.get("Số dư tiền vay", "0")).replace(",", "").replace(".", ""))

            fig, ax = plt.subplots(figsize=(4,3))
            sns.barplot(x=["Thu nhập", "Tiền gửi", "Tiền vay"], y=[income, deposit, loan],
                        ax=ax, palette=["#AE1C3F","#D72638","#F57C00"])
            ax.set_ylabel("Giá trị (VNĐ)")
            ax.set_title("📈 Cơ cấu tài chính khách hàng")
            st.pyplot(fig)
        except Exception:
            st.info("Không đủ dữ liệu để hiển thị biểu đồ.")

    # === KHUNG CHAT ===
    st.markdown("### 💬 Trợ lý AI – Bổ sung hoặc hỏi thêm")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        who = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f"<div class='chat-box'><b>{who}</b> {msg['content']}</div>", unsafe_allow_html=True)

    question = st.chat_input("Nhập câu hỏi hoặc thông tin bổ sung...")
    if question:
        st.session_state["chat_history"].append({"role":"user","content":question})
        prompt = f"Khách hàng: {cust_data}\nNhân viên nói: {question}\nHãy trả lời ngắn gọn, sát thực tế, chỉ hỏi lại nếu cần thông tin bổ sung."
        answer = call_openai(prompt, openai_key, openai_model, creativity)
        st.session_state["chat_history"].append({"role":"assistant","content":answer})
        st.rerun()

else:
    st.info("⬆️ Hãy tải file Excel (sheet KhachHang) để bắt đầu phân tích.")

# FOOTER
st.markdown("<div class='footer'>© 2025 Agribank AI | AgriAI CRM Pro v3.1</div>", unsafe_allow_html=True)
