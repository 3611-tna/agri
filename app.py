import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import concurrent.futures
from openai import OpenAI
from google import genai

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="AgriAI CRM Pro 3.0", layout="wide", page_icon="🤖")

# ========== STYLES ==========
st.markdown("""
<style>
body {background-color:#fafafa;}
h1,h2,h3,h4 {color:#AE1C3F;}
.stButton>button {
    background: linear-gradient(90deg,#AE1C3F,#D72638);
    color:white;font-weight:bold;border:none;padding:0.6em 1.2em;border-radius:8px;
}
.stButton>button:hover {opacity:0.9;transform:scale(1.02);}
.chat-box {background:#fff;border-left:5px solid #AE1C3F;padding:8px;margin:6px 0;border-radius:6px;}
.footer {text-align:center;margin-top:2em;color:#666;font-size:0.9em;}
.analysis-box {background:white;padding:15px;border-radius:10px;box-shadow:0 0 8px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🤖 AGRIAI CRM PRO 3.0")
st.caption("Ứng dụng AI phân tích hành vi & tư vấn khách hàng Agribank – tích hợp OpenAI + Gemini")

# ========== CONFIG ==========
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

# ========== FILE UPLOAD ==========
uploaded = st.file_uploader("📥 Tải file Excel khách hàng (sheet: KhachHang)", type=["xlsx"])

# ========== AI CALLERS ==========
def call_openai(prompt, key, model_name, creativity):
    try:
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia phân tích khách hàng Agribank có 15 năm kinh nghiệm. Trình bày có cấu trúc, ngắn gọn, bám sát thực tế ngân hàng."},
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
    st.success(f"✅ Đã tải dữ liệu gồm {len(df)} khách hàng")

    # Chọn khách hàng
    search = st.text_input("🔍 Nhập tên khách hàng")
    filtered = df[df["Họ tên"].str.contains(search, case=False, na=False)] if search else df
    st.dataframe(filtered, use_container_width=True)

    selected = st.selectbox("👤 Chọn khách hàng cần phân tích", filtered["Họ tên"].tolist())

    cust_data = df[df["Họ tên"] == selected].iloc[0].to_dict()
    col1, col2 = st.columns([1.2, 1])
    with col1:
        if st.button("🚀 Phân tích khách hàng"):
            prompt = f"""
            Dữ liệu khách hàng: {cust_data}
            Hãy phân tích khách hàng này ở 4 khía cạnh sau:
            1️⃣ Tình hình tài chính, thu nhập, mức độ rủi ro.
            2️⃣ Tâm lý, độ tuổi, khu vực, sở thích, xu hướng chi tiêu.
            3️⃣ Sản phẩm/dịch vụ phù hợp nên tư vấn (ví dụ: tiết kiệm, vay vốn, bảo hiểm...).
            4️⃣ Đề xuất chiến lược chăm sóc, giữ chân và phát triển quan hệ.
            Trình bày ngắn gọn, có lý do, sát thực tế Agribank.
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

            # Xuất Excel
            excel_data = export_excel(selected, summary)
            st.download_button("📊 Tải kết quả (Excel)", excel_data, file_name=f"{selected}_AI.xlsx")

    with col2:
        # Biểu đồ minh họa tài chính
        try:
            income = float(str(cust_data.get("Thu nhập", "0")).replace(",", "").replace(".", ""))
            deposit = float(str(cust_data.get("Số dư tiền gửi", "0")).replace(",", "").replace(".", ""))
            loan = float(str(cust_data.get("Số dư tiền vay", "0")).replace(",", "").replace(".", ""))

            fig, ax = plt.subplots(figsize=(4,3))
            sns.barplot(x=["Thu nhập", "Tiền gửi", "Tiền vay"], y=[income, deposit, loan], ax=ax, palette=["#AE1C3F","#C62828","#F57C00"])
            ax.set_ylabel("Giá trị (VNĐ)")
            ax.set_title("Tình hình tài chính KH")
            st.pyplot(fig)
        except Exception as e:
            st.info("Không đủ dữ liệu để hiển thị biểu đồ.")

    # Chat AI
    st.markdown("### 💬 Trợ lý AI – Hỏi thêm / bổ sung dữ liệu")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        who = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f"<div class='chat-box'><b>{who}</b> {msg['content']}</div>", unsafe_allow_html=True)

    question = st.chat_input("Nhập câu hỏi hoặc bổ sung thông tin...")
    if question:
        st.session_state["chat_history"].append({"role":"user","content":question})
        prompt = f"Khách hàng: {cust_data}\nNhân viên nói: {question}\nHãy phản hồi ngắn gọn, thực tế, chỉ hỏi thêm khi cần."
        answer = call_openai(prompt, openai_key, openai_model, creativity)
        st.session_state["chat_history"].append({"role":"assistant","content":answer})
        st.rerun()

else:
    st.info("⬆️ Hãy tải file Excel khách hàng (sheet KhachHang) để bắt đầu.")

# FOOTER
st.markdown("<div class='footer'>© 2025 Agribank AI | AgriAI CRM Pro v3.0</div>", unsafe_allow_html=True)
