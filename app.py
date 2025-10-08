import streamlit as st
import pandas as pd
from io import BytesIO
from google import genai
from google.genai.errors import APIError
from openai import OpenAI

# ==============================
# ⚙️ Cấu hình
# ==============================
st.set_page_config(page_title="🤖 AgriAI CRM – Phân tích & tư vấn khách hàng Agribank", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; color:#AE1C3F;'>🤖 AGRIAI CRM</h1>
<p style='text-align:center; color:gray;'>Hệ thống AI hỗ trợ phân tích – tư vấn – chăm sóc khách hàng Agribank</p>
<hr style='border:1px solid #AE1C3F'>
""", unsafe_allow_html=True)

# ==============================
# 🔑 Cấu hình API Keys và Tùy chọn
# ==============================
with st.expander("🔑 Cấu hình API"):
    col1, col2 = st.columns(2)
    with col1:
        openai_key = st.text_input("🔹 OpenAI API Key:", type="password", placeholder="sk-...")
    with col2:
        gemini_key = st.text_input("🔸 Gemini API Key:", type="password", placeholder="AIzaSy...")
    creativity = st.slider("🎨 Mức độ sáng tạo (0: thực tế – 2: sáng tạo)", 0.0, 2.0, 1.0, 0.1)
    ai_mode = st.radio("🤝 Chế độ AI:", ["Gemini", "OpenAI", "Hybrid (So sánh & tổng hợp)"], horizontal=True)

# ==============================
# 📂 Upload file Excel
# ==============================
uploaded_file = st.file_uploader("📥 Tải file Excel khách hàng Agribank (sheet: KhachHang)", type=["xlsx", "xls"])

# ==============================
# ⚙️ Hàm gọi AI
# ==============================
def call_gemini(prompt, key):
    try:
        client = genai.Client(api_key=key)
        model = "gemini-2.5-flash"
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini lỗi: {e}"

def call_openai(prompt, key, creativity):
    try:
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=creativity
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI lỗi: {e}"

# ==============================
# 🔍 Logic phân tích
# ==============================
def analyze_customer(row, mode, creativity, gemini_key, openai_key):
    prompt = f"""
    Bạn là chuyên gia phân tích khách hàng Agribank. 
    Dựa trên dữ liệu sau, hãy đánh giá và đề xuất hướng tiếp cận, chăm sóc phù hợp:

    Dữ liệu khách hàng:
    {row.to_dict()}

    Hãy trả lời gồm 4 phần:
    1️⃣ Phân tích hành vi & tâm lý (tính cách, sở thích, độ tuổi, khu vực, nghề nghiệp, tôn giáo, chính trị)
    2️⃣ Dự đoán nhu cầu tài chính & hành vi sử dụng sản phẩm Agribank
    3️⃣ Gợi ý sản phẩm phù hợp (tiền gửi, tín dụng, Mobile Banking, QR, POS, thẻ, bảo hiểm…)
    4️⃣ Đề xuất chiến lược tiếp cận & chăm sóc cá nhân hóa.

    Mức độ sáng tạo: {creativity}.
    """

    gemini_text, openai_text = None, None

    if mode in ["Gemini", "Hybrid"]:
        gemini_text = call_gemini(prompt, gemini_key)
    if mode in ["OpenAI", "Hybrid"]:
        openai_text = call_openai(prompt, openai_key, creativity)

    # Nếu hybrid → tổng hợp
    if mode == "Hybrid":
        hybrid_prompt = f"""
        Dưới đây là hai bản phân tích cùng về một khách hàng Agribank:

        🔸 Gemini:
        {gemini_text}

        🔹 OpenAI:
        {openai_text}

        Hãy tổng hợp thành bản đánh giá cuối cùng, chọn ra nội dung hợp lý nhất và khách quan nhất.
        """
        hybrid_result = call_openai(hybrid_prompt, openai_key, 0.7)
        return hybrid_result
    else:
        return gemini_text or openai_text

# ==============================
# 🧠 Giao diện xử lý
# ==============================
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="KhachHang")
        st.success(f"✅ Đã tải {len(df)} khách hàng.")
        st.dataframe(df, use_container_width=True)

        selected_customers = st.multiselect(
            "👥 Chọn khách hàng cần phân tích",
            options=df["Họ tên"].tolist(),
            default=[df["Họ tên"].iloc[0]] if not df.empty else []
        )

        if st.button("🚀 Phân tích khách hàng"):
            if (ai_mode == "Gemini" and not gemini_key) or (ai_mode == "OpenAI" and not openai_key) or (ai_mode == "Hybrid" and (not gemini_key or not openai_key)):
                st.error("⚠️ Vui lòng nhập API key đầy đủ cho chế độ đã chọn.")
            else:
                st.info("🧩 Đang phân tích dữ liệu... vui lòng chờ...")
                results = []
                for _, row in df[df["Họ tên"].isin(selected_customers)].iterrows():
                    analysis = analyze_customer(row, ai_mode, creativity, gemini_key, openai_key)
                    results.append({"Họ tên": row["Họ tên"], "Kết quả phân tích & tư vấn": analysis})
                    st.markdown(f"### 👤 {row['Họ tên']}")
                    st.info(analysis)

                result_df = pd.DataFrame(results)
                output = BytesIO()
                result_df.to_excel(output, index=False, engine="openpyxl")
                st.download_button(
                    label="⬇️ Tải kết quả (Excel)",
                    data=output.getvalue(),
                    file_name="agriAI_tuvan_khachhang.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    except Exception as e:
        st.error(f"⚠️ Lỗi khi đọc file: {e}")
else:
    st.info("⬆️ Vui lòng tải file khách hàng Agribank (sheet: KhachHang) để bắt đầu.")

# ==============================
# 🔚 Footer
# ==============================
st.markdown("""
<footer style='text-align:center; margin-top:40px; padding:10px; background:#AE1C3F; color:white; border-radius:12px;'>
© 2025 Agribank Training & Development — Hệ thống <b>AgriAI CRM</b><br>
Phát triển bởi Bộ phận CNTT Agribank • Tích hợp AI kép (Gemini + OpenAI)
</footer>
""", unsafe_allow_html=True)
