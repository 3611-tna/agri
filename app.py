import streamlit as st
import pandas as pd
from io import BytesIO
from google import genai
from google.genai.errors import APIError

# ✅ Import OpenAI safely (tương thích mọi môi trường)
try:
    from openai import OpenAI
except ImportError:
    import openai
    OpenAI = None

# ==============================
# ⚙️ Cấu hình trang
# ==============================
st.set_page_config(page_title="🤖 AgriAI CRM – Phân tích & tư vấn khách hàng", layout="wide")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; color:#AE1C3F;'>💡 AGRIAI CRM</h1>
<p style='text-align:center; color:gray;'>Hệ thống hỗ trợ chăm sóc và phân tích hành vi khách hàng Agribank bằng AI lai (Gemini + OpenAI)</p>
<hr style='border:1px solid #AE1C3F'>
""", unsafe_allow_html=True)

# ==============================
# 🔑 Cấu hình API & lựa chọn AI
# ==============================
with st.expander("⚙️ Cấu hình hệ thống AI"):
    c1, c2 = st.columns(2)
    with c1:
        openai_key = st.text_input("🔹 OpenAI API Key:", type="password", placeholder="sk-...")
        openai_model = st.selectbox("🔹 Model OpenAI:", ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])
    with c2:
        gemini_key = st.text_input("🔸 Gemini API Key:", type="password", placeholder="AIzaSy...")
        gemini_model = st.selectbox("🔸 Model Gemini:", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
    creativity = st.slider("🎨 Mức độ sáng tạo (0 – 2)", 0.0, 2.0, 1.0, 0.1)
    ai_mode = st.radio("🤝 Chọn chế độ phân tích:", ["Gemini", "OpenAI", "Hybrid (So sánh & hợp nhất)"], horizontal=True)

# ==============================
# 📂 Upload file Excel khách hàng
# ==============================
uploaded_file = st.file_uploader("📥 Tải file Excel dữ liệu khách hàng (sheet: KhachHang)", type=["xlsx", "xls"])

# ==============================
# ⚙️ Hàm gọi AI
# ==============================
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

# ==============================
# 🧠 Hàm phân tích khách hàng
# ==============================
def analyze_customer(row, mode, creativity, gemini_key, gemini_model, openai_key, openai_model):
    base_prompt = f"""
    Bạn là chuyên gia phân tích khách hàng của Agribank. 
    Hãy đánh giá và tư vấn chiến lược tiếp cận dựa trên thông tin sau:

    {row.to_dict()}

    Yêu cầu trả lời gồm 4 phần:
    1️⃣ Phân tích hành vi, tâm lý, sở thích, độ tuổi, khu vực, tôn giáo, chính trị.
    2️⃣ Dự đoán nhu cầu tài chính & hành vi tiêu dùng.
    3️⃣ Gợi ý sản phẩm, dịch vụ Agribank phù hợp (tín dụng, tiết kiệm, số hoá...).
    4️⃣ Chiến lược chăm sóc, tương tác & tiếp cận cá nhân hoá.

    Mức độ sáng tạo: {creativity}
    """

    gemini_text, openai_text = None, None

    if mode in ["Gemini", "Hybrid"] and gemini_key:
        gemini_text = call_gemini(base_prompt, gemini_key, gemini_model)
    if mode in ["OpenAI", "Hybrid"] and openai_key:
        openai_text = call_openai(base_prompt, openai_key, openai_model, creativity)

    if mode == "Hybrid":
        merge_prompt = f"""
        Dưới đây là hai bản phân tích cùng về khách hàng này:

        🔸 Gemini:
        {gemini_text}

        🔹 OpenAI:
        {openai_text}

        Hãy tổng hợp lại thành bản kết luận duy nhất, giữ ý chính hợp lý từ cả hai.
        """
        return call_openai(merge_prompt, openai_key, openai_model, 0.7)
    else:
        return gemini_text or openai_text

# ==============================
# 📊 Hiển thị và xử lý dữ liệu
# ==============================
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="KhachHang")
        st.success(f"✅ Đã tải {len(df)} khách hàng từ file Excel.")
        st.dataframe(df, use_container_width=True)

        selected = st.multiselect(
            "👥 Chọn khách hàng để phân tích",
            options=df["Họ tên"].tolist(),
            default=[df["Họ tên"].iloc[0]] if not df.empty else []
        )

        if st.button("🚀 Phân tích & tư vấn khách hàng"):
            if mode := ai_mode:
                if (mode == "Gemini" and not gemini_key) or \
                   (mode == "OpenAI" and not openai_key) or \
                   (mode == "Hybrid" and (not gemini_key or not openai_key)):
                    st.error("⚠️ Vui lòng nhập đủ API key cho chế độ đã chọn.")
                else:
                    st.info("🔍 Đang phân tích, vui lòng chờ...")
                    results = []
                    for _, row in df[df["Họ tên"].isin(selected)].iterrows():
                        analysis = analyze_customer(row, ai_mode, creativity, gemini_key, gemini_model, openai_key, openai_model)
                        results.append({"Họ tên": row["Họ tên"], "Phân tích & tư vấn": analysis})
                        st.markdown(f"### 👤 {row['Họ tên']}")
                        st.info(analysis)

                    result_df = pd.DataFrame(results)
                    out = BytesIO()
                    result_df.to_excel(out, index=False, engine="openpyxl")
                    st.download_button("⬇️ Tải kết quả (Excel)", out.getvalue(),
                        file_name="agriAI_ketqua.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"⚠️ Lỗi khi đọc file: {e}")
else:
    st.info("⬆️ Hãy tải file Excel (sheet: KhachHang) để bắt đầu phân tích.")

# ==============================
# 🔚 Footer
# ==============================
st.markdown("""
<footer style='text-align:center; margin-top:40px; padding:10px; background:#AE1C3F; color:white; border-radius:12px;'>
© 2025 Agribank Training & Development — <b>AgriAI CRM</b><br>
Phát triển bởi Bộ phận CNTT Agribank • Kết hợp AI kép (Gemini + OpenAI)
</footer>
""", unsafe_allow_html=True)
