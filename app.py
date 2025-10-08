import streamlit as st
import pandas as pd
import time
import concurrent.futures
from io import BytesIO
from google import genai
from google.genai.errors import APIError

# --- OPENAI fallback for all SDK versions ---
try:
    from openai import OpenAI
except ImportError:
    import openai
    OpenAI = None

# ==============================
# ⚙️ PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="🤖 AgriAI CRM PRO",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🎨 CUSTOM STYLE
# ==============================
st.markdown("""
<style>
body {
    background: linear-gradient(180deg,#fff,#f9f9f9);
    font-family: "Segoe UI",sans-serif;
}
h1,h2,h3 { color: #AE1C3F; }
.block-container { padding-top: 1rem; }
footer {visibility: hidden;}
div[data-testid="stMetricValue"] {
    color:#AE1C3F;
}
.analysis-card {
    background-color: #fff6f8;
    border-left: 5px solid #AE1C3F;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.gradient-btn {
    background: linear-gradient(90deg,#AE1C3F,#d7355f);
    color: white;
    border: none;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
    transition: 0.3s;
}
.gradient-btn:hover {
    background: linear-gradient(90deg,#d7355f,#AE1C3F);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🧠 HEADER
# ==============================
st.markdown("""
<h1 style='text-align:center; color:#AE1C3F;'>🤖 AGRIAI CRM PRO</h1>
<p style='text-align:center; color:gray;'>Phân tích & tư vấn khách hàng Agribank bằng AI lai (Gemini + ChatGPT-5)</p>
<hr style='border:1px solid #AE1C3F'>
""", unsafe_allow_html=True)

# ==============================
# 🔧 API CONFIGURATION
# ==============================
with st.expander("⚙️ Cấu hình API Key & Chế độ AI"):
    c1, c2 = st.columns(2)
    with c1:
        openai_key = st.text_input("🔹 OpenAI API Key:", type="password", placeholder="sk-...")
        openai_model = st.selectbox("🔹 Model OpenAI:", [
            "gpt-5", "gpt-4.1-mini", "gpt-4o-mini", "gpt-4-turbo"
        ])
    with c2:
        gemini_key = st.text_input("🔸 Gemini API Key:", type="password", placeholder="AIzaSy...")
        gemini_model = st.selectbox("🔸 Model Gemini:", [
            "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"
        ])
    creativity = st.slider("🎨 Mức độ sáng tạo (0–2)", 0.0, 2.0, 1.0, 0.1)
    ai_mode = st.radio("🤝 Chọn chế độ AI:", ["Gemini", "OpenAI", "Hybrid (lai so sánh)"], horizontal=True)

# ==============================
# 📂 UPLOAD EXCEL
# ==============================
uploaded_file = st.file_uploader("📥 Tải file Excel dữ liệu khách hàng (sheet: KhachHang)", type=["xlsx", "xls"])

# ==============================
# ⚙️ CALL AI FUNCTIONS
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
            import openai
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
# 🧩 ANALYSIS CORE
# ==============================
def analyze_customer(row, mode, creativity, gemini_key, gemini_model, openai_key, openai_model):
    prompt = f"""
    Bạn là chuyên gia phân tích hành vi khách hàng của Agribank.
    Dựa trên dữ liệu sau, hãy phân tích & đề xuất hướng chăm sóc:

    {row.to_dict()}

    Phân tích gồm:
    1️⃣ Tâm lý & hành vi khách hàng (tôn giáo, sở thích, khu vực, hôn nhân...).
    2️⃣ Dự đoán nhu cầu tài chính và xu hướng sản phẩm.
    3️⃣ Đề xuất sản phẩm/dịch vụ phù hợp (gửi tiết kiệm, vay, thẻ, bảo hiểm, QR, Mobile Banking...).
    4️⃣ Đề xuất chiến lược tiếp cận & chăm sóc phù hợp.

    Mức sáng tạo: {creativity}
    """

    gemini_text, openai_text = None, None

    if mode == "Hybrid":
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = []
            if gemini_key: futures.append(pool.submit(call_gemini, prompt, gemini_key, gemini_model))
            if openai_key: futures.append(pool.submit(call_openai, prompt, openai_key, openai_model, creativity))

            for f in concurrent.futures.as_completed(futures, timeout=60):
                try:
                    r = f.result(timeout=60)
                    if "Gemini lỗi" not in r and "OpenAI lỗi" not in r:
                        if not gemini_text: gemini_text = r
                        else: openai_text = r
                except Exception as e:
                    st.warning(f"⚠️ Một AI bị lỗi hoặc quá hạn: {e}")

        if not gemini_text and not openai_text:
            return "⚠️ Không có phản hồi từ cả hai AI."

        if gemini_text and openai_text:
            merge_prompt = f"""
            🔸 Gemini:
            {gemini_text}

            🔹 OpenAI:
            {openai_text}

            Hãy hợp nhất hai bản thành đánh giá cuối cùng, chọn nội dung hợp lý & thực tế nhất.
            """
            return call_openai(merge_prompt, openai_key, openai_model, 0.7)
        else:
            return gemini_text or openai_text

    elif mode == "Gemini":
        return call_gemini(prompt, gemini_key, gemini_model)
    elif mode == "OpenAI":
        return call_openai(prompt, openai_key, openai_model, creativity)

# ==============================
# 🧠 MAIN APP LOGIC
# ==============================
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="KhachHang")
        st.success(f"✅ Đã tải {len(df)} khách hàng từ file Excel.")
        st.dataframe(df, use_container_width=True)

        selected = st.multiselect("👥 Chọn khách hàng cần phân tích", df["Họ tên"].tolist(),
                                  default=[df["Họ tên"].iloc[0]])

        if st.button("🚀 Bắt đầu phân tích", type="primary", use_container_width=True):
            if (ai_mode == "Gemini" and not gemini_key) or \
               (ai_mode == "OpenAI" and not openai_key) or \
               (ai_mode == "Hybrid" and (not gemini_key or not openai_key)):
                st.error("⚠️ Vui lòng nhập API Key cho chế độ đã chọn.")
            else:
                with st.spinner("🧩 Đang phân tích dữ liệu khách hàng..."):
                    results = []
                    for _, row in df[df["Họ tên"].isin(selected)].iterrows():
                        st.markdown(f"<h3>👤 {row['Họ tên']}</h3>", unsafe_allow_html=True)
                        analysis = analyze_customer(row, ai_mode, creativity, gemini_key, gemini_model, openai_key, openai_model)
                        results.append({"Họ tên": row["Họ tên"], "Kết quả phân tích & tư vấn": analysis})
                        st.markdown(f"<div class='analysis-card'>{analysis}</div>", unsafe_allow_html=True)
                        time.sleep(0.3)

                    result_df = pd.DataFrame(results)
                    out = BytesIO()
                    result_df.to_excel(out, index=False, engine="openpyxl")
                    st.download_button("⬇️ Tải kết quả (Excel)", out.getvalue(),
                        file_name="AgriAI_phan_tich.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"⚠️ Lỗi đọc file: {e}")
else:
    st.info("⬆️ Hãy tải file Excel (sheet: KhachHang) để bắt đầu phân tích.")

# ==============================
# 🔚 FOOTER
# ==============================
st.markdown("""
<hr>
<div style='text-align:center;color:white;background:#AE1C3F;padding:10px;border-radius:12px;'>
© 2025 Agribank Training & Development — <b>AgriAI CRM PRO</b><br>
Phát triển bởi Bộ phận CNTT Agribank • Tích hợp AI lai (Gemini + GPT-5)
</div>
""", unsafe_allow_html=True)
