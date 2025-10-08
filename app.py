import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from google import genai

st.set_page_config(page_title="AgriAI CRM – Gemini Only", layout="wide", page_icon="🤖")

st.markdown("""
<style>
h1 {color:#AE1C3F; text-align:center;}
.analysis-box {background:white; padding:20px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

st.title("🌱 AgriAI CRM (Gemini Only)")

with st.expander("⚙️ Cấu hình"):
    gemini_key = st.text_input("🔸 Gemini API Key", type="password")
    gemini_model = st.selectbox("🔸 Model Gemini", ["gemini-2.0-flash", "gemini-1.5-flash"])

uploaded = st.file_uploader("📥 Tải file Excel (sheet KhachHang)", type=["xlsx"])

def call_gemini(prompt, key, model_name):
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=model_name, contents=prompt)
        return resp.text.strip()
    except Exception as e:
        return f"⚠️ Gemini lỗi: {e}"

def export_excel(name, insights, scores_df):
    # Kết hợp phân tích + điểm + bảng dữ liệu
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame({"Khách hàng":[name], "Phân tích & Tư vấn":[insights]}).to_excel(writer, sheet_name="Tư vấn", index=False)
        scores_df.to_excel(writer, sheet_name="Scores", index=False)
    return out.getvalue()

if uploaded:
    df = pd.read_excel(uploaded, sheet_name="KhachHang")
    st.success(f"Đã tải {len(df)} khách hàng")

    search = st.text_input("🔍 Tìm khách hàng")
    filtered = df[df["Họ tên"].str.contains(search, case=False, na=False)] if search else df
    st.dataframe(filtered, use_container_width=True)

    selected = st.selectbox("Chọn KH để phân tích", filtered["Họ tên"].tolist())
    cust = df[df["Họ tên"] == selected].iloc[0]

    if st.button("🚀 Phân tích bằng Gemini"):
        # 1. Gọi Gemini để phân tích về insight
        prompt = f"Dữ liệu khách hàng: {cust.to_dict()}\nHãy phân tích chi tiết & đề xuất sản phẩm phù hợp."
        insight = call_gemini(prompt, gemini_key, gemini_model)

        # 2. Tính score đơn giản (ví dụ: thu nhập + gửi – vay)
        try:
            thu_nhap = float(cust.get("Thu nhập", 0))
            so_du_gui = float(cust.get("Số dư tiền gửi", 0))
            so_du_vay = float(cust.get("Số dư tiền vay", 0))
        except:
            thu_nhap = so_du_gui = so_du_vay = 0

        score = thu_nhap + so_du_gui - so_du_vay
        df_scores = pd.DataFrame({
            "Chỉ tiêu": ["Thu nhập", "Tiền gửi", "Tiền vay", "Điểm rủi ro"],
            "Giá trị": [thu_nhap, so_du_gui, so_du_vay, score]
        })

        st.markdown(f"<div class='analysis-box'><b>🔍 Insight:</b><br>{insight}</div>", unsafe_allow_html=True)
        st.subheader("📊 Điểm & Chỉ số")
        st.dataframe(df_scores, use_container_width=False)

        # 3. Biểu đồ (nếu số liệu hợp lý)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(x=["Thu nhập", "Tiền gửi", "Tiền vay"], y=[thu_nhap, so_du_gui, so_du_vay], ax=ax, palette=["#AE1C3F","#4CAF50","#F44336"])
        ax.set_ylabel("VNĐ")
        ax.set_title("Cơ cấu tài chính")
        st.pyplot(fig)

        # 4. Xuất Excel
        excel_data = export_excel(selected, insight, df_scores)
        st.download_button("📊 Tải kết quả Excel", excel_data, file_name=f"{selected}_insight.xlsx")

else:
    st.info("Hãy tải file Excel khách hàng để bắt đầu")

