# ======================================================
# 🌾 AgriAI CRM PRO 4.2 – Phân tích, dự báo & tư vấn hành động thông minh
# ------------------------------------------------------
# Tác giả: Shine | Agribank Tây Nghệ An | 2025
# Mục tiêu: Phân tích khách hàng - chấm điểm - dự báo - gợi ý hành động cụ thể
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from google import genai
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# ---------- CẤU HÌNH GIAO DIỆN ----------
st.set_page_config(page_title="AgriAI CRM Pro 4.2", layout="wide", page_icon="🌾")

st.markdown("""
<style>
h1,h2,h3,h4 {color:#AE1C3F;}
.stButton>button {
    background:linear-gradient(90deg,#AE1C3F,#D72638);
    color:white;font-weight:bold;border:none;
    border-radius:8px;padding:0.5em 1.2em;
}
.stButton>button:hover {opacity:0.9;transform:scale(1.03);}
.analysis-box {
    background:white;padding:18px;border-radius:10px;
    border-left:6px solid #AE1C3F;
    box-shadow:0 2px 10px rgba(0,0,0,0.1);
    margin-bottom:25px;
}
.footer {text-align:center;color:#777;margin-top:40px;}
</style>
""", unsafe_allow_html=True)

st.title("🌾 AgriAI CRM PRO 4.2 – Dự báo & đề xuất hành động thông minh")
st.caption("Phân tích định lượng + định tính, dự báo khả năng rời bỏ và đề xuất hành động cụ thể cho CBTD Agribank.")

# ---------- CẤU HÌNH AI ----------
with st.expander("⚙️ Cấu hình Gemini"):
    gemini_key = st.text_input("🔸 Gemini API Key", type="password")
    gemini_model = st.selectbox("🔹 Model Gemini", ["gemini-2.0-flash", "gemini-1.5-flash"])
    creativity = st.slider("🎨 Mức sáng tạo (0 - 2)", 0.0, 2.0, 0.8, 0.1)

# ---------- UPLOAD FILE ----------
uploaded = st.file_uploader("📥 Tải file Excel (sheet KhachHang)", type=["xlsx"])

# ---------- HÀM PHỤ TRỢ ----------
def call_gemini(prompt, key, model, creativity):
    """Gọi Gemini AI để sinh nhận định chuyên sâu."""
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=model, contents=prompt, config={"temperature":creativity})
        return resp.text.strip()
    except Exception as e:
        return f"⚠️ Lỗi Gemini: {e}"

def calc_scores(row):
    """Tính điểm rủi ro, tiềm năng, gắn bó"""
    try:
        thu_nhap = float(str(row.get("Thu nhập", 0)).replace(",", "").replace(".", ""))
        gui = float(str(row.get("Số dư tiền gửi", 0)).replace(",", "").replace(".", ""))
        vay = float(str(row.get("Số dư tiền vay", 0)).replace(",", "").replace(".", ""))
        rui_ro = min(100, max(0, (vay / (thu_nhap + 1)) * 60))
        tiem_nang = min(100, (thu_nhap + gui) / 1_000_000 * 10)
        gan_bo = 40 + len(str(row.get("Dịch vụ đang dùng", "")).split(",")) * 10
        if "Thành phố" in str(row.get("Khu vực", "")): gan_bo += 10
        return round(rui_ro,1), round(tiem_nang,1), round(gan_bo,1)
    except:
        return 0,0,0

def suggest_action(r, t, g, churn_prob, loan_prob):
    """Đề xuất hành động cụ thể dựa trên phân tích + dự báo."""
    acts = []
    if churn_prob > 0.7:
        acts.append("⚠️ Cảnh báo: Khách hàng có nguy cơ rời bỏ cao. Cần gọi điện / CSKH trực tiếp.")
    if loan_prob > 0.7:
        acts.append("💰 Có tiềm năng vay thêm – tư vấn gói vay tiêu dùng linh hoạt hoặc vay sản xuất.")
    if t > 80:
        acts.append("📈 Đề xuất gói tiết kiệm dài hạn hoặc đầu tư linh hoạt.")
    if g < 50:
        acts.append("💬 Tăng tương tác qua chương trình tri ân / CSKH định kỳ.")
    if not acts:
        acts.append("✅ Khách hàng ổn định, duy trì chăm sóc định kỳ.")
    return acts

def export_excel(name, summary, df_scores):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame({"Khách hàng":[name], "Phân tích":[summary]}).to_excel(writer, sheet_name="PhanTich", index=False)
        df_scores.to_excel(writer, sheet_name="Scores", index=False)
    return out.getvalue()

# ---------- HÀM DỰ BÁO ----------
def train_predictive_model(df):
    """Huấn luyện nhanh logistic regression để dự báo churn & loan demand"""
    scaler = MinMaxScaler()
    X = df[["Điểm Rủi ro","Điểm Tiềm năng","Điểm Gắn bó"]]
    X_scaled = scaler.fit_transform(X)

    churn_labels = np.where(df["Điểm Gắn bó"] < 50, 1, 0)
    loan_labels = np.where(df["Điểm Tiềm năng"] > 70, 1, 0)

    churn_model = LogisticRegression().fit(X_scaled, churn_labels)
    loan_model = LogisticRegression().fit(X_scaled, loan_labels)

    return churn_model, loan_model, scaler

# ---------- CHƯƠNG TRÌNH CHÍNH ----------
if uploaded:
    df = pd.read_excel(uploaded, sheet_name="KhachHang")
    st.success(f"✅ Đã tải {len(df)} khách hàng.")

    # --- Tính điểm định lượng ---
    scores = [calc_scores(row) for _, row in df.iterrows()]
    df["Điểm Rủi ro"], df["Điểm Tiềm năng"], df["Điểm Gắn bó"] = zip(*scores)

    st.dataframe(df, use_container_width=True)

    # --- Huấn luyện mô hình dự báo ---
    churn_model, loan_model, scaler = train_predictive_model(df)

    # --- Phân tích khách hàng ---
    selected = st.selectbox("👤 Chọn khách hàng cần phân tích", df["Họ tên"].tolist())
    cust = df[df["Họ tên"] == selected].iloc[0].to_dict()

    if st.button("🚀 Phân tích chuyên sâu & Dự báo hành động"):
        r,t,g = calc_scores(cust)
        features = np.array([[r,t,g]])
        X_scaled = scaler.transform(features)

        churn_prob = churn_model.predict_proba(X_scaled)[0][1]
        loan_prob = loan_model.predict_proba(X_scaled)[0][1]
        actions = suggest_action(r,t,g,churn_prob,loan_prob)

        # --- Prompt chuyên sâu nghiệp vụ ---
        prompt = f"""
Bạn là chuyên gia tư vấn khách hàng Agribank có 15 năm kinh nghiệm.
Dưới đây là hồ sơ khách hàng:
{cust}

Chỉ số hệ thống:
- Điểm rủi ro: {r}
- Điểm tiềm năng: {t}
- Điểm gắn bó: {g}
- Xác suất rời bỏ: {round(churn_prob*100,2)}%
- Xác suất vay thêm: {round(loan_prob*100,2)}%

Viết bản PHÂN TÍCH NGHIỆP VỤ theo 4 phần:
1️⃣ Tổng quan tài chính và hành vi khách hàng
2️⃣ Phân tích rủi ro và xu hướng hành động
3️⃣ Dự báo tiềm năng sản phẩm phù hợp (ưu tiên các sản phẩm Agribank thực tế)
4️⃣ Kế hoạch hành động chi tiết cho CBTD trong 1–3 tháng

Viết ngắn gọn, súc tích, thực tế và có tính áp dụng cao.
"""
        ai_text = call_gemini(prompt, gemini_key, gemini_model, creativity)

        # --- Kết quả hiển thị ---
        summary = f"""
### 📌 Phân tích chuyên sâu khách hàng **{selected}**
#### 🔢 Điểm hệ thống
- Rủi ro: {r} | Tiềm năng: {t} | Gắn bó: {g}
- Xác suất rời bỏ: {round(churn_prob*100,2)}% | Xác suất vay thêm: {round(loan_prob*100,2)}%

#### 💡 Nhận định chuyên gia (AI)
{ai_text}

#### 🎯 Hành động đề xuất
{chr(10).join(actions)}
"""
        st.markdown(f"<div class='analysis-box'>{summary}</div>", unsafe_allow_html=True)

        df_scores = pd.DataFrame({
            "Chỉ tiêu":["Rủi ro","Tiềm năng","Gắn bó","Xác suất rời bỏ","Xác suất vay thêm"],
            "Giá trị":[r,t,g,round(churn_prob*100,2),round(loan_prob*100,2)]
        })
        excel = export_excel(selected, summary, df_scores)
        st.download_button("📊 Tải báo cáo chi tiết (Excel)", excel, file_name=f"{selected}_AgriAI_Insight.xlsx")

else:
    st.info("⬆️ Hãy tải file Excel khách hàng (sheet KhachHang) để bắt đầu phân tích.")

st.markdown("<div class='footer'>© 2025 Agribank AI | AgriAI CRM Pro 4.2 – Predictive Edition</div>", unsafe_allow_html=True)
