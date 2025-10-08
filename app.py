# ======================================================
# 🌾 AgriAI CRM PRO 4.2.1 – Bản sửa lỗi huấn luyện, ổn định & thực dụng
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from google import genai
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="AgriAI CRM Pro 4.2.1", layout="wide", page_icon="🌾")

# ---------- GIAO DIỆN ----------
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

st.title("🌾 AgriAI CRM PRO 4.2.1 – Dự báo & hành động thông minh (ổn định)")
st.caption("Tự động phân tích khách hàng, dự báo xu hướng & gợi ý hành động thực tiễn cho CBTD Agribank")

# ---------- CẤU HÌNH ----------
with st.expander("⚙️ Cấu hình Gemini"):
    gemini_key = st.text_input("🔸 Gemini API Key", type="password")
    gemini_model = st.selectbox("🔹 Model Gemini", ["gemini-2.0-flash", "gemini-1.5-flash"])
    creativity = st.slider("🎨 Mức sáng tạo (0 - 2)", 0.0, 2.0, 0.8, 0.1)

uploaded = st.file_uploader("📥 Tải file Excel (sheet KhachHang)", type=["xlsx"])

# ---------- HÀM CỐT LÕI ----------
def call_gemini(prompt, key, model, creativity):
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=model, contents=prompt, config={"temperature":creativity})
        return resp.text.strip()
    except Exception as e:
        return f"⚠️ Lỗi Gemini: {e}"

def calc_scores(row):
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
    acts = []
    if churn_prob > 0.7:
        acts.append("⚠️ Nguy cơ rời bỏ cao → cần CSKH hoặc ưu đãi duy trì.")
    if loan_prob > 0.7:
        acts.append("💰 Tiềm năng vay thêm → gợi ý gói vay tiêu dùng / sản xuất.")
    if t > 80:
        acts.append("📈 Đề xuất gói tiết kiệm linh hoạt hoặc đầu tư dài hạn.")
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

def train_predictive_model(df):
    scaler = MinMaxScaler()
    X = df[["Điểm Rủi ro","Điểm Tiềm năng","Điểm Gắn bó"]]
    X_scaled = scaler.fit_transform(X)

    churn_labels = np.where(df["Điểm Gắn bó"] < 50, 1, 0)
    loan_labels = np.where(df["Điểm Tiềm năng"] > 70, 1, 0)

    # Nếu dữ liệu có <2 lớp, tạo nhãn giả để tránh lỗi
    if len(np.unique(churn_labels)) < 2:
        churn_labels[0] = 1
    if len(np.unique(loan_labels)) < 2:
        loan_labels[-1] = 1

    churn_model = LogisticRegression().fit(X_scaled, churn_labels)
    loan_model = LogisticRegression().fit(X_scaled, loan_labels)

    return churn_model, loan_model, scaler

# ---------- CHƯƠNG TRÌNH CHÍNH ----------
if uploaded:
    df = pd.read_excel(uploaded, sheet_name="KhachHang")
    st.success(f"✅ Đã tải {len(df)} khách hàng.")
    scores = [calc_scores(row) for _, row in df.iterrows()]
    df["Điểm Rủi ro"], df["Điểm Tiềm năng"], df["Điểm Gắn bó"] = zip(*scores)
    st.dataframe(df, use_container_width=True)

    # Huấn luyện mô hình (ổn định)
    churn_model, loan_model, scaler = train_predictive_model(df)

    selected = st.selectbox("👤 Chọn khách hàng để phân tích", df["Họ tên"].tolist())
    cust = df[df["Họ tên"] == selected].iloc[0].to_dict()

    if st.button("🚀 Phân tích chuyên sâu & dự báo hành động"):
        r,t,g = calc_scores(cust)
        features = np.array([[r,t,g]])
        X_scaled = scaler.transform(features)
        churn_prob = churn_model.predict_proba(X_scaled)[0][1]
        loan_prob = loan_model.predict_proba(X_scaled)[0][1]

        actions = suggest_action(r,t,g,churn_prob,loan_prob)

        prompt = f"""
Bạn là chuyên viên tư vấn khách hàng Agribank có 15 năm kinh nghiệm.
Dưới đây là hồ sơ khách hàng:
{cust}

Điểm hệ thống:
- Rủi ro: {r}
- Tiềm năng: {t}
- Gắn bó: {g}
- Xác suất rời bỏ: {round(churn_prob*100,2)}%
- Xác suất vay thêm: {round(loan_prob*100,2)}%

Hãy viết bản PHÂN TÍCH NGHIỆP VỤ gồm:
1️⃣ Tổng quan tài chính & hành vi khách hàng.
2️⃣ Rủi ro & xu hướng hành động.
3️⃣ Gợi ý sản phẩm Agribank phù hợp.
4️⃣ Kế hoạch hành động cụ thể (trong 1–3 tháng).

Ngắn gọn, rõ ràng, chuyên nghiệp và thực tiễn.
"""
        ai_text = call_gemini(prompt, gemini_key, gemini_model, creativity)

        summary = f"""
### 📌 Phân tích khách hàng **{selected}**
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

st.markdown("<div class='footer'>© 2025 Agribank AI | AgriAI CRM Pro 4.2.1</div>", unsafe_allow_html=True)
