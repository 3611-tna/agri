# ======================================================
# 🌾 AgriAI CRM PRO 4.1 – Phân tích & Tư vấn chuyên sâu khách hàng Agribank
# ------------------------------------------------------
# Tác giả: Shine | Agribank Tây Nghệ An | 2025
# Phiên bản: v4.1 (Gemini-only)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from google import genai
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AgriAI CRM Pro 4.1", layout="wide", page_icon="🌾")

# ---------- STYLE ----------
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
    background:white;padding:16px;border-radius:10px;
    border-left:6px solid #AE1C3F;
    box-shadow:0 2px 10px rgba(0,0,0,0.1);
    margin-bottom:20px;
}
.footer {text-align:center;color:#777;margin-top:40px;}
</style>
""", unsafe_allow_html=True)

st.title("🌾 AgriAI CRM PRO 4.1 – Phân tích chuyên sâu & Đề xuất hành động")
st.caption("Phiên bản nghiệp vụ thực tiễn – Phân tích định lượng + định tính, xây dựng chiến lược chăm sóc khách hàng Agribank")

# ---------- CONFIG ----------
with st.expander("⚙️ Cấu hình Gemini AI"):
    gemini_key = st.text_input("🔸 Gemini API Key", type="password")
    gemini_model = st.selectbox("🔹 Model Gemini", ["gemini-2.0-flash", "gemini-1.5-flash"])
    creativity = st.slider("🎨 Mức sáng tạo (0 - 2)", 0.0, 2.0, 0.8, 0.1)

uploaded = st.file_uploader("📥 Tải file Excel (sheet KhachHang)", type=["xlsx"])

# ---------- HÀM PHỤ TRỢ ----------
def call_gemini(prompt, key, model, creativity):
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=model, contents=prompt, generation_config={"temperature":creativity})
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

def suggest_action(r, t, g):
    acts = []
    if t > 80: acts.append("👉 Đề xuất gói tiết kiệm dài hạn hoặc đầu tư linh hoạt.")
    if r > 50: acts.append("⚠️ Theo dõi dư nợ, xem xét tái cơ cấu / gia hạn.")
    if g < 50: acts.append("💬 Tăng tương tác, tổ chức CSKH định kỳ.")
    if t < 40 and g < 50: acts.append("📞 Tiếp cận lại, gợi mở ưu đãi nhỏ để khôi phục quan hệ.")
    if not acts: acts.append("✅ Khách hàng ổn định – duy trì chăm sóc định kỳ.")
    return acts

def export_excel(name, summary, df_scores):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame({"Khách hàng":[name], "Phân tích":[summary]}).to_excel(writer, sheet_name="PhanTich", index=False)
        df_scores.to_excel(writer, sheet_name="Scores", index=False)
    return out.getvalue()

# ---------- MAIN ----------
if uploaded:
    df = pd.read_excel(uploaded, sheet_name="KhachHang")
    st.success(f"✅ Đã tải {len(df)} khách hàng.")

    scores = [calc_scores(row) for _, row in df.iterrows()]
    df["Điểm Rủi ro"], df["Điểm Tiềm năng"], df["Điểm Gắn bó"] = zip(*scores)
    st.dataframe(df, use_container_width=True)

    # ========== CHỌN KHÁCH HÀNG ==========
    selected = st.selectbox("👤 Chọn khách hàng cần phân tích", df["Họ tên"].tolist())
    cust = df[df["Họ tên"] == selected].iloc[0].to_dict()

    # ========== CHẾ ĐỘ SO SÁNH ==========
    compare_mode = st.checkbox("🔁 So sánh với khách hàng khác")
    if compare_mode:
        compare_with = st.selectbox("Chọn khách hàng so sánh", [x for x in df["Họ tên"].tolist() if x != selected])
        cust2 = df[df["Họ tên"] == compare_with].iloc[0].to_dict()
        r1,t1,g1 = calc_scores(cust)
        r2,t2,g2 = calc_scores(cust2)
        comp_df = pd.DataFrame({
            "Chỉ tiêu":["Rủi ro","Tiềm năng","Gắn bó"],
            selected:[r1,t1,g1],
            compare_with:[r2,t2,g2]
        })
        st.subheader("📊 So sánh giữa hai khách hàng")
        st.dataframe(comp_df, use_container_width=False)
        fig, ax = plt.subplots()
        sns.barplot(data=comp_df.melt(id_vars="Chỉ tiêu"), x="Chỉ tiêu", y="value", hue="variable", ax=ax)
        ax.set_title("So sánh điểm giữa hai khách hàng")
        st.pyplot(fig)

    # ========== PHÂN TÍCH CHUYÊN SÂU ==========
    if st.button("🚀 Phân tích chuyên sâu & đề xuất hành động"):
        r,t,g = calc_scores(cust)
        actions = suggest_action(r,t,g)

        # --- PROMPT CHUYÊN NGHIỆP NGHIỆP VỤ AGRIBANK ---
        prompt = f"""
Bạn là chuyên gia phân tích khách hàng Agribank có hơn 15 năm kinh nghiệm.
Dưới đây là dữ liệu khách hàng thực tế:
{cust}

Các chỉ số hệ thống:
- Điểm rủi ro: {r}
- Điểm tiềm năng: {t}
- Điểm gắn bó: {g}

Hãy viết bản PHÂN TÍCH CHUYÊN SÂU theo 4 phần:
1️⃣ **Tổng quan năng lực tài chính:** đánh giá thực tế, phân tích cơ cấu thu nhập, tiền gửi, dư nợ.
2️⃣ **Hành vi & tâm lý khách hàng:** mô tả phong cách, mức độ trung thành, yếu tố vùng miền, nghề nghiệp.
3️⃣ **Định hướng sản phẩm phù hợp:** chọn tối đa 3 sản phẩm Agribank (VD: Tiết kiệm bậc thang, vay tiêu dùng, bảo hiểm ABIC, QR POS...).
4️⃣ **Chiến lược chăm sóc & hành động đề xuất:** nêu cụ thể hành động mà CBTD nên làm trong 1–3 tháng tới.

Lưu ý:
- Không nói chung chung.
- Phải bám sát dữ liệu và chỉ số thực tế.
- Viết ngắn gọn, dễ đọc, như tư vấn nghiệp vụ thật.
"""
        ai_text = call_gemini(prompt, gemini_key, gemini_model, creativity)

        summary = f"""
### 📌 Phân tích chuyên sâu khách hàng **{selected}**
#### 🔢 Điểm hệ thống
- Rủi ro: {r} | Tiềm năng: {t} | Gắn bó: {g}

#### 💡 Nhận định chuyên gia (AI)
{ai_text}

#### 🎯 Hành động đề xuất
{chr(10).join(actions)}
"""
        st.markdown(f"<div class='analysis-box'>{summary}</div>", unsafe_allow_html=True)

        # Xuất Excel
        df_scores = pd.DataFrame({"Chỉ tiêu":["Rủi ro","Tiềm năng","Gắn bó"],"Điểm":[r,t,g]})
        excel = export_excel(selected, summary, df_scores)
        st.download_button("📊 Tải báo cáo chi tiết (Excel)", excel, file_name=f"{selected}_PhanTich_AgriAI.xlsx")

    # ========== PHÂN TÍCH NHÓM ==========
    st.subheader("📈 Phân tích tổng quan nhóm khách hàng")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df["Điểm Tiềm năng"], color="#AE1C3F", kde=True, ax=ax)
        ax.set_title("Phân bố Điểm Tiềm năng")
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.boxplot(df[["Điểm Rủi ro","Điểm Gắn bó"]], ax=ax2)
        ax2.set_title("So sánh Rủi ro & Gắn bó")
        st.pyplot(fig2)

else:
    st.info("⬆️ Hãy tải file Excel khách hàng (sheet KhachHang) để bắt đầu phân tích.")

# ---------- FOOTER ----------
st.markdown("<div class='footer'>© 2025 Agribank AI | AgriAI CRM Pro 4.1</div>", unsafe_allow_html=True)
