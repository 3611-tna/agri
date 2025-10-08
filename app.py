# ======================================================
# 🌾 AgriAI CRM PRO 4.4.0 – Phân tích chuyên sâu & thực tiễn
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from google import genai
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import base64

st.set_page_config(page_title="AgriAI CRM Pro 4.4.0", layout="wide", page_icon="🌾")

# Khởi tạo session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_customer_context" not in st.session_state:
    st.session_state.current_customer_context = None
if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = ""
if "detailed_metrics" not in st.session_state:
    st.session_state.detailed_metrics = {}

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
.metric-box {
    background:#f8f9fa;padding:15px;border-radius:8px;
    border:2px solid #AE1C3F;margin:10px 0;
}
.chat-message {
    padding:12px;border-radius:8px;margin-bottom:10px;
}
.user-message {
    background:#f0f0f0;border-left:4px solid #AE1C3F;
}
.ai-message {
    background:#fff5f7;border-left:4px solid #D72638;
}
.footer {text-align:center;color:#777;margin-top:40px;}
</style>
""", unsafe_allow_html=True)

# Hiển thị logo
try:
    with open("logo.png", "rb") as f:
        logo_data = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{logo_data}" style="width:180px;margin-bottom:20px;">',
            unsafe_allow_html=True
        )
except:
    pass

st.title("🌾 AgriAI CRM PRO 4.4.0 – Phân tích chuyên sâu & thực tiễn")
st.caption("Phân tích chi tiết dựa trên số liệu thực tế, tư vấn chính xác với AI có ngữ cảnh đầy đủ")

# ---------- CẤU HÌNH ----------
with st.expander("⚙️ Cấu hình Gemini"):
    gemini_key = st.text_input("🔸 Gemini API Key", type="password")
    gemini_model = st.selectbox("🔹 Model Gemini", ["gemini-2.0-flash-exp", "gemini-2.0-flash", "gemini-1.5-flash"])
    creativity = st.slider("🎨 Mức sáng tạo (0 - 2)", 0.0, 2.0, 0.7, 0.1)

uploaded = st.file_uploader("📥 Tải file Excel (sheet KhachHang)", type=["xlsx"])

# ---------- HÀM CỐT LÕI ----------
def call_gemini(prompt, key, model, creativity):
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=model, contents=prompt, config={"temperature":creativity})
        return resp.text.strip()
    except Exception as e:
        return f"⚠️ Lỗi Gemini: {e}"

def call_gemini_with_history(user_msg, key, model, creativity, context="", history=[]):
    """Gọi Gemini với lịch sử hội thoại và ngữ cảnh đầy đủ"""
    try:
        client = genai.Client(api_key=key)
        
        system_prompt = f"""Bạn là chuyên viên tư vấn khách hàng NGÂN HÀNG AGRIBANK có 15 năm kinh nghiệm thực tế.

NGUYÊN TẮC TƯ VẤN:
1. Dựa trên SỐ LIỆU CỤ THỂ đã được phân tích (thu nhập, số dư gửi/vay, điểm đánh giá)
2. Đề xuất phải THỰC TẾ, khả thi với sản phẩm Agribank hiện có
3. Nếu thiếu thông tin quan trọng (tuổi, nghề nghiệp, mục tiêu tài chính, tình trạng gia đình), HÃY HỎI NGƯỜI DÙNG
4. Trả lời NGẮN GỌN, TẬP TRUNG vào hành động cụ thể

THÔNG TIN KHÁCH HÀNG VÀ PHÂN TÍCH:
{context}

Dựa trên lịch sử hội thoại và thông tin trên, hãy trả lời câu hỏi một cách chính xác và thực tiễn."""

        full_prompt = system_prompt + "\n\nLỊCH SỬ HỘI THOẠI:\n"
        for msg in history[-8:]:
            full_prompt += f"{msg['role']}: {msg['content']}\n"
        full_prompt += f"\nUser: {user_msg}\nAssistant:"
        
        resp = client.models.generate_content(
            model=model, 
            contents=full_prompt, 
            config={"temperature": creativity}
        )
        return resp.text.strip()
    except Exception as e:
        return f"⚠️ Lỗi Gemini: {e}"

def calc_detailed_scores(row):
    """Tính toán chi tiết các chỉ số với lý do cụ thể"""
    try:
        # Lấy dữ liệu và chuẩn hóa
        thu_nhap = float(str(row.get("Thu nhập", 0)).replace(",", "").replace(".", ""))
        gui = float(str(row.get("Số dư tiền gửi", 0)).replace(",", "").replace(".", ""))
        vay = float(str(row.get("Số dư tiền vay", 0)).replace(",", "").replace(".", ""))
        dich_vu = str(row.get("Dịch vụ đang dùng", ""))
        khu_vuc = str(row.get("Khu vực", ""))
        
        # Tính các chỉ số chi tiết
        metrics = {}
        
        # 1. RỦI RO (0-100)
        ty_le_vay_thu_nhap = (vay / (thu_nhap + 1)) * 100 if thu_nhap > 0 else 0
        rui_ro_base = min(100, ty_le_vay_thu_nhap * 0.6)
        
        # Điều chỉnh dựa trên mức vay tuyệt đối
        if vay > 500_000_000:  # Vay trên 500 triệu
            rui_ro_base += 10
        elif vay > 1_000_000_000:  # Vay trên 1 tỷ
            rui_ro_base += 20
            
        rui_ro = min(100, max(0, rui_ro_base))
        
        metrics['rui_ro'] = round(rui_ro, 1)
        metrics['ty_le_vay_thu_nhap'] = round(ty_le_vay_thu_nhap, 1)
        metrics['muc_vay'] = vay
        
        # Đánh giá mức rủi ro
        if rui_ro < 30:
            metrics['danh_gia_rui_ro'] = "THẤP - An toàn"
        elif rui_ro < 60:
            metrics['danh_gia_rui_ro'] = "TRUNG BÌNH - Cần theo dõi"
        else:
            metrics['danh_gia_rui_ro'] = "CAO - Cần xử lý ưu tiên"
        
        # 2. TIỀM NĂNG (0-100)
        tong_tai_san = thu_nhap + gui
        tiem_nang_base = min(100, (tong_tai_san / 1_000_000) * 10)
        
        # Thưởng điểm cho khách có tiền gửi cao
        if gui > 100_000_000:  # Gửi trên 100 triệu
            tiem_nang_base += 10
        if gui > 500_000_000:  # Gửi trên 500 triệu
            tiem_nang_base += 15
            
        tiem_nang = min(100, tiem_nang_base)
        
        metrics['tiem_nang'] = round(tiem_nang, 1)
        metrics['tong_tai_san'] = tong_tai_san
        metrics['tien_gui'] = gui
        
        # Đánh giá tiềm năng
        if tiem_nang < 40:
            metrics['danh_gia_tiem_nang'] = "THẤP - Khách hàng bình thường"
        elif tiem_nang < 70:
            metrics['danh_gia_tiem_nang'] = "TRUNG BÌNH - Có cơ hội phát triển"
        else:
            metrics['danh_gia_tiem_nang'] = "CAO - Khách hàng VIP, ưu tiên chăm sóc"
        
        # 3. GẮN BÓ (0-100)
        so_dich_vu = len([dv for dv in dich_vu.split(",") if dv.strip()])
        gan_bo = 40 + (so_dich_vu * 10)
        
        # Điều chỉnh theo khu vực
        if "Thành phố" in khu_vuc or "TP" in khu_vuc:
            gan_bo += 10
            metrics['khu_vuc_bonus'] = "Thành phố (+10 điểm)"
        else:
            metrics['khu_vuc_bonus'] = "Nông thôn/khác"
        
        gan_bo = min(100, gan_bo)
        
        metrics['gan_bo'] = round(gan_bo, 1)
        metrics['so_dich_vu'] = so_dich_vu
        metrics['dich_vu_su_dung'] = dich_vu
        
        # Đánh giá gắn bó
        if gan_bo < 50:
            metrics['danh_gia_gan_bo'] = "YẾU - Nguy cơ rời bỏ cao"
        elif gan_bo < 70:
            metrics['danh_gia_gan_bo'] = "TRUNG BÌNH - Cần tăng cường tương tác"
        else:
            metrics['danh_gia_gan_bo'] = "TỐT - Khách hàng trung thành"
        
        return metrics
    except Exception as e:
        return {
            'rui_ro': 0, 'tiem_nang': 0, 'gan_bo': 0,
            'danh_gia_rui_ro': 'Không xác định',
            'danh_gia_tiem_nang': 'Không xác định',
            'danh_gia_gan_bo': 'Không xác định'
        }

def suggest_action_with_reason(metrics, churn_prob, loan_prob):
    """Đề xuất hành động với lý do cụ thể dựa trên số liệu"""
    acts = []
    
    # Phân tích rủi ro rời bỏ
    if churn_prob > 0.7:
        acts.append({
            'priority': '🔴 KHẨN CẤP',
            'action': 'Ngăn chặn rời bỏ khách hàng',
            'reason': f"Xác suất rời bỏ {round(churn_prob*100,1)}% - Điểm gắn bó chỉ {metrics['gan_bo']}",
            'steps': [
                'Liên hệ trong vòng 48h để thăm hỏi',
                'Đề xuất ưu đãi lãi suất tiền gửi +0.3%/năm',
                'Tặng quà tri ân khách hàng thân thiết',
                'Mời tham gia chương trình khách hàng VIP'
            ]
        })
    
    # Phân tích cơ hội cho vay
    if loan_prob > 0.7:
        thu_nhap = metrics.get('tong_tai_san', 0)
        acts.append({
            'priority': '🟢 CƠ HỘI',
            'action': 'Khai thác nhu cầu vay vốn',
            'reason': f"Xác suất vay thêm {round(loan_prob*100,1)}% - Thu nhập {thu_nhap:,.0f} VNĐ",
            'steps': [
                f'Đề xuất gói vay tiêu dùng lãi suất ưu đãi 7.5%/năm',
                f'Hạn mức tối đa: {thu_nhap * 3:,.0f} VNĐ (3 lần thu nhập)',
                'Thời gian xét duyệt nhanh 24h',
                'Không cần tài sản đảm bảo với hạn mức dưới 500 triệu'
            ]
        })
    
    # Phân tích tiềm năng tài chính
    if metrics['tiem_nang'] > 80:
        acts.append({
            'priority': '⭐ ƯU TIÊN',
            'action': 'Nâng cấp lên khách hàng VIP',
            'reason': f"Điểm tiềm năng {metrics['tiem_nang']} - Tiền gửi {metrics['tien_gui']:,.0f} VNĐ",
            'steps': [
                'Chuyển sang quản lý bởi CBTD Senior',
                'Mở gói Tiết kiệm Plus lãi suất 6.8%/năm',
                'Tư vấn chứng chỉ quỹ, trái phiếu doanh nghiệp',
                'Tặng gói bảo hiểm sức khỏe trị giá 5 triệu/năm'
            ]
        })
    
    # Phân tích mức độ gắn bó
    if metrics['gan_bo'] < 50:
        acts.append({
            'priority': '🟡 QUAN TRỌNG',
            'action': 'Tăng cường tương tác và gắn bó',
            'reason': f"Chỉ sử dụng {metrics['so_dich_vu']} dịch vụ - Điểm gắn bó {metrics['gan_bo']}",
            'steps': [
                'Giới thiệu thêm 2-3 dịch vụ phù hợp (Mobile Banking, Visa)',
                'Mời tham gia webinar tài chính hàng tháng',
                'Gửi tin nhắn chăm sóc định kỳ (2 lần/tháng)',
                'Tặng voucher giảm giá 20% phí chuyển khoản'
            ]
        })
    
    # Trường hợp ổn định
    if not acts:
        acts.append({
            'priority': '✅ ỔN ĐỊNH',
            'action': 'Duy trì quan hệ hiện tại',
            'reason': f"Khách hàng ổn định - Rủi ro {metrics['rui_ro']}, Gắn bó {metrics['gan_bo']}",
            'steps': [
                'Chăm sóc định kỳ 1 lần/quý',
                'Thông báo các chương trình khuyến mãi mới',
                'Thu thập feedback về chất lượng dịch vụ',
                'Duy trì lãi suất ưu đãi hiện tại'
            ]
        })
    
    return acts

def export_excel(name, summary, df_scores, metrics):
    """Xuất báo cáo chi tiết ra Excel"""
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        # Sheet 1: Tổng quan
        pd.DataFrame({
            "Khách hàng": [name], 
            "Phân tích": [summary]
        }).to_excel(writer, sheet_name="TongQuan", index=False)
        
        # Sheet 2: Điểm số
        df_scores.to_excel(writer, sheet_name="DiemSo", index=False)
        
        # Sheet 3: Chi tiết số liệu
        metrics_df = pd.DataFrame([
            ["Thu nhập + Tiền gửi", f"{metrics.get('tong_tai_san', 0):,.0f} VNĐ"],
            ["Số dư vay", f"{metrics.get('muc_vay', 0):,.0f} VNĐ"],
            ["Tỷ lệ vay/thu nhập", f"{metrics.get('ty_le_vay_thu_nhap', 0):.1f}%"],
            ["Số dịch vụ sử dụng", metrics.get('so_dich_vu', 0)],
            ["Khu vực", metrics.get('khu_vuc_bonus', '')],
            ["Đánh giá rủi ro", metrics.get('danh_gia_rui_ro', '')],
            ["Đánh giá tiềm năng", metrics.get('danh_gia_tiem_nang', '')],
            ["Đánh giá gắn bó", metrics.get('danh_gia_gan_bo', '')]
        ], columns=["Chỉ tiêu", "Giá trị"])
        metrics_df.to_excel(writer, sheet_name="ChiTietSoLieu", index=False)
    
    return out.getvalue()

def train_predictive_model(df):
    scaler = MinMaxScaler()
    X = df[["Điểm Rủi ro","Điểm Tiềm năng","Điểm Gắn bó"]]
    X_scaled = scaler.fit_transform(X)

    churn_labels = np.where(df["Điểm Gắn bó"] < 50, 1, 0)
    loan_labels = np.where(df["Điểm Tiềm năng"] > 70, 1, 0)

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
    
    # Tính điểm cho tất cả khách hàng
    all_metrics = [calc_detailed_scores(row) for _, row in df.iterrows()]
    df["Điểm Rủi ro"] = [m['rui_ro'] for m in all_metrics]
    df["Điểm Tiềm năng"] = [m['tiem_nang'] for m in all_metrics]
    df["Điểm Gắn bó"] = [m['gan_bo'] for m in all_metrics]
    
    st.dataframe(df, use_container_width=True)

    # Huấn luyện mô hình
    churn_model, loan_model, scaler = train_predictive_model(df)

    selected = st.selectbox("👤 Chọn khách hàng để phân tích", df["Họ tên"].tolist())
    cust = df[df["Họ tên"] == selected].iloc[0].to_dict()

    if st.button("🚀 Phân tích chuyên sâu & dự báo hành động"):
        # Tính toán chi tiết
        metrics = calc_detailed_scores(cust)
        st.session_state.detailed_metrics = metrics
        
        r, t, g = metrics['rui_ro'], metrics['tiem_nang'], metrics['gan_bo']
        features = np.array([[r, t, g]])
        X_scaled = scaler.transform(features)
        churn_prob = churn_model.predict_proba(X_scaled)[0][1]
        loan_prob = loan_model.predict_proba(X_scaled)[0][1]

        # Hiển thị số liệu chi tiết
        st.markdown("### 📊 SỐ LIỆU PHÂN TÍCH CHI TIẾT")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-box'>
            <h4>🔴 RỦI RO: {r}</h4>
            <p><b>{metrics['danh_gia_rui_ro']}</b></p>
            <p>Tỷ lệ vay/thu nhập: <b>{metrics['ty_le_vay_thu_nhap']:.1f}%</b></p>
            <p>Số dư vay: <b>{metrics['muc_vay']:,.0f} VNĐ</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-box'>
            <h4>💎 TIỀM NĂNG: {t}</h4>
            <p><b>{metrics['danh_gia_tiem_nang']}</b></p>
            <p>Tổng tài sản: <b>{metrics['tong_tai_san']:,.0f} VNĐ</b></p>
            <p>Tiền gửi: <b>{metrics['tien_gui']:,.0f} VNĐ</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-box'>
            <h4>🤝 GẮN BÓ: {g}</h4>
            <p><b>{metrics['danh_gia_gan_bo']}</b></p>
            <p>Số dịch vụ: <b>{metrics['so_dich_vu']}</b></p>
            <p>Khu vực: <b>{metrics['khu_vuc_bonus']}</b></p>
            </div>
            """, unsafe_allow_html=True)

        # Dự báo
        st.markdown(f"""
        ### 🎯 DỰ BÁO HÀNH VI
        - **Xác suất rời bỏ:** {round(churn_prob*100,1)}% {'🔴 (CAO)' if churn_prob > 0.7 else '🟢 (THẤP)'}
        - **Xác suất vay thêm:** {round(loan_prob*100,1)}% {'✅ (CÓ CƠ HỘI)' if loan_prob > 0.7 else '⚪ (Thấp)'}
        """)

        # Đề xuất hành động
        actions = suggest_action_with_reason(metrics, churn_prob, loan_prob)
        
        prompt = f"""
Bạn là chuyên viên tư vấn Agribank. Dựa trên SỐ LIỆU CỤ THỂ sau:

THÔNG TIN KHÁCH HÀNG:
{cust}

SỐ LIỆU PHÂN TÍCH:
- Rủi ro: {r} ({metrics['danh_gia_rui_ro']})
- Tiềm năng: {t} ({metrics['danh_gia_tiem_nang']})
- Gắn bó: {g} ({metrics['danh_gia_gan_bo']})
- Tỷ lệ vay/thu nhập: {metrics['ty_le_vay_thu_nhap']:.1f}%
- Số dư vay: {metrics['muc_vay']:,.0f} VNĐ
- Tổng tài sản: {metrics['tong_tai_san']:,.0f} VNĐ
- Số dịch vụ: {metrics['so_dich_vu']}
- Xác suất rời bỏ: {round(churn_prob*100,1)}%
- Xác suất vay thêm: {round(loan_prob*100,1)}%

Hãy viết BẢN PHÂN TÍCH NGẮN GỌN (500-700 từ) gồm:

**1. TỔNG QUAN TÀI CHÍNH** (dựa trên số liệu thu nhập, tiền gửi, tiền vay)
**2. ĐÁNH GIÁ RỦI RO** (giải thích tỷ lệ vay/thu nhập và mức độ rủi ro)
**3. CƠ HỘI PHÁT TRIỂN** (dựa trên tiềm năng tài chính)
**4. KẾ HOẠCH HÀNH ĐỘNG 3 THÁNG** (cụ thể, có deadline)

Viết THỰC TẾ, DỄ HIỂU, TẬP TRUNG vào số liệu và hành động.
"""
        ai_text = call_gemini(prompt, gemini_key, gemini_model, creativity)

        # Lưu ngữ cảnh cho chat
        st.session_state.current_customer_context = cust
        st.session_state.analysis_context = f"""
KHÁCH HÀNG: {selected}
THÔNG TIN: {cust}

SỐ LIỆU PHÂN TÍCH CHI TIẾT:
- Điểm Rủi ro: {r}/100 - {metrics['danh_gia_rui_ro']}
  + Tỷ lệ vay/thu nhập: {metrics['ty_le_vay_thu_nhap']:.1f}%
  + Số dư vay: {metrics['muc_vay']:,.0f} VNĐ
  
- Điểm Tiềm năng: {t}/100 - {metrics['danh_gia_tiem_nang']}
  + Tổng tài sản: {metrics['tong_tai_san']:,.0f} VNĐ
  + Tiền gửi: {metrics['tien_gui']:,.0f} VNĐ
  
- Điểm Gắn bó: {g}/100 - {metrics['danh_gia_gan_bo']}
  + Số dịch vụ: {metrics['so_dich_vu']}
  + Khu vực: {metrics['khu_vuc_bonus']}

DỰ BÁO:
- Xác suất rời bỏ: {round(churn_prob*100,1)}%
- Xác suất vay thêm: {round(loan_prob*100,1)}%

PHÂN TÍCH AI:
{ai_text}
"""

        st.markdown(f"<div class='analysis-box'><h3>📌 PHÂN TÍCH KHÁCH HÀNG: {selected}</h3>{ai_text}</div>", unsafe_allow_html=True)

        # Hiển thị kế hoạch hành động
        st.markdown("### 🎯 KẾ HOẠCH HÀNH ĐỘNG CỤ THỂ")
        for idx, act in enumerate(actions, 1):
            with st.expander(f"{act['priority']} - {act['action']}", expanded=(idx==1)):
                st.markdown(f"**Lý do:** {act['reason']}")
                st.markdown("**Các bước thực hiện:**")
                for step in act['steps']:
                    st.markdown(f"- {step}")

        # Xuất báo cáo
        df_scores = pd.DataFrame({
            "Chỉ tiêu": ["Rủi ro","Tiềm năng","Gắn bó","Xác suất rời bỏ (%)","Xác suất vay thêm (%)"],
            "Giá trị": [r, t, g, round(churn_prob*100,1), round(loan_prob*100,1)],
            "Đánh giá": [
                metrics['danh_gia_rui_ro'],
                metrics['danh_gia_tiem_nang'],
                metrics['danh_gia_gan_bo'],
                "CAO" if churn_prob > 0.7 else "THẤP",
                "CÓ CƠ HỘI" if loan_prob > 0.7 else "Thấp"
            ]
        })
        
        summary_text = f"""
KHÁCH HÀNG: {selected}

ĐIỂM SỐ HỆ THỐNG:
- Rủi ro: {r}/100 - {metrics['danh_gia_rui_ro']}
- Tiềm năng: {t}/100 - {metrics['danh_gia_tiem_nang']}
- Gắn bó: {g}/100 - {metrics['danh_gia_gan_bo']}

DỰ BÁO:
- Xác suất rời bỏ: {round(churn_prob*100,1)}%
- Xác suất vay thêm: {round(loan_prob*100,1)}%

PHÂN TÍCH AI:
{ai_text}

KẾ HOẠCH HÀNH ĐỘNG:
{chr(10).join([f"{act['priority']} - {act['action']}: {act['reason']}" for act in actions])}
"""
        
        excel = export_excel(selected, summary_text, df_scores, metrics)
        st.download_button(
            "📊 Tải báo cáo chi tiết (Excel)", 
            excel, 
            file_name=f"{selected}_AgriAI_Insight_ChiTiet.xlsx",
            use_container_width=True
        )

    # ========== PHẦN CHAT AI ==========
    st.markdown("---")
    st.subheader("💬 Chat với AI Tư vấn")
    
    if st.session_state.analysis_context:
        st.info("✅ AI đã có đầy đủ thông tin khách hàng và số liệu phân tích. Hãy hỏi thêm hoặc cung cấp thông tin bổ sung (tuổi, nghề nghiệp, mục tiêu tài chính...)")
        
        # Gợi ý câu hỏi
        with st.expander("💡 Gợi ý câu hỏi"):
            st.markdown("""
            - "Khách hàng này 35 tuổi, làm nông nghiệp, có nên đề xuất vay mở rộng sản xuất không?"
            - "Giải thích chi tiết tại sao điểm rủi ro cao?"
            - "Họ muốn gửi tiết kiệm dài hạn, nên chọn kỳ hạn nào?"
            - "Con của khách đang học đại học, có gói vay học phí nào không?"
            - "Khách hàng này có quan hệ tốt với chi nhánh, điều chỉnh kế hoạch như thế nào?"
            """)
    else:
        st.warning("⚠️ Hãy phân tích một khách hàng trước để AI có đủ ngữ cảnh tư vấn.")
    
    # Hiển thị lịch sử chat
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "User":
                st.markdown(f"<div class='chat-message user-message'><b>🧑 Bạn:</b> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message ai-message'><b>🤖 AI:</b> {msg['content']}</div>", unsafe_allow_html=True)
    
    # Form chat với auto-clear
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5,1])
        with col1:
            user_input = st.text_input(
                "💭 Nhập câu hỏi hoặc thông tin bổ sung:", 
                key="chat_input_field",
                placeholder="VD: Khách hàng 42 tuổi, kinh doanh nông sản, muốn mở rộng quy mô..."
            )
        with col2:
            send_btn = st.form_submit_button("Gửi", use_container_width=True)
    
    col_clear1, col_clear2 = st.columns([5,1])
    with col_clear2:
        if st.button("🗑️ Xóa lịch sử", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if send_btn and user_input and user_input.strip():
        # Thêm tin nhắn của user
        st.session_state.chat_history.append({"role": "User", "content": user_input.strip()})
        
        # Gọi AI với context và history
        ai_response = call_gemini_with_history(
            user_input.strip(), 
            gemini_key, 
            gemini_model, 
            creativity,
            context=st.session_state.analysis_context,
            history=st.session_state.chat_history
        )
        
        # Thêm phản hồi AI
        st.session_state.chat_history.append({"role": "Assistant", "content": ai_response})
        st.rerun()

else:
    st.info("⬆️ Hãy tải file Excel khách hàng (sheet KhachHang) để bắt đầu phân tích.")
    
    # Hướng dẫn sử dụng
    with st.expander("📖 Hướng dẫn sử dụng"):
        st.markdown("""
        ### Cách sử dụng AgriAI CRM Pro 4.4.0:
        
        **Bước 1:** Cấu hình Gemini API Key
        - Lấy API key tại: https://aistudio.google.com/apikey
        
        **Bước 2:** Tải file Excel khách hàng
        - File cần có sheet tên "KhachHang"
        - Các cột cần thiết: Họ tên, Thu nhập, Số dư tiền gửi, Số dư tiền vay, Dịch vụ đang dùng, Khu vực
        
        **Bước 3:** Chọn khách hàng và phân tích
        - Xem số liệu chi tiết và đánh giá
        - Nhận kế hoạch hành động cụ thể
        - Tải báo cáo Excel đầy đủ
        
        **Bước 4:** Chat với AI để tư vấn sâu hơn
        - Cung cấp thêm thông tin về khách hàng
        - Hỏi về các sản phẩm phù hợp
        - Yêu cầu điều chỉnh kế hoạch
        
        ### Số liệu phân tích dựa trên:
        
        **Điểm Rủi ro (0-100):**
        - Tỷ lệ vay/thu nhập
        - Mức vay tuyệt đối
        - Đánh giá: THẤP (<30), TRUNG BÌNH (30-60), CAO (>60)
        
        **Điểm Tiềm năng (0-100):**
        - Thu nhập + Số dư tiền gửi
        - Mức gửi tiết kiệm
        - Đánh giá: THẤP (<40), TRUNG BÌNH (40-70), CAO (>70)
        
        **Điểm Gắn bó (0-100):**
        - Số lượng dịch vụ đang sử dụng
        - Khu vực sinh sống (thành phố +10 điểm)
        - Đánh giá: YẾU (<50), TRUNG BÌNH (50-70), TỐT (>70)
        
        ### Mô hình dự báo:
        - Sử dụng Logistic Regression
        - Xác suất rời bỏ: dựa trên điểm gắn bó
        - Xác suất vay thêm: dựa trên điểm tiềm năng
        """)

st.markdown("<div class='footer'>© 2025 Agribank AI | AgriAI CRM Pro 4.4.0 - Phân tích chuyên sâu & thực tiễn</div>", unsafe_allow_html=True)
