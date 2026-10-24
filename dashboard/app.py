
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

st.set_page_config(
    page_title="Delivery Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

NAVY = "#14263F"
BLUE = "#2563EB"
RED = "#EF4444"
GREEN = "#10B981"
MUTED = "#64748B"

# ==================== GIAO DIỆN ====================
st.markdown("""
<style>
.stApp {
    background: #F5F7FB;
}
.block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: #14263F;
}
[data-testid="stSidebar"] * {
    color: #EAF0FA !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #172B4D !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #2563EB !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] * {
    color: white !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 7px 0;
}
h1, h2, h3 {
    color: #172B4D !important;
}
.hero {
    background: linear-gradient(110deg,#14263F,#244B85);
    padding: 27px 30px;
    border-radius: 18px;
    margin-bottom: 24px;
}
.hero .eyebrow {
    color: #A9C8FF;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
}
.hero h1 {
    color: white !important;
    font-size: 29px;
    margin: 9px 0;
}
.hero p {
    color: #DCE8F8;
    font-size: 14px;
    margin: 0;
}
.kpi {
    background: white;
    border: 1px solid #E7ECF4;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(15,23,42,.035);
    min-height: 130px;
}
.kpi-label {
    color: #64748B;
    font-size: 13px;
    font-weight: 600;
}
.kpi-value {
    font-size: 30px;
    font-weight: 800;
    color: #14263F;
    margin-top: 12px;
}
.kpi-note {
    color: #94A3B8;
    font-size: 11px;
    margin-top: 5px;
}
.section-heading {
    font-size: 19px;
    font-weight: 750;
    color: #172B4D;
    margin: 22px 0 12px;
}
.insight {
    background: #EFF6FF;
    border-left: 4px solid #2563EB;
    border-radius: 9px;
    padding: 17px 19px;
    color: #1E3A5F;
    line-height: 1.7;
}
[data-testid="stPlotlyChart"] {
    background: white;
    border: 1px solid #E7ECF4;
    border-radius: 15px;
    padding: 10px;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ==================== ĐỌC DỮ LIỆU ====================
@st.cache_data
def load_data():
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent

    data = pd.read_csv(data_dir / "data_analysis.csv")
    prediction = pd.read_csv(data_dir / "prediction_test.csv")

    return data, prediction

df, pred = load_data()

df["trang_thai"] = df["label"].map({
    0: "Không trễ",
    1: "Giao trễ"
})

# ==================== HÀM GIAO DIỆN ====================
def kpi(title, value, note="", color=NAVY):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value" style="color:{color}">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def chart(fig, height=365):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=30, r=25, t=65, b=45),
        font=dict(family="Arial", size=12, color="#475569"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_font=dict(size=15, color=NAVY),
        legend=dict(orientation="h", y=-0.2),
        colorway=[BLUE, RED, GREEN, "#8B5CF6"],
    )
    st.plotly_chart(fig, use_container_width=True)

def insight(message):
    st.markdown(
        f'<div class="insight">{message}</div>',
        unsafe_allow_html=True
    )

def stats_by(column):
    result = data.groupby(column).agg(
        tong_don=("label", "count"),
        don_tre=("label", "sum"),
        ty_le=("label", "mean")
    ).reset_index()
    result["ty_le"] *= 100
    return result

# ==================== MENU ====================
st.sidebar.markdown("## 📦 DELIVERY")
st.sidebar.markdown("### INTELLIGENCE")
st.sidebar.caption("DATA ANALYTICS PLATFORM")
st.sidebar.divider()

pages = [
    "Tổng quan",
    "Khu vực",
    "Vận chuyển",
    "Thời gian",
    "Đơn hàng",
    "Dự báo ML"
]

page = st.sidebar.radio(
    "DANH MỤC",
    pages,
    label_visibility="collapsed"
)

st.sidebar.divider()

# Bộ lọc thu gọn, không hiển thị hàng loạt thẻ
with st.sidebar.expander("⚙️ Bộ lọc dữ liệu", expanded=False):
    years = sorted(df["order_year"].dropna().unique().tolist())
    selected_years = st.multiselect(
        "Năm",
        years,
        default=years
    )

    regions = sorted(df["order_region"].dropna().unique().tolist())
    selected_region = st.selectbox(
        "Khu vực",
        ["Tất cả"] + regions
    )

    modes = sorted(df["shipping_mode"].dropna().unique().tolist())
    selected_mode = st.selectbox(
        "Vận chuyển",
        ["Tất cả"] + modes
    )

data = df[df["order_year"].isin(selected_years)].copy()

if selected_region != "Tất cả":
    data = data[data["order_region"] == selected_region]

if selected_mode != "Tất cả":
    data = data[data["shipping_mode"] == selected_mode]

st.sidebar.caption("DataCo Smart Supply Chain")
st.sidebar.caption("GROUP 6 • BIG DATA")

# ==================== HEADER ====================
st.markdown("""
<div class="hero">
    <div class="eyebrow">SUPPLY CHAIN ANALYTICS</div>
    <h1>Phân tích & Dự báo trễ hạn giao hàng</h1>
    <p>Dashboard phân tích dữ liệu DataCo Smart Supply Chain | PySpark • Random Forest • Plotly</p>
</div>
""", unsafe_allow_html=True)

if data.empty and page != "Dự báo ML":
    st.warning("Không có đơn hàng phù hợp với bộ lọc hiện tại.")
    st.stop()

# ==================== TỔNG QUAN ====================
if page == "Tổng quan":
    total = len(data)
    late = int(data["label"].sum())
    ontime = total - late
    rate = late / total * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi("TỔNG ĐƠN HÀNG", f"{total:,}", "Số đơn trong phạm vi phân tích")
    with c2:
        kpi("ĐƠN GIAO TRỄ", f"{late:,}", "Đơn có nguy cơ trễ", RED)
    with c3:
        kpi("ĐƠN KHÔNG TRỄ", f"{ontime:,}", "Đơn giao đúng hạn", GREEN)
    with c4:
        kpi("TỶ LỆ GIAO TRỄ", f"{rate:.2f}%", "Tỷ lệ trên tổng đơn", BLUE)

    st.markdown(
        '<div class="section-heading">Tình trạng giao hàng</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.05, 1], gap="medium")

    with left:
        fig = go.Figure(go.Pie(
            labels=["Giao trễ", "Không trễ"],
            values=[late, ontime],
            hole=0.76,
            marker_colors=[RED, GREEN],
            textinfo="percent",
            textposition="inside",
            sort=False
        ))
        fig.update_layout(
            title="Cơ cấu trạng thái đơn hàng",
            annotations=[dict(
                text=f"<b>{rate:.2f}%</b><br>Giao trễ",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color=NAVY)
            )]
        )
        chart(fig, 350)

    with right:
        st.markdown("#### Tổng hợp phân tích")
        st.write(
            f"Trong **{total:,} đơn hàng**, có **{late:,} đơn giao trễ**, "
            f"tương ứng **{rate:.2f}%**."
        )
        st.write(
            f"Dữ liệu đang bao gồm **{data['order_region'].nunique()} khu vực** "
            f"và **{data['shipping_mode'].nunique()} phương thức vận chuyển**."
        )

        if rate > 50:
            insight(
                "Tỷ lệ giao trễ đang vượt 50%. "
                "Cần ưu tiên phân tích phương thức vận chuyển và "
                "khu vực có tỷ lệ giao trễ cao."
            )
        else:
            insight(
                "Tỷ lệ giao trễ trong phạm vi đang chọn không vượt 50%. "
                "Có thể tiếp tục phân tích các nhóm đơn hàng có rủi ro cao."
            )

# ==================== KHU VỰC ====================
elif page == "Khu vực":
    st.subheader("Phân tích giao hàng theo khu vực")

    stats = stats_by("order_region")
    top_rate = (
        stats[stats["tong_don"] >= 100]
        .nlargest(10, "ty_le")
        .sort_values("ty_le")
    )

    if not top_rate.empty:
        fig = px.bar(
            top_rate,
            x="ty_le",
            y="order_region",
            orientation="h",
            text="ty_le",
            title="Top 10 khu vực có tỷ lệ giao trễ cao nhất",
            labels={"ty_le": "Tỷ lệ (%)", "order_region": "Khu vực"}
        )
        fig.update_traces(
            marker_color=BLUE,
            texttemplate="%{text:.1f}%"
        )
        chart(fig, 450)
    else:
        st.info("Không đủ khu vực có từ 100 đơn hàng.")

    top_volume = stats.nlargest(10, "tong_don")

    fig = go.Figure()
    fig.add_bar(
        x=top_volume["order_region"],
        y=top_volume["tong_don"],
        name="Tổng đơn",
        marker_color=BLUE
    )
    fig.add_bar(
        x=top_volume["order_region"],
        y=top_volume["don_tre"],
        name="Đơn trễ",
        marker_color=RED
    )
    fig.update_layout(
        title="Số đơn hàng và đơn giao trễ theo khu vực",
        barmode="group"
    )
    chart(fig, 420)

    highest = stats.loc[stats["ty_le"].idxmax()]
    insight(
        f"Khu vực <b>{highest['order_region']}</b> có tỷ lệ giao trễ "
        f"cao nhất trong phạm vi lọc: <b>{highest['ty_le']:.2f}%</b>."
    )

# ==================== VẬN CHUYỂN ====================
elif page == "Vận chuyển":
    st.subheader("Hiệu quả phương thức vận chuyển")

    stats = stats_by("shipping_mode")
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        fig = px.bar(
            stats.sort_values("ty_le", ascending=False),
            x="shipping_mode",
            y="ty_le",
            text="ty_le",
            title="Tỷ lệ giao trễ theo phương thức"
        )
        fig.update_traces(
            marker_color=BLUE,
            texttemplate="%{text:.2f}%"
        )
        chart(fig, 400)

    with c2:
        counts = data.groupby(
            ["shipping_mode", "trang_thai"]
        ).size().reset_index(name="so_don")

        fig = px.bar(
            counts,
            x="shipping_mode",
            y="so_don",
            color="trang_thai",
            barmode="group",
            title="So sánh số đơn trễ và không trễ",
            color_discrete_map={
                "Giao trễ": RED,
                "Không trễ": GREEN
            }
        )
        chart(fig, 400)

    highest = stats.loc[stats["ty_le"].idxmax()]
    insight(
        f"Phương thức <b>{highest['shipping_mode']}</b> có tỷ lệ giao trễ "
        f"cao nhất: <b>{highest['ty_le']:.2f}%</b>."
    )

# ==================== THỜI GIAN ====================
elif page == "Thời gian":
    st.subheader("Xu hướng giao hàng theo thời gian")

    monthly = data.groupby(
        ["order_year", "order_month"]
    ).agg(
        ty_le=("label", "mean")
    ).reset_index()

    monthly["ty_le"] *= 100
    monthly["date"] = pd.to_datetime(
        dict(
            year=monthly["order_year"],
            month=monthly["order_month"],
            day=1
        )
    )

    fig = px.line(
        monthly.sort_values("date"),
        x="date",
        y="ty_le",
        markers=True,
        title="Xu hướng tỷ lệ giao trễ theo tháng",
        labels={"date": "Thời gian", "ty_le": "Tỷ lệ (%)"}
    )
    fig.update_traces(line_color=BLUE, line_width=3)
    chart(fig, 400)

    weekday = stats_by("order_day_of_week")

    fig = px.bar(
        weekday.sort_values("order_day_of_week"),
        x="order_day_of_week",
        y="ty_le",
        text="ty_le",
        title="Tỷ lệ giao trễ theo ngày trong tuần"
    )
    fig.update_traces(
        marker_color=BLUE,
        texttemplate="%{text:.1f}%"
    )
    chart(fig, 370)

    insight(
        "Biểu đồ giúp theo dõi sự thay đổi tỷ lệ giao trễ "
        "theo tháng và theo ngày trong tuần."
    )

# ==================== ĐƠN HÀNG ====================
elif page == "Đơn hàng":
    st.subheader("Phân tích đặc điểm đơn hàng")

    temp = data.copy()

    temp["nhom_gia_tri"] = pd.qcut(
        temp["total_order_value"],
        q=5,
        duplicates="drop"
    )

    value_stats = temp.groupby(
        "nhom_gia_tri", observed=True
    )["label"].mean().reset_index()

    value_stats["label"] *= 100
    value_stats["nhom_gia_tri"] = (
        value_stats["nhom_gia_tri"].astype(str)
    )

    fig = px.bar(
        value_stats,
        x="nhom_gia_tri",
        y="label",
        text="label",
        title="Tỷ lệ giao trễ theo nhóm giá trị đơn hàng",
        labels={"label": "Tỷ lệ (%)", "nhom_gia_tri": "Nhóm giá trị"}
    )
    fig.update_traces(
        marker_color=BLUE,
        texttemplate="%{text:.1f}%"
    )
    chart(fig, 390)

    temp["nhom_so_luong"] = pd.cut(
        temp["total_quantity"],
        bins=[0, 1, 2, 5, 10, np.inf],
        labels=["1", "2", "3–5", "6–10", "Trên 10"]
    )

    quantity_stats = temp.groupby(
        "nhom_so_luong", observed=True
    )["label"].mean().reset_index()

    quantity_stats["label"] *= 100

    fig = px.bar(
        quantity_stats,
        x="nhom_so_luong",
        y="label",
        text="label",
        title="Tỷ lệ giao trễ theo số lượng sản phẩm",
        labels={"label": "Tỷ lệ (%)", "nhom_so_luong": "Số lượng"}
    )
    fig.update_traces(
        marker_color=BLUE,
        texttemplate="%{text:.1f}%"
    )
    chart(fig, 370)

    insight(
        "Phân tích giúp xem xét mối liên hệ giữa giá trị, "
        "số lượng sản phẩm và tỷ lệ giao trễ."
    )

# ==================== DỰ BÁO ML ====================
elif page == "Dự báo ML":
    st.subheader("Đánh giá mô hình Random Forest")
    st.caption(
        "Các chỉ số được đánh giá trên tập Test cố định, "
        "không thay đổi theo bộ lọc phân tích."
    )

    y_true = pred["label"].astype(int)
    y_pred = pred["prediction"].astype(int)
    probabilities = pred["probability_late"]

    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted"
    )
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = auc(fpr, tpr)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi("ACCURACY", f"{accuracy:.2%}")
    with c2:
        kpi("WEIGHTED F1", f"{weighted_f1:.4f}")
    with c3:
        kpi("ROC-AUC", f"{roc_auc:.4f}", color=BLUE)
    with c4:
        kpi("TẬP TEST", f"{len(pred):,}", "Số đơn đánh giá")

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        cm = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        )
        fig = px.imshow(
            cm,
            x=["Không trễ", "Giao trễ"],
            y=["Không trễ", "Giao trễ"],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Ma trận nhầm lẫn"
        )
        fig.update_layout(coloraxis_showscale=False)
        chart(fig, 400)

    with c2:
        fig = go.Figure()
        fig.add_scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"ROC-AUC = {roc_auc:.4f}",
            line=dict(color=BLUE, width=3)
        )
        fig.add_scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Ngẫu nhiên",
            line=dict(color=MUTED, dash="dash")
        )
        fig.update_layout(
            title="Đường cong ROC",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        chart(fig, 400)

    comparison = pd.DataFrame({
        "Mô hình": ["RF 12 đặc trưng", "RF 5 đặc trưng"],
        "Accuracy": [0.6948, 0.6949],
        "Weighted F1": [0.6929, 0.6930],
        "ROC-AUC": [0.7309, 0.7345]
    })

    long_df = comparison.melt(
        id_vars="Mô hình",
        var_name="Chỉ số",
        value_name="Giá trị"
    )

    fig = px.bar(
        long_df,
        x="Chỉ số",
        y="Giá trị",
        color="Mô hình",
        barmode="group",
        text="Giá trị",
        title="So sánh Random Forest 12 và 5 đặc trưng",
        color_discrete_map={
            "RF 12 đặc trưng": "#94A3B8",
            "RF 5 đặc trưng": BLUE
        }
    )
    fig.update_traces(texttemplate="%{text:.4f}")
    chart(fig, 390)

    insight(
        "Mô hình Random Forest 5 đặc trưng đạt "
        f"<b>Accuracy {accuracy:.2%}</b>, "
        f"<b>Weighted F1 {weighted_f1:.4f}</b> và "
        f"<b>ROC-AUC {roc_auc:.4f}</b> trên tập Test."
    )

st.divider()
st.caption(
    "GROUP 6  |  BIG DATA ANALYTICS  |  DATACO SMART SUPPLY CHAIN"
)
