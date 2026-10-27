
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
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
    data = pd.read_csv("data_analysis.csv")
    prediction = pd.read_csv("prediction_test.csv")
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
    "Dự báo ML",
    "Dự báo đơn hàng mới"
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

if data.empty and page not in ["Dự báo ML", "Dự báo đơn hàng mới"]:
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

    # Phân tích đa chiều: khu vực x phương thức vận chuyển
    st.markdown("#### Bản đồ nhiệt tỷ lệ giao trễ theo khu vực và vận chuyển")
    min_orders_heat = st.slider(
        "Số đơn tối thiểu cho mỗi tổ hợp", 10, 300, 50, 10,
        key="heat_min_orders"
    )
    heat_stats = data.groupby(["order_region", "shipping_mode"]).agg(
        tong_don=("label", "count"), ty_le=("label", "mean")
    ).reset_index()
    heat_stats = heat_stats[heat_stats["tong_don"] >= min_orders_heat].copy()
    if not heat_stats.empty:
        heat_stats["ty_le"] *= 100
        heat_rate = heat_stats.pivot(index="order_region", columns="shipping_mode", values="ty_le")
        heat_count = heat_stats.pivot(index="order_region", columns="shipping_mode", values="tong_don")
        heat_rate = heat_rate.sort_index()
        heat_count = heat_count.reindex(index=heat_rate.index, columns=heat_rate.columns)
        fig = go.Figure(go.Heatmap(
            z=heat_rate.to_numpy(), x=heat_rate.columns.tolist(), y=heat_rate.index.tolist(),
            customdata=heat_count.to_numpy(), colorscale="YlOrRd",
            colorbar=dict(title="Tỷ lệ (%)"),
            hovertemplate="Khu vực: %{y}<br>Vận chuyển: %{x}<br>Tỷ lệ trễ: %{z:.2f}%<br>Số đơn: %{customdata:.0f}<extra></extra>"
        ))
        fig.update_layout(title="Tỷ lệ giao trễ theo từng tổ hợp", yaxis=dict(autorange="reversed"))
        chart(fig, max(450, 26 * len(heat_rate) + 130))
    else:
        st.info("Không có tổ hợp đủ số đơn theo ngưỡng đã chọn.")

    # Phân tích sâu theo quốc gia, tránh kết luận từ nhóm có quá ít đơn
    if "order_country" in data.columns:
        st.markdown("#### Phân tích giao trễ theo quốc gia")
        c_top, c_min = st.columns(2)
        with c_top:
            top_n_countries = st.slider("Số quốc gia hiển thị", 5, 20, 10, key="country_top_n")
        with c_min:
            min_orders_country = st.number_input(
                "Số đơn tối thiểu mỗi quốc gia", min_value=1, max_value=10000,
                value=100, step=50, key="country_min_orders"
            )
        country_stats = data.dropna(subset=["order_country"]).groupby("order_country").agg(
            tong_don=("label", "count"), don_tre=("label", "sum"), ty_le=("label", "mean")
        ).reset_index()
        country_stats = country_stats[country_stats["tong_don"] >= min_orders_country].copy()
        if not country_stats.empty:
            country_stats["ty_le"] *= 100
            top_countries = country_stats.nlargest(top_n_countries, "ty_le").sort_values("ty_le")
            fig = px.bar(
                top_countries, x="ty_le", y="order_country", orientation="h",
                color="tong_don", color_continuous_scale="Blues",
                custom_data=["tong_don", "don_tre"],
                labels={"ty_le": "Tỷ lệ giao trễ (%)", "order_country": "Quốc gia", "tong_don": "Tổng đơn"},
                title="Các quốc gia có tỷ lệ giao trễ cao (đủ ngưỡng số đơn)"
            )
            fig.update_traces(hovertemplate="Quốc gia: %{y}<br>Tỷ lệ trễ: %{x:.2f}%<br>Tổng đơn: %{customdata[0]}<br>Đơn trễ: %{customdata[1]}<extra></extra>")
            chart(fig, max(400, top_n_countries * 31 + 130))
        else:
            st.info("Không có quốc gia đủ số đơn theo ngưỡng đã chọn.")

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

    # Xu hướng theo thời gian kết hợp phương thức vận chuyển
    st.markdown("#### So sánh xu hướng giao trễ theo phương thức vận chuyển")
    monthly_mode = data.groupby(["order_year", "order_month", "shipping_mode"]).agg(
        tong_don=("label", "count"), ty_le=("label", "mean")
    ).reset_index()
    monthly_mode["ty_le"] *= 100
    monthly_mode["date"] = pd.to_datetime(dict(
        year=monthly_mode["order_year"], month=monthly_mode["order_month"], day=1
    ))
    mode_options = sorted(monthly_mode["shipping_mode"].dropna().unique().tolist())
    chosen_modes = st.multiselect(
        "Chọn phương thức để so sánh", mode_options, default=mode_options,
        key="trend_shipping_modes"
    )
    monthly_mode = monthly_mode[monthly_mode["shipping_mode"].isin(chosen_modes)]
    if not monthly_mode.empty:
        fig = px.line(
            monthly_mode.sort_values("date"), x="date", y="ty_le", color="shipping_mode",
            markers=True, custom_data=["tong_don"],
            labels={"date": "Tháng đặt hàng", "ty_le": "Tỷ lệ giao trễ (%)", "shipping_mode": "Vận chuyển"},
            title="Xu hướng giao trễ theo tháng của từng phương thức"
        )
        fig.update_traces(hovertemplate="Tháng: %{x|%m/%Y}<br>Tỷ lệ trễ: %{y:.2f}%<br>Số đơn: %{customdata[0]}<extra></extra>")
        chart(fig, 430)
    else:
        st.info("Hãy chọn ít nhất một phương thức vận chuyển.")

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


# ==================== DỰ BÁO ĐƠN HÀNG MỚI ====================
elif page == "Dự báo đơn hàng mới":
    st.subheader("Dự báo nguy cơ giao trễ cho đơn hàng mới")
    st.caption("Dự báo bằng Pipeline Random Forest 5 đặc trưng đã huấn luyện trong Google Colab.")

    model_dir = Path(__file__).resolve().parent.parent / "models" / "random_forest_final"
    if not model_dir.is_dir():
        st.error("Chưa tìm thấy mô hình đã lưu tại: " + str(model_dir))
        st.info("Hãy chạy mục 7.1 trong notebook để lưu Pipeline, sau đó chạy lại Dashboard.")
        st.stop()

    @st.cache_resource(show_spinner="Đang nạp mô hình Spark...")
    def load_spark_pipeline(model_path):
        from pyspark.sql import SparkSession
        from pyspark.ml import PipelineModel
        spark_session = (SparkSession.builder
                         .appName("DataCo-New-Order-Prediction")
                         .master("local[*]")
                         .getOrCreate())
        return spark_session, PipelineModel.load(model_path)

    # Danh sách danh mục từ dữ liệu đã xuất; giữ nguyên giá trị gốc để dự báo.
    modes_new = sorted(df["shipping_mode"].dropna().astype(str).unique().tolist())
    regions_new = sorted(df["order_region"].dropna().astype(str).unique().tolist())
    countries_new = (
        sorted(df["order_country"].dropna().astype(str).unique().tolist())
        if "order_country" in df.columns else []
    )

    if not countries_new:
        st.error("data_analysis.csv chưa có cột order_country. Cần xuất thêm cột này trước khi dự báo.")
        st.stop()

    # Chỉ thay đổi nhãn hiển thị; giá trị truyền vào Pipeline vẫn là dữ liệu gốc.
    COUNTRY_DISPLAY = {
    "Afganist�n": "Afganistán",
    "Arabia Saud�": "Arabia Saudí",
    "Azerbaiy�n": "Azerbaiyán",
    "Banglad�s": "Bangladés",
    "Bar�in": "Baréin",
    "Ben�n": "Benín",
    "But�n": "Bután",
    "B�lgica": "Bélgica",
    "Camer�n": "Camerún",
    "Emiratos �rabes Unidos": "Emiratos Árabes Unidos",
    "Espa�a": "España",
    "Etiop�a": "Etiopía",
    "Gab�n": "Gabón",
    "Hait�": "Haití",
    "Hungr�a": "Hungría",
    "Ir�n": "Irán",
    "Jap�n": "Japón",
    "Kazajist�n": "Kazajistán",
    "Kirguist�n": "Kirguistán",
    "L�bano": "Líbano",
    "M�xico": "México",
    "N�ger": "Níger",
    "Om�n": "Omán",
    "Pakist�n": "Pakistán",
    "Panam�": "Panamá",
    "Pap�a Nueva Guinea": "Papúa Nueva Guinea",
    "Pa�ses Bajos": "Países Bajos",
    "Per�": "Perú",
    "Rep�blica Centroafricana": "República Centroafricana",
    "Rep�blica Checa": "República Checa",
    "Rep�blica Democr�tica del Congo": "República Democrática del Congo",
    "Rep�blica Dominicana": "República Dominicana",
    "Rep�blica de Gambia": "República de Gambia",
    "Rep�blica del Congo": "República del Congo",
    "Sud�n": "Sudán",
    "Sud�n del Sur": "Sudán del Sur",
    "S�hara Occidental": "Sáhara Occidental",
    "Taiw�n": "Taiwán",
    "Tayikist�n": "Tayikistán",
    "Turkmenist�n": "Turkmenistán",
    "Turqu�a": "Turquía",
    "T�nez": "Túnez",
    "Uzbekist�n": "Uzbekistán",
}

    # Ràng buộc quốc gia theo khu vực từ dữ liệu thật
    region_country_map = (
        df[["order_region", "order_country"]].dropna().drop_duplicates()
        .groupby("order_region")["order_country"]
        .apply(lambda x: sorted(x.astype(str).unique().tolist()))
        .to_dict()
    )
    col_a, col_b = st.columns(2)
    with col_a:
        shipping_mode = st.selectbox("Phương thức vận chuyển", modes_new)
        order_region = st.selectbox("Khu vực", regions_new)
        order_country = st.selectbox(
            "Quốc gia", options=region_country_map.get(order_region, []),
            format_func=lambda value: COUNTRY_DISPLAY.get(value, value),
            key=f"country_{order_region}"
        )
    with col_b:
        scheduled_days = st.number_input(
            "Số ngày vận chuyển dự kiến", min_value=0, max_value=30, value=3, step=1
        )
        order_month = st.selectbox("Tháng đặt hàng", list(range(1, 13)))
    with st.form("predict_new_order"):
        submitted = st.form_submit_button("Dự báo nguy cơ giao trễ", type="primary")

    if submitted:
        try:
            spark_session, pipeline = load_spark_pipeline(str(model_dir))
            from pyspark.sql.types import StructType, StructField, StringType, IntegerType
            schema = StructType([
                StructField("shipping_mode", StringType(), False),
                StructField("order_region", StringType(), False),
                StructField("order_country", StringType(), False),
                StructField("scheduled_shipping_days", IntegerType(), False),
                StructField("order_month", IntegerType(), False),
            ])
            row = [(shipping_mode, order_region, order_country, int(scheduled_days), int(order_month))]
            new_order = spark_session.createDataFrame(row, schema)
            prediction_row = pipeline.transform(new_order).select("prediction", "probability").first()
            late = int(prediction_row["prediction"]) == 1
            late_probability = float(prediction_row["probability"][1])
            if late:
                st.error("Kết quả: Có nguy cơ giao trễ")
            else:
                st.success("Kết quả: Dự báo không giao trễ")
            st.metric("Xác suất dự báo giao trễ", f"{late_probability:.2%}")
            st.caption("Đây là ước lượng của mô hình, không phải cam kết về thời gian giao hàng thực tế.")
        except Exception as exc:
            st.error("Không thể chạy dự báo. Kiểm tra môi trường PySpark/Java và mô hình đã lưu.")
            st.exception(exc)

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
