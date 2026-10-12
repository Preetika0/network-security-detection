import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Network Security Attack Detection", page_icon="🛡️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

RAW_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack",
    "level",
]
FEATURE_COLUMNS_RAW = RAW_COLUMNS[:-2]
CAT_COLS = ["protocol_type", "service", "flag"]

DEFAULT_VALUES = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 100,
    "dst_bytes": 0,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 1,
    "srv_count": 1,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 1,
    "dst_host_srv_count": 1,
    "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 1.0,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0,
}

PROTOCOL_OPTIONS = ["tcp", "udp", "icmp"]
SERVICE_OPTIONS = [
    "http", "ftp_data", "smtp", "private", "domain_u", "ftp", "telnet", "eco_i",
    "auth", "other", "gopher", "domain", "courier", "csnet_ns", "discard",
    "echo", "ecr_i", "klogin", "kshell", "ldap", "login", "mtp", "name",
    "netbios_dgm", "netbios_ns", "netbios_ssn", "nnsp", "nntp", "ntp_u",
    "pop_2", "pop_3", "printer", "remote_job", "rje", "shell", "sql_net",
    "ssh", "sunrpc", "systat", "tim_i", "time", "urp_i", "uucp", "uucp_path",
    "vmnet", "whois", "X11", "IRC", "Z39_50", "aol", "bgp", "daytime", "hostnames",
    "http_443", "http_8001", "imap4", "iso_tsap", "link", "pm_dump", "red_i",
    "supdup", "tftp_u", "urh_i",
]
FLAG_OPTIONS = ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "OTH", "RSTOS0"]
QUICK_FIELDS = [
    "protocol_type",
    "service",
    "flag",
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "logged_in",
    "same_srv_rate",
    "dst_host_same_srv_rate",
]

CUSTOM_CSS = """
<style>
    .stApp {
        background: #0B1120;
        color: #E5EEF9;
    }
    .main .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    .shield-header {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #F8FAFC;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #9CA3AF;
        letter-spacing: 0.04em;
        margin-bottom: 0.4rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(34, 211, 238, 0.08);
        border: 1px solid rgba(34, 211, 238, 0.4);
        color: #9AE6F9;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin-top: 0.35rem;
        margin-bottom: 0.8rem;
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22D3EE;
        box-shadow: 0 0 10px rgba(34, 211, 238, 0.9);
    }
    .description {
        color: #CBD5E1;
        font-size: 0.98rem;
        line-height: 1.6;
        margin-top: 0.2rem;
        margin-bottom: 1.1rem;
        max-width: 980px;
    }
    .panel-card {
        background: #111827;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.28);
        margin-bottom: 1rem;
    }
    .empty-state {
        min-height: 340px;
        border-radius: 16px;
        border: 1px dashed rgba(56, 189, 248, 0.32);
        background: rgba(17, 24, 39, 0.72);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 1.8rem 1rem;
        color: #94A3B8;
    }
    .empty-state .icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .empty-state h3 {
        margin: 0.1rem 0 0.4rem 0;
        color: #E2E8F0;
        letter-spacing: 0.08em;
    }
    .empty-state p {
        margin: 0;
        max-width: 360px;
        line-height: 1.5;
        color: #9CA3AF;
    }
    .metric-box {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    .metric-label {
        display: block;
        color: #94A3B8;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .section-title {
        color: #E2E8F0;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 0.7rem;
    }
    div[data-testid="stTabs"] [role="tablist"] {
        gap: 0.5rem;
        margin-bottom: 0.9rem;
    }
    div[data-testid="stTabs"] [role="tab"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        color: #DDE8F5;
        padding: 0.55rem 0.9rem;
    }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: rgba(56, 189, 248, 0.12);
        border-color: rgba(56, 189, 248, 0.35);
        color: #F8FAFC;
    }
    .stButton > button {
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.2rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        background: linear-gradient(135deg, #38BDF8 0%, #22D3EE 100%);
        color: #06263C;
        box-shadow: 0 10px 22px rgba(34, 211, 238, 0.25);
    }
    .stDownloadButton > button {
        border: 1px solid rgba(56, 189, 248, 0.32);
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.8);
        color: #E2E8F0;
    }
    .stAlert {
        border-radius: 12px;
    }
    .stDataFrame {
        border-radius: 12px;
    }
    .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #DDE8F5;
        font-size: 0.83rem;
        font-weight: 600;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox div[role="combobox"], .stSlider div {
        background: rgba(15, 23, 42, 0.9);
        color: #F8FAFC;
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_display_class_name(value):
    mapping = {
        "normal": "Normal",
        "dos": "DoS",
        "probe": "Probe",
        "r2l": "R2L",
        "u2r": "U2R",
    }
    return mapping.get(str(value).strip().lower(), str(value).strip().upper())


@st.cache_resource
def load_artifacts():
    required = [
        MODEL_DIR / "security_attacks_model.keras",
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "feature_columns.pkl",
        MODEL_DIR / "label_encoder.pkl",
    ]
    missing_files = [p.name for p in required if not p.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing_files)
        )

    try:
        model = tf.keras.models.load_model(MODEL_DIR / "security_attacks_model.keras")
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODEL_DIR / "feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)
        with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
    except Exception as exc:  # pragma: no cover - defensive for UI load errors
        raise RuntimeError("Model artifacts could not be loaded.") from exc

    return model, scaler, feature_columns, label_encoder


def build_default_record(overrides=None):
    record = {column: DEFAULT_VALUES.get(column, 0) for column in FEATURE_COLUMNS_RAW}
    if overrides:
        for key, value in overrides.items():
            if key in record:
                record[key] = value
    return record


def prepare_input_dataframe(raw_values):
    record = {}
    for column in FEATURE_COLUMNS_RAW:
        if column in CAT_COLS:
            record[column] = str(raw_values[column]).strip()
        else:
            record[column] = float(raw_values[column])
    return pd.DataFrame([record], columns=FEATURE_COLUMNS_RAW)


def preprocess_dataset_frame(frame, feature_columns, scaler):
    if frame is None or frame.empty:
        raise ValueError("The uploaded dataset is empty.")

    df = frame.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]

    for drop_column in ["attack", "attack_class", "level"]:
        if drop_column in df.columns:
            df = df.drop(columns=[drop_column])

    missing = [column for column in FEATURE_COLUMNS_RAW if column not in df.columns]
    if missing:
        raise ValueError(
            "The uploaded dataset is missing required NSL-KDD feature columns. "
            + ", ".join(missing[:8])
        )

    feature_df = df[FEATURE_COLUMNS_RAW].copy()
    for column in FEATURE_COLUMNS_RAW:
        if column not in CAT_COLS:
            feature_df[column] = pd.to_numeric(feature_df[column], errors="raise")

    encoded = pd.get_dummies(feature_df, columns=CAT_COLS, drop_first=False)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return scaler.transform(encoded)


def load_upload_dataframe(uploaded_file):
    if uploaded_file is None:
        raise ValueError("Please upload a CSV or TXT file first.")

    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python", skipinitialspace=True)
    except Exception:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, header=None)
        except Exception as exc:
            raise ValueError("The uploaded file could not be parsed as a table.") from exc

    if df.empty:
        raise ValueError("The uploaded file does not contain any records.")

    column_names = [str(col).strip().lower() for col in df.columns]
    if "duration" not in column_names and len(df.columns) not in (41, 43):
        raise ValueError(
            "The dataset must contain 41 NSL-KDD traffic features or the original 43-column format."
        )

    if "duration" not in column_names:
        if df.shape[1] == 43:
            df.columns = [str(col).strip().lower() for col in RAW_COLUMNS]
        elif df.shape[1] == 41:
            df.columns = [str(col).strip().lower() for col in FEATURE_COLUMNS_RAW]
        else:
            raise ValueError(
                "The file does not match the expected NSL-KDD structure. "
                "Expected 41 features or 43 columns with attack and level fields."
            )
    else:
        df.columns = [str(col).strip().lower() for col in df.columns]

    if len(df.columns) > 43:
        raise ValueError("The uploaded file has too many columns for the NSL-KDD schema.")

    return df


def analyze_single_record(model, scaler, feature_columns, label_encoder, record_df):
    X = preprocess_dataset_frame(record_df, feature_columns, scaler)
    probabilities = model.predict(X, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = str(label_encoder.inverse_transform([predicted_index])[0]).lower()
    confidence_pct = float(probabilities[predicted_index]) * 100.0
    return {
        "predicted_label": predicted_label,
        "confidence_pct": confidence_pct,
        "probabilities": probabilities,
        "class_names": [get_display_class_name(name) for name in label_encoder.classes_],
    }


try:
    model, scaler, feature_columns, label_encoder = load_artifacts()
    artifacts_loaded = True
except Exception:
    artifacts_loaded = False
    st.error("Unable to load the trained model or preprocessing artifacts. Please ensure the files in the model folder are available and valid.")
    st.stop()

st.markdown(
    '<div class="shield-header">🛡️ <span>NETWORK SENTINEL</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">AI-Powered Network Intrusion Detection System</div>',
    unsafe_allow_html=True,
)
if artifacts_loaded:
    st.markdown(
        '<div class="status-pill"><span class="status-dot"></span>MODEL STATUS: LOADED</div>',
        unsafe_allow_html=True,
    )
st.markdown(
    '<div class="description">Analyze network traffic using a trained Artificial Neural Network to identify normal activity and potential security attacks.</div>',
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1.45, 1.1])

with col_left:
    st.markdown('<div class="panel-card"><div class="section-title">Traffic Analysis</div></div>', unsafe_allow_html=True)
    tabs = st.tabs(["Quick Prediction", "Advanced Prediction", "Upload Dataset"])

    with tabs[0]:
        st.warning(
            "Quick prediction uses a reduced set of traffic indicators. Missing features are filled with documented defaults and the result is approximate.",
            icon="⚠️",
        )
        with st.form("quick_prediction_form"):
            quick_columns = st.columns(2)
            quick_values = {}

            for index, field in enumerate(QUICK_FIELDS):
                column = quick_columns[index % 2]
                if field in CAT_COLS:
                    choices = PROTOCOL_OPTIONS if field == "protocol_type" else SERVICE_OPTIONS if field == "service" else FLAG_OPTIONS
                    quick_values[field] = column.selectbox(
                        field.replace("_", " ").title(),
                        choices,
                        index=choices.index(DEFAULT_VALUES[field]),
                        key=f"quick_{field}",
                    )
                else:
                    if field in {"duration", "src_bytes", "dst_bytes", "count"}:
                        quick_values[field] = column.number_input(
                            field.replace("_", " ").title(),
                            min_value=0.0,
                            value=float(DEFAULT_VALUES[field]),
                            step=1.0 if field in {"duration", "count"} else 10.0,
                            key=f"quick_{field}",
                        )
                    else:
                        quick_values[field] = column.selectbox(
                            field.replace("_", " ").title(),
                            options=[0, 1],
                            index=1 if DEFAULT_VALUES[field] == 1 else 0,
                            key=f"quick_{field}",
                        )

            quick_submit = st.form_submit_button("ANALYZE TRAFFIC", use_container_width=True)
            if quick_submit:
                try:
                    defaults = build_default_record()
                    defaults.update({key: value for key, value in quick_values.items()})
                    input_frame = prepare_input_dataframe(defaults)
                    with st.spinner("Analyzing network traffic..."):
                        result = analyze_single_record(model, scaler, feature_columns, label_encoder, input_frame)
                    st.session_state["analysis"] = {
                        "type": "single",
                        "result": result,
                        "approximate": True,
                    }
                except Exception:
                    st.error("The model could not complete the quick prediction. Please review the selected inputs and try again.")

    with tabs[1]:
        with st.form("advanced_prediction_form"):
            def render_group(title, fields):
                with st.expander(title, expanded=True):
                    group_cols = st.columns(2)
                    for idx, field in enumerate(fields):
                        column = group_cols[idx % 2]
                        if field in CAT_COLS:
                            if field == "protocol_type":
                                choices = PROTOCOL_OPTIONS
                            elif field == "service":
                                choices = SERVICE_OPTIONS
                            else:
                                choices = FLAG_OPTIONS
                            column.selectbox(
                                field.replace("_", " ").title(),
                                choices,
                                index=choices.index(DEFAULT_VALUES[field]),
                                key=f"advanced_{field}",
                            )
                        else:
                            if field in {"serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"}:
                                column.slider(
                                    field.replace("_", " ").title(),
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=float(DEFAULT_VALUES[field]),
                                    step=0.01,
                                    key=f"advanced_{field}",
                                )
                            else:
                                column.number_input(
                                    field.replace("_", " ").title(),
                                    min_value=0.0,
                                    value=float(DEFAULT_VALUES[field]),
                                    step=1.0 if field in {"duration", "count", "srv_count", "dst_host_count", "dst_host_srv_count"} else 0.1,
                                    key=f"advanced_{field}",
                                )

            advanced_fields = [
                "duration",
                "src_bytes",
                "dst_bytes",
                "land",
                "wrong_fragment",
                "urgent",
                "hot",
                "num_failed_logins",
                "logged_in",
                "num_compromised",
                "root_shell",
                "su_attempted",
                "num_root",
                "num_file_creations",
                "num_shells",
                "num_access_files",
                "num_outbound_cmds",
                "is_host_login",
                "is_guest_login",
                "count",
                "srv_count",
                "serror_rate",
                "srv_serror_rate",
                "rerror_rate",
                "srv_rerror_rate",
                "same_srv_rate",
                "diff_srv_rate",
                "srv_diff_host_rate",
                "dst_host_count",
                "dst_host_srv_count",
                "dst_host_same_srv_rate",
                "dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate",
                "dst_host_serror_rate",
                "dst_host_srv_serror_rate",
                "dst_host_rerror_rate",
                "dst_host_srv_rerror_rate",
            ]
            render_group("Basic Connection Information", ["duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot"])
            render_group("Protocol and Service Information", ["protocol_type", "service", "flag"])
            render_group("Traffic Volume", ["count", "srv_count", "num_failed_logins", "logged_in", "num_compromised", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds"])
            render_group("Login and Security Indicators", ["root_shell", "su_attempted", "is_host_login", "is_guest_login"])
            render_group("Connection Statistics", ["serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate"])
            render_group("Destination Host Statistics", ["dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"])

            advanced_submit = st.form_submit_button("ANALYZE TRAFFIC", use_container_width=True)
            if advanced_submit:
                try:
                    advanced_values = build_default_record()
                    for column in FEATURE_COLUMNS_RAW:
                        value = st.session_state.get(f"advanced_{column}", DEFAULT_VALUES.get(column, 0))
                        if column in CAT_COLS:
                            advanced_values[column] = str(value)
                        else:
                            advanced_values[column] = float(value)
                    input_frame = prepare_input_dataframe(advanced_values)
                    with st.spinner("Analyzing network traffic..."):
                        result = analyze_single_record(model, scaler, feature_columns, label_encoder, input_frame)
                    st.session_state["analysis"] = {
                        "type": "single",
                        "result": result,
                        "approximate": False,
                    }
                except Exception:
                    st.error("The model could not complete the advanced prediction. Please review your entries and try again.")

    with tabs[2]:
        uploaded_file = st.file_uploader(
            "Upload NSL-KDD CSV/TXT file",
            type=["csv", "txt"],
            help="Upload the original 41-feature NSL-KDD format, or a 43-column version including attack and level.",
        )
        if uploaded_file is not None:
            try:
                preview_df = load_upload_dataframe(uploaded_file)
                st.caption("Preview")
                st.dataframe(preview_df.head(8), use_container_width=True)
            except ValueError as exc:
                st.error(str(exc))

        dataset_submitted = st.button("ANALYZE DATASET", use_container_width=True)
        if dataset_submitted:
            if uploaded_file is None:
                st.error("Please upload a CSV or TXT file before running the analysis.")
            else:
                try:
                    preview_df = load_upload_dataframe(uploaded_file)
                    with st.spinner("Analyzing network traffic..."):
                        X = preprocess_dataset_frame(preview_df, feature_columns, scaler)
                        probabilities = model.predict(X, verbose=0)
                        predicted_indices = np.argmax(probabilities, axis=1)
                        predicted_labels = label_encoder.inverse_transform(predicted_indices)
                        confidence_values = probabilities.max(axis=1) * 100

                        predicted_display = [get_display_class_name(label) for label in predicted_labels]
                        results = pd.DataFrame(
                            {
                                "Record": np.arange(1, len(predicted_display) + 1),
                                "Predicted class": predicted_display,
                                "Confidence (%)": np.round(confidence_values, 2),
                            }
                        )

                    total_records = len(results)
                    normal_count = int((results["Predicted class"] == "Normal").sum())
                    attack_count = total_records - normal_count
                    normal_pct = (normal_count / total_records) * 100 if total_records else 0.0
                    attack_pct = (attack_count / total_records) * 100 if total_records else 0.0

                    st.session_state["analysis"] = {
                        "type": "dataset",
                        "results": results,
                        "total_records": total_records,
                        "normal_count": normal_count,
                        "attack_count": attack_count,
                        "normal_pct": normal_pct,
                        "attack_pct": attack_pct,
                    }
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("The uploaded dataset could not be processed. Please verify the NSL-KDD feature layout and try again.")

with col_right:
    if "analysis" not in st.session_state:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">🟢</div>
                <h3>READY FOR ANALYSIS</h3>
                <p>Enter network traffic details or upload a dataset to begin detection.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        analysis = st.session_state["analysis"]
        if analysis["type"] == "single":
            result = analysis["result"]
            label = get_display_class_name(result["predicted_label"])
            confidence = result["confidence_pct"]
            is_normal = label == "Normal"
            if analysis.get("approximate"):
                st.info("Quick prediction mode relies on partial traffic inputs and therefore produces an approximate model output.")

            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            col_metric_1, col_metric_2 = st.columns(2)
            with col_metric_1:
                st.markdown(
                    f'<div class="metric-box"><span class="metric-label">Predicted Class</span><div class="metric-value">{label}</div></div>',
                    unsafe_allow_html=True,
                )
            with col_metric_2:
                st.markdown(
                    f'<div class="metric-box"><span class="metric-label">Prediction Confidence</span><div class="metric-value">{confidence:.2f}%</div></div>',
                    unsafe_allow_html=True,
                )
            st.write("")
            if is_normal:
                st.success("Security Classification: normal traffic")
            else:
                st.warning("Security Classification: predicted attack")

            st.markdown('<div class="section-title">Probability Breakdown</div>', unsafe_allow_html=True)
            chart_df = pd.DataFrame({"Probability (%)": np.round(result["probabilities"] * 100, 2)}, index=result["class_names"])
            st.bar_chart(chart_df)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            results = analysis["results"]
            total_records = analysis["total_records"]
            normal_count = analysis["normal_count"]
            attack_count = analysis["attack_count"]
            normal_pct = analysis["normal_pct"]
            attack_pct = analysis["attack_pct"]

            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Model Prediction Summary</div>', unsafe_allow_html=True)
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.markdown(
                    f'<div class="metric-box"><span class="metric-label">Total records</span><div class="metric-value">{total_records}</div></div>',
                    unsafe_allow_html=True,
                )
            with metric_cols[1]:
                st.markdown(
                    f'<div class="metric-box"><span class="metric-label">Normal</span><div class="metric-value">{normal_count} ({normal_pct:.1f}%)</div></div>',
                    unsafe_allow_html=True,
                )
            with metric_cols[2]:
                st.markdown(
                    f'<div class="metric-box"><span class="metric-label">Attack</span><div class="metric-value">{attack_count} ({attack_pct:.1f}%)</div></div>',
                    unsafe_allow_html=True,
                )
            st.write("")
            st.markdown('<div class="section-title">Predicted Class and Confidence</div>', unsafe_allow_html=True)
            st.dataframe(results, use_container_width=True)
            st.markdown('<div class="section-title">Predicted Class Distribution</div>', unsafe_allow_html=True)
            class_distribution = results["Predicted class"].value_counts().reindex(["Normal", "DoS", "Probe", "R2L", "U2R"], fill_value=0)
            st.bar_chart(class_distribution)
            st.download_button(
                label="Download Prediction CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="network_security_predictions.csv",
                mime="text/csv",
            )
            st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Model & Methodology"):
    st.write("Dataset: NSL-KDD")
    st.write("Original traffic features: 41")
    st.write("Preprocessed input features: 122 (verified from the saved feature columns)")
    st.write("Model: Artificial Neural Network")
    st.write("Architecture: 122 → Dense(500, ReLU) → Dropout(0.2) → Dense(5, Softmax)")
    st.write("Optimizer: Adam")
    st.write("Loss: Sparse Categorical Cross-Entropy")
    st.write("Class imbalance handling: class weights during training")
    st.write("Output classes: Normal, DoS, Probe, R2L and U2R")
    st.caption("Predictions are model-generated classifications for educational use and are not verified real-world network detections.")
