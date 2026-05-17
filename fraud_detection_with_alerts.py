import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO, BytesIO
import csv

# Page config
st.set_page_config(page_title="Fraud Detection System", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for premium look
st.markdown("""
<style>
    :root {
        --bg-dark: #0a0e27;
        --bg-secondary: #1a1f3a;
        --text-primary: #ffffff;
        --text-secondary: #b0b5c1;
        --border: rgba(255, 255, 255, 0.1);
        --border-hover: rgba(255, 255, 255, 0.3);
        --danger: #ef4444;
        --success: #10b981;
        --warning: #f59e0b;
    }
    
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-dark);
    }
    
    [data-testid="stMainBlockContainer"] {
        padding: 3rem 2rem;
        background-color: var(--bg-dark);
    }
    
    .main {
        background-color: var(--bg-dark) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    
    p, span, label, [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }
    
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] select,
    [data-testid="stTextInput"] input {
        background-color: var(--bg-secondary) !important;
        border: 0.5px solid var(--border) !important;
        color: var(--text-primary) !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stNumberInput"] input:hover,
    [data-testid="stSelectbox"] select:hover,
    [data-testid="stTextInput"] input:hover {
        border-color: var(--border-hover) !important;
    }
    
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stSelectbox"] select:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: rgba(255, 255, 255, 0.5) !important;
        outline: none !important;
    }
    
    [data-testid="stButton"] button {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        padding: 0.875rem 1.5rem !important;
        border-radius: 8px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    [data-testid="stButton"] button:hover {
        background-color: var(--bg-secondary) !important;
        border-color: var(--border-hover) !important;
    }
    
    [data-testid="stButton"] button:active {
        transform: scale(0.98);
    }
    
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 0.5px solid var(--border) !important;
    }
    
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--text-primary) !important;
        border-bottom: 2px solid var(--text-primary) !important;
    }
    
    [data-testid="stTabs"] button[aria-selected="false"] {
        color: var(--text-secondary) !important;
    }
    
    .metric-card {
        background-color: var(--bg-secondary);
        border: 0.5px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }
    
    [data-testid="stMetric"] {
        background-color: var(--bg-secondary);
        border: 0.5px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem;
    }
    
    .divider {
        border-top: 0.5px solid var(--border);
        margin: 2rem 0;
    }
    
    [data-testid="stSelectbox"] svg,
    [data-testid="stNumberInput"] svg {
        fill: var(--text-secondary) !important;
    }
    
    .alert-new {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 3px solid var(--danger);
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    .alert-confirmed {
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 3px solid var(--warning);
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    .alert-false-positive {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 3px solid var(--success);
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.2);
        color: var(--danger);
    }
    
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: var(--warning);
    }
    
    .badge-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: var(--success);
    }
    
    .plotly-graph {
        background-color: var(--bg-secondary) !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model
try:
    model = joblib.load("fraud_detection_pipeline.pkl")
except:
    st.error("Model file not found. Make sure 'fraud_detection_pipeline.pkl' is in the same directory.")
    st.stop()

# Initialize session state
if "transactions" not in st.session_state:
    st.session_state.transactions = []

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "next_alert_id" not in st.session_state:
    st.session_state.next_alert_id = 1

# ==================== RULES ENGINE ====================
def check_rules(row):
    """
    Rule-based fraud detection
    Returns: (is_fraud, triggered_rules, risk_score)
    """
    rules_triggered = []
    risk_score = 0.0
    
    # Rule 1: Large amount transaction (> 20000)
    if row["amount"] > 20000:
        rules_triggered.append("Large amount transaction (>20000)")
        risk_score += 0.3
    
    # Rule 2: Unusual balance change (new balance much lower than old)
    balance_loss_sender = row["oldbalanceOrg"] - row["newbalanceOrig"]
    if row["oldbalanceOrg"] > 0:
        loss_ratio = balance_loss_sender / row["oldbalanceOrg"]
        if loss_ratio > 0.9:  # Lost 90% of balance
            rules_triggered.append("Unusual balance depletion (>90%)")
            risk_score += 0.25
    
    # Rule 3: Cash out transaction with large amount
    if row["type"] == "CASH_OUT" and row["amount"] > 5000:
        rules_triggered.append("Large cash out transaction")
        risk_score += 0.2
    
    # Rule 4: Receiver has zero balance before and after (suspicious)
    if row["oldbalanceDest"] == 0 and row["newbalanceDest"] == 0:
        rules_triggered.append("Receiver with zero balance (new account)")
        risk_score += 0.15
    
    # Rule 5: Transfer without corresponding balance change
    if row["type"] == "TRANSFER":
        expected_receiver_balance = row["oldbalanceDest"] + row["amount"]
        if row["newbalanceDest"] != expected_receiver_balance:
            rules_triggered.append("Inconsistent balance transfer")
            risk_score += 0.2
    
    # Rule 6: Multiple payment types in sequence (simplified - just check if payment)
    if row["type"] == "PAYMENT" and row["amount"] > 10000:
        rules_triggered.append("Large payment amount")
        risk_score += 0.15
    
    is_fraud = len(rules_triggered) > 0 or risk_score > 0.3
    
    return is_fraud, rules_triggered, min(risk_score, 1.0)  # Cap at 1.0

def generate_alert(transaction_data, ml_prediction, ml_confidence, rules_triggered, rule_risk_score):
    """
    Generate an alert if transaction is fraudulent
    """
    # Combine ML and rule-based scores
    combined_risk = (ml_confidence * 0.6 + rule_risk_score * 100 * 0.4) / 100
    
    if ml_prediction == 1 or len(rules_triggered) > 0:
        alert = {
            "alert_id": f"ALT-{st.session_state.next_alert_id:06d}",
            "transaction_id": f"TXN-{len(st.session_state.transactions):06d}",
            "timestamp": datetime.now(),
            "transaction_type": transaction_data["type"],
            "amount": transaction_data["amount"],
            "risk_score": combined_risk * 100,
            "ml_prediction": "FRAUDULENT" if ml_prediction == 1 else "LEGITIMATE",
            "ml_confidence": ml_confidence,
            "rules_triggered": rules_triggered,
            "status": "New",
        }
        st.session_state.next_alert_id += 1
        return alert
    return None

# ==================== PREDICTION PAGE ====================
def page_prediction():
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.title("Fraud Detection")
    st.markdown("Predict if a transaction is fraudulent or legitimate")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Transaction type**")
        transaction_type = st.selectbox(
            "Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**Amount**")
        amount = st.number_input(
            "Amount",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            label_visibility="collapsed"
        )
    
    st.markdown("**Sender balances**")
    col1, col2 = st.columns(2)
    
    with col1:
        oldbalanceOrg = st.number_input(
            "Old Balance (Sender)",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            label_visibility="collapsed"
        )
    
    with col2:
        newbalanceOrg = st.number_input(
            "New Balance (Sender)",
            min_value=0.0,
            value=9000.0,
            step=1000.0,
            label_visibility="collapsed"
        )
    
    st.markdown("**Receiver balances**")
    col1, col2 = st.columns(2)
    
    with col1:
        oldbalanceDest = st.number_input(
            "Old Balance (Receiver)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            label_visibility="collapsed"
        )
    
    with col2:
        newbalanceDest = st.number_input(
            "New Balance (Receiver)",
            min_value=0.0,
            value=1000.0,
            step=1000.0,
            label_visibility="collapsed"
        )
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    if st.button("Predict", use_container_width=True):
        transaction_data = {
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrg,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }
        
        try:
            # ML prediction
            input_df = pd.DataFrame([transaction_data])
            ml_prediction = model.predict(input_df)[0]
            ml_probability = model.predict_proba(input_df)[0]
            ml_confidence = max(ml_probability) * 100
            
            # Rule-based detection
            is_rule_fraud, rules_triggered, rule_risk_score = check_rules(transaction_data)
            
            # Store transaction
            transaction_record = {
                "timestamp": datetime.now(),
                "type": transaction_type,
                "amount": amount,
                "ml_prediction": "FRAUDULENT" if ml_prediction == 1 else "LEGITIMATE",
                "ml_confidence": ml_confidence,
                "rules_triggered": rules_triggered,
                "rule_risk_score": rule_risk_score
            }
            st.session_state.transactions.append(transaction_record)
            
            # Generate alert if fraudulent
            alert = generate_alert(transaction_data, ml_prediction, ml_confidence, rules_triggered, rule_risk_score)
            if alert:
                st.session_state.alerts.append(alert)
            
            # Display result
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([0.7, 0.3])
            
            with col1:
                st.markdown("**Prediction result**")
                if ml_prediction == 1 or is_rule_fraud:
                    st.error(f"🚨 FRAUDULENT")
                else:
                    st.success(f"✓ LEGITIMATE")
                st.markdown(f"**ML Confidence:** {ml_confidence:.1f}%")
                
                if rules_triggered:
                    st.markdown("**Rules triggered:**")
                    for rule in rules_triggered:
                        st.markdown(f"- {rule}")
                
                if alert:
                    st.info(f"⚠️ Alert created: {alert['alert_id']}")
            
            with col2:
                combined_risk = (ml_confidence * 0.6 + rule_risk_score * 100 * 0.4) / 100
                fig = go.Figure(data=[go.Pie(
                    values=[combined_risk, 100-combined_risk],
                    labels=["Fraud Risk", "Safe"],
                    marker=dict(colors=["#ef4444", "#10b981"]),
                    hole=0.7,
                    textinfo="none"
                )])
                fig.update_layout(
                    height=200,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(26, 31, 58, 0)",
                    plot_bgcolor="rgba(26, 31, 58, 0)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# ==================== ALERTS PAGE ====================
def page_alerts():
    st.title("Alerts Management")
    st.markdown("View and manage fraud detection alerts")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    if len(st.session_state.alerts) == 0:
        st.info("No alerts detected yet. Make predictions to generate alerts.")
        return
    
    # Metrics
    df_alerts = pd.DataFrame(st.session_state.alerts)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", len(df_alerts))
    
    with col2:
        new_count = len(df_alerts[df_alerts["status"] == "New"])
        st.metric("New Alerts", new_count)
    
    with col3:
        confirmed_count = len(df_alerts[df_alerts["status"] == "Confirmed"])
        st.metric("Confirmed", confirmed_count)
    
    with col4:
        fp_count = len(df_alerts[df_alerts["status"] == "False Positive"])
        st.metric("False Positives", fp_count)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.multiselect(
            "Filter by status",
            ["New", "Confirmed", "False Positive"],
            default=["New", "Confirmed"]
        )
    
    with col2:
        risk_threshold = st.slider("Minimum risk score", 0, 100, 50)
    
    with col3:
        st.write("")  # Spacing
    
    # Filter alerts
    filtered_alerts = [
        a for a in st.session_state.alerts 
        if a["status"] in status_filter and a["risk_score"] >= risk_threshold
    ]
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Display alerts
    st.markdown("**Alert Details**")
    
    for alert in sorted(filtered_alerts, key=lambda x: x["timestamp"], reverse=True):
        status_color = {
            "New": "danger",
            "Confirmed": "warning",
            "False Positive": "success"
        }[alert["status"]]
        
        with st.container():
            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
            
            with col1:
                st.markdown(f"""
                **{alert['alert_id']}** | {alert['transaction_id']}
                
                • **Amount:** ${alert['amount']:.2f}
                • **Risk Score:** {alert['risk_score']:.1f}%
                • **Timestamp:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                """)
                
                if alert["rules_triggered"]:
                    st.markdown("**Rules triggered:**")
                    for rule in alert["rules_triggered"]:
                        st.markdown(f"- {rule}")
            
            with col2:
                st.markdown(f"<span class='badge badge-{status_color}'>{alert['status']}</span>", unsafe_allow_html=True)
            
            with col3:
                # Status change buttons
                new_status = st.selectbox(
                    "Change status",
                    ["New", "Confirmed", "False Positive"],
                    key=f"status_{alert['alert_id']}"
                )
                if new_status != alert["status"]:
                    for a in st.session_state.alerts:
                        if a["alert_id"] == alert["alert_id"]:
                            a["status"] = new_status
                    st.rerun()
            
            st.markdown("---")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Export options
    st.markdown("**Export Options**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export to CSV", use_container_width=True):
            csv_data = []
            for alert in st.session_state.alerts:
                csv_data.append({
                    "Alert ID": alert["alert_id"],
                    "Transaction ID": alert["transaction_id"],
                    "Type": alert["transaction_type"],
                    "Amount": f"${alert['amount']:.2f}",
                    "Risk Score": f"{alert['risk_score']:.1f}%",
                    "Rules Triggered": "; ".join(alert["rules_triggered"]) if alert["rules_triggered"] else "None",
                    "Timestamp": alert["timestamp"].strftime('%Y-%m-%d %H:%M:%S'),
                    "Status": alert["status"]
                })
            
            df_export = pd.DataFrame(csv_data)
            csv_buffer = StringIO()
            df_export.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"fraud_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="csv_download"
            )
    
    with col2:
        if st.button("📊 Generate Report", use_container_width=True):
            # Summary report
            report_text = f"""
FRAUD DETECTION ALERTS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
-------
Total Alerts: {len(st.session_state.alerts)}
New Alerts: {len([a for a in st.session_state.alerts if a['status'] == 'New'])}
Confirmed Alerts: {len([a for a in st.session_state.alerts if a['status'] == 'Confirmed'])}
False Positives: {len([a for a in st.session_state.alerts if a['status'] == 'False Positive'])}

STATISTICS
----------
Total Amount at Risk: ${sum(a['amount'] for a in st.session_state.alerts):.2f}
Average Risk Score: {np.mean([a['risk_score'] for a in st.session_state.alerts]):.1f}%
Highest Risk Alert: {max([a['risk_score'] for a in st.session_state.alerts]):.1f}%

ALERT DETAILS
-------------
"""
            
            for alert in sorted(st.session_state.alerts, key=lambda x: x["risk_score"], reverse=True):
                report_text += f"""
Alert ID: {alert['alert_id']}
Transaction ID: {alert['transaction_id']}
Type: {alert['transaction_type']}
Amount: ${alert['amount']:.2f}
Risk Score: {alert['risk_score']:.1f}%
Timestamp: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Status: {alert['status']}
Rules Triggered: {', '.join(alert['rules_triggered']) if alert['rules_triggered'] else 'None'}
---
"""
            
            st.download_button(
                label="Download Report (TXT)",
                data=report_text,
                file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="report_download"
            )

# ==================== DASHBOARD PAGE ====================
def page_dashboard():
    st.title("Dashboard")
    st.markdown("Transaction analytics and statistics")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    if len(st.session_state.transactions) == 0:
        np.random.seed(42)
        sample_transactions = []
        for i in range(30):
            is_fraud = np.random.choice([0, 1], p=[0.85, 0.15])
            sample_transactions.append({
                "timestamp": datetime.now() - timedelta(days=np.random.randint(0, 30)),
                "type": np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]),
                "amount": np.random.uniform(100, 50000),
                "ml_prediction": "FRAUDULENT" if is_fraud else "LEGITIMATE",
                "ml_confidence": np.random.uniform(60, 99) if is_fraud else np.random.uniform(70, 99),
                "rules_triggered": [],
                "rule_risk_score": 0
            })
        st.session_state.transactions = sample_transactions
    
    df_trans = pd.DataFrame(st.session_state.transactions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", len(df_trans), delta="+3 today")
    
    with col2:
        fraud_count = len(df_trans[df_trans["ml_prediction"] == "FRAUDULENT"])
        fraud_rate = (fraud_count / len(df_trans) * 100) if len(df_trans) > 0 else 0
        st.metric("Fraud Detected", fraud_count, delta=f"{fraud_rate:.1f}%")
    
    with col3:
        avg_amount = df_trans["amount"].mean()
        st.metric("Avg Transaction", f"${avg_amount:.2f}", delta="+12%")
    
    with col4:
        avg_confidence = df_trans["ml_confidence"].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%", delta="+2.3%")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Transactions by type**")
        type_counts = df_trans["type"].value_counts()
        fig = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            labels={"x": "Type", "y": "Count"},
            color_discrete_sequence=["#3b82f6"]
        )
        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(26, 31, 58, 0)",
            plot_bgcolor="rgba(26, 31, 58, 0)",
            font=dict(color="#ffffff"),
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    with col2:
        st.markdown("**Fraud vs Legitimate**")
        fraud_counts = df_trans["ml_prediction"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=fraud_counts.index,
            values=fraud_counts.values,
            marker=dict(colors=["#10b981", "#ef4444"]),
            hole=0.4
        )])
        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(26, 31, 58, 0)",
            plot_bgcolor="rgba(26, 31, 58, 0)",
            font=dict(color="#ffffff"),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown("**Recent transactions**")
    df_display = df_trans.sort_values("timestamp", ascending=False).head(10).copy()
    df_display["timestamp"] = df_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    df_display["amount"] = df_display["amount"].apply(lambda x: f"${x:.2f}")
    df_display["ml_confidence"] = df_display["ml_confidence"].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        df_display[["timestamp", "type", "amount", "ml_prediction", "ml_confidence"]].rename(columns={
            "timestamp": "Time",
            "type": "Type",
            "amount": "Amount",
            "ml_prediction": "Status",
            "ml_confidence": "Confidence"
        }),
        use_container_width=True,
        hide_index=True
    )

# ==================== MAIN APP ====================
st.sidebar.title("Fraud Detection")
page = st.sidebar.radio("Navigate", ["Prediction", "Alerts", "Dashboard"])

if page == "Prediction":
    page_prediction()
elif page == "Alerts":
    page_alerts()
elif page == "Dashboard":
    page_dashboard()
