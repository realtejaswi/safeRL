import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="SafeRL-Manipulator Dashboard", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

st.title("SafeRL-Manipulator Dashboard")
st.markdown("Risk-Aware Residual Reinforcement Learning for Autonomous Robotic Manipulation")

st.header("1. BC Pretraining Statistics")
bc_stats_file = ARTIFACTS_DIR / "bc" / "data_stats.json"
if bc_stats_file.exists():
    with open(bc_stats_file, "r") as f:
        bc_stats = json.load(f)
    st.write(f"**Samples Used:** {bc_stats.get('count', 'N/A')}")
    st.write(f"**State Dimension:** {len(bc_stats.get('state_mean', []))}")
    st.write(f"**Action Dimension:** {len(bc_stats.get('action_mean', []))}")
else:
    st.info("BC statistics not found. Train the BC model first.")

st.header("2. RL Evaluation Comparison")
comp_file = ARTIFACTS_DIR / "results" / "comparison.csv"
if comp_file.exists():
    df = pd.read_csv(comp_file)
    st.dataframe(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Success Rate")
        st.bar_chart(df.set_index("method")["success_rate"])
    with col2:
        st.subheader("Collision Rate")
        st.bar_chart(df.set_index("method")["collision_rate"])
    with col3:
        st.subheader("Mean Min Clearance")
        st.bar_chart(df.set_index("method")["mean_min_clearance"])
else:
    st.info("No comparison results found. Run evaluation with multiple methods first.")
    
st.header("3. Latest Plots")
plots_dir = ARTIFACTS_DIR / "plots"
if plots_dir.exists():
    images = list(plots_dir.glob("*.png"))
    if images:
        cols = st.columns(3)
        for i, img_path in enumerate(images):
            cols[i % 3].image(str(img_path), caption=img_path.name)
    else:
        st.info("No plots found.")
else:
    st.info("Plot directory not found.")
