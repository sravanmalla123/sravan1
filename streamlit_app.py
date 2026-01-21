import streamlit as st
from main import orchestrator

st.set_page_config(page_title="Multi Agent AI System", layout="centered")

st.title("🤖 Multi-Agent AI System")


user_input = st.text_area("Enter your topic / query:", height=120)

# Initialize session state
if "result" not in st.session_state:
    st.session_state.result = None

if "selected_output" not in st.session_state:
    st.session_state.selected_output = None

# Run orchestrator once
if st.button("Run Agents"):
    if not user_input.strip():
        st.warning("Please enter a valid query!")
    else:
        with st.spinner("Running agents... Please wait"):
            st.session_state.result = orchestrator(user_input)
            st.session_state.selected_output = "research"  # default show research first

# After running show 3 buttons
if st.session_state.result:
    st.markdown("### Choose Output")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔍 Research"):
            st.session_state.selected_output = "research"

    with col2:
        if st.button("📝 Summary"):
            st.session_state.selected_output = "summary"

    with col3:
        if st.button("✉️ Email"):
            st.session_state.selected_output = "email"

    if st.session_state.selected_output == "research":
        st.subheader("Research Output")
        st.write(st.session_state.result["research_output"])

    elif st.session_state.selected_output == "summary":
        st.subheader("Summary Output")
        st.write(st.session_state.result["summary_output"])

    elif st.session_state.selected_output == "email":
        st.subheader("Email Output")
        st.write(st.session_state.result["email_output"])
