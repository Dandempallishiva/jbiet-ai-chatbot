import streamlit as st
from src.jbiet_chatbot import chatbot

st.title("🎓 JBIET AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask me anything about JBIET...")

if question:

    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chatbot(question)

        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })