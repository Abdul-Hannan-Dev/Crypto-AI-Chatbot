import streamlit as st
from main import agent_executor

st.title("Cryptocurrency Info Agent")

if 'messages' not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                    for i in message['content']:
                        st.markdown(i.upper())
                        st.markdown(message['content'][i])
                        print(message['content'][i])
try:
    if prompt := st.chat_input("Ask me anything about Cryptocurrencies..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = agent_executor(prompt, st.session_state.messages)

                        for i in response:
                            st.markdown(i.upper())
                            st.markdown(response[i])
            st.session_state.messages.append({
                    "role": "user","content": prompt}
                )
            st.session_state.messages.append({'role': "assistant","content": response})
except:
    st.error("An error occurred while processing your request. Please try again.")