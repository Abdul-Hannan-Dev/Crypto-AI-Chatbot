
import streamlit as st
import time

from main import agent_executor


st.set_page_config(
    page_title="Cryptocurrency Info Agent",
    layout="centered"
)

st.title("Cryptocurrency Info Agent")


# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Display response character by character
def stream_response(response: str):
    placeholder = st.empty()
    full_response = ""

    for character in response:
        full_response += character
        placeholder.markdown(full_response + "▌")
        time.sleep(0.01)

    placeholder.markdown(full_response)


# Accept user input
if user_input := st.chat_input(
    "Ask me anything about cryptocurrencies..."
):

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            try:
                history = st.session_state.messages[:-1]

                response = agent_executor(
                    user_input,
                    history
                )

            except Exception as e:
                print(f"Frontend error: {e}")

                response = (
                    "Sorry, I couldn't process your request. "
                    "Please try again.\n\n"
                    "Confidence score: 0.00  \n"
                    "Source: N/A"
                )

        stream_response(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })


# Clear chat history button
if st.session_state.messages:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
