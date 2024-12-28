import streamlit as st
import pandas as pd
from salesbot import RentalChatbot  # Assuming your class is saved as rental_chatbot.py
from logging import Logger as logger

# Initialize the chatbot
chatbot = RentalChatbot("gurgaon_10k.csv")

# Streamlit UI
def main():
    st.title("Rental Property Chatbot")
    st.write("Ask me anything about rental properties! Example queries:")
    st.markdown(
        "- I'm looking for a 2BHK apartment in Gurgaon under 30,000\n"
        "- Show me properties in DLF Phase 4 with 3 bedrooms and a balcony\n"
        "- Find me a fully furnished apartment with at least 1200 sqft area"
    )

    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
            st.session_state['clear'] = True
            st.session_state["messages"] = [
                {"role": "assistant",
                 "content": "Start the conversation"}]
    messages = st.session_state.messages

    for n, msg in enumerate(messages):
        st.chat_message(msg["role"]).write(msg["content"], unsafe_allow_html=True)
    
    if user_message := st.chat_input(placeholder="Ask me something"):
            # logger.debug(f"Input prompt: {prompt}")
            st.session_state.messages.append({"role": "user", "content": user_message})
            st.chat_message("user").write(user_message)
            with st.chat_message("assistant"):
                 with st.spinner('Generating Response...🕒'):
                    #   st.session_state.chat_history.append({"role": "user", "message": user_message})
                      chatbot_response = chatbot.get_response(user_message)
                    #   st.session_state.chat_history.append({"role": "assistant", "message": chatbot_response})
                      st.write(chatbot_response)
                      st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
                   
                    

    
if __name__ == "__main__":
    main()
