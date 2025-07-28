import google.generativeai as genai
import streamlit as st
import os

# Set up your API key
api_key = os.environ.get("GEMINI_API_KEY")

# Check if API key is set
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in environment variables")

# Streamlit app setup
st.title("Gemini Flash Query")

# Prompt input
prompt = st.text_input("Enter your prompt here")

# Button to trigger query
if st.button("Query"):
    if prompt:
        try:
            # Set up the model and make the query
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            st.write(response.text)
        except Exception as e:
            st.error(f"Error querying Gemini: {e}")
    else:
        st.error("Prompt cannot be blank")