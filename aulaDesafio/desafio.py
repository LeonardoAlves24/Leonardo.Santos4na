import streamlit as st
from transformers import pipeline

st.title("Chatbot em Português")

# Usando um modelo disponível
chatbot = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-wikiSQL")

# Histórico
if "history" not in st.session_state:
    st.session_state.history = []

# Interface do usuário
with st.form(key="chat_form"):
    user_input = st.text_input("Você:", "")
    submit_button = st.form_submit_button(label="Enviar")

# Processamento de envio
if submit_button and user_input:
    st.session_state.history.append(("Você", user_input))

    # Adicionando um prefixo para o modelo gerar a resposta
    response = chatbot("chat: " + user_input, max_length=100, do_sample=True)[0]["generated_text"]
    st.session_state.history.append(("Bot", response))

# Exibição do histórico
for sender, msg in st.session_state.history:
    st.markdown(f"**{sender}:** {msg}")
