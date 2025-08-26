import streamlit as st
import openai
from sampler import WordSampler
from prompts import Prompts
from konlpy.tag import Okt
from typing import List

BASE_URL = "http://192.168.18.14:11434/v1"
llm = openai.OpenAI(base_url=BASE_URL, api_key="dummy")
okt = Okt()

def stream_prompt(prompt: str):
    done_thinking = False
    stream = llm.chat.completions.create(
        model="hf.co/Qwen/Qwen3-8B-GGUF:Q8_0",
        stream=True,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048,
    )

    for event in stream: 
        chunk = event.choices[0].delta.content
        if (chunk and done_thinking): 
            yield chunk
        if (chunk == "</think>"):
            done_thinking = True

def non_stream_prompt(prompt: str): 
    text = llm.chat.completions.create(
        model="hf.co/Qwen/Qwen3-8B-GGUF:Q8_0",
        stream=False,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048,
    ).choices[0].message.content

    if (not text): 
        raise ValueError("LLM did not generate text")
    trimmed = text.replace("<think>", "")
    trimmed = trimmed.replace("</think>", "")
    trimmed = trimmed.strip()

    return trimmed

def split_sentence(sentence: str): 
    morphs = okt.pos(sentence, stem=True)
    valid_pos = ['Noun', 'Verb', 'Adjective', 'Adverb']
    vocabs: List[str] = [word for word, tag in morphs if tag in valid_pos]
    return vocabs

# -------- App Init --------
language = "Korean"
sampler = WordSampler("Korean")
samples = ", ".join(sampler.get_samples(1)).lower()

prompts = Prompts(language, samples)

st.title("Translation Checker")

st.markdown("**Target topic:**")
st.write(samples)

st.markdown("**Literal Translation:**")
# Session state
if "question" not in st.session_state:
    st.session_state.question = non_stream_prompt(prompts.get_question_prompt())
    print(st.session_state.question)

    st.session_state.sent_eng = st.write_stream(
        stream_prompt(prompts.get_english_translation(st.session_state.question)) # type: ignore
    )

    new_dict = sampler.get_unknown_vocab(split_sentence(st.session_state.question)) # type: ignore
    st.markdown("**Hints**")
    for key in new_dict.keys():
        st.write("* " + key + ": " + new_dict[key])
else: 
    st.write(st.session_state.sent_eng)
if "answer_result" not in st.session_state:
    st.session_state.answer_result = ""
if "user_answer" not in st.session_state:
    st.session_state.user_answer = ""


with st.form("answer_form"):
    user_answer = st.text_input(
        f"Type {language} translation:",
        value=st.session_state.user_answer,
        placeholder=f"Type the {language} translation here and press Enter",
    )
    submitted = st.form_submit_button("Send")

answer_key_heading = st.empty()
answer_key = st.empty()
out = st.empty()

if submitted and user_answer.strip():
    with answer_key_heading.container():
        st.markdown("**Answer Key**")
    with answer_key.container():
        st.write(st.session_state.question)
    with out.container():
        st.markdown("**Result**")
        full = st.write_stream(
            stream_prompt(prompts.get_answer_prompt(st.session_state.question, user_answer.strip()))
        )
    st.session_state.answer_result = full

if st.session_state.answer_result and not (submitted and user_answer.strip()):
    with out.container():
        st.markdown("**Result**")
        st.write(st.session_state.answer_result)
