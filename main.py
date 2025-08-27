import streamlit as st
import openai
from sampler import WordSampler
from prompts import Prompts
from konlpy.tag import Okt
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

server_url = os.getenv("SERVER_URL")
dict_paths = {
    "indonesian": "~/Downloads/Indonesian_Vocabulary_beginner_to_intermediate_/collection.anki2",
    "korean": os.getenv("ANKI_PATH")
}

llm = openai.OpenAI(base_url=server_url, api_key="dummy")


@st.cache_resource
def get_okt():
    return Okt()
okt = get_okt()

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

def main(): 
    pass

language = "Korean"
sampler = WordSampler("Korean", dict_paths)
samples = ", ".join(sampler.get_samples(1)).lower()

prompts = Prompts(language, samples)

st.title("Translation Checker")

st.markdown("**Target topic:**")
st.write(samples)

st.markdown("**Literal Translation:**")

if "target_sent" not in st.session_state:
    # Generate target language
    with st.spinner("Generating sentence..."):
        st.session_state.target_sent = non_stream_prompt(prompts.get_target_sentence())
        print(st.session_state.target_sent)

    # Generate literal English translation
    st.session_state.sent_eng = st.write_stream(
        stream_prompt(prompts.get_english_translation(st.session_state.target_sent)) # type: ignore
    )

    # Print hints
    new_dict = sampler.get_unknown_vocab(split_sentence(st.session_state.target_sent)) # type: ignore
    if (new_dict.keys()):
        st.markdown("**Hints**")
    for key in new_dict.keys():
        st.write("* " + key + ": " + new_dict[key])
else: 
    st.write(st.session_state.sent_eng)
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""
if "user_answer" not in st.session_state:
    st.session_state.user_answer = ""


if (not st.session_state.submitted):
    with st.form("answer_form"):
        user_answer = st.text_input(
            f"Type {language} translation:",
            value=st.session_state.user_answer,
            placeholder=f"Type the {language} translation here and press Enter",
        )
        submitted = st.form_submit_button("Send")
        st.session_state.submitted = True

answer_key_heading = st.empty()
answer_key = st.empty()
out = st.empty()

if submitted and user_answer.strip():
    with answer_key_heading.container():
        st.markdown("**Answer Key**")
    with answer_key.container():
        st.write(st.session_state.target_sent)
    with out.container():
        st.markdown("**Result**")
        full = st.write_stream(
            stream_prompt(prompts.get_analysis_prompt(st.session_state.sent_eng, user_answer.strip()))
        )
    st.session_state.analysis_result = full

if st.session_state.analysis_result and not (submitted and user_answer.strip()):
    with out.container():
        st.markdown("**Result**")
        st.write(st.session_state.analysis_result)
