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

def generate_sample_topic(sampler):
    if not st.session_state.samples:
        samples = ", ".join(sampler.get_samples(1)).lower()
        st.session_state.samples = samples

def init_state():
    ss = st.session_state
    ss.setdefault("sent_target", "")
    ss.setdefault("sample", "")
    ss.setdefault("sent_eng", "")
    ss.setdefault("analysis_result", "")
    ss.setdefault("user_answer", "")
    ss.setdefault("submitted", False)
    ss.setdefault("samples", "")

# ------- UI ---------

def write_topic(): 
    st.markdown("**Target topic:**")
    st.write(st.session_state.samples)

def write_question(sampler): 
    st.markdown("**Literal Translation:**")
    q_container = st.empty()

    if not st.session_state.sent_target:
        # Generate target language
        with st.spinner("Generating sentence..."):
            st.session_state.sent_target = non_stream_prompt(prompts.get_target_sentence())
            print(st.session_state.sent_target)

        # Generate literal English translation
        with q_container.container():
            st.session_state.sent_eng = st.write_stream(
                stream_prompt(prompts.get_english_translation(st.session_state.sent_target)) # type: ignore
            )

        # Print hints
        new_dict = sampler.get_unknown_vocab(split_sentence(st.session_state.sent_target)) # type: ignore
        if (new_dict.keys()):
            st.markdown("**Hints**")
        for key in new_dict.keys():
            st.write("* " + key + ": " + new_dict[key])
    else: 
        st.write(st.session_state.sent_eng)

    return q_container

def write_input_box(container):
    def handle_user_answer():
        st.session_state.submitted = True

    def handle_regenerate_button():
        with container.container():
            st.session_state.sent_eng = st.write_stream(
                stream_prompt(prompts.get_english_translation(st.session_state.sent_target)) # type: ignore
            )


    cols = st.columns(2)
    with cols[0]: 
        st.text_input(
            f"Type {language} translation:",
            label_visibility="collapsed",
            value=st.session_state.user_answer,
            placeholder=f"Type the {language} translation here and press Enter",
            key="user_answer",
            on_change=handle_user_answer
        )
    with cols[1]: 
        st.button(
            "Regenerate question",
            on_click=handle_regenerate_button
        )

def write_analysis(): 
    if st.session_state.submitted and st.session_state.user_answer.strip():
        st.markdown("**Answer Key**")
        st.write(st.session_state.sent_target)
        st.markdown("**Result**")
        full = st.write_stream(
            stream_prompt(prompts.get_analysis_prompt(st.session_state.sent_eng, st.session_state.user_answer.strip()))
        )
        full = "full analysis"
        st.write(full)
        st.session_state.analysis_result = full


# -------- App --------

language = "Korean"
sampler = WordSampler(language, dict_paths)

init_state()
generate_sample_topic(sampler)

prompts = Prompts(language, st.session_state.samples)

st.title("Translation Checker")
write_topic()
q_container = write_question(sampler)
write_input_box(q_container)
st.divider()
write_analysis()
