import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


parallel_example = {
    "한국어": ["오늘 날씨 어때", "딥러닝 기반의 AI기술이 인기를 끌고 있다."],
    "영어": [
        "How is the weather today",
        "Deep learning-based AI technology is gaining popularity.",
    ],
    "일본어": [
        "今日の天気はどうですか",
        "ディープラーニングベースのAIテクノロジーが人気を集めています。",
    ],
}


def translate_text_using_chatgpt(text: str, src_lang: str, trg_lang: str) -> str:
    def build_fewshot(src_lang, trg_lang):
        messages = []
        for s, t in zip(parallel_example[src_lang], parallel_example[trg_lang]):
            messages.append({"role": "user", "content": s})
            messages.append({"role": "assistant", "content": t})
        return messages

    system_instruction = (
        f"assistant는 번역 앱으로 동작한다. "
        f"{src_lang} 문장을 {trg_lang}으로 자연스럽게 번역하고 "
        f"번역된 텍스트만 출력한다."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        *build_fewshot(src_lang, trg_lang),
        {"role": "user", "content": text},
    ]

    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)

    return response.choices[0].message.content.strip()


st.set_page_config(
    page_title="AI Translation Service",
    page_icon="🌐",
    layout="centered",
)
st.markdown("## 🌐 AI Translation Service")
st.divider()

col1, col2 = st.columns(2)
with col1:
    src_lang = st.selectbox("원본 언어", ["영어", "한국어", "일본어"])
with col2:
    trg_lang = st.selectbox("목표 언어", ["영어", "한국어", "일본어"], index=1)

text = st.text_area("", placeholder="번역할 문장을 입력하세요", height=150)

if st.button("번역", use_container_width=True):
    # 번역 함수를 만들어서 (text, src_lang, trg_lang) -> translated_text
    translated_text = translate_text_using_chatgpt(text, src_lang, trg_lang)
    st.divider()
    st.success(translated_text)
