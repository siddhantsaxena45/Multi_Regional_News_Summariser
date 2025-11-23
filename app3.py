import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import math
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
import nltk
from langdetect import detect, DetectorFactory
from gtts import gTTS
import io
from newspaper import Article

# NLTK setup
DetectorFactory.seed = 0
nltk.download('punkt')
nltk.download('punkt_tab')

# -----------------------------------------------------------
# ADVANCED SCRAPER (Custom → Newspaper3k → Generic)
# -----------------------------------------------------------
def get_article(url):
    art = scrape_custom(url)
    if art and len(art["text"]) > 200:
        return art

    art = scrape_newspaper3k(url)
    if art and len(art["text"]) > 200:
        return art

    return scrape_generic(url)


# ---------------- CUSTOM SCRAPER ---------------- #
def scrape_custom(url):
    try:
        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "News Article"

        # TOI typical structure
        toi_blocks = soup.find_all("div", class_="Normal")
        if not toi_blocks:
            toi_blocks = soup.find_all("div", class_="ga-headlines")

        if toi_blocks:
            text = " ".join(block.get_text(" ", strip=True) for block in toi_blocks)
            return {"title": title, "text": clean_news_text(text)}

        # <article> tag fallback
        article_tag = soup.find("article")
        if article_tag:
            p = [x.get_text(" ", strip=True) for x in article_tag.find_all("p")]
            text = " ".join(p)
            return {"title": title, "text": clean_news_text(text)}

        return None

    except:
        return None


# ---------------- NEWSPAPER3K ---------------- #
def scrape_newspaper3k(url):
    try:
        a = Article(url)
        a.download()
        a.parse()
        return {
            "title": a.title,
            "text": clean_news_text(a.text)
        }
    except:
        return None


# ---------------- GENERIC SCRAPER ---------------- #
def scrape_generic(url):
    try:
        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")

        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "News Article"
        all_p = soup.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in all_p)
        return {"title": title, "text": clean_news_text(text)}

    except:
        return {"title": "News Article", "text": ""}


# ---------------- CLEANING NEWS TEXT ---------------- #
def clean_news_text(text):
    junk_patterns = [
        r"(?i)read more",
        r"(?i)also read",
        r"(?i)toi entertainment desk",
        r"(?i)recommended stories",
        r"(?i)most searched",
        r"(?i)trending",
        r"(?i)click here",
        r"(?i)follow us on",
        r"(?i)10\s+[a-zA-Z]+",
        r"(?i)we let chatgpt",
    ]

    for pattern in junk_patterns:
        text = re.sub(pattern, "", text)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        l = line.strip()
        if len(l) < 20:
            continue
        if l.count(" ") < 3:
            continue
        cleaned.append(l)

    cleaned_text = " ".join(cleaned)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text


# -----------------------------------------------------------
# NLP FUNCTIONS (TF–IDF SUMMARIZATION)
# -----------------------------------------------------------

def clean_text(article_body):
    sentences = []
    article = article_body.split(". ")

    for sentence in article:
        sentence = re.sub(
            r"[^0-9A-Za-z\u0980-\u09FF\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF\u0A80-\u0AFF\u0D00-\u0D7F\s]",
            " ", sentence
        )
        sentence = re.sub("\s+", " ", sentence).strip()
        if sentence:
            sentences.append(sentence)

    return sentences


def cnt_words(sent):
    return len(word_tokenize(sent))


def cnt_in_sent(sentences):
    return [{"id": i + 1, "word_cnt": cnt_words(sent)} for i, sent in enumerate(sentences)]


def freq_dict(sentences):
    data = []
    for i, sent in enumerate(sentences):
        freq = {}
        for word in word_tokenize(sent.lower()):
            freq[word] = freq.get(word, 0) + 1
        data.append({"id": i + 1, "freq_dict": freq})
    return data


def calc_TF(text_data, freq_list):
    tf = []
    for item in freq_list:
        for word in item["freq_dict"]:
            denom = text_data[item["id"] - 1]["word_cnt"] or 1
            tf.append({
                "id": item["id"],
                "key": word,
                "tf_score": item["freq_dict"][word] / denom
            })
    return tf


def calc_IDF(text_data, freq_list):
    idf_list = []
    N = len(text_data) or 1
    for item in freq_list:
        for word in item["freq_dict"]:
            df = sum([word in x["freq_dict"] for x in freq_list])
            idf_list.append({
                "id": item["id"],
                "key": word,
                "idf_score": math.log(N / (df + 1))
            })
    return idf_list


def calc_TFIDF(tf_scores, idf_scores):
    result = []
    idf_lookup = {(i["id"], i["key"]): i["idf_score"] for i in idf_scores}
    for tf in tf_scores:
        key = (tf["id"], tf["key"])
        idf_val = idf_lookup.get(key, 0)
        result.append({
            "id": tf["id"],
            "key": tf["key"],
            "tfidf_score": tf["tf_score"] * idf_val
        })
    return result


def sent_scores(tfidf_scores, sentences, text_data):
    data = []
    for item in text_data:
        score = sum([x["tfidf_score"] for x in tfidf_scores if x["id"] == item["id"]])
        data.append({
            "id": item["id"],
            "score": score,
            "sentence": sentences[item["id"] - 1]
        })
    return data


def summary(sent_data, length):
    sent_data = sorted(sent_data, key=lambda x: x["score"], reverse=True)

    if length == "Low":
        selected = sent_data[:3]
    elif length == "Medium":
        selected = sent_data[:5]
    else:
        selected = sent_data[:7]

    ordered = sorted(selected, key=lambda x: x["id"])
    return ". ".join([s["sentence"] for s in ordered]) + "."


# -----------------------------------------------------------
# AUDIO GENERATION
# -----------------------------------------------------------
def summary_to_tts(text, lang_code="en"):
    try:
        tts = gTTS(text=text, lang=lang_code)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except:
        return None


# -----------------------------------------------------------
# STREAMLIT APP UI
# -----------------------------------------------------------
st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="centered")

st.sidebar.title("News Summarizer 📰")
page = st.sidebar.radio("Choose Option", ["Full Text", "Summary"])

if page == "Full Text":
    st.title("Full Article Extractor")
    url = st.text_input("Enter article URL:")

    if st.button("Extract"):
        article = get_article(url)
        st.subheader(article["title"])
        st.write(article["text"])

else:
    st.title("News Summarizer + Translation + Audio")

    url = st.text_input("Enter Article URL:")

    languages = {
        "English": "en",
        "Hindi": "hi",
        "Marathi": "mr",
        "Tamil": "ta",
        "Telugu": "te",
        "Bengali": "bn"
    }

    lang = st.selectbox("Translate summary into:", list(languages.keys()))
    length = st.radio("Summary Length:", ["Low", "Medium", "High"])
    auto = st.checkbox("Auto-detect language", value=True)

    if st.button("Summarize"):

        article = get_article(url)
        article_text = article["text"]

        if auto:
            try:
                detected = detect(article_text)
                st.info(f"Detected Language: {detected}")
            except:
                st.warning("Language detection failed.")

        sentences = clean_text(article_text)
        text_data = cnt_in_sent(sentences)
        freq_list = freq_dict(sentences)
        tf_scores = calc_TF(text_data, freq_list)
        idf_scores = calc_IDF(text_data, freq_list)
        tfidf_scores = calc_TFIDF(tf_scores, idf_scores)
        sent_data = sent_scores(tfidf_scores, sentences, text_data)

        result = summary(sent_data, length)
        translated = GoogleTranslator(source="auto", target=languages[lang]).translate(result)

        st.subheader(f"Summary in {lang}")
        st.write(translated)

        audio_buf = summary_to_tts(translated, languages[lang])
        if audio_buf:
            st.audio(audio_buf, format="audio/mp3")
            st.download_button("Download MP3", audio_buf, "summary.mp3", "audio/mpeg")
