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

# NLTK
DetectorFactory.seed = 0
nltk.download('punkt')
nltk.download('punkt_tab')

# -----------------------------------------------------------
# CUSTOM STREAMLIT UI THEME (CSS)
# -----------------------------------------------------------
def apply_custom_css():
    st.markdown("""
        <style>

        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
            color: white !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #111827;
            color: white;
            padding-top: 40px;
        }

        .css-1d391kg a {
            color: #38bdf8 !important;
        }

        /* Title */
        h1, h2, h3 {
            color: #ffffff !important;
            font-family: 'Segoe UI', sans-serif;
            font-weight: 600;
        }

        /* Input box */
        .stTextInput>div>div>input {
            background-color: #1f2937;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }

        /* Dropdown */
        .stSelectbox div {
            background-color: #1f2937 !important;
            color: white !important;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #1e3a8a, #3b82f6);
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            font-size: 16px;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #3b82f6, #1d4ed8);
            transform: scale(1.03);
        }

        /* Output card */
        .glass-card {
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(12px);
            padding: 25px;
            border-radius: 15px;
            margin-top: 20px;
            border: 1px solid rgba(255,255,255,0.15);
        }
        /* Prevent empty containers from showing a grey block */
        .glass-card:empty {
            display: none !important;
        }

        /* Only apply card styling to real content */
        .glass-card:not(:empty) {
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(12px);
            padding: 25px;
            border-radius: 15px;
            margin-top: 20px;
            border: 1px solid rgba(255,255,255,0.15);
        }

        /* Footer */
        #footer {
            text-align: center;
            padding: 20px;
            color: #9CA3AF;
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------
# SCRAPING CODE (unchanged)
# -----------------------------------------------------------
def get_article(url):
    art = scrape_custom(url)
    if art and len(art["text"]) > 200:
        return art

    art = scrape_newspaper3k(url)
    if art and len(art["text"]) > 200:
        return art

    return scrape_generic(url)

def scrape_custom(url):
    try:
        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "News Article"

        toi_blocks = soup.find_all("div", class_="Normal")
        if not toi_blocks:
            toi_blocks = soup.find_all("div", class_="ga-headlines")

        if toi_blocks:
            text = " ".join(block.get_text(" ", strip=True) for block in toi_blocks)
            return {"title": title, "text": clean_news_text(text)}

        article_tag = soup.find("article")
        if article_tag:
            p = [x.get_text(" ", strip=True) for x in article_tag.find_all("p")]
            text = " ".join(p)
            return {"title": title, "text": clean_news_text(text)}

        return None
    except:
        return None

def scrape_newspaper3k(url):
    try:
        a = Article(url)
        a.download()
        a.parse()
        return {"title": a.title, "text": clean_news_text(a.text)}
    except:
        return None

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

def clean_news_text(text):
    junk_patterns = [
        r"(?i)read more", r"(?i)also read", r"(?i)toi entertainment desk",
        r"(?i)recommended stories", r"(?i)most searched", r"(?i)trending",
        r"(?i)click here", r"(?i)follow us on", r"(?i)10\s+[a-zA-Z]+",
        r"(?i)we let chatgpt",
    ]
    for pattern in junk_patterns:
        text = re.sub(pattern, "", text)

    lines = text.split("\n")
    cleaned = []
    for l in lines:
        l = l.strip()
        if len(l) >= 20 and l.count(" ") >= 3:
            cleaned.append(l)
    cleaned_text = " ".join(cleaned)
    return re.sub(r"\s+", " ", cleaned_text).strip()


# -----------------------------------------------------------
# NLP + TF-IDF (unchanged)
# -----------------------------------------------------------
def clean_text(article_body):
    sentences = []
    article = article_body.split(". ")
    for sentence in article:
        sentence = re.sub(r"[^0-9A-Za-z\u0980-\u09FF\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF\u0A80-\u0AFF\u0D00-\u0D7F\s]",
                          " ", sentence)
        sentence = re.sub("\s+", " ", sentence).strip()
        if sentence:
            sentences.append(sentence)
    return sentences

def cnt_words(sent): return len(word_tokenize(sent))
def cnt_in_sent(sentences): return [{"id": i + 1, "word_cnt": cnt_words(s)} for i, s in enumerate(sentences)]
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
            tf.append({"id": item["id"], "key": word, "tf_score": item["freq_dict"][word] / denom})
    return tf
def calc_IDF(text_data, freq_list):
    idf_list = []
    N = len(text_data) or 1
    for item in freq_list:
        for word in item["freq_dict"]:
            df = sum([word in x["freq_dict"] for x in freq_list])
            idf_list.append({"id": item["id"], "key": word, "idf_score": math.log(N / (df + 1))})
    return idf_list
def calc_TFIDF(tf, idf):
    idf_lookup = {(i["id"], i["key"]): i["idf_score"] for i in idf}
    result = []
    for item in tf:
        key = (item["id"], item["key"])
        result.append({"id": item["id"], "key": item["key"], "tfidf_score": item["tf_score"] * idf_lookup.get(key, 0)})
    return result
def sent_scores(tfidf_scores, sentences, text_data):
    data = []
    for item in text_data:
        score = sum([x["tfidf_score"] for x in tfidf_scores if x["id"] == item["id"]])
        data.append({"id": item["id"], "score": score, "sentence": sentences[item["id"] - 1]})
    return data
def summary(sent_data, length):
    sent_data = sorted(sent_data, key=lambda x: x["score"], reverse=True)
    sizes = {"Low": 3, "Medium": 5, "High": 7}
    selected = sent_data[:sizes[length]]
    ordered = sorted(selected, key=lambda x: x["id"])
    return ". ".join([s["sentence"] for s in ordered]) + "."


# -----------------------------------------------------------
# AUDIO
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
# STREAMLIT FRONTEND – BRAND: “NewsPolyglot”
# -----------------------------------------------------------
st.set_page_config(page_title="NewsPolyglot", page_icon="🌐", layout="centered")
apply_custom_css()

# Sidebar
st.sidebar.title("🌐 NewsPolyglot")
st.sidebar.markdown("#### Multilingual News Summarizer\nGet clean, translated summaries with audio.")

page = st.sidebar.radio("Choose Mode", ["Full Article", "Summarizer"])

# -----------------------------------------------------------
# FULL ARTICLE PAGE
# -----------------------------------------------------------
if page == "Full Article":

    st.title("📰 Extract Full Article")
    st.markdown("Paste any news URL and extract clean text.")

    url = st.text_input("Enter Article URL")

    if st.button("Extract Article"):
        article = get_article(url)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(article["title"])
        st.write(article["text"])
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# SUMMARY PAGE
# -----------------------------------------------------------
else:
    st.title("🌐 NewsPolyglot — Multilingual News Summarizer")
    st.markdown("Summarize → Translate → Listen")

    url = st.text_input("Enter Article URL")

    languages = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Assamese": "as",
    "Punjabi": "pa",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Odia": "or"
}


    lang = st.selectbox("Translate summary to:", list(languages.keys()))
    length = st.radio("Summary Length:", ["Low", "Medium", "High"])
    auto = st.checkbox("Detect language automatically", value=True)

    if st.button("Generate Summary"):

        article = get_article(url)
        article_text = article["text"]

        if auto:
            try:
                detected = detect(article_text)
                st.info(f"Auto-Detected Language: **{detected}**")
            except:
                st.warning("Language auto-detection failed.")

        # NLP processing
        sentences = clean_text(article_text)
        text_data = cnt_in_sent(sentences)
        freq_list = freq_dict(sentences)
        tf_scores = calc_TF(text_data, freq_list)
        idf_scores = calc_IDF(text_data, freq_list)
        tfidf_scores = calc_TFIDF(tf_scores, idf_scores)
        sent_data = sent_scores(tfidf_scores, sentences, text_data)

        result = summary(sent_data, length)
        translated = GoogleTranslator(source="auto", target=languages[lang]).translate(result)

        # ONLY ONE CARD — remove the empty one
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.subheader(f"📝 Summary ({lang})")
        st.write(translated)

        audio_buf = summary_to_tts(translated, languages[lang])

        if audio_buf:
            st.audio(audio_buf, format="audio/mp3")
            st.download_button("⬇️ Download MP3", audio_buf, "summary.mp3", "audio/mpeg")

        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div id="footer">
NewsPolyglot © 2025 • Made for multilingual journalism ✨
</div>
""", unsafe_allow_html=True)
