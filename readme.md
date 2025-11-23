
# 📰 News Summarizer App

A powerful Streamlit-based application that can:

✔ Scrape full news articles from any website
✔ Automatically clean ads, junk, and TOI-specific unwanted blocks
✔ Summarize the article using TF–IDF NLP
✔ Translate the summary into 6 languages
✔ Generate audio (MP3) from the summary
✔ Provide fallback scraping using Newspaper3k

---

## 🚀 Features

### 🔎 Advanced Web Scraping

* Custom BeautifulSoup scraper
* Special handling for Times of India (TOI) articles
* Fallback to `newspaper3k` if custom scrape fails
* Final fallback generic HTML scraping

### 🧹 Smart Cleaning

Automatically removes:

* TOI Entertainment Desk text
* “Read More / Also Read” sections
* Trending/Recommended sections
* Ads, follow links, clutter

### 🧠 NLP Summarization

Uses TF–IDF scoring to extract key sentences.

### 🌍 Multi-language Translation

Supports:

* English (en)
* Hindi (hi)
* Bengali (bn)
* Marathi (mr)
* Tamil (ta)
* Telugu (te)

### 🔊 Audio Generation

Uses Google Text-to-Speech to produce an MP3 file of the translated summary.

---

## 📦 Installation

Install all dependencies in **one line**:

```bash
pip install streamlit requests beautifulsoup4 nltk deep-translator langdetect gTTS newspaper3k lxml cssselect
```

Additionally fix the new `lxml.clean` requirement:

```bash
pip install lxml_html_clean
```

Download NLTK tokenizer (app automatically runs this):

```python
nltk.download('punkt')
nltk.download('punkt_tab')
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py              # Main Streamlit application
├── README.md           # Documentation
└── requirements.txt    # (Optional) Install dependencies
```

---

## 🧩 How It Works

### 1. **Web Scraping**

The app first tries your custom scraper:

* Detects TOI article blocks (`div.Normal`)
* Removes unwanted HTML sections
* Falls back to `<article>` tag parsing
* Then uses Newspaper3k
* Finally generic scraping

### 2. **Text Cleaning**

Regex removes all unwanted lines.

### 3. **Summarization**

TF–IDF based:

* Tokenization
* Word frequency
* TF computation
* IDF computation
* Sentence scoring
* Selecting top-ranked sentences

### 4. **Translation**

Powered by `deep-translator` (Google Translate backend).

### 5. **Text-to-Speech**

Using `gTTS` with MP3 download support.

---

## 🌐 Supported Websites

Works well on:

* Times of India (TOI)
* NDTV
* ABP News
* Hindustan Times
* The Hindu
* Indian Express
* BBC
* CNN
* Any blog, news article, or HTML page with `<p>` tags

---

## 🛠 Future Enhancements

(You can request these anytime)

* AI summarization (HuggingFace models)
* Metadata extraction (author, date, tags)
* Extract images + captions
* PDF export

---

## 🤝 Credits

Built using:

* **Streamlit**
* **BeautifulSoup4**
* **Newspaper3k**
* **NLTK**
* **gTTS**
* **deep-translator**


