# 🎓 AI in Indonesian Education - YouTube Comments Clustering & Topic Modeling

This repository contains my **unsupervised NLP project** that clusters **4,000 YouTube comments** (filtered down to ~1,679 education-relevant ones) discussing the **impact of AI on education in Indonesia**. The goal is to discover the main personas / opinion groups in public discourse, without any pre-labeled data.

👉 **Note:** All comment data belongs to its original authors (YouTube users). This repo is **for educational / portfolio purposes only**.
👉 The **scraping pipeline, EDA, preprocessing, clustering, and topic modeling** are my own work.

---

## ✨ Features

- End-to-end unsupervised NLP pipeline: **YouTube scraping -> keyword filter -> EDA -> preprocessing -> text representation -> clustering -> topic modeling -> persona analysis**
- **Two-stage scraping strategy** with priority + fallback video pools and a relevance scoring function (AI + education + Indonesia + impact keywords)
- **Two text representations** compared:
  - **TF-IDF** (unigram + bigram, sparse bag-of-words)
  - **SBERT embeddings** (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim semantic vectors)
- **Two unsupervised models** compared:
  - **K-Means** (with `k` grid search, evaluated via silhouette score + elbow / inertia)
  - **BERTopic** (UMAP -> HDBSCAN -> c-TF-IDF, evaluated via coherence `c_v` + topic diversity)
- **Hyperparameter tuning** for BERTopic (`n_neighbors`, `min_topic_size`) with coherence as the objective
- **Persona analysis** via qualitative sampling of 10 random comments per cluster/topic with a consistency re-check (different seed)
- Indonesian-specific preprocessing: custom stopwords from EDA, slang normalization, outlier chunking for very long comments

---

## ⚙️ Tech Stack

- **Language:** Python 3
- **Scraping:** YouTube Data API v3 (via `requests`), oEmbed (for video titles without an API key)
- **NLP / Preprocessing:** `re`, custom Indonesian stopword list derived from EDA
- **Text Representation:** `scikit-learn` (TF-IDF), `sentence-transformers` (SBERT multilingual)
- **Clustering:** `scikit-learn` (KMeans, silhouette score)
- **Topic Modeling:** `bertopic`, `umap-learn`, `hdbscan`
- **Coherence evaluation:** `gensim` (CoherenceModel, `c_v`)
- **EDA / Viz:** `pandas`, `numpy`, `matplotlib`, `IPython.display`

---

## 📊 Dataset

- **Source:** YouTube comments from Indonesian videos about AI and education, scraped via YouTube Data API v3
- **Strategy:**
  - 7 **priority videos** (hand-picked, directly about AI + education + Indonesia)
  - 8 **fallback videos** (broader AI / education topics, used only if the priority pool doesn't hit 4,000 "KEEP" comments)
  - Each comment is scored by a simple regex-based **relevance function** (AI pattern + education pattern + Indonesia pattern + impact pattern), and only comments with `score >= 6` are kept
- **Sizes at each stage:**

| Stage | Rows | Description |
|---|:---:|---|
| Raw scrape (KEEP pool) | 4,000 | Comments passing the relevance threshold |
| After education keyword refilter (`edu_hit`) | 1,679 | Contains at least one of 40+ Indonesian education keywords |
| After dedup + outlier chunking | 1,876 | Long comments (>205 words, P95) split into ~120-word chunks with 20-word overlap |

- **Target theme:** "Dampak kecerdasan buatan dalam pendidikan di Indonesia" (Impact of AI on education in Indonesia)

---

## 🔍 Key Insights from EDA

1. **Raw scrapes are noisy even with keyword filters.** Before the education refilter, the biggest video (*"MENGAPA PARA PAKAR AI MULAI KETAKUTAN DENGAN AI??"*) contributed 1,040 comments, but most of them were general AI takes, not about education. After the education keyword filter, it dropped to 208, and videos that are actually about **AI in education** (e.g. *Davyn Sudirdjo - AI Untuk Pendidikan*, *STEM*, *Kick Andy*) rose to the top. This is a nice example of why keyword refiltering on top of scraping matters.

2. **Comment length is heavily right-skewed.** Median = 36 words, P95 = 205 words, but the longest comment is **1,367 words**. A handful of very long essays dragged the mean way above the median. I handled this by chunking only the outliers (>P95), not every comment, to keep semantic units coherent.

3. **Unigrams alone are misleading.** Before adding stopwords, the top unigrams were all function words (`dan`, `yang`, `di`, `itu`, `ini`). After adding a custom Indonesian stopword list derived from the EDA itself, the top tokens finally became meaningful: `ai`, `pendidikan`, `guru`, `belajar`, `anak`, `indonesia`. **Bigrams** were even more useful: `dunia pendidikan`, `sistem pendidikan`, `penggunaan ai`, `menggunakan ai` jumped straight to the top.

4. **Education discourse in Indonesia is bimodal.** Two big themes keep appearing side by side: (a) **AI as a tool for students** (helpful but makes them lazy, copy-paste, joki tugas/skripsi), and (b) **the Indonesian education system itself** (teacher welfare, curriculum, access in rural areas, policy, public figures). This foreshadowed what clustering would later find.

5. **A surprise topic emerged: assignment submissions.** BERTopic surfaced a topic that turned out to be **homework answers for a school assignment** (lots of `nama-kelas-soal` templates, heavy `XI MPLB` class tag). These are AI-in-education content *technically*, but they're not organic public opinion - they're students answering a worksheet in the YouTube comments. An insight I wouldn't have found without topic modeling.

---

## 🧪 Clustering & Topic Modeling Results

### Phase 1 - K-Means on TF-IDF vs SBERT

I ran K-Means with `k` ∈ {2, 3, ..., 9} on both representations and evaluated with silhouette (cosine) and inertia (elbow).

| Representation | Best k (silhouette) | Best silhouette | Notes |
|---|:---:|:---:|---|
| TF-IDF | 8 | **~0.014** | Near-zero silhouette, clusters barely separable |
| **SBERT** | **2** | **~0.13** | Much cleaner separation, k=4 also competitive |

👉 **TF-IDF failed here.** Silhouette scores around 0.01 mean the clusters are essentially overlapping, this is the classical problem with TF-IDF on short, noisy, semantically-overlapping comments: two reviews saying the same thing with different words look completely different in TF-IDF space.

👉 **SBERT + K-Means at k=2** gave the most defensible baseline (silhouette ≈ 0.13, ~47/53 split, no tiny "garbage" cluster).

### Phase 2 - Persona Analysis on K-Means (SBERT, k=2)

| Cluster | Persona | Size | Consistency (re-sampled with seed=123) |
|:---:|---|:---:|:---:|
| 0 | "AI as a study tool + worries about thinking skills" - students using ChatGPT for homework, fear of becoming lazy / unable to think critically | 886 (47%) | ~60% of samples on-persona |
| 1 | "Indonesian education system + policy + public figures" - broader critique of curriculum, teacher welfare, corruption, access inequality | 990 (53%) | ~90% of samples on-persona |

👉 Cluster 1 is much more coherent than Cluster 0. Cluster 0 catches real AI-in-education complaints but also drifts into politics/religion/philosophy because SBERT places those in similar semantic space as "big-picture AI worries."

### Phase 3 - BERTopic (SBERT + UMAP + HDBSCAN + c-TF-IDF)

**Baseline** (`n_neighbors=15`, `min_topic_size=20`, reduced to 2 topics):
- 2 topics after reduction, but extremely imbalanced (1,141 vs 33) and **noise rate = 37.4%**

**After tuning** (grid over `n_neighbors` ∈ {10, 30} × `min_topic_size` ∈ {15, 30}):

| Config | Coherence (c_v) | Topic diversity | Noise rate |
|---|:---:|:---:|:---:|
| Baseline (15, 20) | 0.7439 | - | 37.4% |
| **Best: (10, 30)** | **0.7769** | high | **drastically lower** |

Best config produced two clearly interpretable topics:

| Topic | Persona | Consistency |
|:---:|---|:---:|
| 0 | "Macro discussion of education + opinion essays" (mixed AI takes, political figures, teacher/lecturer welfare, gen critique) | mostly consistent but top-words dominated by filler Indonesian words (`harus`, `semua`, `gak`, `kalo`) |
| 1 | "AI tools in learning - homework assignment template" (students answering a worksheet: definition of AI, tools like ChatGPT/Gemini/Canva, pros/cons) | very consistent but **not organic public opinion** - this is one school's assignment |

---

## 🧠 Lessons Learned

- **Silhouette on text is almost always tiny.** A silhouette of 0.13 *feels* bad, but for noisy short-text clustering it's actually among the better results you'll realistically get. Don't chase 0.5+ silhouettes - they usually mean you've collapsed everything into two trivial clusters.
- **SBERT >> TF-IDF for short noisy text.** The ~10x silhouette gap (0.01 -> 0.13) is exactly the kind of evidence that says "switch representation" rather than "tune the model more."
- **Topic modeling finds things clustering can't.** K-Means told me there are two personas. BERTopic told me *one of those personas is actually a school assignment being dumped into the comments*. That insight alone justifies running both.
- **Custom Indonesian stopwords matter more than model choice.** Filler words like `harus`, `semua`, `gak`, `kalo`, `aja`, `sih` dominated topic top-words even after tuning. A stronger stopword list (+ slang normalization `gak -> tidak`, `kalo -> kalau`, `yg -> yang`) is the highest-ROI next step.
- **Human-judgment consistency checks are underrated.** Re-sampling with a new seed and re-labeling the cluster's persona told me where my clusters are *actually* messy (Cluster 0 at ~60%) vs where they're solid (Cluster 1 at ~90%) - much more honest than just reporting silhouette.

---

## 🔮 Future Work

1. **Reduce noise before modeling** - stronger Indonesian stopwords + slang normalization, drop very short comments (< 5 words), filter out purely political / religious rants that got dragged in.
2. **Separate the "school assignment" template** as its own class before topic modeling, so organic public opinion isn't contaminated by homework dumps.
3. **Try UMAP -> HDBSCAN directly** (outside BERTopic) to allow non-spherical clusters that K-Means can't represent.
4. **Aspect-based analysis** - split each comment into aspects (tool usage, pedagogy, policy, ethics) since many Indonesian commenters mix all four in one long comment.
5. **Expand the video pool and the time window** to get a more representative picture of public opinion, currently the dataset is dominated by a handful of high-engagement videos.

---

## 🗂️ Repository Structure

```
ai-education-indonesia-clustering/
├─ README.md
├─ YoutubeScrapper.ipynb               # Step 1: scraping with priority + fallback + relevance scoring
├─ EDA_Preprocess_Model.ipynb          # Full end-to-end notebook (EDA -> preprocessing -> TF-IDF/SBERT -> KMeans -> BERTopic)
├─ data/
│  ├─ yt_comments_raw_4000.csv         # Raw scrape output (4000 comments that passed relevance threshold)
│  ├─ yt_comments_prioritized_4000.csv # With relevance_score column
│  ├─ yt_comments_keep_only.csv        # With video_title + comment URLs added
│  ├─ yt_comments_backup_pool.csv      # Backup pool (12000) in case refilter loses too many
│  ├─ yt_comments_edu_filtered.csv     # After education keyword refilter (1679 rows)
│  └─ yt_ai_pendidikan_comments.csv    # Final enriched dataset used for modeling
└─ requirements.txt
```

---

## 📒 Notebooks Overview

| Notebook | What's inside |
|---|---|
| `YoutubeScrapper.ipynb` | Scrapes ~4,000 comments from 15+ Indonesian YouTube videos using YouTube Data API v3, applies a 4-pattern relevance score (AI / Edu / Indonesia / Impact), keeps only comments with `score >= 6`, dumps a backup pool of 12,000 in case the KEEP pool needs topping up |
| `EDA_Preprocess_Model.ipynb` | Full pipeline: EDA (comment length, per-video distribution, top n-grams), education keyword refilter (1679/4000 kept), custom stopwords + slang handling, outlier chunking, TF-IDF + SBERT, K-Means grid search (k=2..9), persona analysis, BERTopic baseline + tuning, coherence evaluation, final conclusions |

---

## 🚀 How to Reproduce

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/ai-education-indonesia-clustering.git
cd ai-education-indonesia-clustering

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your YouTube API key (do NOT hard-code it in the notebook!)
export YOUTUBE_API_KEY="your_api_key_here"

# 4. (Optional) Re-scrape fresh comments
jupyter notebook YoutubeScrapper.ipynb

# 5. Run the main pipeline
jupyter notebook EDA_Preprocess_Model.ipynb
```

---

## License

- Code in this repository is released under the **MIT License**.
- YouTube comment data is scraped via the YouTube Data API v3 and belongs to its respective users / Google. This repository does **not claim ownership** of any comment content, it is included only for **non-commercial educational / portfolio purposes** under fair-use research context. If you are a rights holder and want content removed, please open an issue.
- Video thumbnails, titles, and channel names that appear in the dataset are the property of their respective owners.
