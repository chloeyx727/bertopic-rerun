import base64
import gzip
import importlib.metadata as md
import json
import os
import threading
import traceback
from collections import Counter

from flask import Flask, jsonify

RESULT = {"status": "starting"}
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "analysis_status": RESULT.get("status", "unknown")})


@app.get("/result.json")
def result_json():
    return jsonify(RESULT)


def package_versions():
    names = [
        "bertopic",
        "sentence-transformers",
        "umap-learn",
        "hdbscan",
        "scikit-learn",
        "pandas",
        "jieba",
        "torch",
        "flask",
    ]
    out = {}
    for name in names:
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = None
    return out


def run_analysis():
    global RESULT
    try:
        RESULT = {"status": "loading_data"}

        import jieba
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP

        encoded = os.environ.get("DATA_GZ_B64")
        if not encoded:
            raise RuntimeError("DATA_GZ_B64 environment variable is missing")

        raw = gzip.decompress(base64.b64decode(encoded))
        records = json.loads(raw.decode("utf-8"))

        if len(records) != 140:
            raise RuntimeError(f"Expected 140 documents, found {len(records)}")

        participants = [str(r["participant"]) for r in records]
        prompts = [str(r["prompt"]) for r in records]
        texts = [str(r["text"]).strip() for r in records]

        n_participants = len(set(participants))
        if n_participants != 28:
            raise RuntimeError(f"Expected 28 participants, found {n_participants}")
        if any(not t for t in texts):
            raise RuntimeError("At least one computational document is blank")

        per_participant = Counter(participants)
        if set(per_participant.values()) != {5}:
            raise RuntimeError(
                "Expected exactly 5 reflective documents per participant; "
                f"distribution={dict(per_participant)}"
            )

        # A modest Chinese stop-word list is used only at topic-representation stage.
        # It is intentionally compact and does not alter the sentence-transformer embeddings.
        stop_words = list({
            "的", "了", "和", "是", "我", "也", "在", "就", "都", "而", "及", "与", "着", "或",
            "我们", "你", "你们", "他们", "她们", "它", "它们", "这", "那", "这些", "那些", "自己",
            "可以", "会", "有", "很", "更", "能", "让", "把", "被", "对", "中", "上", "下", "里", "到",
            "用", "使用", "进行", "通过", "因为", "所以", "但是", "如果", "还是", "以及", "并且", "对于",
            "方面", "时候", "之后", "之前", "目前", "本次", "一些", "这种", "这个", "那些", "就是", "比较",
            "非常", "可能", "需要", "觉得", "认为", "同时", "已经", "主要", "相关", "内容", "视频", "制作",
            "ai", "AI", "人工智能"
        })

        RESULT = {"status": "loading_embedding_model"}
        embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=10,
            min_samples=None,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        vectorizer_model = CountVectorizer(
            tokenizer=jieba.lcut,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 1),
            min_df=1,
            max_df=1.0,
            stop_words=stop_words,
        )

        ctfidf_model = ClassTfidfTransformer(
            bm25_weighting=False,
            reduce_frequent_words=False,
        )

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            top_n_words=10,
            nr_topics=None,
            calculate_probabilities=False,
            low_memory=False,
            verbose=True,
        )

        RESULT = {"status": "fitting_model"}
        topics, _ = topic_model.fit_transform(texts)
        topics = [int(t) for t in topics]
        counts = Counter(topics)
        non_outlier_topics = sorted(t for t in counts if t != -1)

        keyword_map = {}
        for topic_id in non_outlier_topics:
            vals = topic_model.get_topic(topic_id) or []
            keyword_map[str(topic_id)] = [
                {"term": str(term), "weight": float(weight)}
                for term, weight in vals[:10]
            ]

        prompt_composition = {}
        participant_composition = {}
        for topic_id in sorted(counts):
            indices = [i for i, t in enumerate(topics) if t == topic_id]
            prompt_composition[str(topic_id)] = dict(
                sorted(Counter(prompts[i] for i in indices).items())
            )
            participant_composition[str(topic_id)] = len(
                set(participants[i] for i in indices)
            )

        outlier_count = int(counts.get(-1, 0))
        result = {
            "status": "complete",
            "analysis_label": "default-based reconstructed BERTopic rerun; not recovery of the historical run",
            "n_documents": len(texts),
            "n_participants": n_participants,
            "documents_per_participant": {
                "min": min(per_participant.values()),
                "max": max(per_participant.values()),
                "mean": sum(per_participant.values()) / len(per_participant),
            },
            "n_topics_excluding_outlier": len(non_outlier_topics),
            "topic_counts": {str(k): int(v) for k, v in sorted(counts.items())},
            "outlier_count": outlier_count,
            "outlier_percent": round(100.0 * outlier_count / len(texts), 4),
            "keywords": keyword_map,
            "prompt_composition": prompt_composition,
            "unique_participants_per_topic": participant_composition,
            "parameters": {
                "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "umap": {
                    "n_neighbors": 15,
                    "n_components": 5,
                    "min_dist": 0.0,
                    "metric": "cosine",
                    "random_state": 42,
                },
                "hdbscan": {
                    "min_cluster_size": 10,
                    "min_samples": None,
                    "metric": "euclidean",
                    "cluster_selection_method": "eom",
                    "prediction_data": True,
                },
                "vectorizer": {
                    "tokenizer": "jieba.lcut",
                    "token_pattern": None,
                    "lowercase": False,
                    "ngram_range": [1, 1],
                    "min_df": 1,
                    "max_df": 1.0,
                    "chinese_stopword_filter": True,
                },
                "ctfidf": {
                    "bm25_weighting": False,
                    "reduce_frequent_words": False,
                },
                "bertopic": {
                    "top_n_words": 10,
                    "nr_topics": None,
                    "calculate_probabilities": False,
                    "low_memory": False,
                    "reduce_outliers_called": False,
                },
            },
            "versions": package_versions(),
            "privacy": "Aggregate output only; no reflection text or direct identifiers are exposed.",
        }
        RESULT = result

        with open("/tmp/result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("ANALYSIS_COMPLETE")
        print(json.dumps(result, ensure_ascii=False))

    except Exception as exc:
        RESULT = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print("ANALYSIS_ERROR")
        print(json.dumps(RESULT, ensure_ascii=False))


if __name__ == "__main__":
    threading.Thread(target=run_analysis, daemon=True).start()
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=True)
