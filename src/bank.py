"""Bank marketing: which customers should the bank call?

A Portuguese bank phoned 45,211 customers about a term deposit; 11.7% said yes.
The model ranks customers by how likely they are to say yes, so the bank can
call the most promising ones first.

Call duration is left out on purpose: it is only known once the call has
ended, so it can't be used to decide who to call.

    python src/bank.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bank-full.csv"
RESULTS, IMAGES = ROOT / "results", ROOT / "images"

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
NUMERIC = ["age", "balance", "campaign", "pdays", "previous", "day"]
FEATURES = CATEGORICAL + NUMERIC   # 'duration' deliberately excluded (only known after the call)


def load(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["subscribed"] = (df["y"] == "yes").astype(int)
    return df


def split(df: pd.DataFrame):
    """Same 85/15 stratified split as the original notebook."""
    return train_test_split(df[FEATURES], df["subscribed"], test_size=0.15, random_state=42,
                            stratify=df["subscribed"])


def call_top(y_true, scores, share: float) -> dict:
    """If the bank calls only the top `share` of customers ranked by the model."""
    y_true = np.asarray(y_true)
    k = int(round(len(y_true) * share))
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return {"conversion": float(y_true[top].mean()), "share_of_all_yes": float(y_true[top].sum() / y_true.sum())}


def gains_curve(y_true, scores):
    """Share of all subscribers reached (y) as the bank works down the ranked list (x)."""
    y_true = np.asarray(y_true)[np.argsort(-np.asarray(scores), kind="stable")]
    x = np.arange(1, len(y_true) + 1) / len(y_true)
    return x, np.cumsum(y_true) / y_true.sum()


# ---------- the original notebook, reproduced ----------

def original_tree(df: pd.DataFrame) -> dict:
    """Original preprocessing (ordinal codes, duration and poutcome dropped) and the tuned tree."""
    d = df.copy()
    q99 = d["balance"].quantile(0.99)
    d["balance_class"] = pd.cut(d["balance"], [-np.inf, 72, 448, 1428, q99, np.inf], right=False,
                                labels=["E", "D", "C", "B", "A"]).astype(str)
    d["previous_bool"] = d["previous"] != 0
    X = d.drop(columns=["y", "subscribed", "balance", "duration", "previous", "day", "poutcome", "pdays"])
    X = OrdinalEncoder().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, d["subscribed"], test_size=0.15, random_state=42,
                                              stratify=d["subscribed"])
    grid = {"max_depth": [5, 10, 15, 20, 25, 30, None], "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 3], "min_samples_leaf": [1, 2]}
    pred = GridSearchCV(DecisionTreeClassifier(random_state=42), grid, cv=10, n_jobs=-1).fit(X_tr, y_tr).predict(X_te)
    return {
        # classification_report(pred, y_test) in the original swaps these two:
        "as_reported_recall": round(recall_score(pred, y_te), 3),
        "as_reported_precision": round(precision_score(pred, y_te), 3),
        "true_recall": round(recall_score(y_te, pred), 3),
        "true_precision": round(precision_score(y_te, pred), 3),
        "accuracy": round(accuracy_score(y_te, pred), 3),
        "always_no_accuracy": round(float(1 - y_te.mean()), 3),
    }


# ---------- models ----------

def models() -> dict:
    one_hot = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                                 ("num", StandardScaler(), NUMERIC)])
    one_hot_dense = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
                                       ("num", "passthrough", NUMERIC)])
    return {
        "Decision tree (depth 5)": make_pipeline(one_hot, DecisionTreeClassifier(max_depth=5, random_state=0)),
        "Logistic regression": make_pipeline(one_hot, LogisticRegression(max_iter=3000)),
        "Random forest": make_pipeline(one_hot, RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=-1)),
        "Gradient boosting": make_pipeline(one_hot_dense, HistGradientBoostingClassifier(random_state=0)),
    }


def evaluate(df: pd.DataFrame):
    X_tr, X_te, y_tr, y_te = split(df)
    rows, scores = [], {}
    rows.append({"model": "Random calling", "roc_auc": 0.5, "avg_precision": round(float(y_te.mean()), 3),
                 "top10_conversion": round(float(y_te.mean()), 3), "top10_share_of_yes": 0.1,
                 "top20_share_of_yes": 0.2})
    for name, model in models().items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        scores[name] = p
        t10, t20 = call_top(y_te, p, 0.10), call_top(y_te, p, 0.20)
        rows.append({"model": name, "roc_auc": round(roc_auc_score(y_te, p), 3),
                     "avg_precision": round(average_precision_score(y_te, p), 3),
                     "top10_conversion": round(t10["conversion"], 3),
                     "top10_share_of_yes": round(t10["share_of_all_yes"], 3),
                     "top20_share_of_yes": round(t20["share_of_all_yes"], 3)})
    return pd.DataFrame(rows), y_te, scores


def future_period_check(df: pd.DataFrame) -> dict:
    """Rows are in date order (May 2008 to Nov 2010). Train on the first 80%, test on the last 20%."""
    cut = int(len(df) * 0.8)
    model = models()["Gradient boosting"]
    model.fit(df[FEATURES].iloc[:cut], df["subscribed"].iloc[:cut])
    p = model.predict_proba(df[FEATURES].iloc[cut:])[:, 1]
    y = df["subscribed"].iloc[cut:]
    return {"roc_auc": round(roc_auc_score(y, p), 3), "test_yes_rate": round(float(y.mean()), 3),
            "train_yes_rate": round(float(df["subscribed"].iloc[:cut].mean()), 3)}


# ---------- charts ----------

def plot_gains(y_te, scores):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([0, 1], [0, 1], color="#9aa6b5", linestyle="--", label="Random calling")
    for (name, p), color in zip(scores.items(), ["#c9a227", "#7fa7c9", "#1f5f8b", "#0b3a5c"]):
        x, g = gains_curve(y_te, p)
        ax.plot(x, g, color=color, linewidth=1.6, label=name)
    best = max(scores, key=lambda k: roc_auc_score(y_te, scores[k]))
    t = call_top(y_te, scores[best], 0.2)
    ax.scatter([0.2], [t["share_of_all_yes"]], color="#0b3a5c", zorder=5)
    ax.annotate(f"Call the top 20%:\nreach {t['share_of_all_yes']:.0%} of all 'yes' customers",
                (0.2, t["share_of_all_yes"]), xytext=(0.32, 0.38), fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#5b6675"})
    ax.set_xlabel("Share of customers called (highest-ranked first)")
    ax.set_ylabel("Share of all subscribers reached")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Gains chart: who to call first", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGES / "gains_chart.png", dpi=150)
    plt.close(fig)


def plot_drivers(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    po = df.groupby("poutcome")["subscribed"].mean().reindex(["success", "other", "failure", "unknown"])
    labels = {"success": "Said yes last time", "other": "Other", "failure": "Said no last time",
              "unknown": "Not contacted before"}
    axes[0].barh([labels[i] for i in po.index][::-1], po.values[::-1],
                 color=["#9aa6b5", "#9aa6b5", "#9aa6b5", "#1f5f8b"])
    avg = df["subscribed"].mean()
    for y, v in enumerate(po.values[::-1]):
        axes[0].text(max(v, avg) + 0.012, y, f"{v:.0%}", va="center", fontsize=9)
    axes[0].axvline(avg, color="#b4473c", linestyle="--", linewidth=1)
    axes[0].text(avg + 0.005, 3.45, f"average {avg:.1%}", color="#b4473c", fontsize=8, va="bottom")
    axes[0].set_title("Yes rate by previous campaign result", loc="left", fontsize=11)
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_xlim(0, 0.75)

    block = np.arange(len(df)) // (len(df) // 10)
    rate = df.groupby(np.minimum(block, 9))["subscribed"].mean()
    axes[1].plot(rate.index + 1, rate.values, marker="o", color="#1f5f8b")
    axes[1].set_xticks(range(1, 11))
    axes[1].set_xlabel("Tenth of the campaign (in date order, May 2008 to Nov 2010)")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[1].set_ylim(0, rate.max() * 1.1)
    axes[1].set_title("Yes rate rose sharply over time", loc="left", fontsize=11)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGES / "what_drives_yes.png", dpi=150)
    plt.close(fig)


def main():
    df = load()
    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    original = original_tree(df)
    results, y_te, scores = evaluate(df)
    future = future_period_check(df)
    results.to_csv(RESULTS / "results.csv", index=False)
    (RESULTS / "original_tree_check.json").write_text(json.dumps(original, indent=2))
    (RESULTS / "future_period_check.json").write_text(json.dumps(future, indent=2))
    plot_gains(y_te, scores)
    plot_drivers(df)
    print("Original tree:", original)
    print("Future-period check:", future)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
