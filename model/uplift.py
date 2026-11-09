"""T-learner uplift modeli.

İki model eğitilir: biri indirim alanlar, biri kontrol grubu üzerinde.
uplift = P(yolculuk | indirim) − P(yolculuk | indirim yok)

Skorlar 2 katlı çapraz tahminle (cross-fitting) üretilir: her satır, kendisini hiç
görmemiş modellerle skorlanır. Gizli gerçek (`hidden_truth`) yalnızca doğrulamada okunur.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from config import ROOT, SEED, get_engine

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATURE_SQL = """
SELECT a.user_id, a.campaign_id, a.treated, o.rode,
       u.sehir, u.yas_grubu, u.kullanilan_hizmet,
       u.son_30_gun_yolculuk, u.uygulama_yasi_gun
FROM assignments a
JOIN outcomes o ON o.user_id = a.user_id AND o.campaign_id = a.campaign_id
JOIN users u ON u.user_id = a.user_id
"""
CATEGORICAL = ["campaign_id", "sehir", "yas_grubu", "kullanilan_hizmet"]
NUMERIC = ["son_30_gun_yolculuk", "uygulama_yasi_gun"]
QINI_PNG = ROOT / "docs" / "qini.png"


def load_training_frame(engine=None) -> pd.DataFrame:
    return pd.read_sql(FEATURE_SQL, engine or get_engine())


def _features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df[CATEGORICAL].astype(str), dtype=float)
    return pd.concat([X, df[NUMERIC].astype(float)], axis=1)


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=SEED)


def fit_t_learner(X: pd.DataFrame, treated: np.ndarray, y: np.ndarray):
    """Tedavi ve kontrol grupları için ayrı birer model eğitir."""
    m_t = _new_model().fit(X[treated == 1], y[treated == 1])
    m_c = _new_model().fit(X[treated == 0], y[treated == 0])
    return m_t, m_c


def cross_fit_scores(df: pd.DataFrame, n_splits: int = 2) -> pd.DataFrame:
    """Her satır için p_treated, p_control ve uplift skorlarını döndürür."""
    X, t, y = _features(df), df["treated"].to_numpy(), df["rode"].to_numpy()
    p_t, p_c = np.zeros(len(df)), np.zeros(len(df))
    for train_idx, test_idx in KFold(n_splits, shuffle=True, random_state=SEED).split(X):
        m_t, m_c = fit_t_learner(X.iloc[train_idx], t[train_idx], y[train_idx])
        p_t[test_idx] = m_t.predict_proba(X.iloc[test_idx])[:, 1]
        p_c[test_idx] = m_c.predict_proba(X.iloc[test_idx])[:, 1]
    return pd.DataFrame({
        "user_id": df["user_id"], "campaign_id": df["campaign_id"],
        "p_treated": p_t, "p_control": p_c, "uplift": p_t - p_c,
    })


def qini_curve(uplift: np.ndarray, treated: np.ndarray, y: np.ndarray, points: int = 100):
    """Skora göre sıralanmış kitlede kümülatif ek yolculuk (Qini eğrisi)."""
    order = np.argsort(-uplift)
    t, y = treated[order], y[order]
    cum_t, cum_c = np.cumsum(t), np.cumsum(1 - t)
    cum_yt, cum_yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    gain = cum_yt - cum_yc * np.divide(cum_t, cum_c, out=np.zeros(len(t)), where=cum_c > 0)
    idx = np.linspace(0, len(t) - 1, points).astype(int)
    return (idx + 1) / len(t), gain[idx]


def qini_coefficient(frac: np.ndarray, gain: np.ndarray) -> float:
    """Model eğrisi ile rastgele çizgi arasındaki alan (normalize)."""
    random_line = frac * gain[-1]
    return float(np.trapezoid(gain - random_line, frac) / max(abs(gain[-1]), 1e-9))


def validate(scores: pd.DataFrame, df: pd.DataFrame, engine=None, save_plot: bool = True) -> dict:
    """Tahmini uplift'i gizli gerçekle karşılaştırır ve Qini grafiğini çizer."""
    truth = pd.read_sql("SELECT user_id, campaign_id, true_uplift FROM hidden_truth", engine or get_engine())
    merged = scores.merge(truth, on=["user_id", "campaign_id"])
    corr = float(np.corrcoef(merged["uplift"], merged["true_uplift"])[0, 1])
    frac, gain = qini_curve(scores["uplift"].to_numpy(), df["treated"].to_numpy(), df["rode"].to_numpy())
    qini = qini_coefficient(frac, gain)
    if save_plot:
        _plot_qini(frac, gain)
    return {"korelasyon": round(corr, 3), "qini": round(qini, 3)}


def _plot_qini(frac: np.ndarray, gain: np.ndarray, path: Path = QINI_PNG) -> None:
    path.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(frac * 100, gain, color="#1f6feb", lw=2, label="Uplift modeline göre hedefleme")
    ax.plot([0, 100], [0, gain[-1]], color="#8b949e", ls="--", label="Rastgele hedefleme")
    ax.set_xlabel("İndirim gönderilen kitle (% , skora göre sıralı)")
    ax.set_ylabel("Kümülatif ek yolculuk")
    ax.set_title("Qini eğrisi: doğru kişiyi seçmek ne kazandırıyor?")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train_and_score(engine=None, save_plot: bool = True) -> dict:
    """Modeli eğitir, skorları `uplift_scores` tablosuna yazar, doğrulama metriklerini döner."""
    engine = engine or get_engine()
    df = load_training_frame(engine)
    scores = cross_fit_scores(df)
    scores.to_sql("uplift_scores", engine, if_exists="replace", index=False, chunksize=20_000)
    metrics = validate(scores, df, engine, save_plot)
    print(f"Uplift skorları yazıldı ({len(scores):,} satır). "
          f"Gerçek uplift ile korelasyon: {metrics['korelasyon']}, Qini katsayısı: {metrics['qini']}")
    return metrics
