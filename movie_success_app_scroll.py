"""
Movie Success Predictor – Premium Tkinter UI (Scrollable Inputs)
Based on: tmdb_movie_success_classification_template_v7_threshold_optimization.ipynb

What this version fixes
- Adds a smooth vertical Scrollbar for the Inputs panel so you can always see all fields.
- Keeps ONLY the most important pre-release inputs used by the model (no leakage features).

Files expected next to this script (for first run training):
- tmdb_5000_movies.csv
- tmdb_5000_credits.csv

Artifacts saved for next runs:
- movie_success_model.joblib
- movie_success_meta.json

Optional logo:
- lion.png next to this script (auto-resized)

Run:
    python movie_success_app_scroll.py
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import f1_score

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

import tkinter as tk
from tkinter import ttk, messagebox


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "movie_success_model.joblib"
META_PATH = APP_DIR / "movie_success_meta.json"
MOVIES_CSV = APP_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = APP_DIR / "tmdb_5000_credits.csv"


# ----------------------------
# Data helpers (same logic as notebook/template)
# ----------------------------
def safe_eval_list(x):
    if not isinstance(x, str) or x.strip() == "":
        return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def first_name_from_json_list(x, default="Unknown"):
    v = safe_eval_list(x)
    if len(v) > 0 and isinstance(v[0], dict) and "name" in v[0]:
        return v[0]["name"]
    return default


def list_len(x):
    return len(safe_eval_list(x))


def extract_director(crew_str):
    crew = safe_eval_list(crew_str)
    for person in crew:
        if isinstance(person, dict) and person.get("job") == "Director":
            return person.get("name", "Unknown")
    return "Unknown"


def keep_top_n(series: pd.Series, n: int = 50) -> pd.Series:
    top = series.value_counts().head(n).index
    return series.where(series.isin(top), other="Other")


def oof_proba_groupkfold(estimator, X, y, groups, n_splits: int = 5) -> np.ndarray:
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(X), dtype=float)

    Xr = X.reset_index(drop=True)
    yr = y.reset_index(drop=True)
    gr = pd.Series(groups).reset_index(drop=True)

    for train_idx, val_idx in gkf.split(Xr, yr, gr):
        est = clone(estimator)
        est.fit(Xr.iloc[train_idx], yr.iloc[train_idx])
        oof[val_idx] = est.predict_proba(Xr.iloc[val_idx])[:, 1]
    return oof


def best_threshold_from_proba(y_true: pd.Series, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        score = f1_score(y_true, y_pred, average="weighted")
        if score > best_score:
            best_score = score
            best_t = float(t)
    return float(best_t)


@dataclass
class Artifacts:
    model: object
    threshold: float
    numeric_features: List[str]
    categorical_features: List[str]
    top_genres: List[str]
    top_companies: List[str]
    top_languages: List[str]
    top_directors: List[str]


def build_dataset() -> pd.DataFrame:
    if not MOVIES_CSV.exists() or not CREDITS_CSV.exists():
        raise FileNotFoundError(
            "CSV files not found.\n\n"
            "Please place these files next to the script:\n"
            f" - {MOVIES_CSV.name}\n"
            f" - {CREDITS_CSV.name}\n"
        )

    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)
    df = movies.merge(credits, left_on="id", right_on="movie_id", how="left")

    if "title_x" in df.columns:
        df = df.rename(columns={"title_x": "title"})
    if "title_y" in df.columns:
        df = df.drop(columns=["title_y"])

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month

    df["main_genre"] = df["genres"].apply(first_name_from_json_list)
    df["num_genres"] = df["genres"].apply(list_len)

    df["top_company"] = df["production_companies"].apply(first_name_from_json_list)
    df["num_production_companies"] = df["production_companies"].apply(list_len)

    df["is_english"] = (df["original_language"] == "en").astype(int)

    df["cast_size"] = df["cast"].apply(list_len)
    df["crew_size"] = df["crew"].apply(list_len)
    df["director_name"] = df["crew"].apply(extract_director)

    # Target: composite success score (profit/rating/votes/popularity) like in the template
    df["profit"] = df["revenue"] - df["budget"]
    df["profit_pos"] = df["profit"].clip(lower=0)

    P = np.log(df["profit_pos"] + 1)
    V = np.log(df["vote_count"] + 1)
    Pop = np.log(df["popularity"] + 1)
    Q = df["vote_average"] / 10

    df["FilmSuccessScore"] = 0.4 * P + 0.3 * Q + 0.2 * V + 0.1 * Pop
    threshold_target = df["FilmSuccessScore"].median()
    df["success"] = (df["FilmSuccessScore"] >= threshold_target).astype(int)

    # Categorical grouping
    df["director_group"] = keep_top_n(df["director_name"], n=80)
    df["company_group"] = keep_top_n(df["top_company"], n=80)
    df["lang_group"] = keep_top_n(df["original_language"], n=30)
    df["genre_group"] = keep_top_n(df["main_genre"], n=20)

    return df


def train_and_save_artifacts() -> Artifacts:
    if joblib is None:
        raise RuntimeError("joblib is required. Install it with: pip install joblib")

    df = build_dataset()

    # ✅ Keep ONLY the most important pre-release features (no leakage)
    numeric_features = [
        "budget",
        "runtime",
        "release_year",
        "release_month",
        "num_genres",
        "num_production_companies",
        "cast_size",
        "crew_size",
        "is_english",
    ]
    categorical_features = ["genre_group", "company_group", "lang_group", "director_group"]

    X = df[numeric_features + categorical_features].copy()
    y = df["success"].copy()
    groups = df["director_group"]

    X_train, _, y_train, _, g_train, _ = train_test_split(
        X, y, groups, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    log_reg = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear")),
        ]
    )

    param_grid_lr = {"model__C": [0.01, 0.1, 1, 10, 50]}
    cv_group = GroupKFold(n_splits=5)

    gs_lr = GridSearchCV(
        log_reg, param_grid=param_grid_lr, scoring="f1_weighted", cv=cv_group, n_jobs=-1
    )
    gs_lr.fit(X_train, y_train, groups=g_train)
    best_lr = gs_lr.best_estimator_

    oof = oof_proba_groupkfold(best_lr, X_train, y_train, g_train, n_splits=5)
    best_t = best_threshold_from_proba(y_train.reset_index(drop=True), oof)

    best_lr.fit(X, y)

    top_genres = list(pd.Series(df["genre_group"]).value_counts().index)
    top_companies = list(pd.Series(df["company_group"]).value_counts().index)
    top_languages = list(pd.Series(df["lang_group"]).value_counts().index)
    top_directors = list(pd.Series(df["director_group"]).value_counts().index)

    joblib.dump(best_lr, MODEL_PATH)
    meta = {
        "threshold": best_t,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "top_genres": top_genres,
        "top_companies": top_companies,
        "top_languages": top_languages,
        "top_directors": top_directors,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return Artifacts(
        model=best_lr,
        threshold=float(best_t),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        top_genres=top_genres,
        top_companies=top_companies,
        top_languages=top_languages,
        top_directors=top_directors,
    )


def load_artifacts() -> Artifacts:
    if joblib is None:
        raise RuntimeError("joblib is required. Install it with: pip install joblib")

    if MODEL_PATH.exists() and META_PATH.exists():
        model = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        return Artifacts(
            model=model,
            threshold=float(meta["threshold"]),
            numeric_features=list(meta["numeric_features"]),
            categorical_features=list(meta["categorical_features"]),
            top_genres=list(meta["top_genres"]),
            top_companies=list(meta["top_companies"]),
            top_languages=list(meta["top_languages"]),
            top_directors=list(meta["top_directors"]),
        )

    return train_and_save_artifacts()


def group_or_other(value: str, allowed: List[str]) -> str:
    v = (value or "").strip()
    if v == "":
        return "Other"
    return v if v in allowed else "Other"


def predict_success(art: Artifacts, inputs: Dict[str, object]) -> Tuple[int, float]:
    row = {}
    for f in art.numeric_features:
        row[f] = inputs.get(f, np.nan)
    for f in art.categorical_features:
        row[f] = inputs.get(f, "Other")

    X_row = pd.DataFrame([row])
    proba = float(art.model.predict_proba(X_row)[:, 1][0])
    pred = int(proba >= art.threshold)
    return pred, proba


# ----------------------------
# UI (Premium style + Scrollbar)
# ----------------------------
class ModernStyle:
    BG = "#070A12"
    CARD = "#0C1324"
    CARD_2 = "#0A1020"
    BORDER = "#24324A"
    TEXT = "#EAF0FF"
    MUTED = "#9AA7BD"
    GOLD = "#C7A02E"
    EMERALD = "#2CCB8A"
    BURGUNDY = "#B23A48"
    BLUE = "#3B82F6"

    FONT_TITLE = ("Segoe UI", 17, "bold")
    FONT_H1 = ("Segoe UI", 13, "bold")
    FONT_BODY = ("Segoe UI", 9)
    FONT_BODY_B = ("Segoe UI", 9, "bold")
    FONT_SMALL = ("Segoe UI", 8)


class ScrollableFrame(ttk.Frame):
    """A scrollable container (Canvas + inner Frame)."""
    def __init__(self, container, bg: str):
        super().__init__(container)

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self._window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        # Make inner frame width follow the canvas width
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel bindings
        self._bind_mousewheel()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _bind_mousewheel(self):
        # Windows / macOS
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        # Linux (common)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")
        # Touchpad horizontal can be ignored; we only vertical scroll.

    def _on_mousewheel(self, event):
        # On Windows event.delta is multiples of 120, on mac it's smaller
        delta = event.delta
        if delta == 0:
            return
        step = int(-1 * (delta / 120))
        if step == 0:
            step = -1 if delta > 0 else 1
        self.canvas.yview_scroll(step, "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Movie Success Predictor")
        self.configure(bg=ModernStyle.BG)

        # Fit to screen, but allow resizing
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = max(980, int(sw * 0.92))
        h = max(650, int(sh * 0.88))
        self.geometry(f"{w}x{h}")
        self.resizable(True, True)

        try:
            self.state("zoomed")  # Windows maximize if possible
        except Exception:
            pass

        self._set_ttk_theme()

        try:
            self.art = load_artifacts()
            self.status_text = "Model ready ✅"
        except Exception as e:
            self.art = None
            self.status_text = "Model not ready ❌"
            messagebox.showerror("Model error", str(e))

        self._build_layout()

    def _set_ttk_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=ModernStyle.BG, foreground=ModernStyle.TEXT, font=ModernStyle.FONT_BODY)
        style.configure("TFrame", background=ModernStyle.BG)
        style.configure("Card.TFrame", background=ModernStyle.CARD, relief="flat")

        style.configure("TLabel", background=ModernStyle.BG, foreground=ModernStyle.TEXT)
        style.configure("Title.TLabel", font=ModernStyle.FONT_TITLE, foreground=ModernStyle.GOLD, background=ModernStyle.BG)
        style.configure("Muted.TLabel", foreground=ModernStyle.MUTED, background=ModernStyle.BG, font=ModernStyle.FONT_SMALL)
        style.configure("CardMuted.TLabel", foreground=ModernStyle.MUTED, background=ModernStyle.CARD, font=ModernStyle.FONT_SMALL)
        style.configure("CardH1.TLabel", font=ModernStyle.FONT_H1, foreground=ModernStyle.TEXT, background=ModernStyle.CARD)

        style.configure(
            "TEntry",
            fieldbackground=ModernStyle.CARD_2,
            background=ModernStyle.CARD_2,
            foreground=ModernStyle.TEXT,
            bordercolor=ModernStyle.BORDER,
            lightcolor=ModernStyle.BORDER,
            darkcolor=ModernStyle.BORDER,
            padding=(10, 6),
        )
        style.configure(
            "TCombobox",
            fieldbackground=ModernStyle.CARD_2,
            background=ModernStyle.CARD_2,
            foreground=ModernStyle.TEXT,
            arrowcolor=ModernStyle.GOLD,
            padding=(10, 5),
        )
        style.map("TCombobox",
                  fieldbackground=[("readonly", ModernStyle.CARD_2)],
                  foreground=[("readonly", ModernStyle.TEXT)])

        style.configure(
            "TButton",
            font=ModernStyle.FONT_BODY_B,
            padding=(14, 9),
            background=ModernStyle.BLUE,
            foreground="white",
            borderwidth=0,
        )
        style.map(
            "TButton",
            background=[("active", ModernStyle.GOLD), ("disabled", ModernStyle.BORDER)],
            foreground=[("active", ModernStyle.BG)],
        )
        style.configure("Secondary.TButton", background=ModernStyle.CARD_2, foreground=ModernStyle.TEXT)
        style.configure("TSeparator", background=ModernStyle.BORDER)
        style.configure("Horizontal.TProgressbar", troughcolor=ModernStyle.CARD_2, background=ModernStyle.GOLD, thickness=12)

    def _draw_logo(self, parent: ttk.Frame):
        png = Path(__file__).with_name("lion.png")
        if png.exists():
            try:
                self._logo_img = tk.PhotoImage(file=str(png))
                target_h = 100
                w, h = self._logo_img.width(), self._logo_img.height()
                if h > 0:
                    factor = max(1, int(round(h / target_h)))
                    self._logo_img = self._logo_img.subsample(factor, factor)
                ttk.Label(parent, image=self._logo_img).pack(side="left", anchor="w")
                return
            except Exception:
                pass

        c = tk.Canvas(parent, width=28, height=28, bg=ModernStyle.BG, highlightthickness=0)
        c.pack(side="left", anchor="w")
        x0, y0, x1, y1 = 2, 2, 26, 26
        r = 6
        c.create_rectangle(x0 + r, y0, x1 - r, y1, fill=ModernStyle.CARD, outline=ModernStyle.BORDER, width=1)
        c.create_rectangle(x0, y0 + r, x1, y1 - r, fill=ModernStyle.CARD, outline=ModernStyle.BORDER, width=1)
        c.create_arc(x0, y0, x0 + 2 * r, y0 + 2 * r, start=90, extent=90, fill=ModernStyle.CARD, outline=ModernStyle.BORDER, width=1)
        c.create_arc(x1 - 2 * r, y0, x1, y0 + 2 * r, start=0, extent=90, fill=ModernStyle.CARD, outline=ModernStyle.BORDER, width=1)
        c.create_arc(x0, y1 - 2 * r, x0 + 2 * r, y1, start=180, extent=90, fill=ModernStyle.CARD, outline=ModernStyle.BORDER, width=1)
        c.create_arc(x1 - 2 * r, y1 - 2 * r, x1, y1, start=270, extent=90, fill=ModernStyle.CARD, outline=ModernStyle.BORDER, width=1)
        c.create_oval(17, 6, 21, 10, fill=ModernStyle.GOLD, outline="")
        c.create_text(13.5, 16, text="M", fill=ModernStyle.TEXT, font=("Segoe UI", 11, "bold"))

    def _build_layout(self):
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        # Header
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 10))

        left = ttk.Frame(header)
        left.pack(side="left", fill="x", expand=True)

        title_row = ttk.Frame(left)
        title_row.pack(anchor="w", fill="x")

        self._draw_logo(title_row)
        ttk.Label(title_row, text="Movie Success Predictor", style="Title.TLabel").pack(
            side="left", anchor="w", padx=(10, 0)
        )
        ttk.Label(
            left,
            text="Predict if a movie is likely to be a SUCCESS using a ML model trained on TMDB 5000.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        right = ttk.Frame(header)
        right.pack(side="right")
        ttk.Label(right, text=self.status_text, style="Muted.TLabel").pack(anchor="e")

        # Body grid
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3, uniform="col")
        body.columnconfigure(1, weight=2, uniform="col")
        body.rowconfigure(0, weight=1)

        # Cards
        inputs_card = ttk.Frame(body, style="Card.TFrame", padding=12)
        inputs_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        output_card = ttk.Frame(body, style="Card.TFrame", padding=12)
        output_card.grid(row=0, column=1, sticky="nsew")

        self._build_inputs(inputs_card)
        self._build_output(output_card)

    def _build_inputs(self, parent: ttk.Frame):
        ttk.Label(parent, text="Inputs (scrollable)", style="CardH1.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Only the most important pre-release features.", style="CardMuted.TLabel").pack(
            anchor="w", pady=(4, 10)
        )

        # ✅ Scrollable form area
        scroll = ScrollableFrame(parent, bg=ModernStyle.CARD)
        scroll.pack(fill="both", expand=True)

        form = scroll.scrollable_frame
        form.columnconfigure(0, weight=1, uniform="formcol")
        form.columnconfigure(1, weight=1, uniform="formcol")

        # Defaults
        self.vars = {
            "budget": tk.StringVar(value="50000000"),
            "runtime": tk.StringVar(value="120"),
            "release_year": tk.StringVar(value="2025"),
            "release_month": tk.StringVar(value="7"),
            "num_genres": tk.StringVar(value="2"),
            "num_production_companies": tk.StringVar(value="1"),
            "cast_size": tk.StringVar(value="10"),
            "crew_size": tk.StringVar(value="50"),
            "is_english": tk.IntVar(value=1),
            "lang_group": tk.StringVar(value="en"),
            "genre_group": tk.StringVar(value="Other"),
            "company_group": tk.StringVar(value="Other"),
            "director_group": tk.StringVar(value="Other"),
        }

        if self.art:
            self.genre_values = [g for g in self.art.top_genres if g != "Other"] + ["Other"]
            self.lang_values = [l for l in self.art.top_languages if l != "Other"] + ["Other"]
            self.company_values = [c for c in self.art.top_companies if c != "Other"] + ["Other"]
            self.director_values = [d for d in self.art.top_directors if d != "Other"] + ["Other"]
        else:
            self.genre_values = ["Action", "Drama", "Comedy", "Other"]
            self.lang_values = ["en", "fr", "es", "Other"]
            self.company_values = ["Other"]
            self.director_values = ["Other"]

        # Two-column grid
        r = 0
        r = self._field(form, r, 0, "Budget (USD)", "budget", hint="e.g., 50000000")
        r = self._field(form, r, 1, "Runtime (min)", "runtime", hint="e.g., 120")

        r = self._field(form, r, 0, "Release year", "release_year", hint="1900–2100")
        r = self._field(form, r, 1, "Release month", "release_month", hint="1–12")

        r = self._combo(form, r, 0, "Main genre", "genre_group", values=self.genre_values)
        r = self._combo(form, r, 1, "Language", "lang_group", values=self.lang_values, on_change=self._sync_english_flag)

        r = self._combo(form, r, 0, "Top company", "company_group", values=self.company_values)
        r = self._combo(form, r, 1, "Director group", "director_group", values=self.director_values)

        r = self._field(form, r, 0, "# Genres", "num_genres", hint="0–50")
        r = self._field(form, r, 1, "# Production companies", "num_production_companies", hint="0–50")

        r = self._field(form, r, 0, "Cast size", "cast_size", hint="0–500")
        r = self._field(form, r, 1, "Crew size", "crew_size", hint="0–2000")

        # Checkbox row
        chk = ttk.Frame(form, style="Card.TFrame")
        chk.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        ttk.Checkbutton(
            chk,
            text="Is English (auto-set from language)",
            variable=self.vars["is_english"]
        ).pack(anchor="w")

        # Buttons (outside scroll, fixed at bottom of Inputs card)
        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(10, 0))

        ttk.Button(btns, text="Predict", command=self.on_predict).pack(side="right")
        ttk.Button(btns, text="Reset", style="Secondary.TButton", command=self.on_reset).pack(side="right", padx=(0, 10))

        note = ttk.Label(
            parent,
            text=("Tip: If it says CSV files are missing, put tmdb_5000_movies.csv and "
                  "tmdb_5000_credits.csv next to this script, then restart."),
            style="Muted.TLabel",
            wraplength=720,
            justify="left",
        )
        note.pack(anchor="w", pady=(8, 0))

    def _field(self, parent, row, col, label, key, hint=""):
        container = ttk.Frame(parent, style="Card.TFrame")
        container.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else 10, 0), pady=6)
        container.columnconfigure(0, weight=1)

        ttk.Label(container, text=label, style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(container, textvariable=self.vars[key]).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(container, text=(hint or " "), style="CardMuted.TLabel").grid(row=2, column=0, sticky="w", pady=(2, 0))
        return row + 1

    def _combo(self, parent, row, col, label, key, values, on_change=None, hint=""):
        container = ttk.Frame(parent, style="Card.TFrame")
        container.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else 10, 0), pady=6)
        container.columnconfigure(0, weight=1)

        ttk.Label(container, text=label, style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        cb = ttk.Combobox(container, textvariable=self.vars[key], values=values, state="readonly")
        cb.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        if on_change:
            cb.bind("<<ComboboxSelected>>", lambda _e: on_change())
        ttk.Label(container, text=(hint or " "), style="CardMuted.TLabel").grid(row=2, column=0, sticky="w", pady=(2, 0))
        return row + 1

    def _sync_english_flag(self):
        lang = (self.vars["lang_group"].get() or "").strip().lower()
        self.vars["is_english"].set(1 if lang == "en" else 0)

    def _build_output(self, parent: ttk.Frame):
        ttk.Label(parent, text="Result", style="CardH1.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Prediction based on your inputs (probability + threshold).",
            style="CardMuted.TLabel",
            wraplength=420,
            justify="left",
        ).pack(anchor="w", pady=(4, 8))

        ttk.Separator(parent).pack(fill="x", pady=8)

        self.badge = ttk.Label(parent, text="—", font=("Segoe UI", 16, "bold"))
        self.badge.pack(anchor="w", pady=(2, 8))

        self.proba_label = ttk.Label(parent, text="Probability of success: —", style="Muted.TLabel")
        self.proba_label.pack(anchor="w")

        self.pb = ttk.Progressbar(parent, orient="horizontal", mode="determinate", maximum=100, value=0)
        self.pb.pack(fill="x", pady=(8, 6))

        self.thr_label = ttk.Label(parent, text="Threshold used: —", style="Muted.TLabel")
        self.thr_label.pack(anchor="w", pady=(4, 0))

        self.details = tk.Text(
            parent,
            height=13,
            bg=ModernStyle.CARD_2,
            fg=ModernStyle.TEXT,
            insertbackground=ModernStyle.GOLD,
            relief="flat",
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
            wrap="word",
            font=("Consolas", 8),
        )
        self.details.pack(fill="both", expand=True, pady=(10, 0))
        self.details.insert("1.0", "Fill the form and click Predict.\n")
        self.details.configure(state="disabled")

    def on_reset(self):
        defaults = {
            "budget": "50000000",
            "runtime": "120",
            "release_year": "2025",
            "release_month": "7",
            "num_genres": "2",
            "num_production_companies": "1",
            "cast_size": "10",
            "crew_size": "50",
            "lang_group": "en",
            "genre_group": "Other",
            "company_group": "Other",
            "director_group": "Other",
        }
        for k, v in defaults.items():
            if k in self.vars:
                self.vars[k].set(v)
        self._sync_english_flag()
        self._set_result(None)

    def _coerce_int(self, key: str, minv: int | None = None, maxv: int | None = None) -> int:
        s = (self.vars[key].get() or "").strip()
        if s == "":
            raise ValueError(f"{key} is required.")
        v = int(float(s))
        if minv is not None and v < minv:
            raise ValueError(f"{key} must be >= {minv}.")
        if maxv is not None and v > maxv:
            raise ValueError(f"{key} must be <= {maxv}.")
        return v

    def _coerce_float(self, key: str, minv: float | None = None) -> float:
        s = (self.vars[key].get() or "").strip()
        if s == "":
            raise ValueError(f"{key} is required.")
        v = float(s)
        if minv is not None and v < minv:
            raise ValueError(f"{key} must be >= {minv}.")
        return v

    def on_predict(self):
        if not self.art:
            messagebox.showerror("Model not ready", "Model artifacts could not be loaded or trained.")
            return

        try:
            budget = self._coerce_float("budget", minv=0.0)
            runtime = self._coerce_float("runtime", minv=1.0)
            release_year = self._coerce_int("release_year", minv=1900, maxv=2100)
            release_month = self._coerce_int("release_month", minv=1, maxv=12)
            num_genres = self._coerce_int("num_genres", minv=0, maxv=50)
            num_prod = self._coerce_int("num_production_companies", minv=0, maxv=50)
            cast_size = self._coerce_int("cast_size", minv=0, maxv=500)
            crew_size = self._coerce_int("crew_size", minv=0, maxv=2000)

            lang = self.vars["lang_group"].get()
            genre = self.vars["genre_group"].get()
            company = self.vars["company_group"].get()
            director = self.vars["director_group"].get()

            inputs = {
                "budget": budget,
                "runtime": runtime,
                "release_year": release_year,
                "release_month": release_month,
                "num_genres": num_genres,
                "num_production_companies": num_prod,
                "cast_size": cast_size,
                "crew_size": crew_size,
                "is_english": int(self.vars["is_english"].get()),
                "lang_group": group_or_other(lang, self.art.top_languages),
                "genre_group": group_or_other(genre, self.art.top_genres),
                "company_group": group_or_other(company, self.art.top_companies),
                "director_group": group_or_other(director, self.art.top_directors),
            }

            pred, proba = predict_success(self.art, inputs)
            self._set_result((pred, proba, inputs))

        except Exception as e:
            messagebox.showerror("Invalid input", str(e))

    def _set_result(self, result):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")

        if result is None:
            self.badge.configure(text="—", foreground=ModernStyle.MUTED)
            self.proba_label.configure(text="Probability of success: —")
            self.pb.configure(value=0)
            self.thr_label.configure(text="Threshold used: —")
            self.details.insert("1.0", "Fill the form and click Predict.\n")
            self.details.configure(state="disabled")
            return

        pred, proba, inputs = result
        pct = int(round(proba * 100))
        self.pb.configure(value=pct)

        if pred == 1:
            self.badge.configure(text="SUCCESS ✅", foreground=ModernStyle.EMERALD)
        else:
            self.badge.configure(text="FAILURE ❌", foreground=ModernStyle.BURGUNDY)

        self.proba_label.configure(text=f"Probability of success: {proba:.3f}  ({pct}%)")
        self.thr_label.configure(text=f"Threshold used: {self.art.threshold:.2f}")

        lines = [
            "Inputs used by the model (pre-release, most important):",
            f"  budget={inputs['budget']}",
            f"  runtime={inputs['runtime']}",
            f"  release_year={inputs['release_year']}",
            f"  release_month={inputs['release_month']}",
            f"  num_genres={inputs['num_genres']}",
            f"  num_production_companies={inputs['num_production_companies']}",
            f"  cast_size={inputs['cast_size']}",
            f"  crew_size={inputs['crew_size']}",
            f"  is_english={inputs['is_english']}",
            "",
            "Categorical groups:",
            f"  genre_group={inputs['genre_group']}",
            f"  company_group={inputs['company_group']}",
            f"  lang_group={inputs['lang_group']}",
            f"  director_group={inputs['director_group']}",
            "",
            "Note:",
            "  Model trained on TMDB 5000. Target is derived from a composite score",
            "  (profit, rating, votes, popularity) computed during training only.",
        ]
        self.details.insert("1.0", "\n".join(lines))
        self.details.configure(state="disabled")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
