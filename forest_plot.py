from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.figure
import pandas as pd
from scipy.stats import chi2

REQUIRED_COLUMNS = [
    "Название исследования",
    "К-во объектов в исследовании",
    "Среднее",
    "Ширина 95% доверительного интервала",
]

Z_95 = 1.959964

MODEL_LABELS = {
    "fixed": "фиксированная модель",
    "random": "случайная модель",
}

MODEL_METHOD_NOTES = {
    "fixed": "Метод: инверсная дисперсия (fixed-effect)",
    "random": "Метод: DerSimonian–Laird (random-effects)",
}


class DataFormatError(ValueError):
    pass


@dataclass
class Study:
    name: str
    n: float
    mean: float
    ci_width: float

    @property
    def half_width(self) -> float:
        return self.ci_width / 2

    @property
    def lower(self) -> float:
        return self.mean - self.half_width

    @property
    def upper(self) -> float:
        return self.mean + self.half_width

    @property
    def se(self) -> float:
        return self.half_width / Z_95


@dataclass
class Heterogeneity:
    q: float
    df: int
    p_value: float
    i2: float
    tau2: float


def load_studies(path: str | Path) -> tuple[list[Study], list[str]]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise DataFormatError(f"Неподдерживаемый формат файла: {path.suffix}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataFormatError(
            "В файле отсутствуют обязательные колонки: " + ", ".join(missing)
        )

    studies: list[Study] = []
    warnings: list[str] = []

    for pos, (_, row) in enumerate(df.iterrows()):
        file_row = pos + 2
        name_raw, n_raw, mean_raw, width_raw = (row[c] for c in REQUIRED_COLUMNS)

        if any(pd.isna(v) for v in (name_raw, n_raw, mean_raw, width_raw)):
            warnings.append(f"Строка {file_row}: пропущены данные — строка пропущена")
            continue

        try:
            n = float(n_raw)
            mean = float(mean_raw)
            ci_width = float(width_raw)
        except (TypeError, ValueError):
            warnings.append(f"Строка {file_row}: нечисловые данные в числовой колонке — строка пропущена")
            continue

        if n <= 0:
            warnings.append(f"Строка {file_row}: количество объектов должно быть положительным — строка пропущена")
            continue
        if ci_width < 0:
            warnings.append(f"Строка {file_row}: ширина доверительного интервала отрицательна — строка пропущена")
            continue
        if not float(n).is_integer():
            warnings.append(f"Строка {file_row}: количество объектов не целое число — округлено до {round(n)}")
            n = float(round(n))

        studies.append(Study(name=str(name_raw), n=n, mean=mean, ci_width=ci_width))

    if not studies:
        raise DataFormatError("Файл не содержит ни одной корректной строки данных.")

    warnings.extend(_duplicate_name_warnings(studies))
    warnings.extend(_outlier_warnings(studies))

    return studies, warnings


def _duplicate_name_warnings(studies: list[Study]) -> list[str]:
    counts = Counter(s.name for s in studies)
    return [
        f"Название исследования повторяется: «{name}» ({count} раза) — проверьте, не задвоены ли данные"
        for name, count in counts.items() if count > 1
    ]


def _outlier_warnings(studies: list[Study], z_threshold: float = 2.5) -> list[str]:
    if len(studies) < 3:
        return []
    means = [s.mean for s in studies]
    avg = statistics.mean(means)
    spread = statistics.pstdev(means)
    if spread == 0:
        return []
    result = []
    for s in studies:
        z = abs(s.mean - avg) / spread
        if z > z_threshold:
            result.append(
                f"Исследование «{s.name}»: среднее сильно отличается от остальных "
                f"(возможный выброс, z≈{z:.1f}) — проверьте данные"
            )
    return result


def _fixed_weights(studies: list[Study]) -> list[float]:
    return [1 / (s.se ** 2) if s.se > 0 else 0.0 for s in studies]


def compute_heterogeneity(studies: list[Study]) -> Heterogeneity:
    weights = _fixed_weights(studies)
    total_w = sum(weights)
    fixed_mean = sum(w * s.mean for w, s in zip(weights, studies)) / total_w
    q = sum(w * (s.mean - fixed_mean) ** 2 for w, s in zip(weights, studies))
    df = len(studies) - 1

    if df <= 0:
        return Heterogeneity(q=q, df=0, p_value=1.0, i2=0.0, tau2=0.0)

    p_value = float(chi2.sf(q, df))
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0

    sum_w2 = sum(w ** 2 for w in weights)
    c = total_w - sum_w2 / total_w if total_w > 0 else 0.0
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0

    return Heterogeneity(q=q, df=df, p_value=p_value, i2=i2, tau2=tau2)


def compute_weights(studies: list[Study], model: str, het: Heterogeneity | None = None) -> list[float]:
    if model == "random":
        het = het or compute_heterogeneity(studies)
        return [1 / (s.se ** 2 + het.tau2) if (s.se ** 2 + het.tau2) > 0 else 0.0 for s in studies]
    return _fixed_weights(studies)


@dataclass
class MetaAnalysisReport:
    studies: list[Study]
    total: Study
    weight_percents: list[float]
    heterogeneity: Heterogeneity
    model: str


def analyze(studies: list[Study], model: str = "fixed") -> MetaAnalysisReport:
    het = compute_heterogeneity(studies)
    weights = compute_weights(studies, model, het)
    total_weight = sum(weights)
    pooled_mean = sum(w * s.mean for w, s in zip(weights, studies)) / total_weight
    pooled_se = (1 / total_weight) ** 0.5
    total = Study(
        name="Итого",
        n=sum(s.n for s in studies),
        mean=pooled_mean,
        ci_width=2 * Z_95 * pooled_se,
    )
    weight_percents = [100 * w / total_weight for w in weights]
    return MetaAnalysisReport(
        studies=studies, total=total, weight_percents=weight_percents,
        heterogeneity=het, model=model,
    )


def build_figure(studies: list[Study], model: str = "fixed") -> matplotlib.figure.Figure:
    report = analyze(studies, model)
    total, het = report.total, report.heterogeneity
    n_rows = len(studies) + 2

    fig = matplotlib.figure.Figure(figsize=(11, 0.55 * n_rows + 2.6), dpi=150, layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1.4], height_ratios=[n_rows, 0.55], wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis("off")

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax_text.set_facecolor("#ffffff")

    y_positions = list(range(len(studies), 0, -1))
    max_w = max(report.weight_percents) or 1

    for y, s, w in zip(y_positions, studies, report.weight_percents):
        marker_size = 50 + 260 * (w / max_w)
        ax.plot([s.lower, s.upper], [y, y], color="#2f6f9f", linewidth=1.6, zorder=2)
        ax.scatter(
            [s.mean], [y],
            s=marker_size, marker="s", color="#2f6f9f",
            zorder=3, edgecolors="#1c4a68", linewidths=0.8,
        )

    diamond_y = 0
    diamond_x = [total.lower, total.mean, total.upper, total.mean]
    diamond_y_pts = [diamond_y, diamond_y + 0.28, diamond_y, diamond_y - 0.28]
    ax.fill(diamond_x, diamond_y_pts, color="#c0392b", zorder=3)

    ax.axhline(diamond_y + 0.6, color="#cccccc", linewidth=1, zorder=1)

    all_labels = [s.name for s in studies] + ["Итого"]
    all_y = y_positions + [diamond_y]
    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=10)

    ax.axvline(total.mean, color="#c0392b", linestyle="--", linewidth=1, alpha=0.5, zorder=0)

    ax.set_ylim(diamond_y - 1, len(studies) + 1)
    ax.set_xlabel("Значение (среднее и 95% ДИ)", fontsize=10)
    ax.set_title("Форест-плот", fontsize=14, fontweight="bold", pad=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    ax_text.set_xlim(0, 1)
    ax_text.axis("off")
    ax_text.text(0, len(studies) + 0.7, "N", fontsize=9, fontweight="bold", family="monospace")
    ax_text.text(0.18, len(studies) + 0.7, "Среднее [95% ДИ]", fontsize=9, fontweight="bold", family="monospace")
    ax_text.text(0.82, len(studies) + 0.7, "Вес", fontsize=9, fontweight="bold", family="monospace")

    for y, s, w in zip(y_positions, studies, report.weight_percents):
        ax_text.text(0, y, f"{s.n:.0f}", fontsize=9, family="monospace", va="center")
        ax_text.text(0.18, y, f"{s.mean:.2f} [{s.lower:.2f}; {s.upper:.2f}]", fontsize=9, family="monospace", va="center")
        ax_text.text(0.82, y, f"{w:.1f}%", fontsize=9, family="monospace", va="center")

    ax_text.text(0, diamond_y, f"{total.n:.0f}", fontsize=9, family="monospace", va="center", fontweight="bold")
    ax_text.text(
        0.18, diamond_y, f"{total.mean:.2f} [{total.lower:.2f}; {total.upper:.2f}]",
        fontsize=9, family="monospace", va="center", fontweight="bold",
    )
    ax_text.text(0.82, diamond_y, "100.0%", fontsize=9, family="monospace", va="center", fontweight="bold")

    stats_line = (
        f"Модель: {MODEL_LABELS[model]}   "
        f"I² = {het.i2:.1f}%   Q = {het.q:.2f} (ст.св.={het.df}, p={het.p_value:.3f})   τ² = {het.tau2:.3f}"
    )
    ax_stats.text(0, 0.5, stats_line, fontsize=9, color="#555555", va="center")

    return fig


def build_export_rows(studies: list[Study], model: str = "fixed") -> tuple[list[dict], Heterogeneity]:
    report = analyze(studies, model)
    rows = []
    for s, w in zip(studies, report.weight_percents):
        rows.append({
            "Исследование": s.name,
            "N": s.n,
            "Среднее": s.mean,
            "Нижняя граница 95% ДИ": round(s.lower, 4),
            "Верхняя граница 95% ДИ": round(s.upper, 4),
            "SE": round(s.se, 4),
            "Вес, %": round(w, 2),
        })
    total = report.total
    rows.append({
        "Исследование": "Итого (пул)",
        "N": total.n,
        "Среднее": round(total.mean, 4),
        "Нижняя граница 95% ДИ": round(total.lower, 4),
        "Верхняя граница 95% ДИ": round(total.upper, 4),
        "SE": None,
        "Вес, %": 100.0,
    })
    return rows, report.heterogeneity
