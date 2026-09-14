from __future__ import annotations

import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from forest_plot import (
    MODEL_LABELS,
    MODEL_METHOD_NOTES,
    DataFormatError,
    REQUIRED_COLUMNS,
    Study,
    analyze,
    build_export_rows,
    build_figure,
    load_studies,
)

ctk.set_appearance_mode("light")

UI_MODEL_LABELS = {
    "fixed": "Фиксированная модель",
    "random": "Случайная модель",
}

BG = "#eef1f6"
CARD = "#ffffff"
BORDER = "#e2e5ec"
TEXT_MAIN = "#1f2430"
TEXT_MUTED = "#6b7280"
ACCENT = "#2f6fed"
ACCENT_HOVER = "#2456c4"
SECONDARY = "#eef2ff"
SECONDARY_HOVER = "#dde5ff"
SECONDARY_TEXT = "#2f3a56"
GOOD = "#1a7f4e"
WARN = "#b8560f"


class ForestPlotApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Форест-плот — построитель графиков")
        self.geometry("1180x800")
        self.minsize(960, 640)
        self.configure(fg_color=BG)

        self.studies: list[Study] = []
        self.canvas: FigureCanvasTkAgg | None = None
        self.model: str = "fixed"

        self._build_layout()

    def _card(self, parent, **kwargs) -> ctk.CTkFrame:
        defaults = dict(fg_color=CARD, corner_radius=14, border_width=1, border_color=BORDER)
        defaults.update(kwargs)
        return ctk.CTkFrame(parent, **defaults)

    def _primary_button(self, parent, text, command, **kwargs) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent, text=text, command=command,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#ffffff",
            corner_radius=9, height=36, font=ctk.CTkFont(size=13, weight="bold"),
            **kwargs,
        )

    def _secondary_button(self, parent, text, command, **kwargs) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent, text=text, command=command,
            fg_color=SECONDARY, hover_color=SECONDARY_HOVER, text_color=SECONDARY_TEXT,
            text_color_disabled="#9aa3bd",
            corner_radius=9, height=36, font=ctk.CTkFont(size=13),
            **kwargs,
        )

    def _build_layout(self) -> None:
        header = self._card(self, corner_radius=16)
        header.pack(side="top", fill="x", padx=18, pady=(18, 10))

        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.pack(side="top", fill="x", padx=18, pady=(14, 4))

        ctk.CTkLabel(
            title_row, text="Форест-плот",
            font=ctk.CTkFont(size=21, weight="bold"), text_color=TEXT_MAIN,
        ).pack(side="left")
        ctk.CTkLabel(
            title_row, text="  построение графиков для мета-анализа",
            font=ctk.CTkFont(size=13), text_color=TEXT_MUTED,
        ).pack(side="left")

        self.status_label = ctk.CTkLabel(
            title_row, text="Файл не загружен", text_color=TEXT_MUTED, font=ctk.CTkFont(size=12),
        )
        self.status_label.pack(side="right")

        controls_row = ctk.CTkFrame(header, fg_color="transparent")
        controls_row.pack(side="top", fill="x", padx=18, pady=(4, 16))

        self.load_button = self._primary_button(controls_row, "📂  Загрузить файл", self.on_load_file)
        self.load_button.pack(side="left", padx=(0, 8))

        self.save_button = self._secondary_button(controls_row, "💾  Сохранить график", self.on_save_plot)
        self.save_button.configure(state="disabled")
        self.save_button.pack(side="left", padx=8)

        self.export_button = self._secondary_button(controls_row, "📤  Экспорт таблицы (CSV)", self.on_export_csv)
        self.export_button.configure(state="disabled")
        self.export_button.pack(side="left", padx=8)

        ctk.CTkLabel(
            controls_row, text="Модель:", font=ctk.CTkFont(size=13), text_color=TEXT_MUTED,
        ).pack(side="left", padx=(20, 8))

        self.model_selector = ctk.CTkSegmentedButton(
            controls_row, values=list(UI_MODEL_LABELS.values()), command=self.on_model_change,
            selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
            unselected_color="#4b5568", unselected_hover_color="#3a4354",
            text_color="#ffffff", font=ctk.CTkFont(size=12),
        )
        self.model_selector.set(UI_MODEL_LABELS["fixed"])
        self.model_selector.pack(side="left")

        self.stats_card = self._card(self, corner_radius=12)
        self.stats_card.pack(side="top", fill="x", padx=18, pady=(0, 10))
        self.stats_label = ctk.CTkLabel(
            self.stats_card, text="Показатели гетерогенности появятся после загрузки данных",
            text_color=TEXT_MUTED, font=ctk.CTkFont(size=12), anchor="w", justify="left",
        )
        self.stats_label.pack(side="left", padx=16, pady=10, fill="x")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(side="top", fill="both", expand=True, padx=18, pady=(0, 18))

        preview_frame = self._card(body)
        preview_frame.pack(side="top", fill="x", pady=(0, 12))

        preview_header = ctk.CTkFrame(preview_frame, fg_color="transparent")
        preview_header.pack(side="top", fill="x", padx=16, pady=(14, 2))
        ctk.CTkLabel(
            preview_header, text="Предпросмотр данных", font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")
        ctk.CTkLabel(
            preview_header, text="  двойной клик по ячейке — исправить значение",
            text_color=TEXT_MUTED, font=ctk.CTkFont(size=11),
        ).pack(side="left")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Forest.Treeview", background=CARD, fieldbackground=CARD,
            foreground=TEXT_MAIN, rowheight=26, borderwidth=0, font=("TkDefaultFont", 10),
        )
        style.configure(
            "Forest.Treeview.Heading", background="#f4f6fb", foreground=TEXT_MUTED,
            font=("TkDefaultFont", 10, "bold"), borderwidth=0,
        )
        style.map(
            "Forest.Treeview",
            background=[("selected", "#dbe6ff")],
            foreground=[("selected", TEXT_MAIN)],
        )

        self.tree = ttk.Treeview(
            preview_frame, columns=REQUIRED_COLUMNS, show="headings", height=6, style="Forest.Treeview",
        )
        for col in REQUIRED_COLUMNS:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=180, anchor="center")
        self.tree.pack(fill="x", padx=16, pady=(6, 16))
        self.tree.bind("<Double-1>", self.on_tree_double_click)
        self._edit_entry: tk.Entry | None = None

        self.plot_card = self._card(body)
        self.plot_card.pack(side="top", fill="both", expand=True)

        self.plot_scroll = ctk.CTkScrollableFrame(self.plot_card, fg_color="transparent")
        self.plot_scroll.pack(fill="both", expand=True, padx=6, pady=6)
        self.plot_frame = self.plot_scroll

        self.placeholder = ctk.CTkLabel(
            self.plot_card,
            text="Загрузите файл с данными, чтобы построить график",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=14),
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def on_load_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите файл с данными",
            filetypes=[("Таблица Excel/CSV", "*.xlsx *.xls *.csv"), ("Все файлы", "*.*")],
        )
        if not path:
            return

        try:
            studies, warnings = load_studies(path)
        except DataFormatError as exc:
            messagebox.showerror("Ошибка формата данных", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Не удалось прочитать файл", str(exc))
            return

        self.studies = studies
        self._refresh_preview()
        self._refresh_plot()
        self.status_label.configure(text=f"Загружено исследований: {len(studies)}", text_color=GOOD)
        self.save_button.configure(state="normal")
        self.export_button.configure(state="normal")

        if warnings:
            messagebox.showwarning(
                "Часть строк пропущена",
                f"Пропущено строк: {len(warnings)}\n\n" + "\n".join(warnings),
            )

    def _refresh_preview(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for s in self.studies:
            self.tree.insert("", "end", values=(s.name, s.n, s.mean, s.ci_width))

    def on_tree_double_click(self, event: tk.Event) -> None:
        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)
        if not row_id or not col_id:
            return

        col_index = int(col_id.replace("#", "")) - 1
        row_index = self.tree.index(row_id)
        bbox = self.tree.bbox(row_id, col_id)
        if not bbox:
            return
        x, y, width, height = bbox

        if self._edit_entry is not None:
            self._edit_entry.destroy()

        current_value = self.tree.set(row_id, REQUIRED_COLUMNS[col_index])
        entry = tk.Entry(self.tree, relief="solid", borderwidth=1)
        entry.insert(0, current_value)
        entry.select_range(0, "end")
        entry.focus()
        entry.place(x=x, y=y, width=width, height=height)
        self._edit_entry = entry

        def commit(_event=None) -> None:
            new_value = entry.get()
            entry.destroy()
            self._edit_entry = None
            self._apply_cell_edit(row_index, col_index, new_value)

        def cancel(_event=None) -> None:
            entry.destroy()
            self._edit_entry = None

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        entry.bind("<Escape>", cancel)

    def _apply_cell_edit(self, row_index: int, col_index: int, new_value: str) -> None:
        study = self.studies[row_index]
        try:
            if col_index == 0:
                if not new_value.strip():
                    raise ValueError("Название не может быть пустым")
                study.name = new_value.strip()
            elif col_index == 1:
                n = float(new_value)
                if n <= 0:
                    raise ValueError("К-во объектов должно быть положительным")
                study.n = n
            elif col_index == 2:
                study.mean = float(new_value)
            elif col_index == 3:
                width = float(new_value)
                if width < 0:
                    raise ValueError("Ширина ДИ не может быть отрицательной")
                study.ci_width = width
        except ValueError as exc:
            messagebox.showerror("Некорректное значение", str(exc))

        self._refresh_preview()
        self._refresh_plot()

    def on_model_change(self, selected_label: str) -> None:
        self.model = next(key for key, label in UI_MODEL_LABELS.items() if label == selected_label)
        if self.studies:
            self._refresh_plot()

    def _refresh_plot(self) -> None:
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        self.placeholder.place_forget()

        figure = build_figure(self.studies, model=self.model)
        figure.set_dpi(100)
        self.canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(padx=6, pady=6)
        self._current_figure = figure

        report = analyze(self.studies, model=self.model)
        het = report.heterogeneity
        high = het.i2 >= 50
        self.stats_label.configure(
            text=(
                f"I² = {het.i2:.1f}%    Q = {het.q:.2f} (ст.св.={het.df}, p={het.p_value:.3f})    "
                f"τ² = {het.tau2:.3f}    —    "
                + (
                    "высокая гетерогенность между исследованиями: рекомендуется случайная модель"
                    if high else "низкая/умеренная гетерогенность между исследованиями"
                )
            ),
            text_color=WARN if high else GOOD,
        )
        self.stats_card.configure(border_color=WARN if high else BORDER)

    def on_export_csv(self) -> None:
        if not self.studies:
            return
        path = filedialog.asksaveasfilename(
            title="Экспортировать таблицу расчётов",
            defaultextension=".csv",
            filetypes=[("Таблица CSV", "*.csv")],
        )
        if not path:
            return

        rows, het = build_export_rows(self.studies, model=self.model)
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
                writer.writerow({})
                writer.writerow({"Исследование": f"Модель: {MODEL_LABELS[self.model]}"})
                writer.writerow({"Исследование": MODEL_METHOD_NOTES[self.model]})
                writer.writerow({"Исследование": f"I² = {het.i2:.1f}%"})
                writer.writerow({"Исследование": f"Q = {het.q:.2f}, ст.св. = {het.df}, p = {het.p_value:.4f}"})
                writer.writerow({"Исследование": f"tau² = {het.tau2:.4f}"})
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Не удалось сохранить файл", str(exc))
            return
        messagebox.showinfo("Готово", f"Таблица сохранена:\n{path}")

    def on_save_plot(self) -> None:
        if not self.studies:
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить график",
            defaultextension=".png",
            filetypes=[("Изображение PNG", "*.png"), ("Документ PDF", "*.pdf")],
        )
        if not path:
            return
        try:
            self._current_figure.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Не удалось сохранить файл", str(exc))
            return
        messagebox.showinfo("Готово", f"График сохранён:\n{path}")


if __name__ == "__main__":
    app = ForestPlotApp()
    app.mainloop()
