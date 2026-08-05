import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import pickle
import re
import os
import sys


class GameSentimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎮 Game Review Sentiment Analyzer")
        self.root.geometry("900x720")
        self.root.configure(bg="#1f2a44")

        self.model = None
        self.vectorizer = None
        self.stop_words = None
        self.stemmer = None

        self.build_ui()
        self.load_model()

    # ================= LOAD MODEL =================
    def load_model(self):
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "game_sentiment_model.pkl")

        if not os.path.exists(path):
            messagebox.showerror("Error", "Model not found. Run model_train.py first.")
            sys.exit(1)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.vectorizer = data["vectorizer"]
        self.stop_words = data["stop_words"]
        self.stemmer = data["stemmer"]

        self.status.config(text="✅ Model Ready", fg="#2ecc71")
        self.enable_controls()

    # ================= UI =================
    def build_ui(self):

        # ===== HEADER =====
        header = tk.Frame(self.root, bg="#111827", height=70)
        header.pack(fill="x")

        tk.Label(
            header,
            text="🎮  Game Review Sentiment Analyzer",
            font=("Segoe UI", 26, "bold"),
            fg="white",
            bg="#111827"
        ).pack(pady=14)

        # ===== MAIN CONTAINER =====
        container = tk.Frame(self.root, bg="#1f2a44")
        container.pack(fill="both", expand=True, padx=30, pady=20)

        # ===== INPUT CARD =====
        self.input_card = tk.Frame(container, bg="#2b3a5c")
        self.input_card.pack(fill="x", pady=10)

        tk.Label(
            self.input_card, text="📝 Enter Game Review",
            font=("Segoe UI", 15, "bold"),
            bg="#2b3a5c", fg="white"
        ).pack(anchor="w", padx=15, pady=(12, 5))

        self.textbox = scrolledtext.ScrolledText(
            self.input_card, height=5, wrap=tk.WORD,
            font=("Segoe UI", 12), state="disabled",
            bg="#f8fafc", relief=tk.FLAT
        )
        self.textbox.pack(fill="x", padx=15, pady=8)

        btn_frame = tk.Frame(self.input_card, bg="#2b3a5c")
        btn_frame.pack(pady=10)

        self.example_btn = tk.Button(btn_frame, text="Example", width=12, command=self.example, state="disabled", bg="#64748b", fg="white")
        self.analyze_btn = tk.Button(btn_frame, text="Analyze", width=12, command=self.analyze, state="disabled", bg="#3b82f6", fg="white")
        self.clear_btn = tk.Button(btn_frame, text="Clear", width=12, command=self.clear, state="disabled", bg="#f97316", fg="white")
        self.cm_btn = tk.Button(btn_frame, text="Confusion Matrix", width=18, command=self.show_cm, bg="#a855f7", fg="white")

        self.example_btn.pack(side="left", padx=8)
        self.analyze_btn.pack(side="left", padx=8)
        self.clear_btn.pack(side="left", padx=8)
        self.cm_btn.pack(side="left", padx=8)

        # ===== RESULT CARD =====
        self.result_card = tk.Frame(container, bg="#111827")
        self.result_card.pack(fill="both", expand=True, pady=15)

        self.status = tk.Label(
            self.result_card,
            text="Loading model...",
            font=("Segoe UI", 14),
            bg="#111827",
            fg="#facc15"
        )
        self.status.pack(pady=60)

        # ===== PROGRESS BAR STYLES =====
        style = ttk.Style()
        style.theme_use("default")
        style.configure("pos.Horizontal.TProgressbar", background="#22c55e")
        style.configure("neg.Horizontal.TProgressbar", background="#ef4444")

    def enable_controls(self):
        self.textbox.config(state="normal")
        self.example_btn.config(state="normal")
        self.analyze_btn.config(state="normal")
        self.clear_btn.config(state="normal")

    # ================= FUNCTIONS =================
    def example(self):
        txt = "Amazing gameplay, beautiful graphics and very fun to play."
        self.textbox.delete("1.0", tk.END)
        self.textbox.insert("1.0", txt)

    def clear(self):
        self.textbox.delete("1.0", tk.END)
        for w in self.result_card.winfo_children():
            w.destroy()
        self.status = tk.Label(
            self.result_card,
            text="Enter a game review and click Analyze",
            font=("Segoe UI", 14),
            bg="#111827",
            fg="#94a3b8"
        )
        self.status.pack(pady=60)

    # ================= NLP =================
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        tokens = text.split()
        return " ".join(
            self.stemmer.stem(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        )

    def analyze(self):
        text = self.textbox.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter a review.")
            return

        clean = self.preprocess(text)
        X = self.vectorizer.transform([clean])

        pred = int(self.model.predict(X)[0])
        probs = self.model.predict_proba(X)[0]

        pos_prob = probs[1]
        neg_prob = probs[0]

        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        self.show_result(label, pos_prob, neg_prob)

    # ================= RESULT DISPLAY =================
    def show_result(self, label, pos_p, neg_p):
        for w in self.result_card.winfo_children():
            w.destroy()

        is_pos = label == "POSITIVE"
        color = "#22c55e" if is_pos else "#ef4444"
        emoji = "😊" if is_pos else "😡"

        tk.Label(
            self.result_card,
            text=f"{emoji}  {label}",
            font=("Segoe UI", 38, "bold"),
            fg=color,
            bg="#111827"
        ).pack(pady=(30, 10))

        tk.Label(
            self.result_card,
            text="Prediction Confidence",
            font=("Segoe UI", 13),
            fg="white",
            bg="#111827"
        ).pack()

        bar = ttk.Progressbar(
            self.result_card, length=420, mode="determinate",
            style="pos.Horizontal.TProgressbar" if is_pos else "neg.Horizontal.TProgressbar"
        )
        bar.pack(pady=10)
        bar["value"] = max(pos_p, neg_p) * 100

        tk.Label(
            self.result_card,
            text=f"{max(pos_p, neg_p)*100:.1f} %",
            font=("Segoe UI", 14, "bold"),
            fg=color,
            bg="#111827"
        ).pack(pady=5)

        # Dual bars
        tk.Label(self.result_card, text="Positive", bg="#111827", fg="#22c55e").pack()
        pbar = ttk.Progressbar(self.result_card, length=300, mode="determinate", style="pos.Horizontal.TProgressbar")
        pbar.pack(pady=3)
        pbar["value"] = pos_p * 100

        tk.Label(self.result_card, text="Negative", bg="#111827", fg="#ef4444").pack()
        nbar = ttk.Progressbar(self.result_card, length=300, mode="determinate", style="neg.Horizontal.TProgressbar")
        nbar.pack(pady=3)
        nbar["value"] = neg_p * 100

    # ================= CONFUSION MATRIX =================
    def show_cm(self):
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "confusion_matrix.png")

        if not os.path.exists(path):
            messagebox.showerror("Error", "Confusion matrix not found.")
            return

        win = tk.Toplevel(self.root)
        win.title("Confusion Matrix")
        win.configure(bg="#0f172a")

        img = Image.open(path).resize((520, 420))
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=photo, bg="#0f172a")
        lbl.image = photo
        lbl.pack(padx=15, pady=15)


# ================= MAIN =================
def main():
    root = tk.Tk()
    GameSentimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
