"""ИИ-ревизор ОСИ: Streamlit-приложение (мастер из 4 шагов)."""
import json
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from osi.analyze import (analyze, apply_uploads, load_files,
                         summary_markdown)
from osi.classifier import CategoryClassifier
from osi.receipts import read_receipt, read_statement
from osi.vocab import CATEGORY_RU

HISTORY_DIR = Path("history")
SAMPLES = Path("data/samples")
SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "info": "🔵"}

st.set_page_config(page_title="ИИ-ревизор ОСИ", page_icon="🧾", layout="wide")


@st.cache_resource(show_spinner="Загружаю модели…")
def get_classifier():
    path = Path("models/category.joblib")
    if path.exists():
        try:
            return CategoryClassifier.load(path)
        except Exception:                      # noqa: BLE001
            pass                               # другая версия sklearn
    return CategoryClassifier().fit()


@st.cache_resource(show_spinner="Загружаю OCR (первый раз ~1 мин)…")
def get_reader():
    from osi.ingest import get_ocr_reader
    try:
        return get_ocr_reader()
    except RuntimeError:
        return None


def state():
    defaults = {"step": 0, "save_history": False, "sid": uuid.uuid4().hex[:8],
                "analysis": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    return st.session_state


def save_to_workdir(files):
    folder = Path(tempfile.mkdtemp(prefix="osi_"))
    paths = []
    for item in files:
        target = folder / Path(item.name).name
        target.write_bytes(item.getbuffer())
        paths.append(target)
    return paths


def sample_paths(prefix):
    return sorted(SAMPLES.glob(f"{prefix}*.pdf"))


def save_history(analysis, sid):
    HISTORY_DIR.mkdir(exist_ok=True)
    data = {"risk": analysis.risk, "score": analysis.score,
            "reports": [r.label for r in analysis.reports],
            "findings": [{"severity": f.severity, "title": f.title,
                          "detail": f.detail} for f in analysis.findings]}
    (HISTORY_DIR / f"{sid}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1))


def sidebar(s):
    st.sidebar.title("🧾 ИИ-ревизор ОСИ")
    steps = ["Приветствие", "Загрузка данных", "Уточняющие вопросы",
             "Итоговый отчёт"]
    for i, name in enumerate(steps):
        mark = "➡️" if i == s.step else ("✅" if i < s.step else "▫️")
        st.sidebar.write(f"{mark} {name}")
    st.sidebar.divider()
    if s.save_history and HISTORY_DIR.exists():
        past = sorted(HISTORY_DIR.glob("*.json"))
        if past:
            st.sidebar.caption("История проверок")
            for file in past[-5:]:
                info = json.loads(file.read_text())
                st.sidebar.write(f"• {', '.join(info['reports'][:2])}… "
                                 f"риск: {info['risk']}")
        if st.sidebar.button("🗑 Удалить историю"):
            for file in HISTORY_DIR.glob("*.json"):
                file.unlink()
            st.sidebar.success("История удалена")
    if st.sidebar.button("↺ Начать заново"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def step_greeting(s):
    st.title("Здравствуйте! Я ИИ-ревизор ОСИ")
    st.write(
        "Помогаю проверить отчёты объединения собственников имущества: "
        "проверю арифметику и остатки, найду необычные расходы, подскажу, "
        "какие документы стоит запросить, и подготовлю итоговый отчёт.")
    st.subheader("Сохранять историю проверок?")
    st.caption("По умолчанию — нет. Загруженные файлы не сохраняются в любом "
               "случае; в истории хранятся только выводы проверки, и её можно "
               "удалить в любой момент.")
    choice = st.radio("История", ["Нет, не сохранять", "Да, сохранять выводы"],
                      index=1 if s.save_history else 0,
                      label_visibility="collapsed")
    s.save_history = choice.startswith("Да")
    if st.button("Далее →", type="primary"):
        s.step = 1
        st.rerun()


def run_analysis(paths, s):
    reader_needed = any(p.suffix.lower() in (".jpg", ".jpeg", ".png")
                        for p in paths)
    with st.status("Читаю документы…", expanded=True) as status:
        reader = get_reader() if reader_needed else None
        st.write(f"Файлов: {len(paths)}")
        reports, receipts, skipped = load_files(paths, reader)
        st.write(f"Отчётов: {len(reports)}, подтверждающих документов: "
                 f"{len(receipts)}")
        status.update(label="Проверяю…")
        s.analysis = analyze(reports, receipts, get_classifier(), skipped)
        status.update(label="Готово", state="complete")
    if s.save_history:
        save_history(s.analysis, s.sid)


def step_upload(s):
    st.title("Загрузите ваши данные")
    st.write("Месячные и годовые отчёты ОСИ (PDF, скан или фото), чеки и "
             "другие документы. Выписки банка — CSV/XLSX (по желанию).")
    files = st.file_uploader(
        "Файлы", accept_multiple_files=True,
        type=["pdf", "jpg", "jpeg", "png", "csv", "xlsx"])
    col1, col2 = st.columns(2)
    if col1.button("Проанализировать", type="primary", disabled=not files):
        run_analysis(save_to_workdir(files), s)
        s.step = 2
        st.rerun()
    if SAMPLES.exists() and sample_paths("demo_otchet"):
        if col2.button("Попробовать на демо-данных (вымышленные)"):
            run_analysis(sample_paths("demo_otchet"), s)
            s.step = 2
            st.rerun()


def step_questions(s):
    a = s.analysis
    st.title("Уточняющие вопросы")
    for name, err in a.skipped:
        st.warning(f"Не удалось прочитать {name}: {err}")
    st.metric("Найдено замечаний", len(a.findings))
    open_q = a.open_questions
    if open_q:
        st.write("Чтобы подтвердить расходы, загрузите документы:")
        for q in open_q:
            st.markdown(f"**❓ {q.text}**")
    else:
        st.success("Все вопросы закрыты документами.")
    more = st.file_uploader("Загрузить документы по вопросам",
                            accept_multiple_files=True, key="more",
                            type=["pdf", "jpg", "jpeg", "png", "csv", "xlsx"])
    col1, col2, col3 = st.columns(3)
    if col1.button("Учесть документы", disabled=not more):
        paths = save_to_workdir(more)
        reader = get_reader() if any(
            p.suffix.lower() in (".jpg", ".jpeg", ".png") for p in paths) else None
        new = []
        for path in paths:
            if path.suffix.lower() in (".csv", ".xlsx"):
                new += read_statement(path)
            else:
                new.append(read_receipt(path, reader))
        apply_uploads(a, new)
        st.rerun()
    if SAMPLES.exists() and sample_paths("demo_chek"):
        if col2.button("Добавить демо-чеки"):
            apply_uploads(a, [read_receipt(p) for p in sample_paths("demo_chek")])
            st.rerun()
    if col3.button("К итоговому отчёту →", type="primary"):
        s.step = 3
        st.rerun()
    if a.receipts:
        with st.expander("Прочитанные документы"):
            st.dataframe(pd.DataFrame([{
                "файл": r.filename, "сумма": r.total, "дата": r.day,
                "продавец": r.vendor} for r in a.receipts]))


def step_report(s):
    a = s.analysis
    st.title("Итоговый отчёт")
    c1, c2, c3 = st.columns(3)
    c1.metric("Риск", a.risk, f"{a.score}/100", delta_color="off")
    c2.metric("Замечаний", len([f for f in a.findings if f.severity != "info"]))
    c3.metric("Вопросов закрыто",
              f"{len(a.questions) - len(a.open_questions)}/{len(a.questions)}")
    tab1, tab2, tab3 = st.tabs(["Замечания", "Структура расходов", "Текст отчёта"])
    with tab1:
        if not a.findings:
            st.success("Нарушений не найдено.")
        for f in a.findings:
            with st.expander(f"{SEVERITY_ICON[f.severity]} {f.report} — "
                             f"{f.title}"):
                st.write(f.detail)
                if f.ask:
                    st.caption(f"Что запросить: {f.ask}")
    with tab2:
        frame = a.categories
        if frame is not None and not frame.empty:
            by_cat = (frame.assign(категория=frame["категория"].map(CATEGORY_RU))
                      .groupby("категория")["сумма"].sum().sort_values())
            st.bar_chart(by_cat)
            with st.expander("Как модель отнесла строки"):
                st.dataframe(frame.drop(columns=["idx"]))
    with tab3:
        text = summary_markdown(a)
        st.markdown(text)
        st.download_button("⬇ Скачать отчёт (.md)", text,
                           file_name="revizor_report.md")
    if st.button("← К вопросам"):
        s.step = 2
        st.rerun()


def main():
    s = state()
    sidebar(s)
    if s.step == 0:
        step_greeting(s)
    elif s.step == 1:
        step_upload(s)
    elif s.step == 2 and s.analysis:
        step_questions(s)
    elif s.step == 3 and s.analysis:
        step_report(s)
    else:
        s.step = 1
        st.rerun()


main()
