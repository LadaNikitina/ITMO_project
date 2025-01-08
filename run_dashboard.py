import os
import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

import subprocess

db_name = 'dialogs.db'

# Проверка наличия базы данных
if not os.path.exists(db_name):
    st.warning(f"База данных '{db_name}' не найдена. Запуск скриптов для создания...")
    scripts = [
        'data_for_pretrain/pretrain_collecting.py',
        'data_for_pretrain/pretrain_preprocessing.py',
        'data_for_downstream/downstream_collecting.py',
        'create_db.py'
    ]

    for script in scripts:
        try:
            st.text(f'Запуск {script}...')
            subprocess.run(['python', script], check=True)
            st.text(f'{script} выполнен успешно!')
        except subprocess.CalledProcessError as e:
            st.error(f"Ошибка при выполнении {script}: {e}")
            st.stop()
else:
    st.success(f"База данных '{db_name}' уже существует. Пропуск запуска скриптов.")

conn = sqlite3.connect(db_name)

st.title("Аналитическая платформа по диалогам")

tab1, tab2, tab3 = st.tabs(["Dialogs", "Pairs", "Downstream"])

with tab1:
    st.header("Таблица диалогов для предобучения")

    # Загрузка данных
    dialogs_df = pd.read_sql("SELECT * FROM dialogs", conn)
    st.dataframe(dialogs_df.head(10))

    # Метрики
    st.subheader("Основные метрики")
    st.write(f"Количество записей: {len(dialogs_df)}")
    avg_length = dialogs_df['dialog_text'].apply(lambda x: len(x.split('\n'))).mean()
    st.write(f"Средняя длина диалогов (число реплик): {avg_length:.2f}")

    # Распределение по split_name
    st.subheader("Распределение по split_name")
    split_counts = dialogs_df['split_name'].value_counts()
    st.bar_chart(split_counts)

    # Количество диалогов по датасетам
    st.subheader("Распределение по dataset_name")
    dataset_counts = dialogs_df['dataset_name'].value_counts()
    st.bar_chart(dataset_counts)

    # Длина диалогов
    st.subheader("Распределение длины диалогов")
    dialogs_df['dialog_length'] = dialogs_df['dialog_text'].apply(lambda x: len(x.split('\n')))
    st.bar_chart(dialogs_df['dialog_length'].value_counts().sort_index())

    # Часто встречающиеся слова
    st.subheader("Облако слов по всем диалогам")
    all_text = ' '.join(dialogs_df['dialog_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    st.image(wordcloud.to_array())

with tab2:
    st.header("Таблица пар реплик")

    # Загрузка данных
    pairs_df = pd.read_sql("SELECT * FROM pairs", conn)
    st.dataframe(pairs_df.head(10))

    # Метрики
    st.subheader("Основные метрики")
    st.write(f"Количество пар реплик: {len(pairs_df)}")
    avg_first_length = pairs_df['first_utterance'].apply(lambda x: len(x.split())).mean()
    avg_second_length = pairs_df['second_utterance'].apply(lambda x: len(x.split())).mean()
    st.write(f"Средняя длина первой реплики (число слов): {avg_first_length:.2f}")
    st.write(f"Средняя длина второй реплики (число слов): {avg_second_length:.2f}")

    # Длина первой реплики
    st.subheader("Распределение числа слов первой реплики")
    pairs_df['first_length'] = pairs_df['first_utterance'].apply(lambda x: len(x.split()))
    st.bar_chart(pairs_df['first_length'].value_counts().sort_index())

    # Длина второй реплики
    st.subheader("Распределение числа слов второй реплики")
    pairs_df['second_length'] = pairs_df['second_utterance'].apply(lambda x: len(x.split()))
    st.bar_chart(pairs_df['second_length'].value_counts().sort_index())

    # Распределение по числу слов первой и второй реплик
    st.subheader("Распределение по числу слов первой и второй реплик")
    fig, ax = plt.subplots()
    sns.scatterplot(x=pairs_df['first_length'], y=pairs_df['second_length'], ax=ax)
    ax.set_xlabel('Длина первой реплики')
    ax.set_ylabel('Длина второй реплики')
    st.pyplot(fig)

with tab3:
    st.header("Таблица Downstream")

    # Загрузка данных
    downstream_df = pd.read_sql("SELECT * FROM downstream", conn)
    st.dataframe(downstream_df.head(10))

    # Метрики
    st.subheader("Основные метрики")
    st.write(f"Количество записей: {len(downstream_df)}")
    num_languages = downstream_df['language'].nunique()
    num_labels = downstream_df['label'].nunique()
    st.write(f"Число уникальных языков: {num_languages}")
    st.write(f"Число классов (меток): {num_labels}")

    label_dist = downstream_df['language'].value_counts(normalize=True) * 100
    st.write("Распределение по языкам (в %):")
    st.write(label_dist)

    # Распределение по языкам
    st.subheader("Распределение по языкам")
    language_counts = downstream_df['language'].value_counts()
    st.bar_chart(language_counts)

    # Распределение по split
    st.subheader("Распределение по split")
    split_counts = downstream_df['split'].value_counts()
    st.bar_chart(split_counts)
    
    st.subheader("Распределение по языкам в разрезе сплитов")

    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=downstream_df,
        x='language', 
        hue='split', 
        order=downstream_df['language'].value_counts().index
    )
    plt.title('Распределение по языкам в разрезе сплитов')
    plt.xlabel('Языки')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    plt.legend(title='Распределение по split')
    st.pyplot(plt)
    
    # Кластеризация и эмбеддинги
#     st.subheader("Кластеризация реплик с использованием SentenceTransformer")
    
#     model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
#     # Прогресс-бар
#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     utterances = downstream_df['utterance'].tolist()
#     n = len(utterances)

#     # Размер батча
#     batch_size = 512

#     # Обработка в батчах с прогрессом
#     embeddings = []
#     for i in range(0, n, batch_size):
#         batch = utterances[i : i + batch_size]
#         batch_embeddings = model.encode(batch, show_progress_bar=False)
#         embeddings.extend(batch_embeddings)

#         progress = (i + len(batch)) / n
#         progress_bar.progress(progress)
#         status_text.text(f"Обработано {min(i + batch_size, n)} из {n} реплик")

#     status_text.text("Обработка завершена!")

# #     embeddings = model.encode(downstream_df['utterance'].tolist())
#     pca = PCA(n_components=2)
#     reduced_embeddings = pca.fit_transform(embeddings)

#     fig, ax = plt.subplots()
#     scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
#     plt.colorbar(scatter)
#     st.pyplot(fig)


conn.close()
