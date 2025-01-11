# Проект по предмету "Инжиниринг управления данными ПИШ 2024"

## Введение

Студент: Леднева Дарья Романовна, 1 курс, AI Talent Hub.

Добро пожаловать в репозиторий! Этот проект посвящён формированию датасетов для создания и улучшения диалоговых эмбеддингов — ключевого элемента современных диалоговых систем. Репозиторий разделён на две основные директории, каждая из которых решает важные задачи: подготовка данных для обучения моделей (диалоговых эмбеддеров) и данных для их оценивания.

## 🚀 Цель проекта: подготовка данных для обучения и оценки диалоговых эмбеддеров

Диалоговые эмбеддеры преобразуют сырой текст диалоговых данных в компактное и информативное представление - эмбеддинг, которое можно использовать для решения различных диалоговых задач. Диалоговые эмбеддинги служат фундаментом для построения диалоговых систем, способных:

- Понимать смысловые контексты и намерения пользователя.
- Поддерживать долгие и осмысленные взаимодействия.
- Быстро адаптироваться под разные языки и области применения.

Использование качественных эмбеддингов позволяет существенно улучшить точность задач, таких как классификация намерений, генерация ответов, или поиск в базах знаний.

## 📌 Реализованные этапы по критериям оценивания проекта

1. Сбор данных
   
   ```Реализовано в скриптах "data_for_pretrain/название датасета_preprocessing.py" и "data_for_downstream/название датасета_preprocessing.py". Помимо этого, для того, чтобы было проще воспроизвести эксперимент, данные для предобучения модели были собраны и выгружены в единое хранилище, поскольку далеко не все эти датасеты есть на Hugging Face.```
   
2. Предобработка данных

   ```Реализовано в скриптах "data_for_pretrain/название датасета_preprocessing.py", "data_for_downstream/название датасета_preprocessing.py", "data_for_pretrain/pretrain_collecting.py", "data_for_pretrain/pretrain_preprocessing.py" и "data_for_downstream/downstream_collecting.py".```
   
3. Исследовательский анализ данных

   ```Реализовано в рамках скрипта data_for_pretrain/dataset_analytics.py, который вызывается для аналитики каждого из обрабатываемых датасетов, и папки data_for_pretrain/analytics для датасетов для предобучения модели, а также ноутбука data_for_downstream/downstream_analytics.ipynb, в котором выполняется аналитика данных для оценки моделей.```

4. Определение и обоснование метрик качества данных

``` ```

5. Разработка базы данных для хранения данных

   ```Реализовано в рамках скрипта create_db.py, его запуск через команду python create_db.py позволяет сформировать базу данных SQLite.```
   
6. Автоматизация процессов: оформление пунктов 1, 2 и 5 в отдельный пайплайн для автоматизации
    
   ```Реализовано в рамках скрипта run_pipeline.py, запускающего весь пайплайн сбора данных, предобработки данных и формирования базы данных SQLite.```
   
7. Оформление дашборда

```Реализовано в рамках скрипта run_dashboard.py, запускающего расчет дашборда при помощи streamlit. Демо (запись) дашборда представлено по (ссылке)[https://drive.google.com/file/d/1lIcieReKksGiEPCOzh9ErQCFNrK0zU1d/view?usp=sharing]. Запуск дашборда осуществляется при помощи команды streamlit run run_dashboard.py.```

8. Создание презентации и пояснительной записки о работе

``` ```

## 🗂 Структура репозитория

**data_for_pretrain — подготовка данных для предобучения (pre-train)**

Данные и скрипты из этой папки позволяют создать большой корпус диалогов, который можно использовать для контрастивного или любого другого вида обучения диалоговых эмбеддингов.

Структура папки:
```plaintext
.
├── dialog_datasets/           # Папка с данными для предобработки
├── analytics/                 # Папка с логами и результатами аналитики
├── README.md                  # Инструкция по использованию репозитория
├── camrest676_preprocessing.py # Скрипт предобработки данных для CamRest676
├── data_example.py            # Пример структуры данных для обработки
├── dataset_analytics.py       # Скрипт выполнения аналитики по датасету
├── frames_preprocessing.py    # Скрипт предобработки данных для Frames
├── metalwoz_preprocessing.py  # Скрипт предобработки данных для MetalWOZ
├── msre2e_preprocessing.py    # Скрипт предобработки данных для MSR-E2E
├── multiwoz_preprocessing.py  # Скрипт предобработки данных для MultiWOZ
├── pretrain_collecting.py     # Скрипт для подготовки данных для pretrain - формирование пар реплик
├── pretrain_preprocessing.py  # Скрипт для предобработки данных и объединения датасетов
├── schema_preprocessing.py    # Скрипт предобработки данных для Schema
├── smd_preprocessing.py       # Скрипт предобработки данных для SMD
├── taskmaster_preprocessing.py# Скрипт предобработки данных для Taskmaster
└── woz_preprocessing.py       # Скрипт предобработки данных для WOZ
```

**data_for_downstream — подготовка данных для оценки эмбеддингов**

Эта папка содержит данные и инструменты для оценки качества эмбеддингов на задаче интент-классификации. Здесь реализована обработка популярных диалоговых датасетов и их объединение в единый формат для решения задач классификации намерений (интентов).

Структура папки:
```plaintext
.
├── README.md                    # Описание проекта
├── downstream_analytics.ipynb   # Ноутбук с аналитикой
├── downstream_collecting.py     # Скрипт объединения всех датасетов
│
├── banking77_preprocessing.py   # Обработка датасета Banking77
├── clinc150_preprocessing.py    # Обработка датасета CLINC150
├── massive_preprocessing.py     # Обработка датасета MASSIVE
├── minds14_preprocessing.py     # Обработка датасета MINDS14
├── snips_preprocessing.py       # Обработка датасета SNIPS
│
└── SNIPS/                       # Папка с исходными файлами SNIPS
    ├── README.md
```

## 🛠 Использование
**Подготовка данных для предобучения**

Склонируйте репозиторий. Поместите сырые данные в папку data_for_pretrain/dialog_datasets/.

Запустите скрипты:

```
python data_for_pretrain/pretrain_collecting.py
python data_for_pretrain/pretrain_preprocessing.py
```
Итоговый датасет сохранится в файл output.csv. Больше деталей можно найти в README.md data_for_pretrain.

**Обработка данных для downstream-задач**

Склонируйте репозиторий. Поместите данные для датасета SNIPS в соответствующую папку data_for_downstream/SNIPS/.

Выполните команду:

```
python data_for_downstream/downstream_collecting.py
```

Итоговый датасет будет сохранён в файл downstream_data.csv. Больше деталей можно найти в README.md data_for_downstream.

**Создание базы данных**

Склонируйте репозиторий. Поместите сырые данные в папку data_for_pretrain/dialog_datasets/, а данные для датасета SNIPS в соответствующую папку data_for_downstream/SNIPS/. Выполните два предыдущих шага.

Выполните команду:

```
python create_db.py
```

**Запуск единого пайплайна**

Склонируйте репозиторий. Поместите сырые данные в папку data_for_pretrain/dialog_datasets/, а данные для датасета SNIPS в соответствующую папку data_for_downstream/SNIPS/.

Выполните команду:

```
python run_pipeline.py
```
