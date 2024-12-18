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

1. Поместите сырые данные в папку data_for_pretrain/dialog_datasets/.

2. Запустите скрипты:

```
python data_for_pretrain/pretrain_preprocessing.py
python data_for_pretrain/pretrain_collecting.py
```
Итоговый датасет сохранится в файл output.csv. Больше деталей можно найти в README.md data_for_pretrain.

**Обработка данных для downstream-задач**

1. Поместите данные для датасета SNIPS в соответствующую папку.

Выполните команду:

```
python data_for_downstream/downstream_collecting.py
```

Итоговый датасет будет сохранён в файл downstream_data.csv. Больше деталей можно найти в README.md data_for_downstream.