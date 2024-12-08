# Подготовка датасета для downstream задач

## Описание

Эта директория предназначена для обработки и анализа данных из различных источников для решения downstream задачи интент классификации в диалоговом домене. Основная цель — создать датасет, приведенный к единому формату.

## Структура директории

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

1. Скрипты обработки датасетов

Каждый скрипт отвечает за обработку определённого набора данных:

- banking77_preprocessing.py — обработка датасета Banking77.
- clinc150_preprocessing.py — обработка датасета CLINC150.
- massive_preprocessing.py — обработка датасета MASSIVE.
- minds14_preprocessing.py — обработка датасета MINDS14.
- snips_preprocessing.py — обработка датасета SNIPS.

2. Скрипт объединения датасетов в единый датасет

downstream_collecting.py — основной скрипт для объединения датасетов. Вызывает функции обработки каждого датасета и сохраняет итоговый объединённый датасет в файл downstream_data.csv.

3. Ноутбук с аналитикой

downstream_analytics.ipynb — аналитика над объединённым датасетом downstream_data.csv

## Использование

Для запуска препроцессинга всех датасетов выполните команду: 

```
python3 downstream_collecting.py
```

Итоговый файл downstream_data.csv имеет следующие столбцы:

- utterance — текстовые высказывания
- label — метки классов (с префиксом названия датасета)
- split — разбиение (train, test, validation)
- language — язык данных
- dataset_name — название исходного датасета
