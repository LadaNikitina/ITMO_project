import sqlite3
import pickle
import csv

# Удалим существующую базу данных, если она есть
if os.path.exists(db_name):
    os.remove(db_name)
    print(f'Существующая база данных "{db_name}" удалена.')

# Создадим базу данных SQLite и подключение к ней
conn = sqlite3.connect('dialogs.db')  # Имя базы данных
cursor = conn.cursor()

# Создадим таблицу dialogs полных диалогов в базе данных
cursor.execute('''
CREATE TABLE IF NOT EXISTS dialogs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dialog_text TEXT,
    split_name TEXT,
    dataset_name TEXT
);
''')

# Создадим таблицу pairs пар диалоговых реплик в базе данных
cursor.execute('''
CREATE TABLE IF NOT EXISTS pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_utterance TEXT,
    second_utterance TEXT
);
''')

# Создадим таблицу для downstream задач в базе данных
cursor.execute('''
CREATE TABLE IF NOT EXISTS downstream (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    utterance TEXT,
    label TEXT,
    split TEXT,
    language TEXT,
    dataset_name TEXT
);
''')

# Загрузим данные
with open('data_for_pretrain/full_dialogs.pkl', 'rb') as file:
    data = pickle.load(file)

# Вставим данные в таблицу dialogs
for record in data:
    dialog_text = '\n'.join(record['dialog'])
    split_name = record['split_name']
    dataset_name = record['dataset_name']

    cursor.execute('''
    INSERT INTO dialogs (dialog_text, split_name, dataset_name)
    VALUES (?, ?, ?)
    ''', (dialog_text, split_name, dataset_name))

# Загрузим данные из CSV файла для таблицы pairs
with open('data_for_pretrain/output.csv', 'r', newline = '', encoding = 'utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        cursor.execute('''
        INSERT INTO pairs (first_utterance, second_utterance)
        VALUES (?, ?)
        ''', (row[0], row[1]))

# Загрузим данные из CSV файла для таблицы downstream
with open('data_for_downstream/downstream_data.csv', 'r', newline = '', encoding = 'utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        cursor.execute('''
        INSERT INTO downstream (utterance, label, split, language, dataset_name)
        VALUES (?, ?, ?, ?, ?)
        ''', (row[1], row[2], row[3], row[4], row[5]))

conn.commit()
conn.close()

print("База данных успешно создана!")