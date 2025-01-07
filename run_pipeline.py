import subprocess

scripts = [
    'data_for_pretrain/pretrain_collecting.py',
    'data_for_pretrain/pretrain_preprocessing.py',
    'data_for_downstream/downstream_collecting.py',
    'create_db.py'
]

for script in scripts:
    print(f'Запуск {script}...')
    subprocess.run(['python', script], check=True)
    print(f'{script} выполнен успешно!')