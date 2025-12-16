# 01_data_loading.py
import pandas as pd
import numpy as np
import config
import os


def load_data():
    """
    Загрузка и подготовка данных временного ряда.
    Выполняет пункты 1.1-1.3 задания.
    """
    print("=" * 60)
    print("ШАГ 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("=" * 60)

    try:
        # Загрузка данных из CSV
        df = pd.read_csv(config.DATA_PATH)
        print(f"📊 Файл загружен: {config.DATA_PATH}")

        # Проверка структуры данных
        print(f"Структура данных:")
        print(f"  - Колонки: {list(df.columns)}")
        print(f"  - Всего строк: {len(df)}")

        # Преобразование года в datetime и установка как индекс
        df['year'] = pd.to_datetime(df['year'], format='%Y')
        df.set_index('year', inplace=True)

        # Переименование колонки value для удобства
        df.columns = ['value']

        print(f"\n📅 Временной ряд:")
        print(f"  - Период: {df.index[0].year} - {df.index[-1].year}")
        print(f"  - Всего наблюдений: {len(df)}")
        print(f"  - Частота: {pd.infer_freq(df.index) or 'Годовая'}")

        # Описательная статистика
        print(f"\n📈 Описательная статистика:")
        stats = df['value'].describe()
        for key, val in stats.items():
            print(f"  {key:8}: {val:,.0f}")

        # Сохранение формального представления (пункт 1.3)
        save_formal_table(df)

        return df

    except FileNotFoundError:
        print(f"❌ ОШИБКА: Файл не найден по пути {config.DATA_PATH}")
        print("   Создайте файл austria_outbound_tourism.csv в папке data/")
        print("   Формат файла: year,value")
        return None
    except Exception as e:
        print(f"❌ ОШИБКА при загрузке данных: {e}")
        return None


def save_formal_table(df):
    """
    Сохранение формального представления данных (таблицы)
    Пункт 1.3 задания
    """
    table_content = "ФОРМАЛЬНОЕ ПРЕДСТАВЛЕНИЕ ДАННЫХ\n"
    table_content += "=" * 50 + "\n\n"

    table_content += f"Источник: {config.DATA_DESCRIPTION['source']}\n"
    table_content += f"Показатель: {config.DATA_DESCRIPTION['indicator']}\n"
    table_content += f"Страна: {config.DATA_DESCRIPTION['country']}\n"
    table_content += f"Период: {config.DATA_DESCRIPTION['period']}\n"
    table_content += f"Единицы: {config.DATA_DESCRIPTION['unit']}\n"
    table_content += f"Наблюдений: {config.DATA_DESCRIPTION['observations']}\n\n"

    table_content += "=" * 50 + "\n"
    table_content += "ПЕРВЫЕ 5 НАБЛЮДЕНИЙ:\n"
    table_content += df.head().to_string() + "\n\n"

    table_content += "=" * 50 + "\n"
    table_content += "ПОСЛЕДНИЕ 5 НАБЛЮДЕНИЙ:\n"
    table_content += df.tail().to_string() + "\n\n"

    table_content += "=" * 50 + "\n"
    table_content += "КЛЮЧЕВЫЕ ГОДЫ (ПИКИ И ПАДЕНИЯ):\n"

    # Находим ключевые годы
    max_year = df['value'].idxmax()
    min_year = df['value'].idxmin()
    covid_year = pd.Timestamp('2020-01-01')

    key_years = pd.concat([
        df.loc[[max_year]],
        df.loc[[min_year]],
        df.loc[[covid_year]]
    ])

    table_content += key_years.to_string() + "\n"

    # Сохраняем в файл
    table_path = os.path.join(config.TABLES_DIR, '01_formal_table.txt')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(table_content)

    print(f"✅ Таблица сохранена: {table_path}")

    # Также сохраняем в CSV для удобства
    csv_path = os.path.join(config.TABLES_DIR, '01_formal_table.csv')
    df.to_csv(csv_path)
    print(f"✅ CSV версия: {csv_path}")


def main():
    """Основная функция модуля"""
    df = load_data()
    return df


if __name__ == "__main__":
    df = main()