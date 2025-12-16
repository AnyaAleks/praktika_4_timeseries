# main.py
import sys
import os
import time
from datetime import datetime

# Добавляем папку scripts в путь Python
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))


def print_header():
    """Печать заголовка программы"""
    print("\n" + "=" * 70)
    print("ПРАКТИЧЕСКОЕ ЗАДАНИЕ №4 - ИССЛЕДОВАНИЕ ВРЕМЕННЫХ РЯДОВ")
    print("=" * 70)
    print("Группа: 4315")
    print("Данные: UN Tourism - Отправной туризм из Австрии (2000-2024)")
    print("Автор: [Ваше ФИО]")
    print(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def print_step_header(step_number, step_name):
    """Печать заголовка шага"""
    print(f"\n{'=' * 60}")
    print(f"ШАГ {step_number}: {step_name}")
    print(f"{'=' * 60}")


def run_analysis_steps():
    """Запуск всех шагов анализа"""

    steps = [
        {
            'module': 'data_loading',
            'name': 'ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ',
            'description': 'Пункты 1.1-1.3 задания: загрузка, описание, формальное представление'
        },
        {
            'module': 'visualization',
            'name': 'ВИЗУАЛИЗАЦИЯ ДАННЫХ',
            'description': 'Пункт 1.4: построение графиков и диаграмм'
        },
        {
            'module': 'stationarity',
            'name': 'ПРОВЕРКА НА СТАЦИОНАРНОСТЬ',
            'description': 'Пункт 1.5: автокорреляции и коррелограмма'
        },
        {
            'module': 'decomposition',
            'name': 'ДЕКОМПОЗИЦИЯ ВРЕМЕННОГО РЯДА',
            'description': 'Пункт 1.6: выделение компонент'
        },
        {
            'module': 'anomaly_detection',
            'name': 'ОБНАРУЖЕНИЕ АНОМАЛИЙ',
            'description': 'Пункт 1.7: выявление аномальных уровней'
        },
        {
            'module': 'trend_smoothing',
            'name': 'СГЛАЖИВАНИЕ И ТРЕНД',
            'description': 'Пункт 1.8: методы скользящей средней, выделение тренда'
        },
        {
            'module': 'trend_modeling',
            'name': 'МОДЕЛИРОВАНИЕ ТРЕНДА',
            'description': 'Пункт 1.9: подбор модели, оценка параметров'
        },
        {
            'module': 'residual_analysis',
            'name': 'АНАЛИЗ ОСТАТКОВ',
            'description': 'Пункт 2.1: оценка остаточной компоненты (для оценки 4-5)',
            'optional': True
        }
    ]

    results = {}

    for i, step in enumerate(steps, 1):
        print_step_header(i, step['name'])
        print(f"Описание: {step['description']}")

        if step.get('optional', False):
            print("\n⚠️  Этот шаг опциональный (для оценки 4-5)")
            response = input("Выполнить этот шаг? (y/n): ").strip().lower()
            if response != 'y':
                print(f"Пропускаем шаг {i}...")
                results[step['module']] = {'skipped': True}
                continue

        print(f"\nЗапуск модуля {step['module']}...")

        try:
            # Импортируем модуль из папки scripts
            module = __import__(step['module'])

            # Замер времени выполнения
            start_time = time.time()

            # Выполнение main функции модуля
            if hasattr(module, 'main'):
                step_result = module.main()
                results[step['module']] = step_result
            else:
                print(f"⚠️  Модуль {step['module']} не имеет функции main()")
                results[step['module']] = {'error': 'No main function'}

            elapsed_time = time.time() - start_time
            print(f"✅ Шаг {i} выполнен за {elapsed_time:.2f} секунд")

        except ImportError as e:
            print(f"❌ Ошибка импорта модуля {step['module']}: {e}")
            print(f"   Проверьте наличие файла scripts/{step['module']}.py")
            results[step['module']] = {'error': str(e)}

        except Exception as e:
            print(f"❌ Ошибка при выполнении шага {i}: {e}")
            import traceback
            traceback.print_exc()
            results[step['module']] = {'error': str(e)}

            # Спрашиваем, продолжать ли
            if i < len(steps):
                response = input("\nПроизошла ошибка. Продолжить анализ? (y/n): ").strip().lower()
                if response != 'y':
                    print("Анализ прерван.")
                    break

    return results


def check_dependencies():
    """Проверка наличия необходимых библиотек"""
    print("Проверка зависимостей...")

    required = ['pandas', 'numpy', 'matplotlib', 'scipy', 'statsmodels']

    missing = []
    for lib in required:
        try:
            __import__(lib)
            print(f"  ✅ {lib}")
        except ImportError:
            print(f"  ❌ {lib} - отсутствует")
            missing.append(lib)

    if missing:
        print(f"\n⚠️  Отсутствуют библиотеки: {', '.join(missing)}")
        print("Установите командой: pip install " + " ".join(missing))
        response = input("Продолжить без них? (y/n): ").strip().lower()
        return response == 'y'

    return True


def main():
    """Основная функция"""

    # Печать заголовка
    print_header()

    # Проверка зависимостей
    if not check_dependencies():
        print("❌ Необходимые библиотеки отсутствуют. Установите их и попробуйте снова.")
        return

    print("\n" + "=" * 70)
    print("НАЧАЛО АНАЛИЗА ВРЕМЕННОГО РЯДА")
    print("=" * 70)

    # Запуск всех шагов анализа
    start_time = time.time()
    results = run_analysis_steps()
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЁН!")
    print("=" * 70)
    print(f"Общее время выполнения: {total_time:.2f} секунд")
    print("\n✅ Анализ временного ряда выполнен!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Анализ прерван пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()