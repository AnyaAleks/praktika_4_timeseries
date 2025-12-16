# 04_decomposition.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import config
from data_loading import load_data

def decompose_time_series(df):
    """
    Декомпозиция временного ряда на компоненты:
    - Трендовая
    - Сезонная (если есть)
    - Циклическая
    - Случайная (остатки)
    """
    print("\n" + "=" * 60)
    print("ДЕКОМПОЗИЦИЯ ВРЕМЕННОГО РЯДА")
    print("=" * 60)

    # Поскольку данные годовые, сезонность явно не выражена
    # Используем аддитивную модель для годовых данных
    print("Примечание: Данные годовые, явная сезонность отсутствует.")
    print("Анализируется трендовая, циклическая и случайная компоненты.\n")

    try:
        # Аддитивная декомпозиция
        decomposition = seasonal_decompose(df['value'],
                                           model='additive',
                                           period=1,  # Нет сезонности
                                           extrapolate_trend='freq')

        # Получаем компоненты
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        # Для годовых данных циклическая компонента - это остатки после выделения тренда
        # Более сложный подход: циклическая = ряд минус тренд минут сезонность
        cyclical = df['value'] - trend - seasonal

        print("✅ Декомпозиция выполнена успешно")
        print(f"   Трендовая компонента: {trend.dropna().shape[0]} точек")
        print(f"   Сезонная компонента: {seasonal.dropna().shape[0]} точек")
        print(f"   Остаточная компонента: {residual.dropna().shape[0]} точек")

        return {
            'observed': df['value'],
            'trend': trend,
            'seasonal': seasonal,
            'cyclical': cyclical,
            'residual': residual,
            'decomposition': decomposition
        }

    except Exception as e:
        print(f"❌ Ошибка при декомпозиции: {e}")

        # Альтернативный метод: простая оценка тренда через скользящее среднее
        print("Использую альтернативный метод (скользящее среднее)...")

        window = min(5, len(df) // 4)  # Размер окна
        trend = df['value'].rolling(window=window, center=True).mean()
        seasonal = pd.Series(np.zeros(len(df)), index=df.index)
        residual = df['value'] - trend
        cyclical = residual  # Для годовых данных

        return {
            'observed': df['value'],
            'trend': trend,
            'seasonal': seasonal,
            'cyclical': cyclical,
            'residual': residual,
            'decomposition': None
        }


def identify_cycles(cyclical_component):
    """
    Выделение циклической составляющей
    """
    print("\n" + "=" * 60)
    print("АНАЛИЗ ЦИКЛИЧЕСКОЙ СОСТАВЛЯЮЩЕЙ")
    print("=" * 60)

    # Поиск локальных максимумов и минимумов для идентификации циклов
    from scipy.signal import argrelextrema

    # Преобразуем в массив и убираем NaN
    cyclical_clean = cyclical_component.dropna()

    if len(cyclical_clean) < 5:
        print("Недостаточно данных для анализа циклов")
        return []

    # Находим локальные максимумы
    maxima_idx = argrelextrema(cyclical_clean.values, np.greater, order=2)[0]
    minima_idx = argrelextrema(cyclical_clean.values, np.less, order=2)[0]

    maxima = cyclical_clean.iloc[maxima_idx]
    minima = cyclical_clean.iloc[minima_idx]

    print(f"Найдено локальных максимумов: {len(maxima)}")
    print(f"Найдено локальных минимумов: {len(minima)}")

    if len(maxima) >= 2 and len(minima) >= 2:
        # Оцениваем длительность циклов (между максимумами)
        maxima_dates = maxima.index
        cycle_lengths = []

        for i in range(1, len(maxima_dates)):
            years_diff = (maxima_dates[i].year - maxima_dates[i - 1].year)
            cycle_lengths.append(years_diff)

        if cycle_lengths:
            avg_cycle = np.mean(cycle_lengths)
            print(f"\nСредняя продолжительность цикла: {avg_cycle:.1f} лет")
            print(f"Диапазон: {min(cycle_lengths)} - {max(cycle_lengths)} лет")

            # Определяем фазу текущего цикла
            if not maxima.empty and not minima.empty:
                last_max = maxima_dates[-1]
                last_min = minima.index[-1]

                if last_max > last_min:
                    print(f"Текущая фаза цикла: СНИЖЕНИЕ (последний пик: {last_max.year})")
                else:
                    print(f"Текущая фаза цикла: РОСТ (последний минимум: {last_min.year})")

    return maxima, minima


def plot_decomposition(decomposition_results):
    """
    Построение графиков декомпозиции
    """
    plt.style.use(config.PLOT_STYLE)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # 1. Исходный ряд
    axes[0].plot(decomposition_results['observed'].index,
                 decomposition_results['observed'].values,
                 color=config.COLORS['primary'],
                 label='Исходный ряд',
                 linewidth=2)
    axes[0].set_ylabel('Поездки, тыс.', fontsize=11)
    axes[0].set_title('Исходный временной ряд', fontsize=12)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # 2. Трендовая компонента
    axes[1].plot(decomposition_results['trend'].index,
                 decomposition_results['trend'].values,
                 color=config.COLORS['trend'],
                 label='Тренд',
                 linewidth=2.5)
    axes[1].set_ylabel('Поездки, тыс.', fontsize=11)
    axes[1].set_title('Трендовая компонента', fontsize=12)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    # 3. Сезонная компонента (если есть)
    seasonal_data = decomposition_results['seasonal']
    if seasonal_data.abs().max() > 0.01:  # Если есть значимая сезонность
        axes[2].plot(seasonal_data.index,
                     seasonal_data.values,
                     color=config.COLORS['seasonal'],
                     label='Сезонность',
                     linewidth=2)
        axes[2].set_ylabel('Отклонение', fontsize=11)
        axes[2].set_title('Сезонная компонента', fontsize=12)
    else:
        axes[2].text(0.5, 0.5, 'Сезонная компонента не выявлена\n(годовые данные)',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=axes[2].transAxes,
                     fontsize=11)
        axes[2].set_title('Сезонная компонента', fontsize=12)
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)

    # 4. Циклическая компонента
    axes[3].plot(decomposition_results['cyclical'].index,
                 decomposition_results['cyclical'].values,
                 color=config.COLORS['secondary'],
                 label='Циклическая',
                 linewidth=2)
    axes[3].set_xlabel('Год', fontsize=11)
    axes[3].set_ylabel('Отклонение', fontsize=11)
    axes[3].set_title('Циклическая компонента', fontsize=12)
    axes[3].legend(loc='upper left')
    axes[3].grid(True, alpha=0.3)

    # Добавляем нулевую линию для циклической компоненты
    axes[3].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    # Сохранение
    plot_path = os.path.join(config.PLOTS_DIR, '04_decomposition.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ График декомпозиции сохранён: {plot_path}")

    # Дополнительный график: все компоненты вместе
    fig2, ax = plt.subplots(figsize=(14, 7))

    ax.plot(decomposition_results['observed'].index,
            decomposition_results['observed'].values,
            color=config.COLORS['primary'],
            label='Исходный ряд',
            alpha=0.7,
            linewidth=1.5)

    ax.plot(decomposition_results['trend'].index,
            decomposition_results['trend'].values,
            color=config.COLORS['trend'],
            label='Тренд',
            linewidth=2.5)

    ax.fill_between(decomposition_results['cyclical'].index,
                    decomposition_results['trend'].values,
                    decomposition_results['observed'].values,
                    color=config.COLORS['secondary'],
                    alpha=0.3,
                    label='Циклическая + Случайная')

    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('Количество поездок, тыс.', fontsize=12)
    ax.set_title('Декомпозиция временного ряда: Исходный ряд = Тренд + Циклическая + Случайная',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path2 = os.path.join(config.PLOTS_DIR, '04_decomposition_combined.png')
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Комбинированный график декомпозиции сохранён: {plot_path2}")

    return fig, fig2


def save_decomposition_results(decomposition_results, maxima, minima):
    """
    Сохранение результатов декомпозиции
    """
    results = "РЕЗУЛЬТАТЫ ДЕКОМПОЗИЦИИ ВРЕМЕННОГО РЯДА\n"
    results += "=" * 60 + "\n\n"

    results += "1. КОМПОНЕНТЫ РЯДА:\n"
    results += "-" * 40 + "\n"

    # Статистики по компонентам
    components = ['trend', 'seasonal', 'cyclical', 'residual']
    component_names = ['Тренд', 'Сезонность', 'Циклическая', 'Остатки']

    for comp, name in zip(components, component_names):
        data = decomposition_results[comp].dropna()
        if len(data) > 0:
            results += f"\n{name}:\n"
            results += f"  Среднее: {data.mean():.2f}\n"
            results += f"  Стандартное отклонение: {data.std():.2f}\n"
            results += f"  Минимум: {data.min():.2f}\n"
            results += f"  Максимум: {data.max():.2f}\n"
            results += f"  Количество точек: {len(data)}\n"

    results += "\n" + "=" * 60 + "\n"
    results += "2. АНАЛИЗ ЦИКЛОВ:\n"
    results += "-" * 40 + "\n"

    if maxima is not None and len(maxima) > 0:
        results += f"Локальные максимумы циклической компоненты:\n"
        for idx, (date, value) in enumerate(maxima.items(), 1):
            results += f"  {idx}. {date.year}: {value:.2f}\n"

    if minima is not None and len(minima) > 0:
        results += f"\nЛокальные минимумы циклической компоненты:\n"
        for idx, (date, value) in enumerate(minima.items(), 1):
            results += f"  {idx}. {date.year}: {value:.2f}\n"

    # Выводы
    results += "\n" + "=" * 60 + "\n"
    results += "3. ВЫВОДЫ:\n"
    results += "-" * 40 + "\n"

    # Анализ сезонности
    seasonal_std = decomposition_results['seasonal'].std()
    if seasonal_std < 10:  # Пороговое значение
        results += "• Явная сезонная составляющая не выявлена (данные годовые).\n"
    else:
        results += f"• Обнаружена сезонная составляющая (σ={seasonal_std:.1f}).\n"

    # Анализ тренда
    trend_change = decomposition_results['trend'].iloc[-1] - decomposition_results['trend'].iloc[0]
    if trend_change > 0:
        results += f"• Тренд положительный (+{trend_change:.0f} тыс. поездок за период).\n"
    else:
        results += f"• Тренд отрицательный ({trend_change:.0f} тыс. поездок за период).\n"

    # Анализ циклов
    if maxima is not None and len(maxima) >= 2:
        results += "• Выявлены циклические колебания.\n"
    else:
        results += "• Явные циклические колебания не обнаружены.\n"

    # Сохранение
    results_path = os.path.join(config.TABLES_DIR, '04_decomposition_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(results)

    print(f"✅ Результаты декомпозиции сохранены: {results_path}")

    # Также сохраняем данные компонент в CSV
    components_df = pd.DataFrame({
        'year': decomposition_results['observed'].index,
        'observed': decomposition_results['observed'].values,
        'trend': decomposition_results['trend'].values,
        'seasonal': decomposition_results['seasonal'].values,
        'cyclical': decomposition_results['cyclical'].values,
        'residual': decomposition_results['residual'].values
    })

    csv_path = os.path.join(config.TABLES_DIR, '04_decomposition_components.csv')
    components_df.to_csv(csv_path, index=False)
    print(f"✅ Данные компонент сохранены в CSV: {csv_path}")


def main():
    """Основная функция модуля"""
    print("\n" + "=" * 60)
    print("ШАГ 4: ДЕКОМПОЗИЦИЯ ВРЕМЕННОГО РЯДА")
    print("=" * 60)

    # Загрузка данных
    df = load_data()
    if df is None:
        return

    # 1. Декомпозиция ряда
    decomposition_results = decompose_time_series(df)

    # 2. Анализ циклической составляющей
    maxima, minima = identify_cycles(decomposition_results['cyclical'])

    # 3. Построение графиков
    plot_decomposition(decomposition_results)

    # 4. Сохранение результатов
    save_decomposition_results(decomposition_results, maxima, minima)

    print("\n✅ Декомпозиция временного ряда завершена!")

    # Возвращаем результаты для использования в следующих шагах
    return decomposition_results


if __name__ == "__main__":
    main()