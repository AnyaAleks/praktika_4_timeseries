# 06_trend_smoothing.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from statsmodels.nonparametric.smoothers_lowess import lowess
import os
import config
from data_loading import load_data


def calculate_moving_averages(df, window_sizes=[3, 5]):
    """
    Расчет скользящих средних с разными окнами
    """
    print("\n" + "=" * 60)
    print("РАСЧЕТ СКОЛЬЗЯЩИХ СРЕДНИХ")
    print("=" * 60)

    moving_averages = {}

    for window in window_sizes:
        # Простая скользящая средняя
        sma = df['value'].rolling(window=window, center=True).mean()

        # Взвешенная скользящая средняя (линейные веса)
        weights = np.arange(1, window + 1)
        wma = df['value'].rolling(window=window, center=True).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )

        # Экспоненциальная скользящая средняя
        ema = df['value'].ewm(span=window, adjust=False).mean()

        moving_averages[f'SMA_{window}'] = sma
        moving_averages[f'WMA_{window}'] = wma
        moving_averages[f'EMA_{window}'] = ema

        print(f"\nОкно {window} лет:")
        print(f"  Простая скользящая средняя (SMA): {sma.dropna().shape[0]} точек")
        print(f"  Взвешенная скользящая средняя (WMA): {wma.dropna().shape[0]} точек")
        print(f"  Экспоненциальная скользящая средняя (EMA): {ema.dropna().shape[0]} точек")

    return moving_averages


def calculate_lowess_smoothing(df, frac_values=[0.3, 0.5, 0.7]):
    """
    LOWESS сглаживание (локально взвешенная регрессия)
    """
    print("\n" + "=" * 60)
    print("LOWESS СГЛАЖИВАНИЕ")
    print("=" * 60)

    lowess_results = {}

    # Преобразуем даты в числовой формат для LOWESS
    x = np.arange(len(df))
    y = df['value'].values

    for frac in frac_values:
        smoothed = lowess(y, x, frac=frac, return_sorted=False)
        lowess_results[f'LOWESS_{int(frac * 100)}'] = pd.Series(smoothed, index=df.index)

        print(f"LOWESS с frac={frac}:")
        print(f"  Сглаженных точек: {len(smoothed)}")
        print(f"  Стандартное отклонение остатков: {np.std(y - smoothed):.1f}")

    return lowess_results


def calculate_savitzky_golay(df, window_sizes=[5, 7], poly_orders=[2, 3]):
    """
    Сглаживание фильтром Савицкого-Голая
    """
    print("\n" + "=" * 60)
    print("ФИЛЬТР САВИЦКОГО-ГОЛАЯ")
    print("=" * 60)

    sg_results = {}

    y = df['value'].values

    for window in window_sizes:
        for order in poly_orders:
            if window > order:  # Условие корректности
                try:
                    smoothed = signal.savgol_filter(y, window_length=window, polyorder=order)
                    key = f'SG_{window}_{order}'
                    sg_results[key] = pd.Series(smoothed, index=df.index)

                    print(f"Окно {window}, порядок {order}:")
                    print(f"  Успешно сглажено")
                except:
                    print(f"Окно {window}, порядок {order}:")
                    print(f"  Невозможно применить (window ≤ polyorder)")

    return sg_results


def identify_trend(df, smoothing_results):
    """
    Анализ наличия тренда
    """
    print("\n" + "=" * 60)
    print("АНАЛИЗ НАЛИЧИЯ ТРЕНДА")
    print("=" * 60)

    # Берем наиболее сглаженную версию (LOWESS 70%)
    if 'LOWESS_70' in smoothing_results:
        smoothed_series = smoothing_results['LOWESS_70']
    elif 'SMA_5' in smoothing_results:
        smoothed_series = smoothing_results['SMA_5']
    else:
        # Берем первую доступную сглаженную серию
        smoothed_series = list(smoothing_results.values())[0]

    # Проверяем монотонность тренда
    diff = smoothed_series.diff().dropna()

    positive_trend = (diff > 0).sum()
    negative_trend = (diff < 0).sum()
    zero_trend = (diff == 0).sum()

    total_points = len(diff)

    print(f"Анализ изменений сглаженного ряда:")
    print(f"  Всего изменений: {total_points}")
    print(f"  Положительных: {positive_trend} ({positive_trend / total_points * 100:.1f}%)")
    print(f"  Отрицательных: {negative_trend} ({negative_trend / total_points * 100:.1f}%)")
    print(f"  Нулевых: {zero_trend} ({zero_trend / total_points * 100:.1f}%)")

    # Определяем тип тренда
    if positive_trend / total_points > 0.6:
        trend_type = "ВОСХОДЯЩИЙ (положительный)"
    elif negative_trend / total_points > 0.6:
        trend_type = "НИСХОДЯЩИЙ (отрицательный)"
    elif abs(positive_trend - negative_trend) / total_points < 0.3:
        trend_type = "БЕЗ ЯВНОГО ТРЕНДА (стационарный)"
    else:
        trend_type = "СМЕШАННЫЙ (нестабильный)"

    print(f"\nТИП ТРЕНДА: {trend_type}")

    # Оцениваем силу тренда через наклон линейной регрессии
    X = np.arange(len(smoothed_series.dropna())).reshape(-1, 1)
    y_clean = smoothed_series.dropna().values

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X, y_clean)

    slope = lr.coef_[0]  # Ежегодное изменение
    r_squared = lr.score(X, y_clean)

    print(f"\nЛинейная аппроксимация тренда:")
    print(f"  Наклон: {slope:.1f} тыс. поездок/год")
    print(f"  R²: {r_squared:.3f}")
    print(f"  Начальное значение: {lr.intercept_:.0f}")
    print(f"  Уравнение: y = {lr.intercept_:.0f} + {slope:.1f} * t")

    return {
        'trend_type': trend_type,
        'slope': slope,
        'r_squared': r_squared,
        'smoothed_series': smoothed_series,
        'trend_model': lr
    }


def plot_smoothing_results(df, moving_averages, lowess_results, sg_results, trend_analysis):
    """
    Построение графиков результатов сглаживания
    """
    plt.style.use(config.PLOT_STYLE)

    # Создаем фигуру с несколькими подграфиками
    fig = plt.figure(figsize=(18, 12))

    # 1. Основное сравнение методов сглаживания
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df.index, df['value'],
             color=config.COLORS['primary'],
             alpha=0.5,
             linewidth=1.5,
             label='Исходный ряд')

    # Добавляем несколько методов сглаживания
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    methods_to_plot = ['SMA_3', 'SMA_5', 'LOWESS_50', 'LOWESS_70']

    for i, method in enumerate(methods_to_plot):
        if method in moving_averages:
            series = moving_averages[method]
            color = colors[i % len(colors)]
        elif method in lowess_results:
            series = lowess_results[method]
            color = colors[i % len(colors)]
        else:
            continue

        ax1.plot(df.index, series,
                 color=color,
                 linewidth=2,
                 label=method)

    ax1.set_xlabel('Год', fontsize=11)
    ax1.set_ylabel('Количество поездок, тыс.', fontsize=11)
    ax1.set_title('Сравнение методов сглаживания', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Детальный анализ скользящих средних
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df.index, df['value'],
             color=config.COLORS['primary'],
             alpha=0.3,
             linewidth=1,
             label='Исходный')

    # Показываем разные окна SMA
    for window in [3, 5]:
        key = f'SMA_{window}'
        if key in moving_averages:
            ax2.plot(df.index, moving_averages[key],
                     linewidth=2,
                     label=f'SMA (окно={window})')

    # Показываем EMA для сравнения
    if 'EMA_5' in moving_averages:
        ax2.plot(df.index, moving_averages['EMA_5'],
                 linewidth=2,
                 linestyle='--',
                 label='EMA (окно=5)')

    ax2.set_xlabel('Год', fontsize=11)
    ax2.set_ylabel('Количество поездок, тыс.', fontsize=11)
    ax2.set_title('Скользящие средние', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 3. LOWESS сглаживание с разными параметрами
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(df.index, df['value'],
             color=config.COLORS['primary'],
             alpha=0.3,
             linewidth=1,
             label='Исходный')

    # Показываем разные уровни сглаживания LOWESS
    for frac in [30, 50, 70]:
        key = f'LOWESS_{frac}'
        if key in lowess_results:
            ax3.plot(df.index, lowess_results[key],
                     linewidth=2,
                     label=f'LOWESS (frac={frac / 100})')

    ax3.set_xlabel('Год', fontsize=11)
    ax3.set_ylabel('Количество поездок, тыс.', fontsize=11)
    ax3.set_title('LOWESS сглаживание', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # 4. Выделенный тренд и остатки
    ax4 = plt.subplot(2, 2, 4)

    # Тренд
    smoothed_series = trend_analysis['smoothed_series']
    ax4.plot(df.index, smoothed_series,
             color=config.COLORS['trend'],
             linewidth=3,
             label='Выделенный тренд')

    # Остатки (разница между исходным и трендом)
    residuals = df['value'] - smoothed_series
    ax4.fill_between(df.index, smoothed_series, df['value'],
                     color=config.COLORS['residual'],
                     alpha=0.3,
                     label='Отклонения от тренда')

    # Линейная аппроксимация тренда
    X = np.arange(len(smoothed_series.dropna()))
    y_trend = trend_analysis['trend_model'].predict(X.reshape(-1, 1))
    trend_dates = smoothed_series.dropna().index

    ax4.plot(trend_dates, y_trend,
             color='black',
             linestyle='--',
             linewidth=2,
             label=f'Линейный тренд (наклон: {trend_analysis["slope"]:.1f}/год)')

    ax4.set_xlabel('Год', fontsize=11)
    ax4.set_ylabel('Количество поездок, тыс.', fontsize=11)
    ax4.set_title(f'Выделение тренда ({trend_analysis["trend_type"]})', fontsize=12)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Сохранение
    plot_path = os.path.join(config.PLOTS_DIR, '06_trend_smoothing.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Графики сглаживания сохранены: {plot_path}")

    # Дополнительный график: только лучший метод сглаживания
    fig2, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df.index, df['value'],
            color=config.COLORS['primary'],
            alpha=0.7,
            linewidth=2,
            marker='o',
            markersize=4,
            label='Исходный ряд')

    # Используем LOWESS 70% как наиболее сглаженный
    best_smoothed = lowess_results.get('LOWESS_70',
                                       moving_averages.get('SMA_5',
                                                           list(moving_averages.values())[0]))

    ax.plot(df.index, best_smoothed,
            color=config.COLORS['trend'],
            linewidth=3,
            label='Сглаженный ряд (LOWESS 70%)')

    # Заполнение области между рядами
    ax.fill_between(df.index, best_smoothed, df['value'],
                    where=df['value'] >= best_smoothed,
                    color='green',
                    alpha=0.2,
                    label='Выше тренда')

    ax.fill_between(df.index, best_smoothed, df['value'],
                    where=df['value'] < best_smoothed,
                    color='red',
                    alpha=0.2,
                    label='Ниже тренда')

    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('Количество поездок, тыс.', fontsize=12)
    ax.set_title('Сглаживание временного ряда: выделение трендовой компоненты',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path2 = os.path.join(config.PLOTS_DIR, '06_best_smoothing.png')
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ График лучшего сглаживания сохранён: {plot_path2}")

    return fig, fig2


def save_smoothing_results(df, moving_averages, lowess_results, sg_results, trend_analysis):
    """
    Сохранение результатов сглаживания
    """
    results = "РЕЗУЛЬТАТЫ СГЛАЖИВАНИЯ И ВЫДЕЛЕНИЯ ТРЕНДА\n"
    results += "=" * 60 + "\n\n"

    results += "1. МЕТОДЫ СГЛАЖИВАНИЯ:\n"
    results += "-" * 40 + "\n"

    # Скользящие средние
    results += "\nСКОЛЬЗЯЩИЕ СРЕДНИЕ:\n"
    for key in sorted(moving_averages.keys()):
        series = moving_averages[key]
        valid_data = series.dropna()
        if len(valid_data) > 0:
            results += f"  {key}: {len(valid_data)} точек, среднее: {valid_data.mean():.1f}\n"

    # LOWESS
    results += "\nLOWESS СГЛАЖИВАНИЕ:\n"
    for key in sorted(lowess_results.keys()):
        series = lowess_results[key]
        valid_data = series.dropna()
        if len(valid_data) > 0:
            results += f"  {key}: {len(valid_data)} точек, среднее: {valid_data.mean():.1f}\n"

    # Савицкого-Голая
    if sg_results:
        results += "\nФИЛЬТР САВИЦКОГО-ГОЛАЯ:\n"
        for key in sorted(sg_results.keys()):
            series = sg_results[key]
            valid_data = series.dropna()
            if len(valid_data) > 0:
                results += f"  {key}: {len(valid_data)} точек\n"

    results += "\n" + "=" * 60 + "\n"
    results += "2. АНАЛИЗ ТРЕНДА:\n"
    results += "-" * 40 + "\n\n"

    results += f"ТИП ТРЕНДА: {trend_analysis['trend_type']}\n\n"
    results += "ЛИНЕЙНАЯ АППРОКСИМАЦИЯ:\n"
    results += f"  Уравнение: y = {trend_analysis['trend_model'].intercept_:.0f} "
    results += f"+ {trend_analysis['slope']:.2f} * t\n"
    results += f"  Наклон: {trend_analysis['slope']:.2f} тыс. поездок/год\n"
    results += f"  Коэффициент детерминации R²: {trend_analysis['r_squared']:.4f}\n"

    # Интерпретация силы тренда
    slope = trend_analysis['slope']
    if abs(slope) > 100:
        strength = "ОЧЕНЬ СИЛЬНЫЙ"
    elif abs(slope) > 50:
        strength = "СИЛЬНЫЙ"
    elif abs(slope) > 20:
        strength = "УМЕРЕННЫЙ"
    elif abs(slope) > 5:
        strength = "СЛАБЫЙ"
    else:
        strength = "ОЧЕНЬ СЛАБЫЙ"

    results += f"  СИЛА ТРЕНДА: {strength}\n"

    # Прогноз на основе линейного тренда
    model = trend_analysis['trend_model']
    last_year = 2024
    forecast_years = 3

    results += f"\nПРОГНОЗ НА {forecast_years} ЛЕТ ВПЕРЕД:\n"

    for i in range(1, forecast_years + 1):
        year = last_year + i
        t = len(moving_averages['SMA_3'].dropna()) + i - 1  # Продолжаем временную ось
        forecast = model.predict([[t]])[0]
        results += f"  {year}: {forecast:.0f} тыс. поездок\n"

    results += "\n" + "=" * 60 + "\n"
    results += "3. ВЫВОДЫ:\n"
    results += "-" * 40 + "\n\n"

    if "ВОСХОДЯЩИЙ" in trend_analysis['trend_type']:
        results += "• Наблюдается устойчивый положительный тренд\n"
        results += "• Количество туристических поездок из Австрии растёт\n"
        results += "• Рост составляет примерно "
        results += f"{abs(slope):.0f} тыс. поездок в год\n"
    elif "НИСХОДЯЩИЙ" in trend_analysis['trend_type']:
        results += "• Наблюдается отрицательный тренд\n"
        results += "• Количество туристических поездок снижается\n"
        results += "• Снижение составляет примерно "
        results += f"{abs(slope):.0f} тыс. поездок в год\n"
    else:
        results += "• Явного тренда не обнаружено\n"
        results += "• Ряд демонстрирует стационарное поведение\n"
        results += "• Значимых изменений за период не наблюдается\n"

    results += f"• Качество аппроксимации тренда: "
    results += f"{'высокое' if trend_analysis['r_squared'] > 0.7 else 'умеренное' if trend_analysis['r_squared'] > 0.3 else 'низкое'}\n"
    results += "• Для прогнозирования рекомендуется использовать методы, "
    results += "учитывающие циклические колебания\n"

    # Сохранение
    results_path = os.path.join(config.TABLES_DIR, '06_smoothing_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(results)

    print(f"✅ Результаты сглаживания сохранены: {results_path}")

    # Сохраняем сглаженные данные в CSV
    smoothed_data = pd.DataFrame({'year': df.index})
    smoothed_data['original'] = df['value'].values

    # Добавляем основные методы сглаживания
    for key in ['SMA_3', 'SMA_5', 'LOWESS_50', 'LOWESS_70']:
        if key in moving_averages:
            smoothed_data[key] = moving_averages[key].values
        elif key in lowess_results:
            smoothed_data[key] = lowess_results[key].values

    csv_path = os.path.join(config.TABLES_DIR, '06_smoothed_data.csv')
    smoothed_data.to_csv(csv_path, index=False)
    print(f"✅ Сглаженные данные сохранены в CSV: {csv_path}")


def main():
    """Основная функция модуля"""
    print("\n" + "=" * 60)
    print("ШАГ 6: СГЛАЖИВАНИЕ И ВЫДЕЛЕНИЕ ТРЕНДА")
    print("=" * 60)

    # Загрузка данных
    df = load_data()
    if df is None:
        return

    # 1. Расчет скользящих средних
    moving_averages = calculate_moving_averages(df, window_sizes=[3, 5])

    # 2. LOWESS сглаживание
    lowess_results = calculate_lowess_smoothing(df, frac_values=[0.3, 0.5, 0.7])

    # 3. Фильтр Савицкого-Голая
    sg_results = calculate_savitzky_golay(df, window_sizes=[5, 7], poly_orders=[2, 3])

    # 4. Анализ тренда
    trend_analysis = identify_trend(df, {**moving_averages, **lowess_results, **sg_results})

    # 5. Построение графиков
    plot_smoothing_results(df, moving_averages, lowess_results, sg_results, trend_analysis)

    # 6. Сохранение результатов
    save_smoothing_results(df, moving_averages, lowess_results, sg_results, trend_analysis)

    print("\n✅ Сглаживание и выделение тренда завершено!")

    # Возвращаем результаты
    return {
        'moving_averages': moving_averages,
        'lowess_results': lowess_results,
        'sg_results': sg_results,
        'trend_analysis': trend_analysis
    }


if __name__ == "__main__":
    main()