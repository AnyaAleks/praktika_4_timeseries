# 05_anomaly_detection.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import os
import config
from data_loading import load_data


def detect_anomalies_statistical(df):
    """
    Обнаружение аномалий статистическими методами
    """
    print("\n" + "=" * 60)
    print("СТАТИСТИЧЕСКОЕ ОБНАРУЖЕНИЕ АНОМАЛИЙ")
    print("=" * 60)

    anomalies = {}

    # 1. Метод межквартильного размаха (IQR)
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    iqr_anomalies = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
    anomalies['IQR'] = {
        'anomalies': iqr_anomalies,
        'bounds': (lower_bound, upper_bound),
        'method': 'Межквартильный размах (IQR)'
    }

    print(f"Метод IQR:")
    print(f"  Q1: {Q1:.0f}, Q3: {Q3:.0f}, IQR: {IQR:.0f}")
    print(f"  Границы: [{lower_bound:.0f}, {upper_bound:.0f}]")
    print(f"  Найдено аномалий: {len(iqr_anomalies)}")
    if not iqr_anomalies.empty:
        print("  Годы аномалий:", list(iqr_anomalies.index.year))

    # 2. Метод Z-score (стандартизованные отклонения)
    z_scores = np.abs(stats.zscore(df['value']))
    z_threshold = 2.0  # Порог 2 стандартных отклонения

    z_anomalies = df[z_scores > z_threshold]
    anomalies['Z-score'] = {
        'anomalies': z_anomalies,
        'z_scores': pd.Series(z_scores, index=df.index),
        'threshold': z_threshold,
        'method': 'Z-score'
    }

    print(f"\nМетод Z-score (порог: {z_threshold}σ):")
    print(f"  Найдено аномалий: {len(z_anomalies)}")
    if not z_anomalies.empty:
        print("  Годы аномалий:", list(z_anomalies.index.year))
        for year, z in zip(z_anomalies.index.year, z_scores[z_scores > z_threshold]):
            print(f"    {year}: Z-score = {z:.2f}")

    # 3. Метод на основе скользящих статистик
    window = 5
    rolling_mean = df['value'].rolling(window=window, center=True).mean()
    rolling_std = df['value'].rolling(window=window, center=True).std()

    # Заполняем NaN на краях
    rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
    rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')

    # Аномалии - отклонения более чем на 2 стандартных отклонения от скользящего среднего
    rolling_anomalies = df[np.abs(df['value'] - rolling_mean) > 2 * rolling_std]

    anomalies['Rolling'] = {
        'anomalies': rolling_anomalies,
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
        'method': 'Скользящие статистики'
    }

    print(f"\nМетод скользящих статистик (окно: {window} лет):")
    print(f"  Найдено аномалий: {len(rolling_anomalies)}")
    if not rolling_anomalies.empty:
        print("  Годы аномалий:", list(rolling_anomalies.index.year))

    # 4. Объединение всех методов
    all_anomaly_indices = set()
    for method, result in anomalies.items():
        if not result['anomalies'].empty:
            all_anomaly_indices.update(result['anomalies'].index)

    combined_anomalies = df.loc[list(all_anomaly_indices)]

    print(f"\nОБЪЕДИНЕННЫЕ РЕЗУЛЬТАТЫ:")
    print(f"  Всего уникальных аномальных лет: {len(combined_anomalies)}")
    print(f"  Аномальные годы: {sorted([y.year for y in combined_anomalies.index])}")

    anomalies['Combined'] = {
        'anomalies': combined_anomalies,
        'method': 'Объединенный результат'
    }

    return anomalies


def contextual_anomaly_detection(df):
    """
    Контекстное обнаружение аномалий (с учетом известных событий)
    """
    print("\n" + "=" * 60)
    print("КОНТЕКСТНОЕ ОБНАРУЖЕНИЕ АНОМАЛИЙ")
    print("=" * 60)

    contextual_anomalies = []

    # Известные события, которые могли вызвать аномалии
    known_events = {
        2020: ("Пандемия COVID-19", "Глобальный кризис туризма"),
        2008: ("Мировой финансовый кризис", "Сокращение расходов на туризм"),
        2001: ("Теракты 11 сентября", "Влияние на международные поездки")
    }

    # Рассчитываем ожидаемые значения на основе тренда
    # Простой линейный тренд для оценки ожидаемых значений
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['value'].values

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    expected = model.predict(X)

    # Рассчитываем отклонения от ожидаемого
    deviations = df['value'] - expected
    deviation_pct = (deviations / expected) * 100

    # Порог для аномалий: отклонение более чем на 20%
    threshold_pct = 20
    significant_deviations = deviation_pct.abs() > threshold_pct

    print(f"Контекстный анализ (отклонение от линейного тренда > {threshold_pct}%):")

    for idx, (date, row) in enumerate(df.iterrows()):
        year = date.year
        actual = row['value']
        expected_val = expected[idx]
        dev_pct = deviation_pct.iloc[idx]

        if significant_deviations.iloc[idx]:
            event_info = known_events.get(year, ("Неизвестное событие", "Требует исследования"))
            contextual_anomalies.append({
                'year': year,
                'actual': actual,
                'expected': expected_val,
                'deviation_pct': dev_pct,
                'event': event_info[0],
                'explanation': event_info[1]
            })

            print(f"  {year}:")
            print(f"    Фактическое: {actual:.0f}, Ожидаемое: {expected_val:.0f}")
            print(f"    Отклонение: {dev_pct:.1f}%")
            print(f"    Событие: {event_info[0]}")
            print(f"    Объяснение: {event_info[1]}")

    return pd.DataFrame(contextual_anomalies)


def plot_anomalies(df, anomalies_results, contextual_anomalies):
    """
    Построение графиков с выделенными аномалиями
    """
    plt.style.use(config.PLOT_STYLE)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. График временного ряда с аномалиями (IQR метод)
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['value'],
             color=config.COLORS['primary'],
             linewidth=2,
             marker='o',
             markersize=5,
             label='Временной ряд')

    # Выделение аномалий IQR
    iqr_anomalies = anomalies_results['IQR']['anomalies']
    if not iqr_anomalies.empty:
        ax1.scatter(iqr_anomalies.index, iqr_anomalies['value'],
                    color=config.COLORS['anomaly'],
                    s=100, zorder=5,
                    label='Аномалии (IQR метод)')

    # Границы IQR
    lower, upper = anomalies_results['IQR']['bounds']
    ax1.axhline(y=lower, color='red', linestyle='--', alpha=0.5, label='Нижняя граница IQR')
    ax1.axhline(y=upper, color='red', linestyle='--', alpha=0.5, label='Верхняя граница IQR')

    ax1.fill_between(df.index, lower, upper, alpha=0.1, color='green', label='Нормальный диапазон')

    ax1.set_xlabel('Год', fontsize=11)
    ax1.set_ylabel('Количество поездок, тыс.', fontsize=11)
    ax1.set_title('Обнаружение аномалий: метод IQR', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. График Z-score
    ax2 = axes[0, 1]
    z_scores = anomalies_results['Z-score']['z_scores']
    ax2.plot(df.index, z_scores,
             color=config.COLORS['secondary'],
             linewidth=2,
             marker='s',
             markersize=5,
             label='Z-score')

    # Пороговые линии
    threshold = anomalies_results['Z-score']['threshold']
    ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Порог (+{threshold}σ)')
    ax2.axhline(y=-threshold, color='red', linestyle='--', label=f'Порог (-{threshold}σ)')

    # Выделение аномальных точек
    z_anomalies = anomalies_results['Z-score']['anomalies']
    if not z_anomalies.empty:
        anomaly_z_scores = z_scores.loc[z_anomalies.index]
        ax2.scatter(z_anomalies.index, anomaly_z_scores,
                    color=config.COLORS['anomaly'],
                    s=100, zorder=5,
                    label='Аномалии')

    ax2.set_xlabel('Год', fontsize=11)
    ax2.set_ylabel('Z-score', fontsize=11)
    ax2.set_title('Обнаружение аномалий: метод Z-score', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 3. График отклонений от тренда
    ax3 = axes[1, 0]

    # Простой линейный тренд
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['value'].values

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)

    deviations = df['value'] - trend

    ax3.plot(df.index, deviations,
             color=config.COLORS['residual'],
             linewidth=2,
             label='Отклонения от тренда')

    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Выделение контекстных аномалий
    if not contextual_anomalies.empty:
        for _, anomaly in contextual_anomalies.iterrows():
            year = anomaly['year']
            dev = deviations.loc[pd.Timestamp(f'{year}-01-01')]
            ax3.scatter(pd.Timestamp(f'{year}-01-01'), dev,
                        color=config.COLORS['anomaly'],
                        s=100, zorder=5,
                        label=anomaly['event'] if anomaly.name == 0 else "")

    ax3.set_xlabel('Год', fontsize=11)
    ax3.set_ylabel('Отклонение от тренда', fontsize=11)
    ax3.set_title('Контекстные аномалии', fontsize=12)

    # Убираем дубликаты в легенде
    handles, labels = ax3.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax3.legend(by_label.values(), by_label.keys(), loc='upper left')

    ax3.grid(True, alpha=0.3)

    # 4. Сводный график всех аномалий
    ax4 = axes[1, 1]

    ax4.plot(df.index, df['value'],
             color=config.COLORS['primary'],
             linewidth=2,
             alpha=0.7,
             label='Временной ряд')

    # Объединенные аномалии
    combined_anomalies = anomalies_results['Combined']['anomalies']
    if not combined_anomalies.empty:
        ax4.scatter(combined_anomalies.index, combined_anomalies['value'],
                    color=config.COLORS['anomaly'],
                    s=150, zorder=5,
                    edgecolors='black',
                    linewidth=2,
                    label=f'Аномалии ({len(combined_anomalies)} лет)')

        # Добавляем подписи
        for date, value in combined_anomalies.iterrows():
            ax4.annotate(f'{date.year}\n{value["value"]:.0f}K',
                         xy=(date, value['value']),
                         xytext=(10, 10),
                         textcoords='offset points',
                         fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white",
                                   edgecolor=config.COLORS['anomaly']))

    ax4.set_xlabel('Год', fontsize=11)
    ax4.set_ylabel('Количество поездок, тыс.', fontsize=11)
    ax4.set_title('Сводка всех обнаруженных аномалий', fontsize=12)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Сохранение
    plot_path = os.path.join(config.PLOTS_DIR, '05_anomaly_detection.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Графики обнаружения аномалий сохранены: {plot_path}")
    return fig


def save_anomaly_results(anomalies_results, contextual_anomalies):
    """
    Сохранение результатов обнаружения аномалий
    """
    results = "РЕЗУЛЬТАТЫ ОБНАРУЖЕНИЯ АНОМАЛЬНЫХ УРОВНЕЙ\n"
    results += "=" * 60 + "\n\n"

    # Статистические методы
    results += "1. СТАТИСТИЧЕСКИЕ МЕТОДЫ:\n"
    results += "-" * 40 + "\n"

    for method_name, method_results in anomalies_results.items():
        if method_name == 'Combined':
            continue

        results += f"\n{method_results['method']}:\n"
        anomalies = method_results['anomalies']

        if anomalies.empty:
            results += "  Аномалии не обнаружены\n"
        else:
            results += f"  Найдено аномалий: {len(anomalies)}\n"
            for date, row in anomalies.iterrows():
                results += f"  • {date.year}: {row['value']:.0f} тыс. поездок\n"

    # Объединенные результаты
    results += "\n" + "=" * 60 + "\n"
    results += "2. ОБЪЕДИНЕННЫЕ РЕЗУЛЬТАТЫ:\n"
    results += "-" * 40 + "\n"

    combined = anomalies_results['Combined']['anomalies']
    if combined.empty:
        results += "Аномалии не обнаружены\n"
    else:
        results += f"Всего аномальных лет: {len(combined)}\n\n"
        for date, row in combined.iterrows():
            results += f"• {date.year}: {row['value']:.0f} тыс. поездок\n"

    # Контекстные аномалии
    results += "\n" + "=" * 60 + "\n"
    results += "3. КОНТЕКСТНЫЕ АНОМАЛИИ:\n"
    results += "-" * 40 + "\n"

    if contextual_anomalies.empty:
        results += "Контекстные аномалии не обнаружены\n"
    else:
        results += "Годы с существенными отклонениями от тренда:\n\n"
        for _, anomaly in contextual_anomalies.iterrows():
            results += f"• {anomaly['year']}:\n"
            results += f"  Фактическое: {anomaly['actual']:.0f}\n"
            results += f"  Ожидаемое: {anomaly['expected']:.0f}\n"
            results += f"  Отклонение: {anomaly['deviation_pct']:.1f}%\n"
            results += f"  Событие: {anomaly['event']}\n"
            results += f"  Объяснение: {anomaly['explanation']}\n\n"

    # Выводы и рекомендации
    results += "\n" + "=" * 60 + "\n"
    results += "4. ВЫВОДЫ И РЕКОМЕНДАЦИИ:\n"
    results += "-" * 40 + "\n"

    if not combined.empty:
        years = [date.year for date in combined.index]
        results += f"• Обнаружены статистические аномалии в годах: {sorted(years)}\n"
        results += "• Рекомендуется провести детальный анализ причин аномалий\n"
        results += "• При прогнозировании следует учитывать возможность подобных событий\n"
        results += "• Аномалии могут указывать на структурные изменения в ряде\n"
    else:
        results += "• Статистически значимых аномалий не обнаружено\n"
        results += "• Ряд демонстрирует устойчивое поведение\n"
        results += "• Прогнозы могут быть более надежными\n"

    # Сохранение
    results_path = os.path.join(config.TABLES_DIR, '05_anomaly_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(results)

    print(f"✅ Результаты обнаружения аномалий сохранены: {results_path}")

    # Сохраняем данные аномалий в CSV
    if not combined.empty:
        combined_df = combined.copy()
        combined_df['year'] = combined_df.index.year
        combined_df['is_anomaly'] = True

        csv_path = os.path.join(config.TABLES_DIR, '05_anomalies.csv')
        combined_df.to_csv(csv_path)
        print(f"✅ Данные аномалий сохранены в CSV: {csv_path}")


def main():
    """Основная функция модуля"""
    print("\n" + "=" * 60)
    print("ШАГ 5: ОБНАРУЖЕНИЕ АНОМАЛЬНЫХ УРОВНЕЙ")
    print("=" * 60)

    # Загрузка данных
    df = load_data()
    if df is None:
        return

    # 1. Статистическое обнаружение аномалий
    anomalies_results = detect_anomalies_statistical(df)

    # 2. Контекстное обнаружение аномалий
    contextual_anomalies = contextual_anomaly_detection(df)

    # 3. Построение графиков
    plot_anomalies(df, anomalies_results, contextual_anomalies)

    # 4. Сохранение результатов
    save_anomaly_results(anomalies_results, contextual_anomalies)

    print("\n✅ Обнаружение аномальных уровней завершено!")

    # Возвращаем результаты
    return {
        'anomalies': anomalies_results,
        'contextual_anomalies': contextual_anomalies
    }


if __name__ == "__main__":
    main()