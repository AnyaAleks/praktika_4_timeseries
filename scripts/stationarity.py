# 03_stationarity.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import config
from data_loading import load_data


def test_stationarity_adf(df):
    """
    Проверка стационарности с помощью теста Дики-Фуллера (ADF)
    """
    print("\n" + "=" * 60)
    print("ТЕСТ ДИКИ-ФУЛЛЕРА (ADF) НА СТАЦИОНАРНОСТЬ")
    print("=" * 60)

    result = adfuller(df['value'].dropna())

    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")

    # Интерпретация
    if result[1] <= config.SIGNIFICANCE_LEVEL:
        print(f"\n✅ Ряд СТАЦИОНАРЕН (p-value = {result[1]:.4f} ≤ {config.SIGNIFICANCE_LEVEL})")
        print("   Нулевая гипотеза о наличии единичного корня отвергается.")
    else:
        print(f"\n❌ Ряд НЕ СТАЦИОНАРЕН (p-value = {result[1]:.4f} > {config.SIGNIFICANCE_LEVEL})")
        print("   Нулевая гипотеза о наличии единичного корня не отвергается.")

    return result


def calculate_autocorrelations(df, nlags=10):
    """
    Расчет автокорреляций и частичных автокорреляций
    """
    print("\n" + "=" * 60)
    print("РАСЧЕТ АВТОКОРРЕЛЯЦИЙ")
    print("=" * 60)

    # Автокорреляционная функция (ACF)
    acf_values = acf(df['value'], nlags=nlags, fft=True)
    pacf_values = pacf(df['value'], nlags=nlags)

    print(f"Лаг |     ACF     |    PACF    ")
    print("-" * 35)
    for i in range(1, nlags + 1):
        print(f"{i:3d} | {acf_values[i]:10.4f} | {pacf_values[i]:10.4f}")

    # Проверка значимости автокорреляций
    print(f"\nПроверка значимости (95% доверительный интервал):")
    print(f"  Критическое значение для ACF: ±{1.96 / np.sqrt(len(df)):.4f}")

    significant_lags_acf = np.where(np.abs(acf_values[1:]) > 1.96 / np.sqrt(len(df)))[0] + 1
    significant_lags_pacf = np.where(np.abs(pacf_values[1:]) > 1.96 / np.sqrt(len(df)))[0] + 1

    if len(significant_lags_acf) > 0:
        print(f"  Значимые лаги ACF: {list(significant_lags_acf)}")
    else:
        print(f"  Нет значимых лагов ACF")

    if len(significant_lags_pacf) > 0:
        print(f"  Значимые лаги PACF: {list(significant_lags_pacf)}")
    else:
        print(f"  Нет значимых лагов PACF")

    return acf_values, pacf_values


def plot_correlogram(df, acf_values, pacf_values, nlags=10):
    """
    Построение коррелограммы (ACF и PACF графики)
    """
    plt.style.use(config.PLOT_STYLE)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # График ACF
    plot_acf(df['value'], lags=nlags, ax=ax1,
             color=config.COLORS['primary'],
             title='Автокорреляционная функция (ACF)')
    ax1.set_xlabel('Лаг')
    ax1.set_ylabel('ACF')
    ax1.grid(True, alpha=0.3)

    # График PACF
    plot_pacf(df['value'], lags=nlags, ax=ax2,
              color=config.COLORS['secondary'],
              title='Частичная автокорреляционная функция (PACF)')
    ax2.set_xlabel('Лаг')
    ax2.set_ylabel('PACF')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Сохранение
    plot_path = os.path.join(config.PLOTS_DIR, '03_correlogram.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Коррелограмма сохранена: {plot_path}")
    return fig


def plot_rolling_statistics(df):
    """
    Построение графиков скользящих статистик для визуальной оценки стационарности
    """
    plt.style.use(config.PLOT_STYLE)

    # Расчет скользящих статистик
    rolling_mean = df['value'].rolling(window=config.SMOOTHING_WINDOW).mean()
    rolling_std = df['value'].rolling(window=config.SMOOTHING_WINDOW).std()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # 1. Исходный ряд и скользящее среднее
    ax1.plot(df.index, df['value'], label='Исходный ряд',
             color=config.COLORS['primary'], alpha=0.7)
    ax1.plot(df.index, rolling_mean, label=f'Скользящее среднее (окно={config.SMOOTHING_WINDOW})',
             color=config.COLORS['trend'], linewidth=2.5)
    ax1.set_xlabel('Год', fontsize=11)
    ax1.set_ylabel('Количество поездок, тыс.', fontsize=11)
    ax1.set_title('Исходный ряд и скользящее среднее', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Скользящее стандартное отклонение
    ax2.plot(df.index, rolling_std, label='Скользящее стандартное отклонение',
             color=config.COLORS['secondary'], linewidth=2)
    ax2.axhline(y=df['value'].std(), color='r', linestyle='--',
                label=f'Общее std: {df["value"].std():.1f}')
    ax2.set_xlabel('Год', fontsize=11)
    ax2.set_ylabel('Стандартное отклонение', fontsize=11)
    ax2.set_title('Изменчивость ряда (скользящее стандартное отклонение)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Сохранение
    plot_path = os.path.join(config.PLOTS_DIR, '03_rolling_statistics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Графики скользящих статистик сохранены: {plot_path}")
    return fig


def save_stationarity_results(adf_result, acf_values, pacf_values):
    """
    Сохранение результатов проверки стационарности
    """
    results = "РЕЗУЛЬТАТЫ ПРОВЕРКИ СТАЦИОНАРНОСТИ РЯДА\n"
    results += "=" * 60 + "\n\n"

    results += "1. ТЕСТ ДИКИ-ФУЛЛЕРА (ADF):\n"
    results += "-" * 40 + "\n"
    results += f"ADF Statistic: {adf_result[0]:.4f}\n"
    results += f"p-value: {adf_result[1]:.4f}\n"
    results += "Критические значения:\n"
    for key, value in adf_result[4].items():
        results += f"  {key}: {value:.4f}\n"

    if adf_result[1] <= config.SIGNIFICANCE_LEVEL:
        results += f"\nВЫВОД: Ряд СТАЦИОНАРЕН (p-value ≤ {config.SIGNIFICANCE_LEVEL})\n"
    else:
        results += f"\nВЫВОД: Ряд НЕ СТАЦИОНАРЕН (p-value > {config.SIGNIFICANCE_LEVEL})\n"

    results += "\n" + "=" * 60 + "\n"
    results += "2. АВТОКОРРЕЛЯЦИОННЫЙ АНАЛИЗ:\n"
    results += "-" * 40 + "\n"

    nlags = len(acf_values) - 1
    results += f"Лаг |     ACF     |    PACF    \n"
    results += "-" * 35 + "\n"
    for i in range(1, min(11, nlags + 1)):
        results += f"{i:3d} | {acf_values[i]:10.4f} | {pacf_values[i]:10.4f}\n"

    # Проверка значимости
    critical_value = 1.96 / np.sqrt(len(acf_values) * 3)  # Примерная оценка
    results += f"\nКритическое значение для значимости: ±{critical_value:.4f}\n"

    # Сохранение
    results_path = os.path.join(config.TABLES_DIR, '03_stationarity_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(results)

    print(f"✅ Результаты проверки стационарности сохранены: {results_path}")


def main():
    """Основная функция модуля"""
    print("\n" + "=" * 60)
    print("ШАГ 3: ПРОВЕРКА РЯДА НА СТАЦИОНАРНОСТЬ")
    print("=" * 60)

    # Загрузка данных
    df = load_data()
    if df is None:
        return

    # 1. Тест Дики-Фуллера
    adf_result = test_stationarity_adf(df)

    # 2. Расчет автокорреляций
    acf_values, pacf_values = calculate_autocorrelations(df, nlags=10)

    # 3. Построение коррелограммы
    plot_correlogram(df, acf_values, pacf_values)

    # 4. Графики скользящих статистик
    plot_rolling_statistics(df)

    # 5. Сохранение результатов
    save_stationarity_results(adf_result, acf_values, pacf_values)

    print("\n✅ Проверка стационарности завершена!")

    # Возвращаем результаты для использования в следующих шагах
    return {
        'is_stationary': adf_result[1] <= config.SIGNIFICANCE_LEVEL,
        'adf_pvalue': adf_result[1],
        'acf_values': acf_values,
        'pacf_values': pacf_values
    }


if __name__ == "__main__":
    main()