# 02_visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import config
from data_loading import load_data

def create_time_series_plot(df):
    """
    Создание линейного графика временного ряда
    Пункт 1.4 задания - основная визуализация
    """
    plt.style.use(config.PLOT_STYLE)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Основной график временного ряда
    ax1.plot(df.index, df['value'],
             marker='o',
             linewidth=2.5,
             markersize=6,
             color=config.COLORS['primary'],
             label='Туристические поездки')

    # Выделение ключевых точек
    key_points = {
        pd.Timestamp('2019-01-01'): ('Докризисный пик', config.COLORS['secondary']),
        pd.Timestamp('2020-01-01'): ('Пандемия COVID-19', config.COLORS['anomaly']),
        pd.Timestamp('2022-01-01'): ('Восстановление', '#45B7D1'),
        pd.Timestamp('2024-01-01'): ('Текущий максимум', '#96CEB4')
    }

    for date, (label, color) in key_points.items():
        if date in df.index:
            value = df.loc[date, 'value']
            ax1.plot(date, value, 'o', markersize=12,
                     color=color, markeredgecolor='white',
                     markeredgewidth=2, label=label)
            # Аннотация
            ax1.annotate(f'{date.year}: {value:,.0f}K',
                         xy=(date, value),
                         xytext=(10, 10),
                         textcoords='offset points',
                         fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white",
                                   edgecolor=color, alpha=0.8))

    ax1.set_xlabel('Год', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Количество поездок, тыс.', fontsize=12, fontweight='bold')
    ax1.set_title('Динамика отправных туристических поездок из Австрии (2000-2024)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.4)
    ax1.legend(loc='upper left', fontsize=10)

    # Форматирование оси X
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. Столбчатая диаграмма (гистограмма)
    years = [int(str(y)[:4]) for y in df.index]
    values = df['value'].values

    bars = ax2.bar(years, values, color=config.COLORS['primary'], alpha=0.7)

    # Выделение столбцов для аномальных лет
    anomaly_years = [2020]  # COVID год
    for i, year in enumerate(years):
        if year in anomaly_years:
            bars[i].set_color(config.COLORS['anomaly'])
            bars[i].set_alpha(0.9)

    ax2.set_xlabel('Год', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Количество поездок, тыс.', fontsize=12, fontweight='bold')
    ax2.set_title('Гистограмма: отправные туристические поездки по годам',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')

    # Добавление значений на столбцы
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 100,
                 f'{int(height):,}', ha='center', va='bottom',
                 fontsize=8, rotation=0)

    plt.tight_layout()

    # Сохранение графика
    plot_path = os.path.join(config.PLOTS_DIR, '02_time_series_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Основной график сохранён: {plot_path}")
    return fig


def create_distribution_plots(df):
    """
    Создание дополнительных графиков распределения
    """
    plt.style.use(config.PLOT_STYLE)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Гистограмма с KDE
    ax1.hist(df['value'], bins=8, density=True,
             alpha=0.6, color=config.COLORS['primary'],
             edgecolor='black', label='Гистограмма')

    # KDE plot
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(df['value'])
    x_range = np.linspace(df['value'].min(), df['value'].max(), 100)
    ax1.plot(x_range, kde(x_range), color=config.COLORS['secondary'],
             linewidth=2, label='KDE')

    ax1.set_xlabel('Количество поездок, тыс.', fontsize=11)
    ax1.set_ylabel('Плотность', fontsize=11)
    ax1.set_title('Распределение значений (гистограмма + KDE)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot
    ax2.boxplot(df['value'], vert=True, patch_artist=True,
                boxprops=dict(facecolor=config.COLORS['primary'], alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Количество поездок, тыс.', fontsize=11)
    ax2.set_title('Box plot распределения', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Добавление выбросов
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['value'] < (Q1 - 1.5 * IQR)) |
                  (df['value'] > (Q3 + 1.5 * IQR))]

    if not outliers.empty:
        ax2.text(0.95, 0.95, f'Выбросы: {len(outliers)}',
                 transform=ax2.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # 3. Кумулятивная сумма
    cumulative = df['value'].cumsum()
    ax3.plot(df.index, cumulative,
             color=config.COLORS['secondary'],
             linewidth=2, marker='o')
    ax3.fill_between(df.index, 0, cumulative,
                     alpha=0.3, color=config.COLORS['secondary'])
    ax3.set_xlabel('Год', fontsize=11)
    ax3.set_ylabel('Кумулятивная сумма, тыс.', fontsize=11)
    ax3.set_title('Кумулятивная сумма поездок', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. Изменение по годам (годовой прирост)
    annual_change = df['value'].pct_change() * 100
    colors = ['green' if x >= 0 else 'red' for x in annual_change]
    ax4.bar(df.index[1:].year, annual_change[1:], color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Год', fontsize=11)
    ax4.set_ylabel('Изменение, %', fontsize=11)
    ax4.set_title('Годовое процентное изменение', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Сохранение
    plot_path = os.path.join(config.PLOTS_DIR, '02_distribution_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Графики распределения сохранены: {plot_path}")
    return fig


def save_visualization_description():
    """
    Сохранение описания визуализаций для отчёта
    """
    description = """ВИЗУАЛИЗАЦИЯ ДАННЫХ ВРЕМЕННОГО РЯДА

1. Линейный график временного ряда:
   - По оси X: годы (2000-2024)
   - По оси Y: количество туристических поездок (тыс.)
   - Выделены ключевые точки: пик 2019, падение 2020, восстановление 2022-2024

2. Столбчатая диаграмма (гистограмма):
   - Каждый столбец соответствует году
   - Высота столбца - количество поездок
   - Выделен 2020 год (аномалия - пандемия COVID-19)

3. Дополнительные графики распределения:
   - Гистограмма с KDE: оценка плотности распределения значений
   - Box plot: визуализация медианы, квартилей и выбросов
   - Кумулятивная сумма: общий объём поездок за период
   - Годовые изменения: процентный прирост/падение по годам

Все графики сохранены в формате PNG с высоким разрешением (300 DPI)."""

    desc_path = os.path.join(config.TABLES_DIR, '02_visualization_description.txt')
    with open(desc_path, 'w', encoding='utf-8') as f:
        f.write(description)

    print(f"✅ Описание визуализаций сохранено: {desc_path}")


def main():
    """Основная функция модуля"""
    print("\n" + "=" * 60)
    print("ШАГ 2: ВИЗУАЛИЗАЦИЯ ДАННЫХ")
    print("=" * 60)

    # Загрузка данных
    df = load_data()
    if df is None:
        return

    # Создание графиков
    create_time_series_plot(df)
    create_distribution_plots(df)
    save_visualization_description()

    print("\n✅ Визуализация данных завершена!")
    print("   Все графики сохранены в папке outputs/plots/")


if __name__ == "__main__":
    main()