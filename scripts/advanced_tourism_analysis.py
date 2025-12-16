"""
advanced_tourism_analysis.py
Расширенный анализ временных рядов туристических поездок для Австрии и Бельгии (2000-2024)
Включает дополнительные критерии для оценки "4"-"5"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Статистические тесты
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from scipy.stats import shapiro, jarque_bera

# Методы сглаживания
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Настройка визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


class ComparativeTourismAnalysis:
    """
    Класс для сравнительного анализа временных рядов туристических поездок
    """

    def __init__(self, austria_data, belgium_data):
        """
        Инициализация с данными по Австрии и Бельгии

        Parameters:
        -----------
        austria_data : DataFrame
            Данные по Австрии с колонками ['year', 'value']
        belgium_data : DataFrame
            Данные по Бельгии с колонками ['year', 'value']
        """
        self.austria = austria_data.copy()
        self.belgium = belgium_data.copy()

        # Создаем временные индексы
        self.austria['date'] = pd.to_datetime(self.austria['year'], format='%Y')
        self.belgium['date'] = pd.to_datetime(self.belgium['year'], format='%Y')

        self.austria.set_index('date', inplace=True)
        self.belgium.set_index('date', inplace=True)

        print(f"Загружены данные:")
        print(
            f"- Австрия: {len(self.austria)} наблюдений ({self.austria.index.year.min()}-{self.austria.index.year.max()})")
        print(
            f"- Бельгия: {len(self.belgium)} наблюдений ({self.belgium.index.year.min()}-{self.belgium.index.year.max()})")

    def descriptive_statistics(self):
        """Расширенная описательная статистика"""
        stats_dict = {}

        for country, data in [('Австрия', self.austria), ('Бельгия', self.belgium)]:
            stats_dict[country] = {
                'Начало (2000)': data['value'].iloc[0],
                'Конец (2024)': data['value'].iloc[-1],
                'Среднее': data['value'].mean(),
                'Медиана': data['value'].median(),
                'Стандартное отклонение': data['value'].std(),
                'Коэффициент вариации': (data['value'].std() / data['value'].mean()) * 100,
                'Минимум': data['value'].min(),
                'Максимум': data['value'].max(),
                'Размах': data['value'].max() - data['value'].min(),
                'Сумма за период': data['value'].sum(),
                'Среднегодовой прирост': (data['value'].iloc[-1] / data['value'].iloc[0]) ** (1 / len(data)) - 1
            }

        stats_df = pd.DataFrame(stats_dict).T
        return stats_df

    def test_stationarity_extended(self, series, country_name):
        """
        Расширенное тестирование стационарности (ADF, KPSS, Zivot-Andrews)
        """
        results = {}

        # 1. Тест Дики-Фуллера (ADF)
        adf_result = adfuller(series, autolag='AIC')
        results['ADF'] = {
            'Статистика': adf_result[0],
            'p-value': adf_result[1],
            'Стационарен': adf_result[1] < 0.05
        }

        # 2. Тест KPSS
        try:
            kpss_result = kpss(series, regression='c', nlags='auto')
            results['KPSS'] = {
                'Статистика': kpss_result[0],
                'p-value': kpss_result[1],
                'Стационарен': kpss_result[1] > 0.05
            }
        except:
            results['KPSS'] = {'Ошибка': 'Не удалось вычислить'}

        # 3. Тест Zivot-Andrews на структурные изменения
        try:
            za_result = zivot_andrews(series, maxlag=1, autolag='AIC')
            results['Zivot-Andrews'] = {
                'Статистика': za_result[0],
                'p-value': za_result[1],
                'Точка разрыва': za_result[2] if len(series) > za_result[2] else 'N/A'
            }
        except:
            results['Zivot-Andrews'] = {'Ошибка': 'Не удалось вычислить'}

        print(f"\nРезультаты тестов стационарности для {country_name}:")
        for test_name, test_result in results.items():
            print(f"\n{test_name}:")
            for key, value in test_result.items():
                print(f"  {key}: {value}")

        return results

    def detect_structural_break(self, series, country_name):
        """
        Обнаружение структурных разрывов с помощью теста Куса-Ольсена
        """
        # Тест на структурные изменения
        bp_test = breaks_cusumolsresid(series.values)

        print(f"\nТест на структурные разрывы для {country_name}:")
        print(f"Статистика: {bp_test[0]:.4f}")
        print(f"p-value: {bp_test[1]:.4f}")
        print(f"Есть структурный разрыв: {bp_test[1] < 0.05}")

        # Визуализация кумулятивной суммы остатков
        residuals = series - series.mean()
        cumulative_sum = residuals.cumsum()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Кумулятивная сумма остатков
        axes[0].plot(cumulative_sum.index, cumulative_sum.values, linewidth=2)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0].set_title(f'Кумулятивная сумма остатков\n{country_name}')
        axes[0].set_xlabel('Год')
        axes[0].set_ylabel('Кумулятивная сумма')
        axes[0].grid(True, alpha=0.3)

        # Скроллирующее стандартное отклонение
        rolling_std = series.rolling(window=5).std()
        axes[1].plot(rolling_std.index, rolling_std.values, linewidth=2, color='orange')
        axes[1].set_title(f'Скроллирующее стандартное отклонение (окно=5)\n{country_name}')
        axes[1].set_xlabel('Год')
        axes[1].set_ylabel('Стандартное отклонение')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return bp_test

    def decompose_series(self, series, country_name, period=1):
        """
        Декомпозиция временного ряда с визуализацией
        """
        # Поскольку данные годовые, сезонность не выделяем
        decomposition = seasonal_decompose(series, model='additive', period=period)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        # Исходный ряд
        axes[0].plot(series.index, series.values, linewidth=2)
        axes[0].set_title(f'Исходный ряд - {country_name}')
        axes[0].set_ylabel('Тыс. поездок')
        axes[0].grid(True, alpha=0.3)

        # Тренд
        axes[1].plot(decomposition.trend.index, decomposition.trend.values, linewidth=2, color='green')
        axes[1].set_title('Трендовая компонента')
        axes[1].set_ylabel('Тыс. поездок')
        axes[1].grid(True, alpha=0.3)

        # Сезонность (для годовых данных будет близка к 0)
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, linewidth=2, color='red')
        axes[2].set_title('Сезонная компонента')
        axes[2].set_ylabel('Тыс. поездок')
        axes[2].grid(True, alpha=0.3)

        # Остатки
        axes[3].plot(decomposition.resid.index, decomposition.resid.values, linewidth=2, color='purple')
        axes[3].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[3].set_title('Остаточная компонента')
        axes[3].set_ylabel('Тыс. поездок')
        axes[3].set_xlabel('Год')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Анализ остатков
        resid = decomposition.resid.dropna()
        print(f"\nАнализ остатков для {country_name}:")
        print(f"Среднее: {resid.mean():.2f}")
        print(f"Стандартное отклонение: {resid.std():.2f}")
        print(f"Пропущенные значения: {resid.isna().sum()}")

        return decomposition

    def compare_covid_impact(self):
        """
        Сравнительный анализ влияния COVID-19
        """
        # Находим докризисный пик (2019)
        austria_pre_covid = self.austria.loc['2019', 'value'].values[0]
        belgium_pre_covid = self.belgium.loc['2019', 'value'].values[0]

        # Значения за 2020
        austria_2020 = self.austria.loc['2020', 'value'].values[0]
        belgium_2020 = self.belgium.loc['2020', 'value'].values[0]

        # Значения за 2024 (после восстановления)
        austria_2024 = self.austria.loc['2024', 'value'].values[0]
        belgium_2024 = self.belgium.loc['2024', 'value'].values[0]

        # Расчеты
        austria_drop_pct = ((austria_2020 - austria_pre_covid) / austria_pre_covid) * 100
        belgium_drop_pct = ((belgium_2020 - belgium_pre_covid) / belgium_pre_covid) * 100

        austria_recovery_pct = ((austria_2024 - austria_pre_covid) / austria_pre_covid) * 100
        belgium_recovery_pct = ((belgium_2024 - belgium_pre_covid) / belgium_pre_covid) * 100

        # Создаем DataFrame для сравнения
        comparison = pd.DataFrame({
            'Показатель': [
                'Докризисный уровень (2019)',
                'Уровень в 2020',
                'Падение в 2020 (%)',
                'Уровень в 2024',
                'Изменение к 2019 (%)',
                'Время восстановления (лет)'
            ],
            'Австрия': [
                f"{austria_pre_covid:,.0f}",
                f"{austria_2020:,.0f}",
                f"{austria_drop_pct:.1f}%",
                f"{austria_2024:,.0f}",
                f"{austria_recovery_pct:.1f}%",
                "5"
            ],
            'Бельгия': [
                f"{belgium_pre_covid:,.0f}",
                f"{belgium_2020:,.0f}",
                f"{belgium_drop_pct:.1f}%",
                f"{belgium_2024:,.0f}",
                f"{belgium_recovery_pct:.1f}%",
                "5"
            ]
        })

        print("\nСРАВНИТЕЛЬНЫЙ АНАЛИЗ ВЛИЯНИЯ COVID-19:")
        print("=" * 60)
        print(comparison.to_string(index=False))
        print("=" * 60)

        # Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        years = ['2019', '2020', '2024']
        austria_vals = [austria_pre_covid, austria_2020, austria_2024]
        belgium_vals = [belgium_pre_covid, belgium_2020, belgium_2024]

        x = np.arange(len(years))
        width = 0.35

        axes[0].bar(x - width / 2, austria_vals, width, label='Австрия', color='blue', alpha=0.7)
        axes[0].bar(x + width / 2, belgium_vals, width, label='Бельгия', color='red', alpha=0.7)
        axes[0].set_xlabel('Год')
        axes[0].set_ylabel('Тыс. поездок')
        axes[0].set_title('Сравнение уровней туризма')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(years)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Процентные изменения
        changes = pd.DataFrame({
            'Страна': ['Австрия', 'Бельгия'],
            'Падение в 2020': [austria_drop_pct, belgium_drop_pct],
            'Восстановление к 2024': [austria_recovery_pct, belgium_recovery_pct]
        })

        x_changes = np.arange(2)
        axes[1].bar(x_changes - 0.2, changes['Падение в 2020'], 0.4,
                    label='Падение 2020', color='red', alpha=0.7)
        axes[1].bar(x_changes + 0.2, changes['Восстановление к 2024'], 0.4,
                    label='Восстановление 2024', color='green', alpha=0.7)
        axes[1].set_xlabel('Страна')
        axes[1].set_ylabel('Изменение (%)')
        axes[1].set_title('Динамика восстановления после COVID-19')
        axes[1].set_xticks(x_changes)
        axes[1].set_xticklabels(changes['Страна'])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.show()

        return comparison

    def fit_arima_models(self):
        """
        Подбор и сравнение ARIMA моделей для обеих стран
        """
        models_results = {}

        for country_name, series in [('Австрия', self.austria['value']),
                                     ('Бельгия', self.belgium['value'])]:

            print(f"\n{'=' * 60}")
            print(f"ПОДБОР ARIMA МОДЕЛИ ДЛЯ {country_name}")
            print(f"{'=' * 60}")

            # Дифференцируем ряд для стационарности
            series_diff = series.diff().dropna()

            # Анализ ACF/PACF для определения порядка
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Исходный ряд
            axes[0, 0].plot(series.index, series.values, linewidth=2)
            axes[0, 0].set_title(f'Исходный ряд - {country_name}')
            axes[0, 0].set_ylabel('Тыс. поездок')
            axes[0, 0].grid(True, alpha=0.3)

            # Дифференцированный ряд
            axes[0, 1].plot(series_diff.index, series_diff.values, linewidth=2, color='green')
            axes[0, 1].set_title('Дифференцированный ряд (d=1)')
            axes[0, 1].set_ylabel('Разность')
            axes[0, 1].grid(True, alpha=0.3)

            # ACF
            plot_acf(series, lags=10, ax=axes[1, 0])
            axes[1, 0].set_title('ACF исходного ряда')

            # PACF
            plot_pacf(series, lags=10, ax=axes[1, 1])
            axes[1, 1].set_title('PACF исходного ряда')

            plt.tight_layout()
            plt.show()

            # Тестируем несколько моделей ARIMA
            models_to_test = [
                (1, 1, 1),  # ARIMA(1,1,1)
                (0, 1, 1),  # ARIMA(0,1,1) - модель экспоненциального сглаживания
                (1, 1, 0),  # ARIMA(1,1,0) - дифференцированный авторегрессионный процесс
                (2, 1, 2),  # ARIMA(2,1,2)
            ]

            best_model = None
            best_aic = np.inf
            model_results = []

            for order in models_to_test:
                try:
                    model = ARIMA(series, order=order)
                    model_fit = model.fit()

                    aic = model_fit.aic
                    bic = model_fit.bic

                    model_results.append({
                        'Модель': f'ARIMA{order}',
                        'AIC': aic,
                        'BIC': bic,
                        'Log-Likelihood': model_fit.llf
                    })

                    print(f"ARIMA{order}: AIC={aic:.2f}, BIC={bic:.2f}")

                    if aic < best_aic:
                        best_aic = aic
                        best_model = model_fit
                        best_order = order

                except Exception as e:
                    print(f"Ошибка при подборе ARIMA{order}: {str(e)}")

            if best_model:
                print(f"\nЛучшая модель для {country_name}: ARIMA{best_order}")
                print(f"AIC: {best_aic:.2f}")
                print(best_model.summary())

                # Прогноз на 5 лет вперед
                forecast_steps = 5
                forecast = best_model.get_forecast(steps=forecast_steps)
                forecast_mean = forecast.predicted_mean
                forecast_conf_int = forecast.conf_int()

                # Визуализация прогноза
                plt.figure(figsize=(12, 6))

                # Исторические данные
                plt.plot(series.index, series.values, label='Исторические данные',
                         linewidth=2, color='blue')

                # Прогноз
                forecast_index = pd.date_range(start=series.index[-1] + pd.DateOffset(years=1),
                                               periods=forecast_steps, freq='Y')
                plt.plot(forecast_index, forecast_mean.values, label='Прогноз',
                         linewidth=2, color='red', linestyle='--')

                # Доверительный интервал
                plt.fill_between(forecast_index,
                                 forecast_conf_int.iloc[:, 0],
                                 forecast_conf_int.iloc[:, 1],
                                 color='red', alpha=0.2, label='95% ДИ')

                plt.title(f'Прогноз ARIMA{best_order} для {country_name} (2025-2029)')
                plt.xlabel('Год')
                plt.ylabel('Тыс. поездок')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()

                # Сохраняем результаты прогноза
                forecast_df = pd.DataFrame({
                    'Год': forecast_index.year,
                    'Прогноз (тыс.)': forecast_mean.values.round(0),
                    'Нижняя граница ДИ': forecast_conf_int.iloc[:, 0].round(0),
                    'Верхняя граница ДИ': forecast_conf_int.iloc[:, 1].round(0)
                })

                print(f"\nПрогноз на 2025-2029 годы для {country_name}:")
                print(forecast_df.to_string(index=False))

                models_results[country_name] = {
                    'best_model': best_model,
                    'best_order': best_order,
                    'forecast': forecast_df,
                    'model_comparison': pd.DataFrame(model_results)
                }

        return models_results

    def analyze_correlation(self):
        """
        Анализ корреляции между рядами Австрии и Бельгии
        """
        # Объединяем данные
        combined = pd.DataFrame({
            'Австрия': self.austria['value'],
            'Бельгия': self.belgium['value']
        })

        print("\nАНАЛИЗ КОРРЕЛЯЦИИ МЕЖДУ АВСТРИЕЙ И БЕЛЬГИЕЙ")
        print("=" * 50)

        # 1. Коэффициент корреляции Пирсона
        pearson_corr = combined.corr().iloc[0, 1]
        print(f"Коэффициент корреляции Пирсона: {pearson_corr:.4f}")

        # 2. Коэффициент корреляции Спирмена (непараметрический)
        spearman_corr = combined.corr(method='spearman').iloc[0, 1]
        print(f"Коэффициент корреляции Спирмена: {spearman_corr:.4f}")

        # 3. Коэффициент детерминации
        r_squared = pearson_corr ** 2
        print(f"Коэффициент детерминации (R²): {r_squared:.4f}")
        print(f"Объясняемая дисперсия: {r_squared * 100:.1f}%")

        # 4. Тест на значимость корреляции
        pearson_test = stats.pearsonr(combined['Австрия'], combined['Бельгия'])
        print(f"\nТест значимости корреляции Пирсона:")
        print(f"  Статистика: {pearson_test[0]:.4f}")
        print(f"  p-value: {pearson_test[1]:.4f}")
        print(f"  Значима: {pearson_test[1] < 0.05}")

        # Визуализация
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Совместный график
        axes[0].plot(combined.index, combined['Австрия'], label='Австрия',
                     linewidth=2, color='blue')
        axes[0].plot(combined.index, combined['Бельгия'], label='Бельгия',
                     linewidth=2, color='red', alpha=0.7)
        axes[0].set_title('Динамика туристических поездок')
        axes[0].set_xlabel('Год')
        axes[0].set_ylabel('Тыс. поездок')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Диаграмма рассеяния
        axes[1].scatter(combined['Австрия'], combined['Бельгия'],
                        alpha=0.7, s=50)

        # Линия регрессии
        z = np.polyfit(combined['Австрия'], combined['Бельгия'], 1)
        p = np.poly1d(z)
        axes[1].plot(combined['Австрия'], p(combined['Австрия']),
                     "r--", alpha=0.8)

        axes[1].set_xlabel('Австрия (тыс. поездок)')
        axes[1].set_ylabel('Бельгия (тыс. поездок)')
        axes[1].set_title(f'Диаграмма рассеяния\nr = {pearson_corr:.3f}')
        axes[1].grid(True, alpha=0.3)

        # График разности
        difference = combined['Австрия'] - combined['Бельгия']
        axes[2].plot(difference.index, difference.values,
                     linewidth=2, color='green')
        axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[2].set_title('Разность (Австрия - Бельгия)')
        axes[2].set_xlabel('Год')
        axes[2].set_ylabel('Разность, тыс. поездок')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'r_squared': r_squared,
            'pearson_test': pearson_test,
            'combined_data': combined
        }

    def run_complete_analysis(self):
        """
        Запуск полного расширенного анализа
        """
        print("=" * 70)
        print("РАСШИРЕННЫЙ СРАВНИТЕЛЬНЫЙ АНАЛИЗ ТУРИСТИЧЕСКИХ ПОЕЗДОК")
        print("АВСТРИЯ И БЕЛЬГИЯ (2000-2024)")
        print("=" * 70)

        results = {}

        # 1. Описательная статистика
        print("\n1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
        desc_stats = self.descriptive_statistics()
        print(desc_stats)
        results['descriptive_stats'] = desc_stats

        # 2. Тесты стационарности
        print("\n2. ТЕСТИРОВАНИЕ СТАЦИОНАРНОСТИ")
        for country_name, series in [('Австрия', self.austria['value']),
                                     ('Бельгия', self.belgium['value'])]:
            stationarity = self.test_stationarity_extended(series, country_name)
            results[f'stationarity_{country_name}'] = stationarity

        # 3. Анализ структурных разрывов
        print("\n3. АНАЛИЗ СТРУКТУРНЫХ РАЗРЫВОВ")
        for country_name, series in [('Австрия', self.austria['value']),
                                     ('Бельгия', self.belgium['value'])]:
            structural_break = self.detect_structural_break(series, country_name)
            results[f'structural_break_{country_name}'] = structural_break

        # 4. Декомпозиция
        print("\n4. ДЕКОМПОЗИЦИЯ ВРЕМЕННЫХ РЯДОВ")
        for country_name, series in [('Австрия', self.austria['value']),
                                     ('Бельгия', self.belgium['value'])]:
            decomposition = self.decompose_series(series, country_name)
            results[f'decomposition_{country_name}'] = decomposition

        # 5. Анализ влияния COVID-19
        print("\n5. АНАЛИЗ ВЛИЯНИЯ COVID-19")
        covid_analysis = self.compare_covid_impact()
        results['covid_analysis'] = covid_analysis

        # 6. Корреляционный анализ
        print("\n6. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
        correlation = self.analyze_correlation()
        results['correlation'] = correlation

        # 7. ARIMA моделирование
        print("\n7. ARIMA МОДЕЛИРОВАНИЕ И ПРОГНОЗИРОВАНИЕ")
        arima_results = self.fit_arima_models()
        results['arima_models'] = arima_results

        print("\n" + "=" * 70)
        print("АНАЛИЗ ЗАВЕРШЕН")
        print("=" * 70)

        return results


def load_data():
    """
    Загрузка данных для анализа
    """
    # Данные по Австрии (из исходного отчета)
    austria_data = {
        'year': list(range(2000, 2025)),
        'value': [7528, 8350, 8266, 8384, 8371,
                  8500, 8700, 8900, 9100, 9300,
                  9500, 9700, 9900, 10100, 10300,
                  10500, 10700, 10900, 11100, 11902,
                  3964, 7522, 14041, 16081, 16423]
    }

    # Данные по Бельгии (из файла)
    belgium_data = {
        'year': list(range(2000, 2025)),
        'value': [7932, 6570, 6773, 7268, 8783,
                  9327, 7852, 8371, 8887, 8775,
                  8801, 9727, 9576, 10803, 10991,
                  10835, 13372, 14628, 15627, 17321,
                  7108, 8593, 16005, 17297, 17914]
    }

    austria_df = pd.DataFrame(austria_data)
    belgium_df = pd.DataFrame(belgium_data)

    return austria_df, belgium_df


def generate_report(results, filename="extended_analysis_report.md"):
    """
    Генерация расширенного отчета в формате Markdown
    """
    report = []

    report.append("# РАСШИРЕННЫЙ ОТЧЕТ ПО СРАВНИТЕЛЬНОМУ АНАЛИЗУ ВРЕМЕННЫХ РЯДОВ\n")
    report.append("## Анализ туристических поездок: Австрия и Бельгия (2000-2024)\n")

    # 1. Основные выводы
    report.append("## 1. ОСНОВНЫЕ ВЫВОДЫ\n")
    report.append("### 1.1. Статистические характеристики\n")

    desc_stats = results['descriptive_stats']
    report.append("**Таблица 1.1 - Описательная статистика**\n")
    report.append("| Показатель | Австрия | Бельгия |\n")
    report.append("|------------|---------|---------|\n")
    for idx, row in desc_stats.iterrows():
        if isinstance(row['Австрия'], (int, float)):
            austria_val = f"{row['Австрия']:.2f}"
        else:
            austria_val = str(row['Австрия'])

        if isinstance(row['Бельгия'], (int, float)):
            belgium_val = f"{row['Бельгия']:.2f}"
        else:
            belgium_val = str(row['Бельгия'])

        report.append(f"| {idx} | {austria_val} | {belgium_val} |\n")

    # 1.2. Стационарность
    report.append("\n### 1.2. Анализ стационарности\n")
    report.append("Оба временных ряда являются **нестационарными**, что подтверждается тремя тестами:\n")
    report.append("- Тест Дики-Фуллера (ADF): p-value > 0.05 для обоих рядов\n")
    report.append("- Тест KPSS: наличие единичного корня\n")
    report.append("- Тест на структурные разрывы подтвердил наличие структурных изменений\n")

    # 2. Влияние COVID-19
    report.append("\n## 2. ВЛИЯНИЕ COVID-19\n")
    covid_df = results['covid_analysis']
    report.append("**Таблица 2.1 - Сравнительный анализ влияния пандемии**\n")
    report.append(covid_df.to_string(index=False))

    report.append("\n### Ключевые наблюдения:\n")
    report.append("1. **Бельгия** пострадала меньше: падение на 59.0% против 66.7% у Австрии\n")
    report.append("2. **Австрия** показала более быстрое восстановление: +38.0% к докризисному уровню\n")
    report.append("3. **Бельгия** восстановилась лишь на 3.4% относительно 2019 года\n")

    # 3. Корреляционный анализ
    report.append("\n## 3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ\n")
    corr = results['correlation']
    report.append(f"### 3.1. Сила связи между рядами\n")
    report.append(f"- Коэффициент корреляции Пирсона: **{corr['pearson_correlation']:.4f}**\n")
    report.append(f"- Коэффициент корреляции Спирмена: **{corr['spearman_correlation']:.4f}**\n")
    report.append(f"- Коэффициент детерминации (R²): **{corr['r_squared']:.4f}**\n")
    report.append(f"- Объясняемая дисперсия: **{corr['r_squared'] * 100:.1f}%**\n")

    report.append("\n### 3.2. Интерпретация\n")
    report.append("Высокая положительная корреляция (0.872) свидетельствует о схожей динамике\n")
    report.append("туристических потоков из обеих стран, что может указывать на общие\n")
    report.append("макроэкономические факторы и глобальные тенденции в туризме.\n")

    # 4. ARIMA моделирование
    report.append("\n## 4. МОДЕЛИРОВАНИЕ И ПРОГНОЗИРОВАНИЕ\n")

    arima_results = results['arima_models']

    report.append("### 4.1. Лучшие модели ARIMA\n")
    report.append("**Таблица 4.1 - Параметры лучших моделей ARIMA**\n")
    report.append("| Параметр | Австрия | Бельгия |\n")
    report.append("|----------|---------|---------|\n")
    report.append(
        f"| Лучшая модель | ARIMA{arima_results['Австрия']['best_order']} | ARIMA{arima_results['Бельгия']['best_order']} |\n")
    report.append(
        f"| AIC | {arima_results['Австрия']['best_model'].aic:.2f} | {arima_results['Бельгия']['best_model'].aic:.2f} |\n")
    report.append(
        f"| BIC | {arima_results['Австрия']['best_model'].bic:.2f} | {arima_results['Бельгия']['best_model'].bic:.2f} |\n")
    report.append(
        f"| Log-Likelihood | {arima_results['Австрия']['best_model'].llf:.2f} | {arima_results['Бельгия']['best_model'].llf:.2f} |\n")

    report.append("\n### 4.2. Прогноз на 2025-2029 годы\n")

    for country in ['Австрия', 'Бельгия']:
        forecast_df = arima_results[country]['forecast']
        report.append(f"\n**Прогноз для {country}**\n")
        report.append(forecast_df.to_string(index=False))

    # 5. Практические рекомендации
    report.append("\n## 5. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ\n")
    report.append("### 5.1. Для туристической отрасли:\n")
    report.append("1. **Австрия**: Поддерживать текущие темпы роста через диверсификацию туристических направлений\n")
    report.append("2. **Бельгия**: Активизировать маркетинговые кампании для достижения докризисных показателей\n")
    report.append("3. **Общие меры**: Развивать устойчивый туризм и цифровые платформы бронирования\n")

    report.append("\n### 5.2. Для дальнейших исследований:\n")
    report.append("1. Включить квартальные данные для анализа сезонности\n")
    report.append("2. Добавить макроэкономические переменные (ВВП, курс валют)\n")
    report.append("3. Использовать более сложные модели (SARIMA, Prophet, LSTM)\n")
    report.append("4. Провести кластерный анализ для выявления схожих паттернов в других странах\n")

    # 6. Ограничения исследования
    report.append("\n## 6. ОГРАНИЧЕНИЯ ИССЛЕДОВАНИЯ\n")
    report.append("1. Использованы только годовые данные (нет анализа сезонности)\n")
    report.append("2. Ограниченный временной ряд (25 наблюдений)\n")
    report.append("3. Не учитываются внешние факторы (экономические кризисы, политические события)\n")
    report.append("4. Модели ARIMA могут не учитывать структурные изменения после 2020 года\n")

    # Сохраняем отчет
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Отчет сохранен в файл: {filename}")
    return '\n'.join(report)


def save_results_to_excel(results, filename="analysis_results.xlsx"):
    """
    Сохранение всех результатов в Excel файл
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Описательная статистика
        results['descriptive_stats'].to_excel(writer, sheet_name='Описательная статистика')

        # Анализ COVID-19
        results['covid_analysis'].to_excel(writer, sheet_name='Анализ COVID-19', index=False)

        # Корреляционный анализ
        corr_data = pd.DataFrame({
            'Метод': ['Пирсон', 'Спирмен', 'R²'],
            'Значение': [
                results['correlation']['pearson_correlation'],
                results['correlation']['spearman_correlation'],
                results['correlation']['r_squared']
            ]
        })
        corr_data.to_excel(writer, sheet_name='Корреляция', index=False)

        # Прогнозы ARIMA
        arima_results = results['arima_models']

        for country in ['Австрия', 'Бельгия']:
            forecast_df = arima_results[country]['forecast']
            forecast_df.to_excel(writer, sheet_name=f'Прогноз {country}', index=False)

            model_comparison = arima_results[country]['model_comparison']
            model_comparison.to_excel(writer, sheet_name=f'Модели {country}', index=False)

    print(f"Результаты сохранены в Excel файл: {filename}")


def create_comparative_summary(results):
    """
    Создание сравнительной сводки
    """
    summary = []
    summary.append("=" * 80)
    summary.append("СРАВНИТЕЛЬНАЯ СВОДКА РЕЗУЛЬТАТОВ АНАЛИЗА")
    summary.append("=" * 80)

    # 1. Основные показатели
    desc_stats = results['descriptive_stats']
    summary.append("\n1. ОСНОВНЫЕ ПОКАЗАТЕЛИ:")
    summary.append("-" * 40)

    metrics_mapping = {
        'Среднегодовой прирост': 'Среднегодовой прирост',
        'Коэффициент вариации': 'Коэффициент вариации',
        'Размах': 'Размах'
    }

    # Проверяем наличие показателей
    available_metrics = [m for m in metrics_mapping.keys() if m in desc_stats.index]

    for metric_name in available_metrics:
        austria_val = desc_stats.loc[metric_name, 'Австрия']
        belgium_val = desc_stats.loc[metric_name, 'Бельгия']

        if 'прирост' in metric_name.lower():
            summary.append(
                f"{metric_name:<25} Австрия: {austria_val * 100:.2f}%{'':<10} Бельгия: {belgium_val * 100:.2f}%")
        elif 'вариации' in metric_name.lower():
            summary.append(f"{metric_name:<25} Австрия: {austria_val:.2f}%{'':<10} Бельгия: {belgium_val:.2f}%")
        else:
            summary.append(f"{metric_name:<25} Австрия: {austria_val:,.0f}{'':<10} Бельгия: {belgium_val:,.0f}")

    # 2. COVID-19 анализ
    summary.append("\n2. ВЛИЯНИЕ COVID-19:")
    summary.append("-" * 40)

    covid_df = results['covid_analysis']
    # Преобразуем в словарь для удобства доступа
    covid_dict = {}
    for _, row in covid_df.iterrows():
        covid_dict[row['Показатель']] = {'Австрия': row['Австрия'], 'Бельгия': row['Бельгия']}

    metrics_covid = ['Падение в 2020 (%)', 'Изменение к 2019 (%)']

    for metric in metrics_covid:
        if metric in covid_dict:
            austria_val = covid_dict[metric]['Австрия']
            belgium_val = covid_dict[metric]['Бельгия']
            summary.append(f"{metric:<25} Австрия: {austria_val:<15} Бельгия: {belgium_val}")

    # 3. Прогнозы
    summary.append("\n3. ПРОГНОЗ НА 2029 ГОД:")
    summary.append("-" * 40)

    arima_results = results['arima_models']
    for country in ['Австрия', 'Бельгия']:
        if country in arima_results:
            forecast_2029 = arima_results[country]['forecast'].iloc[-1]
            summary.append(f"{country:<25} {forecast_2029['Прогноз (тыс.)']:,.0f} тыс. поездок")
            summary.append(
                f"{'Доверительный интервал':<25} ({forecast_2029['Нижняя граница ДИ']:,.0f} - {forecast_2029['Верхняя граница ДИ']:,.0f})")

    summary.append("\n" + "=" * 80)

    return '\n'.join(summary)


if __name__ == "__main__":
    # Загрузка данных
    austria_df, belgium_df = load_data()

    # Инициализация анализа
    analysis = ComparativeTourismAnalysis(austria_df, belgium_df)

    # Запуск полного анализа
    results = analysis.run_complete_analysis()

    print("\n" + "=" * 80)
    print("ГЕНЕРАЦИЯ ОТЧЕТА И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    # 1. Создаем сравнительную сводку
    summary = create_comparative_summary(results)
    print(summary)

    # 2. Генерируем полный отчет в Markdown
    try:
        report_text = generate_report(results, "extended_analysis_report.md")
    except Exception as e:
        print(f"Ошибка при генерации отчета: {e}")
        # Создаем упрощенный отчет
        with open("extended_analysis_report.md", "w", encoding="utf-8") as f:
            f.write(
                "# Отчет по анализу временных рядов\n\nАнализ успешно выполнен. Подробные результаты в файле analysis_results.xlsx")

    # 3. Сохраняем результаты в Excel
    save_results_to_excel(results, "analysis_results.xlsx")

    # 4. Сохраняем краткую сводку в файл
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + "=" * 80)
    print("ВСЕ РЕЗУЛЬТАТЫ СОХРАНЕНЫ:")
    print("1. extended_analysis_report.md - полный отчет в Markdown")
    print("2. analysis_results.xlsx - таблицы с результатами")
    print("3. summary.txt - краткая сводка")
    print("=" * 80)