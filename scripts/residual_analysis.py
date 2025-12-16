# 08_residual_analysis.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot
import warnings

warnings.filterwarnings('ignore')
import os
import config
from data_loading import load_data
from trend_modeling import main as model_trend


def calculate_residuals(df, trend_model):
    """
    Расчет остатков модели тренда
    """
    print("\n" + "=" * 60)
    print("АНАЛИЗ ОСТАТОЧНОЙ КОМПОНЕНТЫ")
    print("=" * 60)

    # Получаем прогноз модели
    if trend_model['type'] == 'linear':
        model = trend_model['model']
        X = np.arange(len(df)).reshape(-1, 1)
        predictions = model.predict(X)

    elif 'polynomial' in trend_model['type']:
        model, poly = trend_model['model']
        X = np.arange(len(df)).reshape(-1, 1)
        X_poly = poly.transform(X)
        predictions = model.predict(X_poly)

    elif trend_model['type'] == 'exponential':
        func, params = trend_model['model']
        t = np.arange(len(df))
        predictions = func(t, *params)

    elif trend_model['type'] == 'logarithmic':
        func, params = trend_model['model']
        t = np.arange(len(df))
        predictions = func(t, *params)

    # Рассчитываем остатки
    residuals = df['value'].values - predictions

    print(f"Остатки модели '{trend_model['type']}':")
    print(f"  Количество: {len(residuals)}")
    print(f"  Среднее: {residuals.mean():.2f}")
    print(f"  Стандартное отклонение: {residuals.std():.2f}")
    print(f"  Минимум: {residuals.min():.2f}")
    print(f"  Максимум: {residuals.max():.2f}")

    return pd.Series(residuals, index=df.index, name='residuals'), predictions


def test_residual_normality(residuals):
    """
    Проверка остатков на нормальность распределения
    """
    print("\n" + "=" * 60)
    print("ПРОВЕРКА НОРМАЛЬНОСТИ РАСПРЕДЕЛЕНИЯ ОСТАТКОВ")
    print("=" * 60)

    results = {}

    # 1. Тест Шапиро-Уилка
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    results['shapiro'] = {
        'statistic': shapiro_stat,
        'p_value': shapiro_p,
        'is_normal': shapiro_p > config.SIGNIFICANCE_LEVEL
    }

    print(f"Тест Шапиро-Уилка:")
    print(f"  Статистика W: {shapiro_stat:.4f}")
    print(f"  p-value: {shapiro_p:.4f}")
    print(f"  Нормальность: {'ДА' if shapiro_p > 0.05 else 'НЕТ'}")

    # 2. Тест Д'Агостино K²
    dagostino_stat, dagostino_p = stats.normaltest(residuals)
    results['dagostino'] = {
        'statistic': dagostino_stat,
        'p_value': dagostino_p,
        'is_normal': dagostino_p > config.SIGNIFICANCE_LEVEL
    }

    print(f"\nТест Д'Агостино K²:")
    print(f"  Статистика: {dagostino_stat:.4f}")
    print(f"  p-value: {dagostino_p:.4f}")
    print(f"  Нормальность: {'ДА' if dagostino_p > 0.05 else 'НЕТ'}")

    # 3. Асимметрия и эксцесс
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    results['descriptive'] = {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'mean': residuals.mean(),
        'std': residuals.std()
    }

    print(f"\nОписательные статистики:")
    print(f"  Асимметрия: {skewness:.4f} (0 для нормального)")
    print(f"  Эксцесс: {kurtosis:.4f} (0 для нормального)")
    print(f"  Среднее: {residuals.mean():.4f} (должно быть около 0)")
    print(f"  Стандартное отклонение: {residuals.std():.4f}")

    # Интерпретация
    print(f"\nИНТЕРПРЕТАЦИЯ:")
    if shapiro_p > 0.05 and dagostino_p > 0.05:
        print("  ✅ Остатки распределены нормально")
        print("  ✅ Предпосылка регрессионного анализа выполняется")
    else:
        print("  ❌ Остатки НЕ распределены нормально")
        print("  ❌ Нарушена предпосылка регрессионного анализа")
        print("  ⚠️  Могут потребоваться преобразования данных")

    return results


def test_residual_autocorrelation(residuals):
    """
    Проверка остатков на автокорреляцию
    """
    print("\n" + "=" * 60)
    print("ПРОВЕРКА АВТОКОРРЕЛЯЦИИ ОСТАТКОВ")
    print("=" * 60)

    results = {}

    # 1. Тест Дарбина-Уотсона
    dw_stat = durbin_watson(residuals)
    results['durbin_watson'] = {
        'statistic': dw_stat,
        'interpretation': ''
    }

    print(f"Тест Дарбина-Уотсона:")
    print(f"  Статистика DW: {dw_stat:.4f}")

    # Интерпретация DW
    if dw_stat < 1.5:
        interpretation = "положительная автокорреляция"
    elif dw_stat > 2.5:
        interpretation = "отрицательная автокорреляция"
    else:
        interpretation = "нет автокорреляции"

    results['durbin_watson']['interpretation'] = interpretation
    print(f"  Интерпретация: {interpretation}")

    # 2. Тест Люнга-Бокса
    lb_test = acorr_ljungbox(residuals, lags=[5, 10], return_df=True)
    results['ljung_box'] = lb_test

    print(f"\nТест Люнга-Бокса:")
    print(f"  Lag 5: статистика={lb_test.loc[5, 'lb_stat']:.4f}, "
          f"p-value={lb_test.loc[5, 'lb_pvalue']:.4f}")
    print(f"  Lag 10: статистика={lb_test.loc[10, 'lb_stat']:.4f}, "
          f"p-value={lb_test.loc[10, 'lb_pvalue']:.4f}")

    # Проверка значимости
    has_autocorr = any(lb_test['lb_pvalue'] < config.SIGNIFICANCE_LEVEL)

    print(f"\nИНТЕРПРЕТАЦИЯ:")
    if not has_autocorr:
        print("  ✅ Нет статистически значимой автокорреляции")
        print("  ✅ Предпосылка о независимости остатков выполняется")
    else:
        print("  ❌ Обнаружена статистически значимая автокорреляция")
        print("  ❌ Нарушена предпосылка о независимости остатков")
        print("  ⚠️  Может потребоваться ARIMA или другие модели")

    return results, has_autocorr


def test_heteroscedasticity(residuals, predictions):
    """
    Проверка остатков на гомоскедастичность
    """
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ГОМОСКЕДАСТИЧНОСТИ ОСТАТКОВ")
    print("=" * 60)

    # Тест Бройша-Пагана
    # Подготовка данных для теста
    X = np.column_stack([np.ones_like(predictions), predictions])  # Константа + прогнозы

    try:
        # Тест Бройша-Пагана
        bp_test = het_breuschpagan(residuals, X)

        results = {
            'lm_statistic': bp_test[0],
            'lm_p_value': bp_test[1],
            'f_statistic': bp_test[2],
            'f_p_value': bp_test[3]
        }

        print(f"Тест Бройша-Пагана:")
        print(f"  LM статистика: {bp_test[0]:.4f}")
        print(f"  LM p-value: {bp_test[1]:.4f}")
        print(f"  F статистика: {bp_test[2]:.4f}")
        print(f"  F p-value: {bp_test[3]:.4f}")

        # Проверка на гетероскедастичность
        is_homoscedastic = bp_test[1] > config.SIGNIFICANCE_LEVEL

        print(f"\nИНТЕРПРЕТАЦИЯ:")
        if is_homoscedastic:
            print("  ✅ Остатки гомоскедастичны (постоянная дисперсия)")
            print("  ✅ Предпосылка о постоянстве дисперсии выполняется")
        else:
            print("  ❌ Остатки гетероскедастичны (непостоянная дисперсия)")
            print("  ❌ Нарушена предпосылка о постоянстве дисперсии")
            print("  ⚠️  Может потребоваться взвешенная регрессия")

        return results, is_homoscedastic

    except Exception as e:
        print(f"❌ Ошибка при тесте гетероскедастичности: {e}")

        # Альтернативный метод: графический анализ
        print("\nГрафический анализ гетероскедастичности:")
        residuals_abs = np.abs(residuals)

        from scipy.stats import spearmanr
        corr, p_value = spearmanr(predictions, residuals_abs)

        print(f"  Корреляция Спирмена: {corr:.4f}")
        print(f"  p-value: {p_value:.4f}")

        is_homoscedastic = p_value > config.SIGNIFICANCE_LEVEL

        return {'spearman_corr': corr, 'spearman_p': p_value}, is_homoscedastic


def analyze_residual_patterns(residuals, predictions):
    """
    Анализ паттернов в остатках
    """
    print("\n" + "=" * 60)
    print("АНАЛИЗ ПАТТЕРНОВ В ОСТАТКАХ")
    print("=" * 60)

    patterns = {}

    # 1. Проверка на тренд в остатках
    t = np.arange(len(residuals))
    slope, intercept, r_value, p_value, std_err = stats.linregress(t, residuals)

    patterns['trend_in_residuals'] = {
        'slope': slope,
        'p_value': p_value,
        'has_trend': p_value < config.SIGNIFICANCE_LEVEL
    }

    print(f"Проверка тренда в остатках:")
    print(f"  Наклон: {slope:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Тренд: {'ДА' if p_value < 0.05 else 'НЕТ'}")

    # 2. Проверка на сезонность в остатках (для годовых данных - циклические паттерны)
    # Автокорреляционная функция
    nlags = min(10, len(residuals) // 2)
    acf_values = np.array([residuals.autocorr(lag=i) for i in range(1, nlags + 1)])

    # Проверка значимости автокорреляций
    significant_lags = np.where(np.abs(acf_values) > 1.96 / np.sqrt(len(residuals)))[0] + 1

    patterns['seasonality'] = {
        'acf_values': acf_values,
        'significant_lags': significant_lags,
        'has_seasonality': len(significant_lags) > 0
    }

    print(f"\nПроверка сезонности/цикличности:")
    print(f"  Значимые лаги: {list(significant_lags) if len(significant_lags) > 0 else 'нет'}")
    print(f"  Сезонность: {'ДА' if len(significant_lags) > 0 else 'НЕТ'}")

    # 3. Проверка на зависимость от времени
    # Корреляция остатков с временем
    time_corr, time_p = stats.pearsonr(t, np.abs(residuals))

    patterns['time_dependence'] = {
        'correlation': time_corr,
        'p_value': time_p,
        'is_dependent': time_p < config.SIGNIFICANCE_LEVEL
    }

    print(f"\nЗависимость от времени:")
    print(f"  Корреляция: {time_corr:.4f}")
    print(f"  p-value: {time_p:.4f}")
    print(f"  Зависимость: {'ДА' if time_p < 0.05 else 'НЕТ'}")

    # 4. Общий вывод о паттернах
    has_problematic_patterns = (
            patterns['trend_in_residuals']['has_trend'] or
            patterns['seasonality']['has_seasonality'] or
            patterns['time_dependence']['is_dependent']
    )

    print(f"\nОБЩИЙ ВЫВОД О ПАТТЕРНАХ:")
    if not has_problematic_patterns:
        print("  ✅ Остатки не демонстрируют проблемных паттернов")
        print("  ✅ Модель адекватно описывает данные")
    else:
        print("  ⚠️  Обнаружены проблемные паттерны в остатках")
        print("  ⚠️  Модель может быть неадекватной")

    return patterns, has_problematic_patterns


def plot_residual_analysis(residuals, predictions, test_results):
    """
    Построение графиков анализа остатков
    """
    plt.style.use(config.PLOT_STYLE)

    # Создаем фигуру с несколькими подграфиками
    fig = plt.figure(figsize=(18, 12))

    # 1. График остатков во времени
    ax1 = plt.subplot(2, 3, 1)

    ax1.plot(residuals.index, residuals.values,
             color=config.COLORS['residual'],
             linewidth=2,
             marker='o',
             markersize=4,
             label='Остатки')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Скользящее среднее остатков
    residuals_sma = residuals.rolling(window=3, center=True).mean()
    ax1.plot(residuals.index, residuals_sma,
             color='red',
             linewidth=1.5,
             label='Скользящее среднее')

    ax1.set_xlabel('Год', fontsize=11)
    ax1.set_ylabel('Остатки', fontsize=11)
    ax1.set_title('Остатки модели во времени', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. QQ-plot для проверки нормальности
    ax2 = plt.subplot(2, 3, 2)

    qqplot(residuals.values, line='45', ax=ax2, marker='o', markersize=4)

    ax2.set_xlabel('Теоретические квантили', fontsize=11)
    ax2.set_ylabel('Выборочные квантили', fontsize=11)
    ax2.set_title('QQ-plot для проверки нормальности', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. Гистограмма распределения остатков
    ax3 = plt.subplot(2, 3, 3)

    n, bins, patches = ax3.hist(residuals.values, bins=8,
                                color=config.COLORS['residual'],
                                alpha=0.7,
                                edgecolor='black',
                                density=True)

    # Нормальное распределение с теми же параметрами
    from scipy.stats import norm
    mu, std = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    p = norm.pdf(x, mu, std)
    ax3.plot(x, p, 'k', linewidth=2, label='Нормальное распределение')

    ax3.set_xlabel('Остатки', fontsize=11)
    ax3.set_ylabel('Плотность', fontsize=11)
    ax3.set_title('Распределение остатков', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Остатки vs Прогнозы (проверка гетероскедастичности)
    ax4 = plt.subplot(2, 3, 4)

    ax4.scatter(predictions, residuals.values,
                color=config.COLORS['residual'],
                alpha=0.7,
                s=50)

    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Линия регрессии
    if len(predictions) > 1:
        z = np.polyfit(predictions, residuals.values, 1)
        p = np.poly1d(z)
        ax4.plot(predictions, p(predictions),
                 color='red',
                 linewidth=1.5,
                 label=f'Тренд: y={z[0]:.3f}x+{z[1]:.3f}')

    ax4.set_xlabel('Прогнозы модели', fontsize=11)
    ax4.set_ylabel('Остатки', fontsize=11)
    ax4.set_title('Остатки vs Прогнозы (гетероскедастичность)', fontsize=12)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    # 5. Автокорреляционная функция остатков
    ax5 = plt.subplot(2, 3, 5)

    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals.values, lags=10, ax=ax5,
             color=config.COLORS['secondary'])

    ax5.set_xlabel('Лаг', fontsize=11)
    ax5.set_ylabel('Автокорреляция', fontsize=11)
    ax5.set_title('Автокорреляционная функция остатков', fontsize=12)
    ax5.grid(True, alpha=0.3)

    # 6. Кумулятивные остатки
    ax6 = plt.subplot(2, 3, 6)

    cumulative_residuals = residuals.cumsum()
    ax6.plot(residuals.index, cumulative_residuals,
             color=config.COLORS['secondary'],
             linewidth=2)

    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Проверка на случайное блуждание
    # Если кумулятивные остатки выходят за пределы ±2σ, может быть систематическая ошибка
    n = len(residuals)
    critical_value = 1.36 * np.sqrt(n)  # Для теста Кульбака-Лейблера

    ax6.axhline(y=cumulative_residuals.max(), color='red',
                linestyle='--', linewidth=1, alpha=0.5,
                label=f'Максимум: {cumulative_residuals.max():.1f}')
    ax6.axhline(y=cumulative_residuals.min(), color='blue',
                linestyle='--', linewidth=1, alpha=0.5,
                label=f'Минимум: {cumulative_residuals.min():.1f}')

    ax6.set_xlabel('Год', fontsize=11)
    ax6.set_ylabel('Кумулятивная сумма', fontsize=11)
    ax6.set_title('Кумулятивные остатки (CUSUM)', fontsize=12)
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Сохранение
    plot_path = os.path.join(config.PLOTS_DIR, '08_residual_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Графики анализа остатков сохранены: {plot_path}")
    return fig


def save_residual_analysis_results(all_test_results, model_adequacy):
    """
    Сохранение результатов анализа остатков
    """
    results = "АНАЛИЗ ОСТАТОЧНОЙ КОМПОНЕНТЫ МОДЕЛИ\n"
    results += "=" * 60 + "\n\n"

    results += "1. ПРОВЕРКА ПРЕДПОСЫЛОК РЕГРЕССИОННОГО АНАЛИЗА:\n"
    results += "-" * 40 + "\n\n"

    # Нормальность
    normality = all_test_results.get('normality', {})
    if normality:
        results += "НОРМАЛЬНОСТЬ РАСПРЕДЕЛЕНИЯ:\n"
        if 'shapiro' in normality:
            shapiro = normality['shapiro']
            results += f"  Тест Шапиро-Уилка: W={shapiro['statistic']:.4f}, "
            results += f"p={shapiro['p_value']:.4f}\n"
            results += f"  Вывод: {'Нормальное' if shapiro['is_normal'] else 'Ненормальное'}\n"

        if 'dagostino' in normality:
            dago = normality['dagostino']
            results += f"  Тест Д'Агостино: статистика={dago['statistic']:.4f}, "
            results += f"p={dago['p_value']:.4f}\n"
            results += f"  Вывод: {'Нормальное' if dago['is_normal'] else 'Ненормальное'}\n"

        if 'descriptive' in normality:
            desc = normality['descriptive']
            results += f"  Асимметрия: {desc['skewness']:.4f} "
            results += f"(должно быть около 0)\n"
            results += f"  Эксцесс: {desc['kurtosis']:.4f} "
            results += f"(должно быть около 0)\n"

    # Автокорреляция
    autocorr = all_test_results.get('autocorrelation', {})
    if autocorr:
        results += "\nАВТОКОРРЕЛЯЦИЯ:\n"
        if 'durbin_watson' in autocorr:
            dw = autocorr['durbin_watson']
            results += f"  Тест Дарбина-Уотсона: DW={dw['statistic']:.4f}\n"
            results += f"  Интерпретация: {dw['interpretation']}\n"

        if 'ljung_box' in autocorr:
            lb = autocorr['ljung_box']
            if not lb.empty:
                results += f"  Тест Люнга-Бокса:\n"
                for lag in lb.index:
                    results += f"    Lag {lag}: статистика={lb.loc[lag, 'lb_stat']:.4f}, "
                    results += f"p={lb.loc[lag, 'lb_pvalue']:.4f}\n"

    # Гетероскедастичность
    hetero = all_test_results.get('heteroscedasticity', {})
    if hetero:
        results += "\nГОМОСКЕДАСТИЧНОСТЬ:\n"
        if 'lm_statistic' in hetero:
            results += f"  Тест Бройша-Пагана:\n"
            results += f"    LM статистика: {hetero['lm_statistic']:.4f}\n"
            results += f"    LM p-value: {hetero['lm_p_value']:.4f}\n"
            results += f"    F статистика: {hetero['f_statistic']:.4f}\n"
            results += f"    F p-value: {hetero['f_p_value']:.4f}\n"
        elif 'spearman_corr' in hetero:
            results += f"  Корреляция Спирмена: {hetero['spearman_corr']:.4f}\n"
            results += f"  p-value: {hetero['spearman_p']:.4f}\n"

    # Паттерны
    patterns = all_test_results.get('patterns', {})
    if patterns:
        results += "\nПАТТЕРНЫ В ОСТАТКАХ:\n"
        if 'trend_in_residuals' in patterns:
            trend = patterns['trend_in_residuals']
            results += f"  Тренд в остатках: наклон={trend['slope']:.4f}, "
            results += f"p={trend['p_value']:.4f}\n"

        if 'seasonality' in patterns:
            season = patterns['seasonality']
            if len(season['significant_lags']) > 0:
                results += f"  Сезонность/цикличность: значимые лаги "
                results += f"{list(season['significant_lags'])}\n"
            else:
                results += f"  Сезонность/цикличность: не обнаружена\n"

        if 'time_dependence' in patterns:
            time_dep = patterns['time_dependence']
            results += f"  Зависимость от времени: корреляция={time_dep['correlation']:.4f}, "
            results += f"p={time_dep['p_value']:.4f}\n"

    results += "\n" + "=" * 60 + "\n"
    results += "2. ОЦЕНКА АДЕКВАТНОСТИ МОДЕЛИ:\n"
    results += "-" * 40 + "\n\n"

    results += f"ОБЩИЙ ВЫВОД ОБ АДЕКВАТНОСТИ МОДЕЛИ:\n"

    if model_adequacy['is_adequate']:
        results += "  ✅ МОДЕЛЬ АДЕКВАТНА\n"
        results += "  • Все основные предпосылки выполняются\n"
        results += "  • Остатки ведут себя случайным образом\n"
        results += "  • Модель можно использовать для прогнозирования\n"
    else:
        results += "  ⚠️  МОДЕЛЬ НЕДОСТАТОЧНО АДЕКВАТНА\n"
        results += "  • Обнаружены нарушения предпосылок:\n"

        issues = []
        if not model_adequacy.get('normality_ok', True):
            issues.append("ненормальность распределения остатков")
        if not model_adequacy.get('autocorr_ok', True):
            issues.append("автокорреляция остатков")
        if not model_adequacy.get('homoscedasticity_ok', True):
            issues.append("гетероскедастичность")
        if not model_adequacy.get('patterns_ok', True):
            issues.append("систематические паттерны в остатках")

        for i, issue in enumerate(issues, 1):
            results += f"    {i}. {issue}\n"

        results += "\n  РЕКОМЕНДАЦИИ:\n"
        results += "  1. Рассмотреть другие спецификации модели\n"
        results += "  2. Использовать преобразования данных\n"
        results += "  3. Применить робастные методы оценивания\n"
        results += "  4. Добавить дополнительные переменные\n"
        results += "  5. Использовать временные ряды (ARIMA и др.)\n"

    results += "\n" + "=" * 60 + "\n"
    results += "3. РЕКОМЕНДАЦИИ ДЛЯ ДАЛЬНЕЙШЕГО АНАЛИЗА:\n"
    results += "-" * 40 + "\n\n"

    if model_adequacy['is_adequate']:
        results += "• Продолжить использование текущей модели\n"
        results += "• Регулярно проверять адекватность на новых данных\n"
        results += "• Рассмотреть добавление сезонных компонент\n"
    else:
        results += "• Исследовать возможность ARIMA-моделирования\n"
        results += "• Проверить наличие структурных изменений в ряде\n"
        results += "• Рассмотреть нелинейные модели тренда\n"
        results += "• Использовать методы машинного обучения для прогнозирования\n"

    # Сохранение
    results_path = os.path.join(config.TABLES_DIR, '08_residual_analysis_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(results)

    print(f"✅ Результаты анализа остатков сохранены: {results_path}")

    # Сохраняем статистики тестов в CSV
    test_stats = []

    for test_name, test_result in all_test_results.items():
        if isinstance(test_result, dict):
            flat_result = {f"{test_name}_{k}": v for k, v in test_result.items()
                           if not isinstance(v, (dict, pd.DataFrame))}
            test_stats.append(flat_result)

    if test_stats:
        import itertools
        all_stats = {}
        for d in test_stats:
            all_stats.update(d)

        stats_df = pd.DataFrame([all_stats])
        csv_path = os.path.join(config.TABLES_DIR, '08_test_statistics.csv')
        stats_df.to_csv(csv_path, index=False)
        print(f"✅ Статистики тестов сохранены: {csv_path}")


def main():
    """Основная функция модуля"""
    print("\n" + "=" * 60)
    print("ШАГ 8: АНАЛИЗ ОСТАТОЧНОЙ КОМПОНЕНТЫ")
    print("=" * 60)

    # Загрузка данных
    df = load_data()
    if df is None:
        return

    # Получаем лучшую модель тренда из предыдущего шага
    # В реальном использовании это можно заменить на загрузку сохраненной модели
    print("Получение лучшей модели тренда...")

    # Для демонстрации используем линейную модель
    from trend_modeling import fit_linear_model
    trend_model = fit_linear_model(df)

    if trend_model is None:
        print("❌ Не удалось получить модель тренда")
        return

    # 1. Расчет остатков
    residuals, predictions = calculate_residuals(df, trend_model)

    # 2. Проверка нормальности
    normality_results = test_residual_normality(residuals)

    # 3. Проверка автокорреляции
    autocorr_results, has_autocorr = test_residual_autocorrelation(residuals)

    # 4. Проверка гетероскедастичности
    hetero_results, is_homoscedastic = test_heteroscedasticity(residuals, predictions)

    # 5. Анализ паттернов
    pattern_results, has_problematic_patterns = analyze_residual_patterns(residuals, predictions)

    # 6. Построение графиков
    plot_residual_analysis(residuals, predictions, {
        'normality': normality_results,
        'autocorrelation': autocorr_results,
        'heteroscedasticity': hetero_results,
        'patterns': pattern_results
    })

    # 7. Оценка адекватности модели
    model_adequacy = {
        'is_adequate': (
                normality_results.get('shapiro', {}).get('is_normal', False) and
                not has_autocorr and
                is_homoscedastic and
                not has_problematic_patterns
        ),
        'normality_ok': normality_results.get('shapiro', {}).get('is_normal', False),
        'autocorr_ok': not has_autocorr,
        'homoscedasticity_ok': is_homoscedastic,
        'patterns_ok': not has_problematic_patterns
    }

    print("\n" + "=" * 60)
    print("ИТОГОВАЯ ОЦЕНКА АДЕКВАТНОСТИ МОДЕЛИ")
    print("=" * 60)

    if model_adequacy['is_adequate']:
        print("✅ МОДЕЛЬ АДЕКВАТНА")
        print("   Все предпосылки выполняются")
    else:
        print("⚠️  МОДЕЛЬ НЕДОСТАТОЧНО АДЕКВАТНА")
        print("   Обнаружены следующие проблемы:")
        if not model_adequacy['normality_ok']:
            print("   • Нарушение нормальности остатков")
        if not model_adequacy['autocorr_ok']:
            print("   • Наличие автокорреляции")
        if not model_adequacy['homoscedasticity_ok']:
            print("   • Гетероскедастичность")
        if not model_adequacy['patterns_ok']:
            print("   • Систематические паттерны в остатках")

    # 8. Сохранение результатов
    all_test_results = {
        'normality': normality_results,
        'autocorrelation': autocorr_results,
        'heteroscedasticity': hetero_results,
        'patterns': pattern_results
    }

    save_residual_analysis_results(all_test_results, model_adequacy)

    print("\n✅ Анализ остаточной компоненты завершён!")

    # Возвращаем результаты
    return {
        'residuals': residuals,
        'predictions': predictions,
        'test_results': all_test_results,
        'model_adequacy': model_adequacy
    }


if __name__ == "__main__":
    main()