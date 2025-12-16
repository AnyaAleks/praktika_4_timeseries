# 07_trend_modeling.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')
import os
import config
from data_loading import load_data


def fit_linear_model(df):
    """
    Подбор линейной модели тренда
    """
    print("\n" + "=" * 60)
    print("ЛИНЕЙНАЯ МОДЕЛЬ ТРЕНДА")
    print("=" * 60)

    # Подготовка данных
    X = np.arange(len(df)).reshape(-1, 1)  # Время как предиктор
    y = df['value'].values

    # Обучение модели
    model = LinearRegression()
    model.fit(X, y)

    # Прогноз
    y_pred = model.predict(X)

    # Метрики качества
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Уравнение: y = {model.intercept_:.2f} + {model.coef_[0]:.2f} * t")
    print(f"Наклон: {model.coef_[0]:.2f} тыс. поездок/год")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")
    print(f"MAE: {mae:.2f}")

    return {
        'model': model,
        'predictions': y_pred,
        'metrics': {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2
        },
        'type': 'linear'
    }


def fit_polynomial_model(df, degree=2):
    """
    Подбор полиномиальной модели тренда
    """
    print(f"\n" + "=" * 60)
    print(f"ПОЛИНОМИАЛЬНАЯ МОДЕЛЬ {degree}-Й СТЕПЕНИ")
    print("=" * 60)

    # Подготовка данных
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['value'].values

    # Полиномиальные признаки
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Обучение модели
    model = LinearRegression()
    model.fit(X_poly, y)

    # Прогноз
    y_pred = model.predict(X_poly)

    # Метрики качества
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Вывод уравнения
    coefs = model.coef_
    intercept = model.intercept_

    equation = f"y = {intercept:.2f}"
    for i in range(1, degree + 1):
        equation += f" + {coefs[i]:.2f} * t^{i}"

    print(f"Уравнение: {equation}")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")
    print(f"MAE: {mae:.2f}")

    return {
        'model': (model, poly),
        'predictions': y_pred,
        'metrics': {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2
        },
        'type': f'polynomial_degree_{degree}',
        'equation': equation
    }


def fit_exponential_model(df):
    """
    Подбор экспоненциальной модели тренда
    """
    print("\n" + "=" * 60)
    print("ЭКСПОНЕНЦИАЛЬНАЯ МОДЕЛЬ ТРЕНДА")
    print("=" * 60)

    # Экспоненциальная модель: y = a * exp(b * t)
    def exp_func(t, a, b):
        return a * np.exp(b * t)

    # Подготовка данных
    t = np.arange(len(df))
    y = df['value'].values

    try:
        # Подбор параметров
        params, params_covariance = curve_fit(exp_func, t, y,
                                              p0=[y[0], 0.01],  # Начальные приближения
                                              maxfev=5000)

        a, b = params

        # Прогноз
        y_pred = exp_func(t, a, b)

        # Метрики качества
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"Уравнение: y = {a:.2f} * exp({b:.4f} * t)")
        print(f"Параметр роста (b): {b:.4f}")
        print(f"Начальное значение (a): {a:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {np.sqrt(mse):.2f}")
        print(f"MAE: {mae:.2f}")

        # Годовой темп роста в процентах
        growth_rate = (np.exp(b) - 1) * 100
        print(f"Годовой темп роста: {growth_rate:.2f}%")

        return {
            'model': (exp_func, params),
            'predictions': y_pred,
            'metrics': {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2
            },
            'type': 'exponential',
            'params': params,
            'growth_rate': growth_rate
        }

    except Exception as e:
        print(f"❌ Ошибка при подборе экспоненциальной модели: {e}")
        return None


def fit_logarithmic_model(df):
    """
    Подбор логарифмической модели тренда
    """
    print("\n" + "=" * 60)
    print("ЛОГАРИФМИЧЕСКАЯ МОДЕЛЬ ТРЕНДА")
    print("=" * 60)

    # Логарифмическая модель: y = a + b * ln(t+1)
    def log_func(t, a, b):
        return a + b * np.log(t + 1)

    # Подготовка данных
    t = np.arange(len(df))
    y = df['value'].values

    try:
        # Подбор параметров
        params, params_covariance = curve_fit(log_func, t, y, p0=[y[0], 1])
        a, b = params

        # Прогноз
        y_pred = log_func(t, a, b)

        # Метрики качества
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"Уравнение: y = {a:.2f} + {b:.2f} * ln(t+1)")
        print(f"Параметр a: {a:.2f}")
        print(f"Параметр b: {b:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {np.sqrt(mse):.2f}")
        print(f"MAE: {mae:.2f}")

        return {
            'model': (log_func, params),
            'predictions': y_pred,
            'metrics': {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2
            },
            'type': 'logarithmic'
        }

    except Exception as e:
        print(f"❌ Ошибка при подборе логарифмической модели: {e}")
        return None


def compare_models(models):
    """
    Сравнение всех подобранных моделей
    """
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ ТРЕНДА")
    print("=" * 60)

    comparison = []

    for model_name, model_info in models.items():
        if model_info is not None:
            comparison.append({
                'model': model_name,
                'type': model_info['type'],
                'r2': model_info['metrics']['r2'],
                'rmse': model_info['metrics']['rmse'],
                'mae': model_info['metrics']['mae'],
                'mse': model_info['metrics']['mse']
            })

    # Создаем DataFrame для сравнения
    comparison_df = pd.DataFrame(comparison)

    if not comparison_df.empty:
        # Сортируем по R² (по убыванию) и RMSE (по возрастанию)
        comparison_df = comparison_df.sort_values(['r2', 'rmse'],
                                                  ascending=[False, True])

        print("\nРейтинг моделей (лучшие сверху):")
        print("-" * 80)
        print(f"{'Модель':<20} {'Тип':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'MSE':<10}")
        print("-" * 80)

        for _, row in comparison_df.iterrows():
            print(f"{row['model']:<20} {row['type']:<20} {row['r2']:<10.4f} "
                  f"{row['rmse']:<10.2f} {row['mae']:<10.2f} {row['mse']:<10.2f}")

    return comparison_df


def select_best_model(models, comparison_df):
    """
    Выбор лучшей модели на основе метрик
    """
    if comparison_df.empty:
        print("❌ Нет моделей для сравнения")
        return None

    # Выбираем модель с максимальным R² и минимальным RMSE
    best_model_name = comparison_df.iloc[0]['model']
    best_model_info = models[best_model_name]

    print(f"\n" + "=" * 60)
    print(f"ВЫБРАНА ЛУЧШАЯ МОДЕЛЬ: {best_model_name.upper()}")
    print("=" * 60)

    print(f"Тип модели: {best_model_info['type']}")
    print(f"R²: {best_model_info['metrics']['r2']:.4f}")
    print(f"RMSE: {best_model_info['metrics']['rmse']:.2f}")
    print(f"MAE: {best_model_info['metrics']['mae']:.2f}")

    # Интерпретация качества
    r2 = best_model_info['metrics']['r2']
    if r2 > 0.9:
        quality = "ОТЛИЧНОЕ"
    elif r2 > 0.7:
        quality = "ХОРОШЕЕ"
    elif r2 > 0.5:
        quality = "УДОВЛЕТВОРИТЕЛЬНОЕ"
    elif r2 > 0.3:
        quality = "СЛАБОЕ"
    else:
        quality = "НЕДОСТАТОЧНОЕ"

    print(f"КАЧЕСТВО АППРОКСИМАЦИИ: {quality}")

    return best_model_info, best_model_name


def plot_trend_models(df, models, best_model_info, best_model_name):
    """
    Построение графиков моделей тренда
    """
    plt.style.use(config.PLOT_STYLE)

    # Создаем фигуру с несколькими подграфиками
    fig = plt.figure(figsize=(18, 12))

    # 1. Все модели на одном графике
    ax1 = plt.subplot(2, 2, 1)

    # Исходные данные
    ax1.plot(df.index, df['value'],
             color=config.COLORS['primary'],
             linewidth=2,
             marker='o',
             markersize=4,
             label='Исходный ряд',
             alpha=0.7)

    # Цвета для разных моделей
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']

    # Рисуем все модели
    for i, (model_name, model_info) in enumerate(models.items()):
        if model_info is not None:
            color = colors[i % len(colors)]
            linestyle = '-' if model_name == best_model_name else '--'
            linewidth = 3 if model_name == best_model_name else 1.5

            ax1.plot(df.index, model_info['predictions'],
                     color=color,
                     linestyle=linestyle,
                     linewidth=linewidth,
                     label=f'{model_name} (R²={model_info["metrics"]["r2"]:.3f})')

    ax1.set_xlabel('Год', fontsize=11)
    ax1.set_ylabel('Количество поездок, тыс.', fontsize=11)
    ax1.set_title('Сравнение моделей тренда', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Лучшая модель с доверительным интервалом
    ax2 = plt.subplot(2, 2, 2)

    ax2.plot(df.index, df['value'],
             color=config.COLORS['primary'],
             linewidth=2,
             marker='o',
             markersize=4,
             label='Исходный ряд',
             alpha=0.7)

    # Прогноз лучшей модели
    y_pred = best_model_info['predictions']
    ax2.plot(df.index, y_pred,
             color=config.COLORS['trend'],
             linewidth=3,
             label=f'{best_model_name} (лучшая)')

    # Остатки
    residuals = df['value'] - y_pred
    std_residuals = residuals.std()

    # Доверительный интервал (95%)
    ci_upper = y_pred + 1.96 * std_residuals
    ci_lower = y_pred - 1.96 * std_residuals

    ax2.fill_between(df.index, ci_lower, ci_upper,
                     color=config.COLORS['trend'],
                     alpha=0.2,
                     label='95% доверительный интервал')

    ax2.set_xlabel('Год', fontsize=11)
    ax2.set_ylabel('Количество поездок, тыс.', fontsize=11)
    ax2.set_title(f'Лучшая модель: {best_model_name}', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 3. Остатки лучшей модели
    ax3 = plt.subplot(2, 2, 3)

    # Остатки как столбчатая диаграмма
    bars = ax3.bar(df.index, residuals,
                   color=np.where(residuals >= 0, 'green', 'red'),
                   alpha=0.6,
                   edgecolor='black',
                   linewidth=0.5)

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Линия скользящего среднего остатков
    residuals_sma = pd.Series(residuals).rolling(window=3, center=True).mean()
    ax3.plot(df.index, residuals_sma,
             color='blue',
             linewidth=2,
             label='Скользящее среднее остатков')

    ax3.set_xlabel('Год', fontsize=11)
    ax3.set_ylabel('Остатки (разность)', fontsize=11)
    ax3.set_title(f'Остатки модели {best_model_name}', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Прогноз на будущее
    ax4 = plt.subplot(2, 2, 4)

    # Исторические данные
    ax4.plot(df.index, df['value'],
             color=config.COLORS['primary'],
             linewidth=2,
             marker='o',
             markersize=4,
             label='Исторические данные',
             alpha=0.7)

    # Прогноз лучшей модели для исторического периода
    ax4.plot(df.index, y_pred,
             color=config.COLORS['trend'],
             linewidth=2,
             label='Аппроксимация моделью')

    # Прогноз на 5 лет вперед
    forecast_years = 5
    last_year = df.index[-1].year

    # Создаем будущие даты
    future_years = pd.date_range(start=f'{last_year + 1}-01-01',
                                 periods=forecast_years,
                                 freq='Y')

    # В зависимости от типа модели делаем прогноз
    if best_model_info['type'] == 'linear':
        model = best_model_info['model']
        # Продолжаем временную ось
        t_future = np.arange(len(df), len(df) + forecast_years).reshape(-1, 1)
        y_future = model.predict(t_future)

    elif 'polynomial' in best_model_info['type']:
        model, poly = best_model_info['model']
        t_future = np.arange(len(df), len(df) + forecast_years).reshape(-1, 1)
        X_future_poly = poly.transform(t_future)
        y_future = model.predict(X_future_poly)

    elif best_model_info['type'] == 'exponential':
        func, params = best_model_info['model']
        t_future = np.arange(len(df), len(df) + forecast_years)
        y_future = func(t_future, *params)

    elif best_model_info['type'] == 'logarithmic':
        func, params = best_model_info['model']
        t_future = np.arange(len(df), len(df) + forecast_years)
        y_future = func(t_future, *params)

    # Рисуем прогноз
    ax4.plot(future_years, y_future,
             color=config.COLORS['secondary'],
             linewidth=2,
             linestyle='--',
             marker='s',
             markersize=5,
             label=f'Прогноз на {forecast_years} лет')

    # Область неопределенности прогноза
    forecast_std = std_residuals * np.sqrt(1 + 1 / len(df))  # Упрощенная формула
    ci_upper_future = y_future + 1.96 * forecast_std
    ci_lower_future = y_future - 1.96 * forecast_std

    ax4.fill_between(future_years, ci_lower_future, ci_upper_future,
                     color=config.COLORS['secondary'],
                     alpha=0.2,
                     label='Доверительный интервал прогноза')

    ax4.set_xlabel('Год', fontsize=11)
    ax4.set_ylabel('Количество поездок, тыс.', fontsize=11)
    ax4.set_title(f'Прогноз на основе модели {best_model_name}', fontsize=12)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Сохранение
    plot_path = os.path.join(config.PLOTS_DIR, '07_trend_models.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Графики моделей тренда сохранены: {plot_path}")

    # Дополнительный график: только лучшая модель
    fig2, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df.index, df['value'],
            color=config.COLORS['primary'],
            linewidth=2.5,
            marker='o',
            markersize=6,
            label='Фактические данные')

    ax.plot(df.index, y_pred,
            color=config.COLORS['trend'],
            linewidth=3,
            label=f'Модель тренда ({best_model_name})')

    # Прогноз
    ax.plot(future_years, y_future,
            color=config.COLORS['secondary'],
            linewidth=2.5,
            linestyle='--',
            marker='s',
            markersize=6,
            label=f'Прогноз ({forecast_years} лет)')

    # Области доверительных интервалов
    ax.fill_between(df.index, ci_lower, ci_upper,
                    color=config.COLORS['trend'],
                    alpha=0.2,
                    label='95% ДИ (история)')

    ax.fill_between(future_years, ci_lower_future, ci_upper_future,
                    color=config.COLORS['secondary'],
                    alpha=0.2,
                    label='95% ДИ (прогноз)')

    ax.set_xlabel('Год', fontsize=12, fontweight='bold')
    ax.set_ylabel('Количество поездок, тыс.', fontsize=12, fontweight='bold')
    ax.set_title(f'Модель тренда и прогноз: {best_model_name}\n'
                 f'R² = {best_model_info["metrics"]["r2"]:.4f}, '
                 f'RMSE = {best_model_info["metrics"]["rmse"]:.1f}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path2 = os.path.join(config.PLOTS_DIR, '07_best_trend_model.png')
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ График лучшей модели сохранён: {plot_path2}")

    # Сохраняем прогноз
    forecast_df = pd.DataFrame({
        'year': future_years.year,
        'forecast': y_future,
        'ci_lower': ci_lower_future,
        'ci_upper': ci_upper_future
    })

    forecast_csv_path = os.path.join(config.TABLES_DIR, '07_forecast.csv')
    forecast_df.to_csv(forecast_csv_path, index=False)
    print(f"✅ Данные прогноза сохранены: {forecast_csv_path}")

    return fig, fig2, forecast_df


def save_modeling_results(models, comparison_df, best_model_info, best_model_name, forecast_df):
    """
    Сохранение результатов моделирования тренда
    """
    results = "РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ ТРЕНДА\n"
    results += "=" * 60 + "\n\n"

    results += "1. СРАВНЕНИЕ МОДЕЛЕЙ:\n"
    results += "-" * 40 + "\n\n"

    if not comparison_df.empty:
        results += f"{'Модель':<20} {'Тип':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10}\n"
        results += "-" * 70 + "\n"

        for _, row in comparison_df.iterrows():
            results += f"{row['model']:<20} {row['type']:<20} {row['r2']:<10.4f} "
            results += f"{row['rmse']:<10.2f} {row['mae']:<10.2f}\n"

    results += "\n" + "=" * 60 + "\n"
    results += "2. ВЫБРАНА ЛУЧШАЯ МОДЕЛЬ:\n"
    results += "-" * 40 + "\n\n"

    if best_model_info is not None:
        results += f"МОДЕЛЬ: {best_model_name.upper()}\n"
        results += f"Тип: {best_model_info['type']}\n"
        results += f"R²: {best_model_info['metrics']['r2']:.4f}\n"
        results += f"RMSE: {best_model_info['metrics']['rmse']:.2f}\n"
        results += f"MAE: {best_model_info['metrics']['mae']:.2f}\n"
        results += f"MSE: {best_model_info['metrics']['mse']:.2f}\n"

        # Дополнительная информация в зависимости от типа модели
        if best_model_info['type'] == 'linear':
            model = best_model_info['model']
            results += f"\nУРАВНЕНИЕ: y = {model.intercept_:.2f} + {model.coef_[0]:.2f} * t\n"
            results += f"Интерпретация: Каждый год количество поездок увеличивается "
            results += f"на {abs(model.coef_[0]):.2f} тыс.\n"

        elif 'polynomial' in best_model_info['type']:
            if 'equation' in best_model_info:
                results += f"\nУРАВНЕНИЕ: {best_model_info['equation']}\n"

        elif best_model_info['type'] == 'exponential':
            if 'params' in best_model_info:
                a, b = best_model_info['params']
                results += f"\nУРАВНЕНИЕ: y = {a:.2f} * exp({b:.4f} * t)\n"
                if 'growth_rate' in best_model_info:
                    results += f"Годовой темп роста: {best_model_info['growth_rate']:.2f}%\n"

        elif best_model_info['type'] == 'logarithmic':
            if 'params' in best_model_info:
                a, b = best_model_info['params']
                results += f"\nУРАВНЕНИЕ: y = {a:.2f} + {b:.2f} * ln(t+1)\n"

    results += "\n" + "=" * 60 + "\n"
    results += "3. ПРОГНОЗ НА БУДУЩЕЕ:\n"
    results += "-" * 40 + "\n\n"

    if forecast_df is not None:
        results += f"{'Год':<10} {'Прогноз':<12} {'Нижняя граница':<15} {'Верхняя граница':<15}\n"
        results += "-" * 55 + "\n"

        for _, row in forecast_df.iterrows():
            results += f"{int(row['year']):<10} {row['forecast']:<12.0f} "
            results += f"{row['ci_lower']:<15.0f} {row['ci_upper']:<15.0f}\n"

    results += "\n" + "=" * 60 + "\n"
    results += "4. ВЫВОДЫ И РЕКОМЕНДАЦИИ:\n"
    results += "-" * 40 + "\n\n"

    if best_model_info is not None:
        r2 = best_model_info['metrics']['r2']

        results += "КАЧЕСТВО МОДЕЛИ:\n"
        if r2 > 0.9:
            results += "• Отличное качество аппроксимации\n"
            results += "• Модель хорошо описывает тенденцию\n"
            results += "• Прогнозы могут быть достаточно точными\n"
        elif r2 > 0.7:
            results += "• Хорошее качество аппроксимации\n"
            results += "• Модель адекватно описывает тренд\n"
            results += "• Прогнозы имеют умеренную точность\n"
        elif r2 > 0.5:
            results += "• Удовлетворительное качество\n"
            results += "• Модель улавливает основные тенденции\n"
            results += "• Прогнозы следует использовать с осторожностью\n"
        else:
            results += "• Низкое качество аппроксимации\n"
            results += "• Модель плохо описывает данные\n"
            results += "• Прогнозы могут быть ненадежными\n"

        results += f"\nРЕКОМЕНДАЦИИ:\n"
        results += "1. Использовать выбранную модель для краткосрочного прогнозирования\n"
        results += "2. Учитывать доверительные интервалы при принятии решений\n"
        results += "3. Регулярно обновлять модель по мере поступления новых данных\n"
        results += "4. Рассмотреть возможность добавления внешних факторов в модель\n"
        results += "5. Проверять остатки на автокорреляцию и нормальность распределения\n"

    # Сохранение
    results_path = os.path.join(config.TABLES_DIR, '07_trend_modeling_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(results)

    print(f"✅ Результаты моделирования тренда сохранены: {results_path}")


def main():
    """Основная функция модуля"""
    print("\n" + "=" * 60)
    print("ШАГ 7: МОДЕЛИРОВАНИЕ ТРЕНДА")
    print("=" * 60)

    # Загрузка данных
    df = load_data()
    if df is None:
        return

    # 1. Подбор различных моделей тренда
    models = {}

    # Линейная модель
    models['linear'] = fit_linear_model(df)

    # Полиномиальные модели
    models['poly_2'] = fit_polynomial_model(df, degree=2)
    models['poly_3'] = fit_polynomial_model(df, degree=3)

    # Экспоненциальная модель
    models['exponential'] = fit_exponential_model(df)

    # Логарифмическая модель
    models['logarithmic'] = fit_logarithmic_model(df)

    # 2. Сравнение моделей
    comparison_df = compare_models(models)

    # 3. Выбор лучшей модели
    best_model_info, best_model_name = select_best_model(models, comparison_df)

    # 4. Построение графиков и прогноз
    forecast_df = None
    if best_model_info is not None:
        _, _, forecast_df = plot_trend_models(df, models, best_model_info, best_model_name)

    # 5. Сохранение результатов
    save_modeling_results(models, comparison_df, best_model_info, best_model_name, forecast_df)

    print("\n✅ Моделирование тренда завершено!")

    # Возвращаем результаты
    return {
        'models': models,
        'comparison': comparison_df,
        'best_model': best_model_info,
        'best_model_name': best_model_name,
        'forecast': forecast_df
    }


if __name__ == "__main__":
    main()