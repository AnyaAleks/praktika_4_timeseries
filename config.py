# config.py
import os

# ========== ПУТИ К ФАЙЛАМ ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'austria_outbound_tourism.csv')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
TABLES_DIR = os.path.join(OUTPUTS_DIR, 'tables')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')

# ========== НАСТРОЙКИ ДАННЫХ ==========
DATA_DESCRIPTION = {
    'source': 'UN Tourism (https://www.untourism.int/)',
    'indicator': 'OUTB_TRIP_TOTL_TOTL_TOUR - outbound trips total overnight visitors',
    'country': 'Austria',
    'period': '2000-2024',
    'unit': 'thousand trips',
    'observations': 25
}

# ========== НАСТРОЙКИ ГРАФИКОВ ==========
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
COLORS = {
    'primary': '#2E86AB',      # Синий основной
    'secondary': '#A23B72',    # Фиолетовый
    'anomaly': '#F18F01',      # Оранжевый для аномалий
    'trend': '#C73E1D',        # Красный для тренда
    'seasonal': '#218380',     # Зеленый для сезонности
    'residual': '#6A5ACD'      # Сливовый для остатков
}

# ========== НАСТРОЙКИ АНАЛИЗА ==========
SMOOTHING_WINDOW = 3           # Окно скользящей средней
SIGNIFICANCE_LEVEL = 0.05      # Уровень значимости

# ========== СЛУЖЕБНЫЕ ФУНКЦИИ ==========
def ensure_directories():
    """Создает все необходимые директории, если они не существуют"""
    directories = [OUTPUTS_DIR, PLOTS_DIR, TABLES_DIR, MODELS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Все директории созданы/проверены")

# Автоматически создаем директории при импорте
ensure_directories()