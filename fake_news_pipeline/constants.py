from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "32_fake_news.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "fake_news_pipeline.joblib"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "plots"
RANDOM_STATE = 42
