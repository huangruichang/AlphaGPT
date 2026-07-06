import torch
import os

class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','ashare')}"
    BATCH_SIZE = 8192
    TRAIN_STEPS = 1000
    MAX_FORMULA_LEN = 12
    TRADE_SIZE_USD = 1000.0
    MIN_MARKET_CAP = 5e9  # 最低市值 50 亿 CNY，低于此视为不可交易
    BASE_FEE = 0.001  # A 股单边费率 0.1%，买卖双边合计约 0.2%，此处取 0.1% 作为基础费率
    INPUT_DIM = 6
