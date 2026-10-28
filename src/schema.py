"""Raw C-MAPSS schema: 2 ids + 3 operating settings + 21 sensors = 26 columns."""

ID_COLS = ["engine_id", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLUMNS = ID_COLS + SETTING_COLS + SENSOR_COLS
NUMERIC_COLS = SETTING_COLS + SENSOR_COLS
