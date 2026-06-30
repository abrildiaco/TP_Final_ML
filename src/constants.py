# ========================= Currency =========================

OFFICIAL_USD_SELL = {
    "2025-05-13": 1150.00,
    "2025-05-14": 1150.00,
    "2025-05-15": 1150.00,
    "2025-05-16": 1160.00,
    "2025-05-19": 1155.00,
    "2025-05-20": 1160.00,
    "2025-05-21": 1160.00,
    "2025-05-22": 1155.00,
    "2025-05-23": 1150.00,
    "2025-05-26": 1160.00,
    "2025-05-27": 1170.00,
    "2025-05-28": 1175.00,
    "2025-05-29": 1195.00,
    "2025-05-30": 1195.00,
}

EXCHANGE_RATE = sum(OFFICIAL_USD_SELL.values()) / len(OFFICIAL_USD_SELL)


# ========================= Premium Brands =========================
PREMIUM_BRANDS = [
    "alfa romeo",
    "audi",
    "bmw",
    "land rover",
    "mercedes benz",
    "porsche",
    "volvo",
]

# ========================= Semantic Mappings =========================

BRAND_MAP = {
    "renault": ["renault", "rrenault"],
    "hyundai": ["hiunday", "hyundai"],
    "volkswagen": ["volkswagen", "vol"],
    "ds": ["ds", "d.s.", "d·s", "ds automobiles"],
    "land rover": ["range rover", "land rover"],
    "great wall motor": ["gwm", "haval"],
    "jetour": ["jetour", "jetur"],
}

MODEL_MAP = {
    "sw4": ["sw4", "hilux sw4"],
    "ds7": ["ds7", "ds7 crossback"],
    "clase ml": ["clase ml", "ml"],
    "santa fe": ["santa fe", "grand santa fe", "grand santa fé"],
    "tiggo 4": ["tiggo 4", "tiggo 4 pro"],
}

COLOR_MAP = {
    "blanco": [
        "blanca",
        "blanco",
        "blanco glaciar",
        "summit white",
        "mineralweiss metallic",
        "blanco nacre tricapa",
        "blanco banquise",
    ],
    "negro": ["negra", "negro", "carbon black", "black meet kettle", "noir perla nera"],
    "amarillo": ["amarilla", "amarillo", "amarrillo"],
    "dorado": ["dorada", "dorado", "champaing"],
    "gris": [
        "gris",
        "gris plata",
        "gris platino",
        "plata",
        "plateado",
        "gray",
        "gris laque",
        "plata bari",
        "prata bari+tet vulc",
        "gris estrella",
        "gris artense",
    ],
    "gris oscuro": [
        "gris oscuro",
        "gris selenium",
        "grafito",
        "granite crystal bc",
        "granite crysta bc",
        "gris titane",
        "gris indy",
        "gris silverstone",
        "skyscraper grau metallic",
    ],
    "marron": ["marron claro", "marron oscuro", "cafe", "marron kodiak"],
    "azul": ["azul", "steel blue", "blue"],
    "rojo": ["rojo", "rojo sunset metalizado"],
    "verde": ["verde", "verde oscuro"],
    "beige": ["beige", "beige techo negro"],
    "celeste": ["celeste", "azul claro"],
    "naranja": ["naranja", "cobre"],
    "violeta": ["violeta", "morado", "morado oscuro"],
    "bordo": ["bordo"],
}

FUEL_TYPE_MAP = {
    "hibrido": ["hibrido", "hibrido/nafta"],
}

TRANSMISSION_MAP = {
    "automatica": [
        "automática",
        "automatica",
        "automatica secuencial",
        "automática secuencial",
        "semiautomatica",
        "semiautomática",
    ],
}

SEMANTIC_CATEGORY_MAPS = {
    "Marca": BRAND_MAP,
    "Modelo": MODEL_MAP,
    "Color": COLOR_MAP,
    "Tipo de combustible": FUEL_TYPE_MAP,
    "Transmisión": TRANSMISSION_MAP,
}


# ========================= Text Extraction =========================

ENGINE_TEXT_MAP = {
    "1.0": ["1.0", "1.0t", "1.0 t", "1.0 tsi", "1.0 200 tsi"],
    "1.3": ["1.3", "1.3 tce", "1.3 tce turbo", "1.3 t270"],
    "1.5": ["1.5", "1.5t", "1.5 t", "1.5 turbo"],
    "1.6": ["1.6", "1.6l", "1.6 16v", "1.6 vti", "1.6 thp"],
    "1.8": ["1.8"],
    "2.0": ["2.0", "2.0 sport", "2.0 sel", "2.0 hse", "2.0 turbo", "2.0 turbonaftero"],
    "2.5": ["2.5"],
    "2.8": ["2.8"],
}

BACKUP_CAMERA_TEXT_MAP = {
    "Camara de retroceso": ["Cámara de retroceso", "Camara de retroceso"],
}

TURBO_PATTERNS = [
    "turbo",
    "turboalimentado",
    "tsi",
    "tdi",
    "tfsi",
    "ecoboost",
    "tce",
    "thp",
    r"\bt\b",
    r"\d\.\d\s*t\b",
]

TRANSMISSION_TEXT_MAP = {
    "automatica": [
        "automatica",
        "automatico",
        "caja automatica",
        "at",
        "at6",
        "at8",
        "at9",
        "cvt",
        "dsg",
        "tiptronic",
        "secuencial",
    ],
    "manual": [
        "manual",
        "caja manual",
        "mt",
    ],
}

INTEREST_TERMS = {
    "papeles": [
        "papeles al dia",
        "documentacion al dia",
        "documentacion en regla",
        "listo para transferir",
        "lista para transferir",
        "sin deudas",
        "libre deuda",
        "libre de deuda",
        "vtv",
        "verificacion policial",
    ],
    "service": [
        "service",
        "services",
        "service oficial",
        "services oficiales",
        "mantenimiento",
        "distribucion hecha",
        "bateria nueva",
        "cubiertas nuevas",
    ],
    "buen_estado": [
        "buen estado",
        "muy buen estado",
        "excelente estado",
        "excelentes condiciones",
        "buenas condiciones",
    ],
    "impecable": [
        "impecable",
        "impecables",
        "sin detalles",
        "como nuevo",
        "como nueva",
        "como 0km",
    ],
    "cuidado": [
        "cuidado",
        "cuidada",
        "muy cuidado",
        "muy cuidada",
        "bien cuidado",
        "bien cuidada",
    ],
    "detalles_uso": [
        "detalles de uso",
        "detalle de uso",
        "detalles esteticos",
        "detalle estetico",
    ],
    "daño": [
        "rayado",
        "rayada",
        "rayones",
        "golpe",
        "golpes",
        "golpeado",
        "golpeada",
        "abollado",
        "abollada",
        "abolladura",
    ],
    "sin_choque": [
        "sin choque",
        "sin choques",
        "nunca chocado",
        "nunca chocada",
    ],
}


# ========================= Feature Engineering =========================

ALL_FEATURE_BLOCKS = [
    "usage",
    "brand_model",
    "premium",
    "cilindrada_missing",
]

DEFAULT_FEATURE_VARIANTS = [
    {
        "name": "baseline",
        "feature_blocks": [],
    },
    {
        "name": "usage",
        "feature_blocks": ["usage"],
    },
    {
        "name": "brand_model",
        "feature_blocks": ["brand_model"],
    },
    {
        "name": "premium_brand",
        "feature_blocks": ["premium"],
    },
    {
        "name": "cilindrada_missing",
        "feature_blocks": ["cilindrada_missing"],
    },
    {
        "name": "all_features",
        "feature_blocks": "all",
    },
    {
        "name": "all_features_without_brand_model_originals",
        "feature_blocks": "all",
        "drop_cols": ["Marca", "Modelo"],
    },
]


# ========================= Experiment Defaults =========================

XGB_BASE_PARAMS = {
    "n_estimators": 350,
    "learning_rate": 0.045,
    "max_depth": 5,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": -1,
}

XGB_REGULARIZED_PARAMS = {
    "n_estimators": 550,
    "learning_rate": 0.035,
    "max_depth": 6,
    "min_child_weight": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.10,
    "reg_lambda": 3.0,
    "random_state": 42,
    "n_jobs": -1,
}

EXPERIMENT_METRIC_FORMAT = {
    "val_rmse": "{:,.2f}",
    "val_mae": "{:,.2f}",
    "val_r2": "{:.4f}",
    "train_rmse": "{:,.2f}",
    "train_mae": "{:,.2f}",
    "train_r2": "{:.4f}",
}

EXPERIMENT_DISPLAY_COLS = [
    "experiment",
    "params",
    "dropped_cols",
    "rare_min_count",
    "n_features",
    "val_rmse",
    "val_mae",
    "val_r2",
    "train_rmse",
    "train_mae",
    "train_r2",
]
