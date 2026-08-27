"""Patches the notebook 01_eda_and_modeling.ipynb in-place to fix import issues."""
import json
from pathlib import Path

nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "01_eda_and_modeling.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

project_root = str(Path(__file__).resolve().parent.parent).replace("\\", "\\\\")
tests_path = project_root + "\\\\tests\\\\make_synthetic_data.py"

# Fix Cell 1: replace sys.path.append('..') with hardcoded absolute path
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "sys.path.append" in src and "from src import config" in src:
            cell["source"] = [
                "import sys\n",
                "\n",
                "# Absolute project root path\n",
                '_project_root = r"' + project_root + '"\n',
                "if _project_root not in sys.path:\n",
                "    sys.path.insert(0, _project_root)\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "from src import config, data_loader, features, model as model_module, evaluation\n",
                "\n",
                "pd.set_option('display.max_columns', 30)\n",
                "plt.rcParams['figure.figsize'] = (11, 4)\n",
            ]
            cell["outputs"] = []
            cell["execution_count"] = None
            print("Fixed Cell 1: sys.path")
            break

# Fix Cell 2: replace 'from tests.make_synthetic_data import' with importlib
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "from tests.make_synthetic_data import build_synthetic_farm" in src:
            cell["source"] = [
                "# ---- Configuration ----\n",
                "USE_SYNTHETIC = True   # set False once you've downloaded the real Kaggle data\n",
                "\n",
                "if USE_SYNTHETIC:\n",
                "    import importlib.util\n",
                "    _spec = importlib.util.spec_from_file_location(\n",
                '        "make_synthetic_data",\n',
                '        r"' + tests_path + '"\n',
                "    )\n",
                "    _mod = importlib.util.module_from_spec(_spec)\n",
                "    _spec.loader.exec_module(_mod)\n",
                "    build_synthetic_farm = _mod.build_synthetic_farm\n",
                "\n",
                '    RAW_DIR = config.PROJECT_ROOT / "data" / "raw" / "Wind Farm A"\n',
                "    if not RAW_DIR.exists():\n",
                "        build_synthetic_farm(RAW_DIR)\n",
                "else:\n",
                "    RAW_DIR = config.RAW_DATA_DIR  # edit in src/config.py, or override here directly\n",
                "\n",
                "paths = data_loader.discover_subdatasets(RAW_DIR)\n",
                'print(f"Found {len(paths)} sub-datasets:")\n',
                "for p in paths:\n",
                '    print(" -", p.name)\n',
            ]
            cell["outputs"] = []
            cell["execution_count"] = None
            print("Fixed Cell 2: synthetic data import")
            break

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Notebook patched successfully at {nb_path}")
