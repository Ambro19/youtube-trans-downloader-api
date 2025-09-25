# backend/tests/conftest.py
# why: ensure `import main` works when running pytest from /backend
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))