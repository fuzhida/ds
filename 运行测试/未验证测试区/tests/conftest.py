import os
import sys

# Ensure tests can import top-level modules like `运行测试.*` when running from this subdirectory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)