import sys
from pathlib import Path

# 저장소 루트를 import 경로에 추가 (어느 위치에서 pytest를 실행해도 동작)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
