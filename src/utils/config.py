
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
# Workspace paths
data_path = root_dir / "data"
src_path = root_dir / "src"
log_path = root_dir / "logs"

test_path = data_path / 'test'
target_API_path = data_path / 'target_API'
llm_cache_path = data_path / "llm_cache"
intermediate_data_path = src_path / "intermediate_data"


API_KEY = 'XXXX'
BASE_URL = "XXXX"
MODEL_NAME =  'XXXX' #'gpt-5' #'gpt-5.4-mini-2026-03-17' # 'claude-sonnet-4-20250514'  #'gpt-5'  # 'gpt-3.5-turbo-0613'

