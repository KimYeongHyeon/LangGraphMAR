#!/bin/bash
# InDuDoNet+ virtual environment activation
# Usage: source baselines/InDuDoNet/activate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

echo "=== InDuDoNet+ environment ==="
python -c "
import torch, odl, astra, numpy, h5py
print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()}')
print(f'  odl={odl.__version__} astra={astra.__version__}')
print(f'  numpy={numpy.__version__} h5py={h5py.__version__}')
"
echo "================================"
