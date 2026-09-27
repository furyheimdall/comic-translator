#!/usr/bin/env bash
# Install MangaTranslator (pinned) into its own venv plus the Korean font pack.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MT_TAG="${MT_TAG:-v1.24.7}"
MT_DIR="$ROOT/engines/mangatranslator"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

if [[ ! -d "$MT_DIR/.git" ]]; then
    git clone --depth 1 --branch "$MT_TAG" https://github.com/meangrinch/MangaTranslator.git "$MT_DIR"
fi
if [[ ! -x "$MT_DIR/.venv/bin/python" ]]; then
    uv venv --python 3.13 "$MT_DIR/.venv"
fi
uv pip install --python "$MT_DIR/.venv/bin/python" torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url "$TORCH_INDEX"
uv pip install --python "$MT_DIR/.venv/bin/python" -r "$MT_DIR/requirements.txt"

# Korean font pack (SIL OFL 1.1). MangaTranslator drops glyphs missing from the
# selected pack, so a Latin comic font would erase Hangul.
FONT_DIR="$ROOT/fonts/korean/NotoSansKR"
mkdir -p "$FONT_DIR"
if [[ ! -s "$FONT_DIR/NotoSansKR[wght].ttf" ]]; then
    curl -fsSL -o "$FONT_DIR/NotoSansKR[wght].ttf" \
        "https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf"
    curl -fsSL -o "$FONT_DIR/OFL.txt" "https://github.com/google/fonts/raw/main/ofl/notosanskr/OFL.txt"
fi

"$MT_DIR/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; print("torch", torch.__version__, torch.cuda.get_device_name(0))'
