#!/bin/bash
set -e

cd frontend

if command -v pnpm >/dev/null 2>&1; then
    PACKAGE_MANAGER="pnpm"
    pnpm install
    pnpm run build
elif command -v npm >/dev/null 2>&1; then
    PACKAGE_MANAGER="npm"
    npm install
    npm run build
else
    echo -e "\033[1;31m\nNeither pnpm nor npm is installed.\n\033[0m" >&2
    exit 1
fi

echo -e "\033[1;32m\nfrontend build success via ${PACKAGE_MANAGER}\033[0m"

cd ../
python3 main.py --port 7860 --host 0.0.0.0
