#!/bin/sh
cd "$(dirname "$0")" || exit 1
. .venv/bin/activate

SOURCE="${1:-$HOME/Downloads/tmp/chrisL.jpg}"
VIDEO="${2:-$HOME/Downloads/tmp/v/angel.mp4}"
OUTPUT="${3:-$HOME/Downloads/tmp/2/can.mp4}"

python facefusion.py headless-run --config-path facefusion-mac.ini \
	-s "${SOURCE}" \
	-t "${VIDEO}" \
	-o "${OUTPUT}" \
	--temp-path "$HOME/Downloads/tmp/tmp"
