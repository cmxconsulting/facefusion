#!/bin/sh
cd "$(dirname "$0")" || exit 1
. .venv/bin/activate

python facefusion.py run --config-path facefusion-mac.ini --open-browser \
	-o ~/Downloads/tmp/2/ \
	--temp-path /Users/cmx/Downloads/tmp/tmp \
	"$@"
