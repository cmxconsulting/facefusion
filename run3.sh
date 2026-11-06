#!/bin/sh
cd "$(dirname "$0")" || exit 1
. .venv/bin/activate

python facefusion.py run --config-path facefusion-mac.ini --open-browser \
	-o ~/Downloads/tmp/2/ \
	--jobs-path /Users/cmx/Downloads/tmp/tmp \
	"$@"
