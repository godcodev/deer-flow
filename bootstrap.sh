#!/bin/bash
# traptest.sh

# Start both of DeerFlow's backend and web UI server.
# If the user presses Ctrl+C, kill them both.

if [ "$1" = "--dev" -o "$1" = "-d" -o "$1" = "dev" ]; then
  echo "Starting DeerFlow in [DEVELOPMENT] mode..."
  echo
	uv run server.py --reload & SERVER_PID=$$!
	cd web && pnpm dev & WEB_PID=$$!
	trap "kill $$SERVER_PID $$WEB_PID" SIGINT SIGTERM
	wait
else
	echo "Starting DeerFlow in [PRODUCTION] mode..."
  echo
	uv run server.py
	cd web && pnpm start
fi
