#!/usr/bin/env bash
# Launch the Streamlit app with the project virtualenv.
#
# Running a bare `streamlit run streamlit_app.py` resolves `streamlit` from
# PATH, which can pick a system Python that lacks this project's dependencies.
# That failure mode shows up as an app that loads forever instead of an error,
# so pin the interpreter explicitly.

set -euo pipefail

cd "$(dirname "$0")"

PY="./venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "error: $PY not found. Create the virtualenv first:" >&2
  echo "  python3 -m venv venv && ./venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

echo "Using $("$PY" --version) at $PY"
exec "$PY" -m streamlit run streamlit_app.py "$@"
