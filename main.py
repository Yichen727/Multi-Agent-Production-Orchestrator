"""MAPO — Multi-Agent Production Orchestrator

Launch the Streamlit UI (sole entry point):
    python main.py
"""

import subprocess
import sys


def main():
    """Launch the Streamlit UI — the only way to interact with MAPO."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py"],
    )


if __name__ == "__main__":
    main()