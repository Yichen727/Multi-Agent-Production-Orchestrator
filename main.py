"""MAPO — Multi-Agent Production Orchestrator

Launch the Streamlit UI:
    python main.py
"""

import subprocess
import sys


def main():
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py"],
    )


if __name__ == "__main__":
    main()