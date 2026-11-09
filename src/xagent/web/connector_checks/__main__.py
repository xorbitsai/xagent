"""``python -m xagent.web.connector_checks``: see ``cli.py``."""

import sys

from dotenv import load_dotenv

from .cli import main

if __name__ == "__main__":
    # Same as the server's own entry point (xagent.web.__main__).
    load_dotenv()
    sys.exit(main())
