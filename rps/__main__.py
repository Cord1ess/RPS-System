"""`python -m rps` runs the command line."""

import sys

from rps.cli.main import main

if __name__ == "__main__":
    sys.exit(main())