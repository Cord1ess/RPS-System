#!/usr/bin/env python3
"""
`python rps.py ...` is the same as `python -m rps ...`, for people who type the file name.
"""

import sys

from rps.cli.main import main

if __name__ == "__main__":
    sys.exit(main())