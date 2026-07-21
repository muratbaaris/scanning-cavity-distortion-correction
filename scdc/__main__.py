"""Allow the package to be executed with ``python -m scdc``."""

import sys

from scdc.cli import main

if __name__ == "__main__":
    sys.exit(main())
