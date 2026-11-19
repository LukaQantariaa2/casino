import multiprocessing
import sys

from domino.main import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main() or 0)
