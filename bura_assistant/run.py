import multiprocessing
import sys

from bura.main import main

if __name__ == "__main__":
    multiprocessing.freeze_support()     # the packaged exe starts the engine's worker processes through itself
    sys.exit(main() or 0)
