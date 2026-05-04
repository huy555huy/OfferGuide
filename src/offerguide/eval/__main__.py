"""``python -m offerguide.eval`` entry point — see runner._cli."""
import sys

from .runner import _cli

if __name__ == "__main__":
    sys.exit(_cli())
