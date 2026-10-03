"""Explicit CLI only; import of the package does not dispatch."""
from .harness import main

if __name__ == "__main__":
    raise SystemExit(main())
