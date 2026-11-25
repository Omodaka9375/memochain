"""Allow ``python -m memochain.server`` in addition to the module scripts."""

from memochain.server.app import main

if __name__ == "__main__":
    main()
