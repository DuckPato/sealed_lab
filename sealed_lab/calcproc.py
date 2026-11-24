"""Child process of one calculation (see `api.calculate_set_isolated`).

argv[1] = JSON {"set_code", "options" (keyword arguments of `api.calculate_set`), "cancel_file", "parent_pid"}. stdout carries
only progress lines "P <fraction>\t<message>"; logging goes to stderr. The calculation is cancelled at its next checkpoint
when `cancel_file` appears or the parent process is gone. (Not through stdin: on Windows a thread blocked reading a pipe
stalled the import of numpy's DLLs in the main thread.) Exit codes: 0 done, 1 error (traceback on stderr), 2 cancelled, 3 set busy;
anything else is a native crash.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import sys
import threading
import traceback

EXIT_ERROR = 1
EXIT_CANCELLED = 2
EXIT_BUSY = 3          # another process is calculating the set (the message is the last stderr line)


def main() -> int:
    from sealed_lab import api
    from sealed_lab.procguard import process_alive
    from sealed_lab.types import Cancelled, SetBusy

    args = json.loads(sys.argv[1])
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    cancel = threading.Event()
    cancel_file, parent = args.get("cancel_file"), args.get("parent_pid")

    def watch() -> None:
        while not cancel.wait(0.5):
            if (cancel_file and os.path.exists(cancel_file)) or (parent and not process_alive(parent)):
                cancel.set()
    threading.Thread(target=watch, name="watch-cancel", daemon=True).start()

    def progress(frac: float, msg: str) -> None:
        print(f"P {frac:.6f}\t{msg}".replace("\n", " "), flush=True)

    try:
        api.calculate_set(args["set_code"], progress=progress, cancel=cancel, **args["options"])
    except Cancelled:
        return EXIT_CANCELLED
    except SetBusy as e:
        print(str(e), file=sys.stderr, flush=True)
        return EXIT_BUSY
    except BaseException:  # noqa: BLE001 - reported to the parent as the traceback
        traceback.print_exc()
        sys.stderr.flush()
        return EXIT_ERROR
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
