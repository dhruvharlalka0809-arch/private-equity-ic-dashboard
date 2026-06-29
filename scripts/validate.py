import subprocess
import sys


COMMANDS = [
    [sys.executable, "-m", "py_compile", "app.py", "src/pe_model.py", "tests/test_pe_model.py"],
    [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
]


def main() -> int:
    for command in COMMANDS:
        print("+", " ".join(command))
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
