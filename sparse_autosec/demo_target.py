import sys


def run(user: str):
    # intentionally vulnerable for demo
    return eval(user)


if __name__ == "__main__":
    payload = sys.argv[1] if len(sys.argv) > 1 else "1+1"
    print(run(payload))
