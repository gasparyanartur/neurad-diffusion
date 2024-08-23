import sys


def main():
    arg = sys.argv[1]
    params = sys.argv[2:]

    print()
    print(sys.argv)
    print()
    print(arg)
    print()

    i_arg = params.index(arg)
    print(i_arg)

    if i_arg == -1:
        raise ValueError(f"Argument {arg} not found in params")

    if i_arg == len(params) - 1:
        raise ValueError(f"No value provided for argument {args.arg}")

    value = params[i_arg + 1]
    print(value)


if __name__ == "__main__":
    main()
