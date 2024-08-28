import sys

HELP_ALIAS = {"-h", "--help"}


def print_help_message():
    print("SLURM array variable reader")
    print("Usage: python slurm_array_var_reader.py <arg> <params>")
    print(
        "Example: \n"
        "\t$ python slurm_array_var_reader.py --model_name --model_name model1 --model_path /path/to/model\n"
        "\tmodel1"
    )


def main():
    if sys.argv[1].lower() in HELP_ALIAS:
        print_help_message()
        return

    arg = sys.argv[1]
    params = sys.argv[2:]

    i_arg = params.index(arg)
    if i_arg == len(params) - 1:  # can't be k,v if v is last
        raise ValueError(f"No value provided for argument {arg}")

    value = params[i_arg + 1]
    print(value)


if __name__ == "__main__":
    main()
