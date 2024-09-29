import itertools as it
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict


class SlurmArrayEntry(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    keys: List[str]
    values: List[List[str]]


class SlurmArrayConfig(BaseModel):
    entries: List[SlurmArrayEntry]


def verify_entry(entry: SlurmArrayEntry):
    if len(entry.keys) != len(entry.values):
        raise ValueError("The number of keys and values in an entry must be the same")

    if len(entry.values) == 0:
        raise ValueError("An entry must have at least one value")

    if not all(len(value) == len(entry.values[0]) for value in entry.values):
        raise ValueError("All values in an entry must have the same length")


def combine_joined_parameters(
    arg_names: List[str], arg_values: List[List[str]], arg_prefix: str = ""
) -> List[str]:
    joined_params = []

    for i_arg in range(len(arg_values[0])):
        params: List[str] = []

        for arg_name, arg_value in zip(arg_names, arg_values):
            params.append(f"{arg_prefix}{arg_name} {arg_value[i_arg]}")

        joined_params.append(" ".join(params))

    return joined_params


def parse_param_list(config: SlurmArrayConfig, arg_prefix: str = "") -> List[str]:
    all_params = []

    for entry in config.entries:
        verify_entry(entry)

        joined_params = combine_joined_parameters(entry.keys, entry.values, arg_prefix)
        all_params.append(joined_params)

    combined_parameters = list(it.product(*all_params))
    joined_parameters = [" ".join(combined) for combined in combined_parameters]

    return joined_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        help="The path to the configs file",
        type=Path,
    )
    parser.add_argument(
        "-i", "--task_id", help="The id to index", type=int, required=False
    )
    parser.add_argument(
        "-s",
        "--size",
        help="The number of entries in configs",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-p",
        "--prefix",
        help="The prefix to add to the arguments",
        required=False,
        default="",
    )

    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        file_content = f.read()

    config = SlurmArrayConfig.model_validate_json(file_content)
    param_list = parse_param_list(config, args.prefix)

    arg_size = len(param_list)

    if args.size:
        print(arg_size)

    elif args.task_id is not None:
        if args.task_id >= arg_size:
            pass

        else:
            print(param_list[args.task_id])

    else:
        raise ValueError("You need to pass either task_id or size")


if __name__ == "__main__":
    main()
