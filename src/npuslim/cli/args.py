import argparse


def build_parser():
    # TODO
    parser = argparse.ArgumentParser(
        prog="npuslim",
        description="NPU model compression toolkit"
    )

    subparsers = parser.add_subparsers(dest="command")

    # ----- quant command -----
    quant_parser = subparsers.add_parser("quant", help="Quant command")
    quant_parser.add_argument("--model", type=str, help="")
    quant_parser.set_defaults(func=run_quant)

    return parser


def run_quant(args):
    print(f"quant, {args.name} from npuslim!")
