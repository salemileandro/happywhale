import sys
import argparse


def parse_command():
    """
    Parse command line arguments and return a dictionary of those arguments. The arguments can be inspected from the
    source code itself or via the command `python3 path_to_script.py --help`
    """
    my_parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    my_parser.add_argument('--epoch', action='store', type=int, default=10, help="Number of epochs (frozen backbone)")
    my_parser.add_argument('--batch', action='store', type=int, default=32, help="Size of batch (frozen backbone)")

    my_parser.add_argument('--lr', action='store', type=float, default=1e-3, help="Learning rate (frozen backbone)")

    my_parser.add_argument('--ft', action='store_true', help="Flag to activate finetuning")
    my_parser.add_argument('--ft-epoch', action='store', type=int, default=10, help="Number of epochs (finetuning)")
    my_parser.add_argument('--ft-batch', action='store', type=int, default=32, help="Size of batch (finetuning)")
    my_parser.add_argument('--ft-lr', action='store', type=float, default=1e-5, help="Learning rate (finetuning)")

    args = my_parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    parse_command()
    print(" ".join(sys.argv))

