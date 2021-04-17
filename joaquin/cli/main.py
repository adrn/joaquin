# Standard library
import argparse
import os
import pathlib
import shutil
import sys

# Third-party
import numpy as np
from threadpoolctl import threadpool_limits

# Package
from .helpers import get_parser
from ..logger import logger


class CLI:
    """To add a new subcommand, just add a new classmethod and a docstring!"""
    _usage = None

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Spectrophotometric parallaxes As A Service',
            usage=self._usage.strip())

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print(f"Unsupported command '{args.command}'")
            parser.print_help()
            sys.exit(1)

        getattr(self, args.command)()

    def make_neighborhoods(self):
        """PCA the spectra to find neighbors and define neighborhoods"""
        from .make_neighborhoods import make_neighborhoods
        parser = get_parser(
            description=(
                "TODO"),
            multiproc_options=False)

        # HACK
        parser.usage = 'joaquin make_neighborhoods' + parser.format_usage()[9:]
        args = parser.parse_args(sys.argv[2:])
        make_neighborhoods(args.config_file)

    def run_thejoker(self):
        """Run The Joker on the input data"""
        from thejoker.logging import logger as joker_logger
        from .run_thejoker import run_thejoker

        parser = get_parser(
            description=(
                "This command is the main workhorse for HQ: it runs The Joker "
                "on all of the input data and caches the samplings."),
            loggers=[logger, joker_logger])
        # HACK
        parser.usage = 'hq run_thejoker' + parser.format_usage()[9:]

        parser.add_argument("-s", "--seed", dest="seed", default=None,
                            type=int, help="Random number seed")
        parser.add_argument("--limit", dest="limit", default=None,
                            type=int, help="Maximum number of stars to process")

        args = parser.parse_args(sys.argv[2:])

        if args.seed is None:
            args.seed = np.random.randint(2**32 - 1)
            logger.log(
                1, f"No random seed specified, so using seed: {args.seed}")

        with threadpool_limits(limits=1, user_api='blas'):
            with args.Pool(**args.Pool_kwargs) as pool:
                run_thejoker(run_path=args.run_path, pool=pool,
                             overwrite=args.overwrite,
                             seed=args.seed, limit=args.limit)

        sys.exit(0)


# Auto-generate the usage block:
cmds = []
maxlen = max([len(name) for name in CLI.__dict__.keys()])
for name, attr in CLI.__dict__.items():
    if not name.startswith('_'):
        cmds.append(f'    {name.ljust(maxlen)}  {attr.__doc__}\n')

CLI._usage = f"""
joaquin <command> [<args>]

Available commands:
{''.join(cmds)}

See more usage information about a given command by running:
    joaquin <command> --help

"""


def main():
    CLI()
