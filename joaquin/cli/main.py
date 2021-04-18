# Standard library
import argparse
import sys

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

    # def make_neighborhoods(self):
    #     """PCA the spectra to find neighbors and define neighborhoods"""
    #     from .make_neighborhoods import make_neighborhoods
    #     parser = get_parser(
    #         description=(
    #             "TODO"),
    #         multiproc_options=False)

    #     # HACK
    #     parser.usage = ('joaquin make_neighborhoods' +
    #                     parser.format_usage()[9:])
    #     args = parser.parse_args(sys.argv[2:])
    #     make_neighborhoods(args.config_file)

    def run(self):
        """Run Joaquin on the pre-generated neighborhoods"""
        from .run import run_pipeline

        parser = get_parser(
            description=(
                "This command is the main workhorse for Joaquin: it runs the "
                "distance training and prediction on all of the stars in the "
                "spectral neighborhoods."),
            loggers=[logger])

        # HACK
        parser.usage = 'joaquin run' + parser.format_usage()[9:]

        parser.add_argument("-i", "--index", dest="neighborhood_index",
                            default=None, type=int,
                            help="If specified, run on just one neighborhood, "
                                 "with the index provided.")

        args = parser.parse_args(sys.argv[2:])

        with args.Pool(**args.Pool_kwargs) as pool:
            run_pipeline(args.config_file, pool=pool,
                         neighborhood_index=args.neighborhood_index)

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
