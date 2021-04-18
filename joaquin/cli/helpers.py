from argparse import ArgumentParser
import logging
import pathlib

from astropy.utils.misc import isiterable

from ..logger import logger


__all__ = ['get_parser']


def set_log_level(args, loggers):

    if args.verbosity == 1:
        log_level = logging.DEBUG

    elif args.verbosity == 2:
        log_level = 1

    elif args.verbosity == 3:
        log_level = 0

    elif args.quietness == 1:
        log_level = logging.WARNING

    elif args.quietness == 2:
        log_level = logging.ERROR

    else:
        log_level = logging.INFO  # default

    if not isiterable(loggers):
        loggers = [loggers]

    for l in loggers:
        l.setLevel(log_level)


def get_parser(description="", loggers=None, multiproc_options=True):
    if loggers is None:
        loggers = [logger]

    class JoaquinArgumentParser(ArgumentParser):

        def parse_args(self, *args, **kwargs):
            parsed = super().parse_args(*args, **kwargs)

            set_log_level(parsed, loggers)

            # deal with multiproc:
            if multiproc_options and parsed.mpi:
                from schwimmbad.mpi import MPIPool
                Pool = MPIPool
                kw = dict()
            elif multiproc_options and parsed.mpiasync:
                from schwimmbad.mpi import MPIAsyncPool
                Pool = MPIAsyncPool
                kw = dict()
            elif multiproc_options and parsed.n_procs > 1:
                from schwimmbad import MultiPool
                Pool = MultiPool
                kw = dict(processes=parsed.n_procs)
            else:
                from schwimmbad import SerialPool
                Pool = SerialPool
                kw = dict()
            parsed.Pool = Pool
            parsed.Pool_kwargs = kw

            return parsed

    # Define parser object
    parser = JoaquinArgumentParser(description=description)

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0,
                          dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0,
                          dest='quietness')

    parser.add_argument("-c", "--config", dest="config_file", required=True,
                        type=str,
                        help="The path to the config file for this run of "
                             "Joaquin.")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true",
                        help="Overwrite any existing results for this run.")

    if multiproc_options:
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--procs", dest="n_procs", default=1,
                           type=int, help="Number of processes.")
        group.add_argument("--mpi", dest="mpi", default=False,
                           action="store_true", help="Run with MPI.")
        group.add_argument("--mpiasync", dest="mpiasync", default=False,
                           action="store_true", help="Run with MPI async.")

    return parser
