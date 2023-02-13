import os
from vcp.common import logger
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def init_logger(args):
    args.log_path = os.path.join(args.log_path, f"vcp_{args.env}_{args.seed}")
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])
    return rank
