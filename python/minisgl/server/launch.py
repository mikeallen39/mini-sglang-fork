from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import traceback
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import build_parallel_infos
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    logger = init_logger(__name__, f"scheduler_rank{args.world_info.rank}")
    scheduler = None
    try:
        with torch.inference_mode():
            scheduler = Scheduler(args)
            scheduler.sync_all_ranks()

            if args.world_info.is_primary():
                ack_queue.put("Scheduler is ready")

            if args.silent_output:
                logging.disable(logging.INFO)

            scheduler.run_forever()
    except KeyboardInterrupt:
        if scheduler is not None and scheduler.world_info.is_primary():
            print()  # for a clean newline after ^C
            logger.info("Scheduler exiting gracefully...")
        if scheduler is not None:
            scheduler.shutdown()
    except Exception:
        logger.error(
            "Scheduler rank %s crashed:\n%s",
            args.world_info.rank,
            traceback.format_exc(),
        )
        raise


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        world_size = server_args.world_info.size
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()

        for i in range(world_size):
            world_info, tp_info, ep_info = build_parallel_infos(
                i, server_args.tp_info.size, server_args.ep_info.size
            )
            new_args = replace(
                server_args,
                world_info=world_info,
                tp_info=tp_info,
                ep_info=ep_info,
                device_id=i,
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"minisgl-rank{i}-scheduler",
            ).start()

        num_tokenizers = server_args.num_tokenizer
        # DeTokenizer, only 1
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
