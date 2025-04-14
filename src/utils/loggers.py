# Copyright (c) EEEM071, University of Surrey

import os
import os.path as osp
import sys

import logging


class Logger:
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(osp.dirname(fpath), exist_ok=True)
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class RankLogger:
    def __init__(self, source_names, target_names):
        self.source_names = source_names
        self.target_names = target_names
        self.logger = {name: {"epoch": [], "rank1": []} for name in self.target_names}

    def write(self, name, epoch, rank1):
        self.logger[name]["epoch"].append(epoch)
        self.logger[name]["rank1"].append(rank1)

    def show_summary(self):
        print("=> Show performance summary")
        for name in self.target_names:
            from_where = "source" if name in self.source_names else "target"
            print(f"{name} ({from_where})")
            for epoch, rank1 in zip(
                self.logger[name]["epoch"], self.logger[name]["rank1"]
            ):
                print(f"- epoch {epoch}\t rank1 {rank1:.1%}")

def setup_logging(filename):
    stream_handler = logging.StreamHandler(sys.stdout)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=filename,
                        filemode='a',
                        level=logging.INFO,
                        format='%(levelname)s %(asctime)s: %(message)s',
                        datefmt='%d-%m-%y %H:%M:%S')
    stream_handler.setFormatter(logging.Formatter('%(levelname)s %(asctime)s: %(message)s', '%d-%m-%y %H:%M:%S'))
    logging.getLogger().addHandler(stream_handler)