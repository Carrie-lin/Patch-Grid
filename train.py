import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging as log
from trainer.trainer import Trainer
from trainer.trainer_shape_space import ShapeSpaceTrainer
from configs.option import parse_options
from configs.open_option import parse_options_open
from configs.shape_space_option import parse_options_ss


log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)

if __name__ == "__main__":

    args, args_str = parse_options()
    log.info(f'Parameters: \n{args_str}')
    log.info(f'Training on {args.dataset_path}')

    if args.train_shape_space:
        args, args_str = parse_options_ss()
        model = ShapeSpaceTrainer(args, args_str)
    else:
        model = Trainer(args, args_str)

    model.train()