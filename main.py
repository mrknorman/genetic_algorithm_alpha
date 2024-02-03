from pathlib import Path
from typing import Union
from copy import deepcopy
import logging
import argparse
import os

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import mixed_precision, layers
from tensorflow.keras import backend as K
from tensorflow.data.experimental import AutoShardPolicy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Local imports:
import gravyflow as gf
from default_genome import return_default_genome
from default_dataset import return_default_dataset_args

def test_model(
        path : Union[Path, None],
        num_tests : int = 32
    ):
    
    max_population : int = 10
    max_num_generations : int = 10
    default_genome = return_default_genome()

    if path is not None:
        population = gf.Population.load(Path(path))
    else:
        population = gf.Population(
            max_population, 
            default_genome
        )

    population.train(num_generations=max_num_generations)
        
if __name__ == "__main__":

    # Read command line arguments:
    parser = argparse.ArgumentParser(
        description = (
            "Train an entire population."
        )
    )
    
    parser.add_argument(
        "--name",
        type = str, 
        default = None,
        help = (
            "Name of population model."
        )
    )

    args = parser.parse_args()
    
    # ---- User parameters ---- #
    # Set logging level:
    logging.basicConfig(level=logging.INFO)

    memory_to_allocate_tf = 500    
    # Test Genetic Algorithm Optimiser:
    with gf.env(
            memory_to_allocate_tf=memory_to_allocate_tf,
        ):
        
        test_model(args.name)
    
        os._exit(1)