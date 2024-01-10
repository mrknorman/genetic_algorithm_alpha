from pathlib import Path
from copy import deepcopy
import logging
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
        num_tests : int = 32
    ):
    
    max_population : int = 10
    max_num_generations : int = 10
    default_genome = return_default_genome()

    population = gf.Population.load()
    if population is None:
        population = gf.Population(
            max_population, 
            default_genome
        )

    population.train(num_generations=max_num_generations)
        
if __name__ == "__main__":

    np.random.seed(100)

    gf.Defaults.set(
        seed=1000,
        num_examples_per_generation_batch=256,
        num_examples_per_batch=32,
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=16.0,
        crop_duration_seconds=0.5,
        scale_factor=1.0E21
    )
    
    # ---- User parameters ---- #
    # Set logging level:
    logging.basicConfig(level=logging.INFO)

    memory_to_allocate_tf = 500    
    # Test Genetic Algorithm Optimiser:
    with gf.env(
            memory_to_allocate_tf=memory_to_allocate_tf,
        ):

        test_model()
    
        os._exit(1)