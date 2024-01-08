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
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = num_tests
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    patience = 1
    max_epochs = 1000
    minimum_snr = 8
    maximum_snr = 15
    cache_segments = False

    max_population : int = 10
    
    num_train_examples : int = int(512)
    num_validation_examples : int = int(512)
    
    dataset_arguments, _, _, ifos = return_default_dataset_args(cache_segments)

    training_config = {
        "num_examples_per_epoc" : num_train_examples,
        "num_validation_examples" : num_validation_examples,
        "patience" : patience,
        "max_epochs" : max_epochs
    }

    default_genome = return_default_genome()
    
    population = gf.Population.load()
    if population is None:
        population = gf.Population(
            max_population, 
            max_population, 
            default_genome,
            training_config,
            deepcopy(dataset_arguments)
        )

    population.train(
        100, 
        deepcopy(dataset_arguments),
        num_validation_examples,
        num_examples_per_batch
    )
        
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

    memory_to_allocate_tf = 10000    
    # Test Genetic Algorithm Optimiser:
    with gf.env(
            memory_to_allocate_tf=memory_to_allocate_tf,
            gpus="6"
        ):

        test_model()
    
        os._exit(1)