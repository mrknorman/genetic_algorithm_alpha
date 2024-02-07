import json
import logging
import argparse
import prctl
from copy import deepcopy
from pathlib import Path
from typing import List
import sys
import os

from tensorflow.keras import losses, optimizers

# Local imports:
import gravyflow as gf
from default_genome import return_default_genome
from default_dataset import return_default_dataset_args

def train_genome(
        heart,
        # Model Arguments:
        model_name : str,
        model_path : str, 
        cache_segments : bool = True,
        # Training Arguments:
        patience : int = 1,
        learning_rate : float = 1.0E-4,
        max_epochs : int = 1000,
        # Dataset Arguments: 
        num_train_examples : int = int(1E5),
        num_validation_examples : int = int(1E4),
        # Manage args
        restart_count : int = 0
    ):

    if restart_count < 1:
        restart_count + 1

    if gf.is_redirected():
        cache_segments = False

    training_arguments, validation_arguments, testing_arguments, ifos = return_default_dataset_args(cache_segments)

    def adjust_features(features, labels):
        labels['INJECTION_MASKS'] = labels['INJECTION_MASKS'][0]
        return features, labels
    
    num_onsource_samples = int(
        (gf.Defaults.onsource_duration_seconds + 2.0*gf.Defaults.crop_duration_seconds)*
        gf.Defaults.sample_rate_hertz
    )
    num_offsource_samples = int(
        gf.Defaults.offsource_duration_seconds*
        gf.Defaults.sample_rate_hertz
    )
        
    input_configs = [
        {
            "name" : gf.ReturnVariables.ONSOURCE.name,
            "shape" : (len(ifos), num_onsource_samples,)
        },
        {
            "name" : gf.ReturnVariables.OFFSOURCE.name,
            "shape" : (len(ifos), num_offsource_samples,)
        }
    ]
    
    output_config = {
        "name" : gf.ReturnVariables.INJECTION_MASKS.name,
        "type" : "binary"
    }

    training_config = {
        "num_examples_per_epoc" : num_train_examples,
        "num_validation_examples" : num_validation_examples,
        "patience" : patience,
        "max_epochs" : max_epochs,
        "model_path" : model_path
    }

    # Load or build model:
    model = gf.Model.load(
        name=model_name,
        model_load_path=model_path,
        load_genome=True,
        num_ifos=len(ifos),
        optimizer=optimizers.Adam(learning_rate=learning_rate), 
        training_config=training_config,
        loss=losses.BinaryCrossentropy(),
        input_configs=input_configs,
        output_config=output_config,
        force_overwrite=(restart_count==0),
        dataset_args=training_arguments
    )
    
    if (restart_count==0):
        model.summary()
    else:
        print(f"Attempt {restart_count + 1}: Restarting training from where we left off...")
        model.summary()

    model.train(
        validate_args=validation_arguments,
        training_config=training_config,
        force_retrain=(restart_count==0), 
        heart=heart
    )

    # Validation configs:
    efficiency_config = {
            "max_scaling" : 15.0, 
            "num_scaling_steps" : 31, 
            "num_examples_per_scaling_step" : 8192
        }
    far_config = {
            "num_seconds" : 1.0E5
        }
    roc_config : dict = {
            "num_examples" : 1.0E5,
            "scaling_ranges" : [
                8.0,
            ]
        } 
    
    model.validate(
        testing_arguments,
        efficiency_config=efficiency_config,
        far_config=far_config,
        roc_config=roc_config,
        model_path=model_path,
        heart=heart
    )

    if heart is not None:
        heart.complete()

    return 0

if __name__ == "__main__":

    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Read command line arguments:
    parser = argparse.ArgumentParser(
        description = (
            "Train a member of the population."
        )
    )
    
    parser.add_argument(
        "--name",
        type = str, 
        default = "Model name. Required.",
        help = (
            "Name of cnn model."
        )
    )
    parser.add_argument(
        "--path",
        type = str, 
        default = "Model name. Required.",
        help = (
            "Name of cnn model."
        )
    )
    parser.add_argument(
        "--gpu",
        type = str, 
        default = None,
        help = (
            "Specify a gpu to use."
        )
    )

    parser.add_argument(
        "--request_memory",
        type = int, 
        default = 6000,
        help = (
            "Specify a how much memory to give tf."
        )
    )

    parser.add_argument(
        "--restart_count",
        type = int, 
        default = 1,
        help = (
            "Number of times model has been trained,"
            " if 0, model will be overwritten."
        )
    )

    args = parser.parse_args()

    # Set parameters based on command line arguments:
    gpu = str(args.gpu)
    memory_to_allocate_tf = args.request_memory
    restart_count = args.restart_count
    name = args.name
    path = Path(args.path)

    # Set process name:
    prctl.set_name(f"gwflow_training_{name}")

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

    # Set up TensorBoard logging directory
    logs = "logs"

    if gf.is_redirected():
        heart = gf.Heart(name)
    else:
        heart = None
    
    with gf.env(
            memory_to_allocate_tf=memory_to_allocate_tf,
            gpus=gpu
        ):
        
        if train_genome(
            heart,
            restart_count=restart_count,
            model_name=name,
            model_path=path
        ) == 0:
            logging.info("Training completed, do a shot!")
            os._exit(0)
        
        else:
            logging.error("Training failed for some reason.")
            os._exit(1)
        
    os._exit(1)
