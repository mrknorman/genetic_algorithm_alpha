from pathlib import Path
import logging
import os

# Local imports:
import gravyflow as gf

def return_default_injection_generators():
    
    minimum_snr : float = 8.0
    maximum_snr : float = 15.0
    ifos : List[gf.IFO] = [gf.IFO.L1]

    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    injection_directory_path : Path = current_dir / "injection_parameters"
    
    # Intilise Scaling Method:
    scaling_method : gf.ScalingMethod = gf.ScalingMethod(
        gf.Distribution(
            min_= minimum_snr,
            max_= maximum_snr,
            type_=gf.DistributionType.UNIFORM
        ),
        gf.ScalingTypes.SNR
    )

    # Load injection config:
    phenom_d_generator : gf.cuPhenomDGenerator = gf.WaveformGenerator.load(
        injection_directory_path / "baseline_phenom_d.json", 
        scaling_method=scaling_method,    
        network = None # Single detector
    )
    phenom_d_generator.injection_chance = 0.5
    # Load glitch config:
    wnb_generator : gf.WNBGenerator = gf.WaveformGenerator.load(
        injection_directory_path / "baseline_wnb.json", 
        scaling_method=scaling_method,    
        network = None # Single detector
    )
    wnb_generator.injection_chance = 0.5

    return [phenom_d_generator, wnb_generator]

def return_default_dataset_args(
    cache_segments : bool
    ):

    ifos : List[gf.IFO] = [gf.IFO.L1]

    # Setup ifo data acquisition object:
    ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
        gf.ObservingRun.O3, 
        gf.DataQuality.BEST, 
        [
            gf.DataLabel.NOISE,
            gf.DataLabel.GLITCHES
        ],
        gf.SegmentOrder.RANDOM,
        cache_segments=cache_segments,
        force_acquisition=False,
        logging_level=logging.ERROR
    )
    
    # Initilise noise generator wrapper:
    noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
        ifo_data_obtainer=ifo_data_obtainer,
        noise_type=gf.NoiseType.REAL,
        ifos=ifos
    )

    # Set requested data to be used as model input:
    input_variables = [
        gf.ReturnVariables.ONSOURCE,
        gf.ReturnVariables.OFFSOURCE
    ]
    
    # Set requested data to be used as model output:
    output_variables = [
        gf.ReturnVariables.INJECTION_MASKS
    ]

    dataset_arguments : Dict = {
        # Noise: 
        "noise_obtainer" : noise_obtainer,
        # Injections:
        "injection_generators" : return_default_injection_generators(), 
        # Output configuration:
        "input_variables" : input_variables,
        "output_variables": output_variables
    } # Define injection directory path:

    return dataset_arguments, ifos