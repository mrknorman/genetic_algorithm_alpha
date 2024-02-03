from copy import deepcopy

# Local imports:
import gravyflow as gf
from default_dataset import return_default_waveform_generators

def return_default_genome():

    max_num_initial_layers : int = 8

    # Training genes:
    optimizer = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.CONSTANT, 
                value="adam"
            )
        )
    batch_size = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.POW_TWO, 
                min_=16,
                max_=64,
                dtype=int
            )
        )
    learning_rate = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.LOG, 
                min_=10E-7, 
                max_=10E-3
            )
        )

    # Injection genes:
    waveform_generators = {name : {
        "hp" : gf.HyperInjectionGenerator(
            min_ = gf.HyperParameter(
                gf.Distribution(
                    type_=gf.DistributionType.UNIFORM, 
                    min_=0, 
                    max_=100
                ) 
            ),
            max_ = gf.HyperParameter(
                gf.Distribution(
                        type_=gf.DistributionType.UNIFORM, 
                        min_=0, 
                        max_=100
                    )
            ),
            mean = gf.HyperParameter(
                gf.Distribution(
                    type_=gf.DistributionType.UNIFORM, 
                    min_=0, 
                    max_=100
                )
            ),
            std = gf.HyperParameter(
                gf.Distribution(
                    type_=gf.DistributionType.UNIFORM, 
                    min_=0, 
                    max_=100
                )
            ),
            distribution = gf.HyperParameter(
                gf.Distribution(
                    type_=gf.DistributionType.CHOICE, 
                    possible_values=[
                        gf.DistributionType.UNIFORM,
                        gf.DistributionType.LOG,
                        gf.DistributionType.NORMAL
                    ]
                )
            ),
            chance = gf.HyperParameter(
                gf.Distribution(
                    type_=gf.DistributionType.UNIFORM, 
                    min_=0, 
                    max_=1
                )
            ),
            generator = gf.HyperParameter(
                    gf.Distribution(
                        type_=gf.DistributionType.CONSTANT, 
                        value = gen["generator"]
                )
            )
        ),
        "excluded" : gen["excluded"],
        "exclusive" : gen["exclusive"]
    } for name, gen in return_default_waveform_generators().items() }

    # Noise genes:
    noise_type = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.CHOICE, 
            possible_values=[
                gf.NoiseType.WHITE,
                gf.NoiseType.COLORED,
                gf.NoiseType.PSEUDO_REAL,
                gf.NoiseType.REAL,
            ],
        )
    )
    exclude_real_glitches = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.CHOICE, 
            possible_values=[
                True,
                False
            ]
        )
    )

    # Temporal Genes:
    onsource_duration_seconds = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.CHOICE, 
            possible_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        )
    )
    offsource_duration_seconds = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.UNIFORM, 
            min_=2, 
            max_=32,
            dtype=int
        )
    )
    sample_rate_hertz = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.POW_TWO, 
            min_=256,
            max_=8192,
            dtype=int
        )
    )

    #Feature engineering layers:
    num_layers = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=2, 
                max_=max_num_initial_layers+1, 
                dtype=int
            )
        )
    activations = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.CHOICE, 
                possible_values=[
                    'relu', 
                    'elu', 
                    'sigmoid', 
                    'tanh', 
                    'selu', 
                    'gelu',
                    'swish',
                    'softmax'
                ]
            )
        )
    d_units = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=1, 
                max_=128, 
                dtype=int
            )
        )
    filters = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM,
                min_=1, 
                max_=512, 
                dtype=int
            )
        )
    kernel_size = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=1, 
                max_=64, 
                dtype=int
            )
        )
    strides = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=1, 
                max_=64, 
                dtype=int
            )
        )
    dilation = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=0, 
                max_=64, 
                dtype=int
            )
        )
    pooling_present = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.CHOICE,
                possible_values = [True, False]
            )
        )
    pool_size = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=1, 
                max_=32, 
                dtype=int
            )
        )
    pool_stride = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM, 
                min_=1, 
                max_=32, 
                dtype=int
            )
        )
    dropout_present = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.CHOICE,
                possible_values = [True, False]
            )
        )
    dropout_value = gf.HyperParameter(
            gf.Distribution(
                type_=gf.DistributionType.UNIFORM,
                min_=0, 
                max_=1
            )
        )
    batch_normalisation_present = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.CHOICE,
            possible_values = [True, False]
        )
    )
    default_layer_type = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.CHOICE,
            possible_values=[
                gf.DenseLayer(d_units, dropout_present, dropout_value, batch_normalisation_present, activations),
                gf.ConvLayer(filters, kernel_size, activations, strides, dilation, dropout_present, dropout_value, batch_normalisation_present, pooling_present, pool_size, pool_stride)
            ]
        ),
    )
    whiten_layer = gf.HyperParameter(
        gf.Distribution(
            type_=gf.DistributionType.CHOICE,
            possible_values=[gf.WhitenLayer(), gf.WhitenPassLayer()]
        ),
    )

    layers = [whiten_layer]
    layers += [
        deepcopy(default_layer_type) for i in range(max_num_initial_layers)
    ]

    default_genome = gf.ModelGenome(
        optimizer=optimizer,
        num_layers=num_layers,
        batch_size=batch_size,
        learning_rate=learning_rate,
        injection_generators=waveform_generators,
        noise_type=noise_type,
        exclude_glitches=exclude_real_glitches,
        onsource_duration_seconds=onsource_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        sample_rate_hertz=sample_rate_hertz,
        layer_genomes=layers
    )

    default_genome.randomize()
    default_genome.mutate(0.05)

    return default_genome
