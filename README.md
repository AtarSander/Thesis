# Running experiments

## Working with infinity

For Infinity's codebase relative paths to work correctly we need to run its functions inside the `Infinity` directory (at least that's the easiest way I found). That's why there is `os.chdir("Infinity")`before the initial Infinty configuration. If you follow similar file to structure to one in this repository, you need to then specify file paths relative to `Infinity` as the working directory (as you see, in `experiment_7` all paths after the initial Infinity configuration are specified with first going back to the parent directory, e.g. `../experiment_7/weights/unchanged.pth"`).

## Prototype experiments

To run the experiments using `steering.py` script you need to satisfy a couple of rules:

### Directory structure

You need to either use the existing `experiment_7` directory, or create a new one with the same parent as `Infinity` directory, so:

```
**IAR-ALIGMENT**
├── **data**
├── **experiment_directory**
│       ├── **utils**
│       │── **setup**
│       │── **weights**
│       └── **results**
└── **Infinity**
```

where:

- **data** - place to store json files with prompts for the model
- **utils** - helper function and classes for prototype steering (copy from `experiment_7`)
- **setup** - place to store json steering setups
- **weights** - place to store saved weights for classifier or generated images
- **results** - place to store generated images with side by side generation's comparation
- **Infinity** - infinity codebase, taken from [Infinity repository](https://github.com/FoundationVision/Infinity)

### Infinity configuration

YAML file that configures which Infinity model to use for experiments. It has seven fields which need to match each other for the model to work correctly. Setup for infinity_125M_256x256 can be found in `experiment_7/infinity_configuration.yml`, to change the model see [Infinity readme](https://github.com/FoundationVision/Infinity/blob/main/README.md).

### Experiment configuration

YAML file theat configures the prototype steering setup.

- DATASET - variant of the dataset, (dogs_cats, shapes_colors)
- DATASET_PATH - path to a json file with prompts
- STEERING_TYPE - type of steering to apply (TEXT, VISUAL, BOTH)
- STEERING_LOCATION - on what layers to apply steering (ffn, ca, ca_block)
- VARIANT - whether to steer from class 0 into 1 or the other way around, target class (0, 1)
- BATCH_SIZE - batch size of the experiment's generations
- CFG - cfg of the infinity model
- DISTANCE - type of distance for prototype steering (EUCLIDEAN, MAHALANOBIS)
- STEERING_MODE - whether to apply the steering only to conditional, unconditional or both parts (CONDITIONAL, UNCONDITIONAL, BOTH)
- EXPERIMENT_DIR - main experiments directory
- EXPERIMENT_SETUP - file in **setup** dir specifying the steering locations (see expected format below)
- GRID_PATH - results grid image filename
- CLASSIFIER_WEIGHTS - name of the classifier weights `.pth` file, in **weights** dir
- UNCHANGED_SAVED - name of the file in **weights** dir with saved baseline images so that the baseline is consistent across all of the experiments, if empty program will automatically generate baseline images and save them to `weights/unchanged.pth`
- FID - whether to calculate FID of the generated images (considerably longer generation if on)

### Experiments setup format

In this file we specify the location and strength of steering. Example format:

```
{
        "scales": {
            "0": {
                "layers": [
                    -1
                ],
                "strength_cond": 3.0,
                "strength_uncond": 1.0
            },
            "1": {
                "layers": [
                    -1
                ],
                "strength_cond": 5.0,
                "strength_uncond": 2.0
            },
            "2": {
                "layers": [
                    0,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9
                ],
                "strength_cond": 3.0,
                "strength_uncond": 1.0
            },
            "3": {
                "layers": [
                    0,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9
                ],
                "strength_cond": 3.0,
                "strength_uncond": 1.0
            },
            "4": {
                "layers": [],
                "strength_cond": 0.0,
                "strength_uncond": 0.0
            },
            "5": {
                "layers": [],
                "strength_cond": 0.0,
                "strength_uncond": 0.0
            },
            "6": {
                "layers": [],
                "strength_cond": 0.0,
                "strength_uncond": 0.0
            }
        },
        "layers_text": [21, 22, 23],
        "strength_text": 5.0
    },
```

For infinity_125M_256x256 there are 7 scales (0-6) and 12 layers (0-11) in which we can specify the steering. We can also select strength different strengths for every layer. `"layers": [-1]` means that steering will be applied in all layers at given scale. You can programatically generate these setup files using `grid_search_steering.py`. Optionally we can also specify strength of the text steering (layers are not working at the moment).

### Running the experiment script !important

After satisfying all of the rules above you can run the experiments script:

```(bash)
python experiment_directory/steering.py --config experiment_directory/experiment_configuration.yml --infinity experiment_directory/infinity_configuration.yml
```

Remember to run it from the parent directory!
