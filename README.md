# BISCUIT <img src="https://i.imgur.com/VGK46TB.png" width="280px" align="right" />
Experimental files to accompany _Uncertainty-Informed Deep Learning Models Enable High-Confidence Predictions for Digital Histopathology_. [Manuscript [ArXiv]](https://arxiv.org/abs/2204.04516)

_**What does BISCUIT do?** Bayesian Inference of Slide-level Confidence via Uncertainty Index Thresholding (BISCUIT) is a uncertainty quantification and thresholding schema used to separate deep learning classification predictions on whole-slide images (WSIs) into low- and high-confidence. Uncertainty is estimated through dropout, which approximates the Bayesian posterior, and thresholds are determined on training data to mitigate data leakage during testing._

## Disclaimer
These files and the accompanying package `biscuit` are considered pre-release. They have been published to assist with the review process. Further updates to expand generalizability and documentation are forthcoming.

# Requirements
- Python >= 3.7
- [Tensorflow](https://tensorflow.org) >=2.7.0 (and associated pre-requisites)
- [Slideflow](https://github.com/jamesdolezal/slideflow) 1.1* (and associated pre-requisites)
- Whole-slide images for training and validation

Please refer to our [Installation instructions](https://slideflow.dev/installation.html) for a guide to installing Slideflow and its preqrequisites.

*_**Note**: At the time of manuscript submission, Slideflow 1.1 has not been fully released. Please use Version 1.1 Release Candidate 1 (`slideflow==1.1.0rc1`)._

# Reproducing Results

## Data preparation
The first step to reproducing results described in our manuscript is downloading whole-slide images (\*.svs files) from [The Cancer Genome Atlas (TCGA) data portal](https://portal.gdc.cancer.gov/), projects TCGA-LUAD and TCGA-LUSC, and slides from the [Clinical Proteomics Tumor Analysis Consortium (CPTAC)](https://proteomics.cancer.gov/data-portal) data portal, projects TCGA-LUAD and TSCA-LSCC.

We use Slideflow for deep learning model training, which organizes data and annotations into [Projects](https://slideflow.dev/project_setup.html). The provided `configure.py` script automatically sets up the TCGA training and CPTAC evaluation projects, using specified paths to the training slides (TCGA) and evaluation slides (CPTAC). This step will also segment the whole-slide images into individual tiles, storing them as `*.tfrecords` for later use.

```
python3 configure.py --train_slides=/path/to/TCGA --val_slides=/path/to/CPTAC
```

Pathologist-annotated regions of interest (ROI) can optionally be used for the training dataset, as described in the [Slideflow documentation](https://slideflow.dev/pipeline.html). To use ROIs, specify the path to the ROI CSV files with the `--roi` argument.

## GAN Training
The next step is training the class-conditional GAN (StyleGAN2) used for generating GAN-Intermediate images. Clone the [StyleGAN2-slideflow](https://github.com/jamesdolezal/stylegan2-slideflow) repository, which has been modified to interface with the `*.tfrecords` storage format Slideflow uses. The GAN will be trained on 512 x 512 pixels images at 400 x 400 micron magnification. Synthetic images will be resized down to the target project size of 299 x 299 pixels and 302 x 302 microns during generation.

Use the `train.py` script **in the StyleGAN2 repository** to train the GAN. Pass the `gan_config.json` file that the `configure.py` script generated earlier to the `--slideflow` flag.

```
python3 train.py --outdir=/path/ --slideflow=/path/to/gan_cofig.json --mirror=1 --cond=1 --augpipe=bgcfnc --metrics=none
```

## Generating GAN images
To create GAN-Intermediate images with latent space embedding interpolation, use the `generate_tfrecords.py` script **in the StyleGAN2-slideflow** repository. Flags that will be relevant include:

- `--network`: Path to network PKL file (saved GAN model)
- `--tiles`: Number of tiles per tfrecord to generate (manuscript uses 1000)
- `--tfrecords`: Number of tfrecords to generate
- `--embed`: Generate intermediate images with class embedding interpolation.
- `--name`: Name format for tfrecords.
- `--class`: Class index, if not using embedding interpolation.
- `--outdir`: Directory in which to save tfrecords.

For example, to create tfrecords containing synthetic images of class 0 (LUAD / adenocarcinoma):

```
python3 generate_tfrecords.py --network=/path/network.pkl --tiles=1000 --tfrecords=10 --name=gan_luad --class=0 --outdir=gan/
```

To create embedding-interpolated intermediate images:

```
python3 generate_tfrecords.py --network=/path/network.pkl --tiles=1000 --tfrecords=10 --name=gan --embed=1 --outdir=gan/
```

Subsequent steps will assume that the GAN tfrecords are in the folder `gan/`.


## Cross-validation & evaluation
Next, models are trained with `train.py`. Experiments are organized by dataset size, each with a corresponding label. The experimental labels for this project are:

| ID | n_slides |
|----|----------|
| AA | full     |
| U  | 800      |
| T  | 700      |
| S  | 600      |
| R  | 500      |
| A  | 400      |
| L  | 350      |
| M  | 300      |
| N  | 250      |
| D  | 200      |
| O  | 176      |
| P  | 150      |
| Q  | 126      |
| G  | 100      |
| V  | 90       |
| W  | 80       |
| X  | 70       |
| Y  | 60       |
| Z  | 50       |
| ZA | 40       |
| ZB | 30       |
| ZC | 20       |
| ZD | 10       |

Experiments are performed in 6 steps for each dataset size:

1. Train cross-validation (CV) models for up to 10 epochs.
2. Train CV models at the optimal epoch (1).
3. Train UQ models in CV, saving predictions and uncertainty.
4. Train nested-UQ models, saving predictions, for uncertainty threshold determination.
5. Train models at the full dataset size without validation.
6. Perform external evaluation of fully-trained models.

We perform three types of experiments:

- `reg`: Regular experiments with balanced outcomes (LUAD:LUSC).
- `ratio`: Experiments testing varying degrees of class imbalance.
- `gan`: Cross-validation experiments using varying degrees of GAN slides in training/validation sets.

Specify which category of experiment should be run by setting its flag to `True`. Specify the steps to run using the `--steps` flag. For example, to run steps 2-6 for the ratio experiments, do:

```
python3 train.py --steps=2-6 --ratio=True
```

## Viewing results
Once all models have finished training (the published experiment included results from approximately 1000 models, so this may take a while), results can be viewed with the `results.py` script. The same experimental category flags, `--reg`, `--ratio`, and `--gan`, are used to determine which results should be viewed. There are two additional categories of results that can be displayed:

- `--heatmap`: Generate the heatmap shown in Figure 4.
- `--umaps`: Generates UMAPs shown in Figure 5.

Figures and output will then be saved in the `results/` folder. For example:

```
python3 results.py --ratio=True --umaps=True
```

# Custom projects

## Setting up a project
BISCUIT can also be used on your own data in custom projects. To create a new project, follow the [Project Setup](https://slideflow.dev/project_setup.html) instructions in the Slideflow documentation. Briefly, projects are initialized by creating an instance of the `slideflow.Project` class and require a pre-configured set of patient-level annotations in CSV format:

```python
import slideflow as sf

project = sf.Project(
    name='MyProject',
    annotations='/path/to/patient_annotations.csv'
)
```

Once the project is configured, add a new dataset source with paths to whole-slide images, optional tumor Regions of Interest (ROI) files, and destination paths for extracted tiles/tfrecords:

```python
project.add_source(
    name="TCGA_LUNG",
    slides="/path/to/slides",
    roi="/path/to/ROI",
    tiles="/tiles/destination",
    tfrecords="/tfrecords/destination"
)
```

This step should automatically attempt to associate slide names with the patient identifiers in your annotations CSV file. After this step, double check that your annotations file has a `"slide"` column for each annotation entry corresponding to the filename (without extension) of the corresponding slide. You should also ensure that the outcome labels you will be training to are correctly represented in this file.

## Extract tiles from slides
The next step is to [extract tiles](https://slideflow.dev/extract_tiles.html) from whole-slide images, using the `sf.Project.extract_tiles()` function. This will save image tiles in the binary `*.tfrecord` format in the destination folder you previously configured.

```python
project.extract_tiles(
    tile_px=299,  # Tile size in pixels
    tile_um=302   # Tile size in microns
)
```

A PDF report summarizing the tile extraction phase will be saved in the TFRecords directory.

## Train models in cross-validation
Next, train models in cross-validation using uncertainty quantification (UQ), which estimates uncertainty via dropout. Model hyperparameters can be manually configured with `sf.model.ModelParams`. Alternatively, the hyperparameters we used in the above manuscript can be accessed via `biscuit.hp.nature2022`. The `uq` parameter should be set to `True` to enable UQ.

```python
import biscuit

hp = biscuit.hp.nature2022
hp.uq = True
```

Now we can train models using labels provided by the project annotations file. The labels we will train to will be referenced with the argument `outcome`, which should indicate the annotations column header with the outcome labels.

```python
biscuit.train(
    project=project,
    outcome="some_header",  # Annotations header with labels
    hp=hp,                  # Hyperparameters
    label="EXPERIMENT"      # Experiment label/ID
    save_predictions=True   # Saves predictions in CSV format
)
```

## Train nested cross-validation models for UQ thresholds
After the outer cross-validation models have been trained, the inner cross-validation models are trained so that optimal UQ thresholds can be found. Initialize the nested cross-validation training with the following:

```python
biscuit.train_nested_cv(
    project=project,
    outcome="some_header",
    hp=hp,
    label="EXPERIMENT"
)
```

The experimental results for each cross-fold can either be manually viewed by opening `results_log.csv` in each model directory, or with the following functions:

```python
cv_models = biscuit.find_cv(
    project=project,
    label="EXPERIMENT",
    outcome="some_header"
)
for m in cv_models:
    results = biscuit.get_model_results(m, outcome="some_header"))
    print(m, results['pt_auc'])  # Prints patient-level AUC for each model
```

## Calculate UQ thresholds and show results
Finally, UQ thresholds are determined from the previously trained nested cross-validation models. Use `biscuit.thresholds_from_nested_cv()` to calculate optimal thresholds, and then apply these thresholds to the outer cross-validation data, rendering high-confidence predictions.

```python
df, thresh = biscuit.thresholds_from_nested_cv(
    project=project,
    label="EXPERIMENT",
    outcome="some_header"
)
```

`thresh` will be a dictionary of tile- and slide-level UQ thresholds, and the slide-level prediction threshold. `df` is a pandas DataFrame containing the thresholded, high-confidence UQ predictions from outer cross-validation.

```python
>>> print(df)
     id  n_slides  fold       uq  patient_auc  patient_uq_perc  slide_auc  slide_uq_perc
0  TEST     359.0   1.0  include     0.974119         0.909091   0.974119       0.909091
1  TEST     359.0   2.0  include     0.972060         0.840336   0.972060       0.840336
2  TEST     359.0   3.0  include     0.901786         0.873950   0.901786       0.873950
>>> print(thresh)
{'tile_uq': 0.008116906, 'slide_uq': 0.0023400568179163194, 'slide_pred': 0.17693227693333335}
```

## Visualize uncertainty calibration
Plots can be generated showing the relationship between predictions and uncertainty, as shown in Figure 3 of the manuscript. The `biscuit.plot_uq_calibration()` function will generate these plots, which can then be shown using `plt.show()`:

```python
import matplotlib.pyplot as plt

biscuit.plot_uq_calibration(
    project=project,
    label="EXPERIMENT",
    outcome="some_header",
    **thresh  # Pass the thresholds from the prior step
)
```

## Full example
