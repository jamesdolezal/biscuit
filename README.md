# BISCUIT <img src="https://i.imgur.com/VGK46TB.png" width="280px" align="right" />
Experimental files to accompany the manuscript submission  _Uncertainty Estimation Enables High-Confidence Deep Learning Predictions for Histopathology_.

_**What does BISCUIT do?** Bayesian Inference of Slide-level Confidence via Uncertainty Index Thresholding (BISCUIT) is a uncertainty quantification and thresholding schema used to separate deep learning classification predictions on whole-slide images (WSIs) into low- and high-confidence. Uncertainty is estimated through dropout, which approximates the Bayesian posterior, and thresholds are determined on training data to mitigate data leakage during testing._

## Disclaimer
These files and the accompanying package `biscuit` are considered pre-release. They have been published to assist with the review process. Further updates to expand generalizability and documentation are forthcoming.

# Requirements
- Python >= 3.7
- [Tensorflow](https://tensorflow.org) >=2.7.0 (and associated pre-requisites)
- [Slideflow](https://github.com/jamesdolezal/slideflow) 1.1* (and associated pre-requisites)
- Whole-slide images for training and validation

Please refer to our [Installation instructions](https://slideflow.dev/installation.html) for a guide to installing Slideflow and its preqrequisites.

*_**Note**: At the time of manuscript submission, Slideflow 1.1 has not been fully released. Please use Version 1.1 Release Candidate 1 (1.1rc1)._

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

Once all models have finished training (the published experiment included results from over 3000 models, so this may take a while), results can be viewed with the `results.py` script. The same experimental category flags, `--reg`, `--ratio`, and `--gan`, are used to determine which results should be viewed. There are two additional categories of results that can be displayed:

- `--heatmap`: Generate the heatmap shown in Figure 4.
- `--umaps`: Generates UMAPs shown in Figure 5.

Figures and output will then be saved in the `results/` folder. For example:

```
python3 results.py --ratio=True --umaps=True
```
