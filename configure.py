"""Configure Slideflow projects for reproducing published results."""

import os
import click
import slideflow as sf
from biscuit import experiment
from os.path import exists, join, abspath

# -----------------------------------------------------------------------------

@click.command()
@click.option('--train_slides', type=str, help='Directory to training slides, for cross-validation', required=True)
@click.option('--train_anns', type=str, help='Directory to annotation file for training data (CSV)', default='annotations/tcga.csv', show_default=True)
@click.option('--train_roi', type=str, help='Directory to CSV ROI files, for cross-validation')
@click.option('--outcome', type=str, help='Outcome (annotation header) that assigns class labels.', default='cohort', show_default=True)
@click.option('--outcome1', type=str, help='First class label.', default='LUAD', show_default=True)
@click.option('--outcome2', type=str, help='Second class label.', default='LUSC', show_default=True)
@click.option('--val_slides', type=str, help='Directory to external evaluation slides, for evaluation')
@click.option('--val_anns', type=str, help='Directory to annotation file for training data (CSV)', default='annotations/cptac.csv', show_default=True)
def configure_projects(
    train_slides,
    train_anns,
    train_roi,
    outcome,
    outcome1,
    outcome2,
    val_slides=None,
    val_anns=None,
):
    """Configure Slideflow projects for reproducing published results.

    This script uses the provided slides to build Slideflow projects in the
    'projects/' folder of the current working directory. Clinical annotations
    (class labels) are read from the 'annotations/' folder unless otherwise
    specified.

    Training slides from The Cancer Genome Atlas (TCGA) are available at
    https://portal.gdc.cancer.gov/ (projects TCGA-LUAD and TCGA-LUSC).

    Validation slides from the Clinical Proteomics Tumor Analysis Consortium
    (CPTAC) are available at https://proteomics.cancer.gov/data-portal
    (projects CPTAC-LUAD and CPTAC-LSCC).
    """

    # Absolute paths
    train_slides = abspath(train_slides)
    train_anns = abspath(train_anns)
    out = abspath('projects')
    if val_slides:
        val_slides = abspath(train_slides)
    if val_anns:
        val_anns = abspath(val_anns)
    if train_roi:
        train_roi = abspath(train_roi)
    gan_path = abspath('gan')
    if not exists(gan_path):
        os.makedirs(gan_path)

    # --- Set up projects -----------------------------------------------------

    # Set up training project
    if (not exists(join(out, 'training'))
       or not exists(join(out, 'training', 'settings.json'))):
        print("Setting up training project...")
        tP = sf.Project(
            join(out, 'training'),
            sources=['Training'],
            annotations=train_anns
        )
        tP.add_source(
            name='Training',
            slides=train_slides,
            roi=(train_roi if train_roi else train_slides),
            tiles=join(out, 'training', 'tiles'),
            tfrecords=join(out, 'training', 'tfrecords')
        )
        tP.add_source(
            name='LUNG_GAN',
            slides=gan_path,
            roi=gan_path,
            tiles=gan_path,
            tfrecords=gan_path
        )
        print(f"Training project setup at {join(out, 'training')}.")
    else:
        tP = sf.Project(join(out, 'training'))
        print("Loading training project which already exists.")

    # Set up external evaluation project
    if val_slides:
        if not val_anns:
            msg = "If providing evaluation slides, evaluation annotations "
            msg += "must also be provided (--val_anns)"
            raise ValueError(msg)
        if (not exists(join(out, 'evaluation'))
           or not exists(join(out, 'evaluation', 'settings.json'))):
            print("Setting up evaluation project.")
            eP = sf.Project(
                join(out, 'evaluation'),
                sources=['Evaluation'],
                annotations=val_anns
            )
            eP.add_source(
                name='Evaluation',
                slides=val_slides,
                roi=val_slides,
                tiles=join(out, 'evaluation', 'tiles'),
                tfrecords=join(out, 'evaluation', 'tfrecords')
            )
            print(f"Evaluation project setup at {join(out, 'evaluation')}.")
        else:
            eP = sf.Project(join(out, 'evaluation'))
            print("Loading evaluation project which already exists.")

    # --- Perform tile extraction ---------------------------------------------

    print("Extracting tiles from WSIs at 299px, 302um")
    for P in (eP, tP):
        P.extract_tiles(
            tile_px=299,
            tile_um=302,
            qc='both',
            img_format='png'
        )
    print("Extracting tiles from WSIs at 512px, 400um (for GAN training)")
    for P in (eP, tP):
        P.extract_tiles(
            tile_px=512,
            tile_um=400,
            qc='both',
            img_format='png'
        )
    print("Finished tile extraction, project configuration complete.")

    # --- Save GAN training configuration -------------------------------------

    if not exists('gan_config.json'):
        gan_config = {
            "project_path": join(out, 'training'),
            "tile_px": 512,
            "tile_um": 400,
            "model_type": "categorical",
            "outcomes": [outcome],
            "filters": {outcome: [outcome1, outcome2]}
        }
        sf.util.write_json(gan_config, 'gan_config.json')
        print("Wrote GAN configuration to gan_config.json")
    else:
        print("GAN configuration already exists at gan_config.json")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    configure_projects()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
