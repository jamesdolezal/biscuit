import os
import click
import slideflow as sf
from biscuit import experiment
from os.path import exists, join, abspath


# ----------------------------------------------------------------------------

@click.command()
@click.option('--train_slides', type=str, help='Directory to training slides, for cross-validation', required=True)
@click.option('--val_slides', type=str, help='Directory to external evaluation slides, for evaluation')
@click.option('--train_anns', type=str, help='Directory to annotation file for training data (CSV)', default='annotations/tcga.csv', show_default=True)
@click.option('--val_anns', type=str, help='Directory to annotation file for training data (CSV)', default='annotations/cptac.csv', show_default=True)
@click.option('--train_roi', type=str, help='Directory to CSV ROI files, for cross-validation')
def configure_projects(train_slides, train_anns, train_roi, val_slides=None, val_anns=None, out='projects'):

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
            sources='Training',
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
            raise ValueError("If providing evaluation slides, evaluation annotations must also be provided (--val_anns)")
        if (not exists(join(out, 'evaluation'))
           or not exists(join(out, 'evaluation', 'settings.json'))):
            print("Setting up evaluation project.")
            eP = sf.Project(
                join(out, 'evaluation'),
                sources='Evaluation',
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
            qc=True,
            img_format='png'
        )
    print("Extracting tiles from WSIs at 512px, 400um (for GAN training)")
    for P in (eP, tP):
        P.extract_tiles(
            tile_px=512,
            tile_um=400,
            qc=True,
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
            "outcome": [
                experiment.OUTCOME
            ],
            "filters": {
                experiment.OUTCOME: [
                    experiment.OUTCOME1,
                    experiment.OUTCOME2
                ]
            }
        }
        sf.util.write_json(gan_config, 'gan_config.json')
        print("Wrote GAN configuration to gan_config.json")
    else:
        print("GAN configuration already exists at gan_config.json")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    configure_projects()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------