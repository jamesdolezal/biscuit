"""Show results from trained UQ experiments."""

import os
import click
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from statistics import mean
from os.path import join, exists

import slideflow as sf
from biscuit import utils, threshold, Experiment
from biscuit.errors import ModelNotFoundError
from biscuit.experiment import ALL_EXP

# -----------------------------------------------------------------------------

@click.command()
@click.option('--train_project', default='projects/training', type=str, help='Set training project')
@click.option('--eval_project', default='projects/evaluation', type=str, help='Set eval project')
@click.option('--outcome', type=str, help='Outcome (annotation header) that assigns class labels.', default='cohort', show_default=True)
@click.option('--outcome1', type=str, help='First class label.', default='LUAD', show_default=True)
@click.option('--outcome2', type=str, help='Second class label.', default='LUSC', show_default=True)
@click.option('--reg', type=bool, help='Generate results from standard experiments', default=False)
@click.option('--ratio', type=bool, help='Generate results from the unbalanced/ratio experiments', default=False)
@click.option('--gan', type=bool, help='Generate results from the GAN experiments', default=False)
@click.option('--umaps', type=bool, help='Generate UMAPs', default=False)
@click.option('--heatmap', type=bool, help='Generate heatmaps', default=False)
@click.option('--heatmap_slide', type=str, default='C3N-01417-23')
def show_results(
    train_project,
    eval_project,
    outcome,
    outcome1,
    outcome2,
    reg=False,
    ratio=False,
    gan=False,
    umaps=False,
    heatmap=False,
    heatmap_slide='C3N-01417-23'
):
    """Show results from trained experiments.

    This script uses the Slideflow projects configured with configure.py and
    trained with train.py. The projects, which by default are expected to be
    in projects/training and projects/evaluation, can be overwritten by setting
    train_project and eval_project.

    Experiments are divided into the following groups:

    Regular (reg): Standard experiments with balanced ratios of adenocarcinoma
    and squamous cell carcinoma slides. These include the cross-validation
    training experiments and the external evaluation experiments
    shown in Figures 1, 2 and 4.

    Ratio (ratio): Unbalanced experiments testing the effect of varying ratios
    of adenocarcinoma:squamous cell carcinoma. These results are shown in
    Figure 3.

    Heatmaps (heatmap): Generate the heatmap shown in Figure 5. By default,
    this will use the same slide published in the manuscript (CPTAC slide
    C3N-01417-23). You may, however, use a different slide by setting
    the heatmap_slide argument.

    UMAPs (umap): Generate the UMAPs shown in Figure 6.

    GAN (gan): Experiments testing effect of adding synthetic GAN-intermediate
    "slides" into training data, with and without uncertainty thresholding.
    These results are shown in Figure 7.
    """

    if not exists('results'):
        os.makedirs('results')

    # === Configure experiments ===============================================
    if not exists(join(eval_project, 'settings.json')):
        print(f"Evaluation project not found at {eval_project}; ignoring")
        eval_project = []
    else:
        eval_project = [eval_project]
    experiment = Experiment(
        train_project,
        eval_projects=eval_project,
        outcome=outcome,
        outcome1=outcome1,
        outcome2=outcome2,
        outdir='results')

    # Configure regular experiments
    reg1 = experiment.config('{}', ALL_EXP, 1, order='f')
    reg1.update(experiment.config('{}_R', ALL_EXP, 1, order='r'))
    reg2 = experiment.config('{}2', ALL_EXP, 1, order='f', order_col='order2')
    reg2.update(experiment.config('{}_R2', ALL_EXP, 1, order='r', order_col='order2'))
    all_reg = copy.deepcopy(reg1)
    all_reg.update(reg2)

    # Configure 3:1 and 10:1 ratio experiments
    r_list = list('AMDPGZ')
    ratio_3 = experiment.config('{}_3', r_list, 3, order='f')
    ratio_3.update(experiment.config('{}_R_3', r_list, 3, order='r'))
    ratio_10 = experiment.config('{}_10', r_list, 10, order='f')
    ratio_10.update(experiment.config('{}_R_10', r_list, 10, order='r'))

    # GAN experiments
    g = list('RALMNDOPQGWY') + ['ZA', 'ZC']
    gan_exp = {}
    gan_exp.update(experiment.config('{}_g10', g, 1, gan=0.1, order='f'))
    gan_exp.update(experiment.config('{}_R_g10', g, 1, gan=0.1, order='r'))
    gan_exp.update(experiment.config('{}_g20', g, 1, gan=0.2, order='f'))
    gan_exp.update(experiment.config('{}_R_g20', g, 1, gan=0.2, order='r'))
    gan_exp.update(experiment.config('{}_g30', g, 1, gan=0.3, order='f'))
    gan_exp.update(experiment.config('{}_R_g30', g, 1, gan=0.3, order='r'))
    gan_exp.update(experiment.config('{}_g40', g, 1, gan=0.4, order='f'))
    gan_exp.update(experiment.config('{}_R_g40', g, 1, gan=0.4, order='r'))
    gan_exp.update(experiment.config('{}_g50', g, 1, gan=0.5, order='f'))
    gan_exp.update(experiment.config('{}_R_g50', g, 1, gan=0.5, order='r'))

    # === Show results ========================================================
    # --- Full experiment -----------------------------------------------------
    # Figures 1, 2 and 4
    if reg:
        print("Calculating results for regular experiments")
        df, eval_dfs = experiment.results(all_reg, uq=True, plot=True)
        experiment.display(df, eval_dfs, prefix='reg_')
    else:
        print("Skipping results for regular experiments")

    # --- Cross-val ratio results ---------------------------------------------
    # Figure 3
    if ratio:
        print("Calculating results for ratio experiments")
        r1_df, _ = experiment.results(reg1, uq=True, eval=False)
        r3_df, _ = experiment.results(ratio_3, uq=True, eval=False)
        r10_df, _ = experiment.results(ratio_10, uq=True, eval=False)

        r1_df['ratio'] = ['1' for _ in range(len(r1_df))]
        r3_df['ratio'] = ['3' for _ in range(len(r3_df))]
        r10_df['ratio'] = ['10' for _ in range(len(r10_df))]

        df = pd.concat([r1_df, r3_df], axis=0, join='outer', ignore_index=True)
        df = pd.concat([df, r10_df], axis=0, join='outer', ignore_index=True)

        try:
            n_slides_in_r10 = np.unique(r10_df['n_slides'].to_numpy())
            df = df.loc[df['n_slides'].isin(n_slides_in_r10)]
        except KeyError:
            print("Ratio training not yet done - unable to show results")
        else:
            print("Ratio Comparison")
            experiment.display(
                df.loc[df['uq'] != 'include'],
                None,
                hue='ratio',
                palette='Set1',
                prefix='ratio_comparison_'
            )
            print("Ratio 1:3")
            experiment.display(r3_df, None, hue='uq', prefix='ratio3_')
            print("Ratio 1:10")
            experiment.display(r10_df, None, hue='uq', prefix='ratio10_')
            df.to_csv(join('results', 'ratio_results.csv'))
    else:
        print("Skipping results for ratio experiments")

    if umaps or heatmap:
        # Load the external evaluation project and find the fully trained model
        P = experiment.train_project
        if not len(experiment.eval_projects):
            raise ValueError("Evaluation project not configured.")
        cP = experiment.eval_projects[0]
        if not utils.model_exists(P, 'EXP_AA_FULL', outcome=experiment.outcome):
            raise ModelNotFoundError("Couldn't find trained model EXP_AA_FULL")
        aa_model = utils.find_model(P, 'EXP_AA_FULL', outcome=experiment.outcome, epoch=1)

        all_tile_uq_thresh = []
        for k in range(1, 4):
            tile_uq = threshold.from_cv(
                utils.df_from_cv(P, f'EXP_AA_UQ-k{k}', outcome=experiment.outcome, k=5),
                tile_uq='detect',
                slide_uq=None,
                patients=P.dataset().patients()
            )['tile_uq']
            all_tile_uq_thresh += [tile_uq]
        aa_tile_uq_thresh = mean(all_tile_uq_thresh)

    # --- Heatmap -------------------------------------------------------------
    # Figure 5
    if heatmap:
        print("Generating heatmap")
        # Use slide if directly provided
        if os.path.exists(heatmap_slide):
            slide = heatmap_slide
        # Otherwise, search for this name in the eval dataset
        else:
            eval_dts = cP.dataset(
                tile_px=299,
                tile_um=302,
                filters={'slide': [heatmap_slide]}
            )
            matching_slide_paths = eval_dts.slide_paths()
            if not len(matching_slide_paths):
                msg = f"Heatmap: could not find slide {heatmap_slide}"
                raise ValueError(msg)
            slide = matching_slide_paths[0]
        if not exists(join('results', 'heatmap_full')):
            os.makedirs(join('results', 'heatmap_full'))
        if not exists(join('results', 'heatmap_high_confidence')):
            os.makedirs(join('results', 'heatmap_high_confidence'))

        # --- Figure 5a -----
        # Save the regular heatmap with predictions
        hm = sf.Heatmap(slide, aa_model, stride_div=1)
        hm.save(
            join('results', 'heatmap_full'),
            cmap=utils.truncate_colormap(plt.get_cmap('PRGn'), 0.1, 0.9)
        )
        # Save the heatmap with masked, high-confidence predictions
        uq_mask = hm.uncertainty[:, :, 0] > aa_tile_uq_thresh
        hm.logits[uq_mask, :] = [-1, -1]
        hm.save(
            join('results', 'heatmap_high_confidence'),
            cmap=utils.truncate_colormap(plt.get_cmap('PRGn'), 0.1, 0.9)
            )
        # --- Figure 5b -----
        # Save the highest and lowest uncertainty tiles
        if not exists(join('results', 'uq_excl')):
            os.makedirs(join('results', 'uq_excl'))
        if not exists(join('results', 'uq_incl')):
            os.makedirs(join('results', 'uq_incl'))
        interface = sf.model.tensorflow.UncertaintyInterface(aa_model)
        wsi = sf.WSI(slide, 299, 302, roi_method='ignore')
        try:
            gen = wsi.build_generator(
                shuffle=False,
                include_loc='grid',
                show_progress=True
            )
            loc_key = 'loc'
        except TypeError: # Slideflow 1.2+ compatibility
            gen = wsi.build_generator(
                shuffle=False,
                show_progress=True
            )
            loc_key = 'grid'
        for tile in gen():
            image = tile['image']
            if interface.wsi_normalizer:
                norm_image = interface.wsi_normalizer.rgb_to_rgb(image)
            else:
                norm_image = image
            parsed = tf.image.per_image_standardization(norm_image)
            parsed.set_shape([wsi.tile_px, wsi.tile_px, 3])
            logits, uncertainty = interface(tf.expand_dims(parsed, axis=0))
            u = uncertainty[0][0]
            tilename = f"{u:.4f}-{tile[loc_key][0]}-{tile[loc_key][1]}.png"
            if uncertainty[0][0] > aa_tile_uq_thresh:
                img = Image.fromarray(tile['image'])
                img.save(join('results', 'uq_excl', tilename))
            else:
                img = Image.fromarray(tile['image'])
                img.save(join('results', 'uq_incl', tilename))
    else:
        print("Skipping heatmap")

    # --- Plot UMAPs (Figure 6) -----------------------------------------------
    if umaps:
        print("Generating UMAPs")
        filters = {'cohort': ['LUAD', 'LUSC']}
        df = cP.generate_features(aa_model, filters=filters, max_tiles=10, cache='act.pkl')
        mosaic = cP.generate_mosaic(df, filters=filters, umap_cache='umap.pkl', use_norm=False)

        # Figure 6a
        mosaic.save(join('results', 'mosaic.png'))

        # Figure 6b
        mosaic.slide_map.label_by_logits(1)
        mosaic.slide_map.save(join('results', 'umap_preds.svg'), s=10)

        # Figure 6c
        if hasattr(mosaic.slide_map, 'label_by_meta'):
            mosaic.slide_map.label_by_meta('prediction')
        else:  # Slideflow 1.2+ compatibility
            mosaic.slide_map.label('prediction')
        mosaic.slide_map.save(join('results', 'umap_binary_pred.svg'), s=10)

        # Figure 6d
        mosaic.slide_map.label_by_uncertainty()
        mosaic.slide_map.save(
            join('results', 'umap_uncertainty.svg'), s=10, hue_norm=(0, 0.15)
        )

        # Figure 6e
        if hasattr(mosaic.slide_map, 'labels'):
            mosaic.slide_map.labels = mosaic.slide_map.labels < aa_tile_uq_thresh
        else:  # Slideflow 1.2+ compatibility
            mosaic.slide_map.data.label = mosaic.slide_map.data.label < aa_tile_uq_thresh
        mosaic.slide_map.save(join('results', 'umap_confidence.svg'), s=10)

        # Showing ground-truth labels
        labels, _ = cP.dataset().labels('cohort')
        mosaic.slide_map.label_by_slide(labels)
        mosaic.slide_map.save(join('results', 'umap_labels.svg'), s=10)
    else:
        print("Skipping UMAPs")

    # --- Analyze GAN (overview, non-UQ) (Figure 7)----------------------------
    if gan:
        print("Calculating results for GAN experiments")
        gan_df, _ = experiment.results(gan_exp, uq=True, plot=False)
        if not len(gan_df):
            print("Unable to find GAN results.")
        else:
            reg_df, _ = experiment.results(reg1, uq=True, plot=False)
            reg_df = reg_df.loc[((reg_df['uq'] != 'include')
                                & (reg_df['n_slides'] <= 500))]
            reg_df['gan_exp'] = 'none'
            gan_df['gan_exp'] = gan_df['id'].str[-3:]
            gan_df = gan_df.loc[gan_df['uq'] != 'include']
            gan_df = pd.concat([gan_df, reg_df], axis=0, join='outer', ignore_index=True)
            experiment.display(
                gan_df,
                None,
                hue='gan_exp',
                relplot_uq_compare=False,
                prefix='gan_'
            )
            # --- Show GAN results (n=500 with UQ) --------------------------------
            r_exp = experiment.config('{}', ['R'], 1, order='f')
            r_exp.update(experiment.config('{}_R', ['R'], 1, order='r'))
            r_df, _ = experiment.results(r_exp)
            gan_df, _ = experiment.results(gan_exp)
            gan_df = gan_df.loc[gan_df['id'].str[0] == 'R']
            gan_df = pd.concat([gan_df, r_df], axis=0, join='outer', ignore_index=True)
            gan_df = gan_df.loc[gan_df['uq'].isin(['all', 'include'])]
            experiment.display(
                gan_df,
                None,
                boxplot_uq_compare=False,
                ttest_uq_groups=['all', 'include'],
                prefix='gan_uq_'
            )
    else:
        print("Skipping GAN experiment results")

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    show_results()  # pylint: disable=no-value-for-parameter
