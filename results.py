import os
import click
import experiment
import copy
import numpy as np
import utils
import slideflow as sf
import matplotlib.pyplot as plt
import tensorflow as tf
import threshold
from errors import *

from PIL import Image
from statistics import mean
from slideflow.util import log
from os.path import join, exists
from experiment import EXP_NAME_MAP

@click.command()
@click.option('--train_project', type=str, help='Manually specify location of training project')
@click.option('--eval_project', type=str, help='Manually specify location of training project')
@click.option('--reg', type=bool, help='Show results from regular models', default=False, show_default=True)
@click.option('--ratio', type=bool, help='Show results from ratio experiments', default=False, show_default=True)
@click.option('--gan', type=bool, help='Show results from gan experiments', default=False, show_default=True)
@click.option('--umaps', type=bool, help='Show results from gan experiments', default=False, show_default=True)
@click.option('--heatmap', type=bool, help='Show heatmap predictions', default=False, show_default=True)
@click.option('--heatmap_slide', type=str, default='C3N-01417-23', help='Slide for which to generate a heatmap')
def show_results(
    train_project = None,
    eval_project  = None,
    reg           = False,
    ratio         = False,
    gan           = False,
    umaps         = False,
    heatmap       = False,
    heatmap_slide = 'C3N-01417-23' ):

    if not exists('results'):
        os.makedirs('results')

    # === Configure experiments ===============================================

    experiment.OUT = 'results'
    experiment.TRAIN_PATH = join('projects', 'training') if train_project is None else train_project
    experiment.EVAL_PATHS = [join('projects', 'evaluation')] if eval_project is None else [eval_project]

    # Configure regular experiments
    reg1_exp = experiment.config('{}', EXP_NAME_MAP, 1, order='forward')
    reg1_exp.update(experiment.config('{}_R', EXP_NAME_MAP, 1, order='reverse'))
    reg2_exp = experiment.config('{}2', EXP_NAME_MAP, 1, order='forward')
    reg2_exp.update(experiment.config('{}_R2', EXP_NAME_MAP, 1, order='reverse'))
    all_reg_exp = copy.deepcopy(reg1_exp)
    all_reg_exp.update(reg2_exp)

    # Configure 3:1 and 10:1 ratio experiments
    ratio_3 = experiment.config('{}_3', ['A', 'M', 'D', 'P', 'G', 'Z'], 3, order='forward')
    ratio_3.update(experiment.config('{}_R_3', ['A', 'M', 'D', 'P', 'G', 'Z'], 3, order='reverse'))
    ratio_10 = experiment.config('{}_10', ['A', 'M', 'D', 'P', 'G', 'Z'], 10, order='forward')
    ratio_10.update(experiment.config('{}_R_10', ['A', 'M', 'D', 'P', 'G', 'Z'], 10, order='reverse'))

    # GAN experiments
    gan_exp_list = ['R', 'A', 'L', 'M', 'N', 'D', 'O', 'P', 'Q', 'G', 'W', 'Y', 'ZA', 'ZC']
    gan_exp = {}
    gan_exp.update(experiment.config('{}_g10', gan_exp_list, 1, gan=0.1, order='forward'))
    gan_exp.update(experiment.config('{}_R_g10', gan_exp_list, 1, gan=0.1, order='reverse'))
    gan_exp.update(experiment.config('{}_g20', gan_exp_list, 1, gan=0.2, order='forward'))
    gan_exp.update(experiment.config('{}_R_g20', gan_exp_list, 1, gan=0.2, order='reverse'))
    gan_exp.update(experiment.config('{}_g30', gan_exp_list, 1, gan=0.3, order='forward'))
    gan_exp.update(experiment.config('{}_R_g30', gan_exp_list, 1, gan=0.3, order='reverse'))
    gan_exp.update(experiment.config('{}_g40', gan_exp_list, 1, gan=0.4, order='forward'))
    gan_exp.update(experiment.config('{}_R_g40', gan_exp_list, 1, gan=0.4, order='reverse'))
    gan_exp.update(experiment.config('{}_g50', gan_exp_list, 1, gan=0.5, order='forward'))
    gan_exp.update(experiment.config('{}_R_g50', gan_exp_list, 1, gan=0.5, order='reverse'))


    # === Show results ========================================================

    # --- Show results from full experiment -----------------------------------
    # Figures 1 and 3
    if reg:
        experiment.display(*experiment.results(all_reg_exp, uq=True, plot=True), prefix='reg_')

    # --- Show cross-val ratio results ----------------------------------------
    # Figure 2
    if ratio:
        r1_df, _ = experiment.results(reg1_exp, uq=True, eval=False)
        r3_df, _ = experiment.results(ratio_3, uq=True, eval=False)
        r10_df, _ = experiment.results(ratio_10, uq=True, eval=False)

        r1_df['ratio'] = ['1' for _ in range(len(r1_df))]
        r3_df['ratio'] = ['3' for _ in range(len(r3_df))]
        r10_df['ratio'] = ['10' for _ in range(len(r10_df))]

        df = r1_df.append(r3_df, ignore_index=True)
        df = df.append(r10_df, ignore_index=True)

        n_slides_in_r10 = np.unique(r10_df['n_slides'].to_numpy())
        df = df.loc[df['n_slides'].isin(n_slides_in_r10)]
        print("Ratio Comparison")
        experiment.display(df.loc[df['uq']!='include'], None, hue='ratio', palette='Set1', prefix='ratio_comparison_')
        print("Ratio 1:3")
        experiment.display(r3_df, None, hue='uq', prefix='ratio3_')
        print("Ratio 1:10")
        experiment.display(r10_df, None, hue='uq', prefix='ratio10_')

    if umaps or heatmap:
        # Load the external evaluation project and find the fully trained model
        P = sf.Project(experiment.TRAIN_PATH)
        cP = sf.Project(experiment.EVAL_PATHS[0])
        if not utils.model_exists(P, f'EXP_AA_FULL'):
            raise ModelNotFoundError("Could not find trained model EXP_AA_FULL.")
        aa_model = utils.find_model(P, f'EXP_AA_FULL', epoch=1)

        # Get tile uncertainty threshold
        threshold_params = {
            'y_pred_header':        utils.y_pred_header,
            'y_true_header':        utils.y_true_header,
            'uncertainty_header':   utils.uncertainty_header,
            'patients':             P.dataset().patients()
        }
        all_tile_uq_thresh = []
        for k in range(1, 4):
            k_preds = [join(folder, 'tile_predictions_val_epoch1.csv') for folder in utils.find_cv(P, f'EXP_AA_UQ-k{k}', k=5)]
            tile_thresh, _, _, _ = threshold.from_cv(k_preds, tile_uq_thresh='detect', slide_uq_thresh=None, **threshold_params)
            all_tile_uq_thresh += [tile_thresh]
        aa_tile_uncertainty_threshold = mean(all_tile_uq_thresh)

    # Figure 4
    if heatmap:
        matching_slide_paths = cP.dataset(299, 302, filters={'slide': [heatmap_slide]}).slide_paths()
        if not len(matching_slide_paths):
            raise ValueError(f"Could not find matching slide {heatmap_slide} for heatmap")
        slide = matching_slide_paths[0]
        if not exists(join('results', 'heatmap_full')): os.makedirs(join('results', 'heatmap_full'))
        if not exists(join('results', 'heatmap_high_confidence')): os.makedirs(join('results', 'heatmap_high_confidence'))

        # --- Figure 4a -----
        # Save the regular heatmap with predictions
        hm = sf.Heatmap(slide, aa_model, stride_div=1, uq=True)
        hm.save(join('results', 'heatmap_full'), cmap=utils.truncate_colormap(plt.get_cmap('PRGn'), 0.1, 0.9))

        # Save the heatmap with masked, high-confidence predictions
        uq_mask = hm.uncertainty > aa_tile_uncertainty_threshold
        hm.logits[uq_mask, :] = [-1, -1]
        hm.save(join('results', 'heatmap_high_confidence'), cmap=utils.truncate_colormap(plt.get_cmap('PRGn'), 0.1, 0.9))

        # --- Figure 4b -----
        # Save the highest and lowest uncertainty tiles
        if not exists(join('results', 'uq_excl')):
            os.makedirs(join('results', 'uq_excl'))
        if not exists(join('results', 'uq_incl')):
            os.makedirs(join('results', 'uq_incl'))
        uq = sf.model.tensorflow.UncertaintyInterface(aa_model)
        wsi = sf.WSI(slide, 299, 302, roi_method='ignore')
        gen = wsi.build_generator(shuffle=False, include_loc='grid', show_progress=True)
        for tile in gen():
            image = tile['image']
            if uq.wsi_normalizer:
                norm_image = uq.wsi_normalizer.rgb_to_rgb(image)
            else:
                norm_image = image
            parsed_image = tf.image.per_image_standardization(norm_image)
            parsed_image.set_shape([wsi.tile_px, wsi.tile_px, 3])
            uq_out = uq(tf.expand_dims(parsed_image, axis=0))[0]
            uncertainty = uq_out[-1]
            tilename = f"{uncertainty:.4f}-{tile['loc'][0]}-{tile['loc'][1]}.png"
            if uncertainty > aa_tile_uncertainty_threshold:
                Image.fromarray(tile['image']).save(join('results', 'uq_excl', tilename))
            else:
                Image.fromarray(tile['image']).save(join('results', 'uq_incl', tilename))

    # Figure 5
    if umaps:
        # --- Plot UMAPs ------------------------------------------------------

        df = cP.generate_features(aa_model, max_tiles=10)
        mosaic = cP.generate_mosaic(df)

        # Figure 5a
        mosaic.save(join('results', 'mosaic.png'))

        # Figure 5b
        mosaic.slide_map.label_by_logits(1)
        mosaic.slide_map.save(join('results', 'umap_preds.svg'), s=10)

        # Figure 5c
        mosaic.slide_map.label_by_meta('prediction')
        mosaic.slide_map.save(join('results', 'umap_binary_pred.svg'), s=10)

        # Figure 5d
        mosaic.slide_map.label_by_logits(2)
        mosaic.slide_map.save(join('results', 'umap_uncertainty.svg'), s=10, hue_norm=(0, 0.15))

        # Figure 5e
        mosaic.slide_map.labels = mosaic.slide_map.labels < aa_tile_uncertainty_threshold
        mosaic.slide_map.save(join('results', 'umap_confidence.svg'), s=10)

    # --- Analyze GAN (overview, non-UQ) --------------------------------------
    # Figure 6
    if gan:
        gan_df, _ = experiment.results(gan_exp, uq=False, plot=False)
        reg_df, _ = experiment.results(reg1_exp, uq=False, plot=False)
        reg_df = reg_df.loc[((reg_df['uq'] != 'include') & (reg_df['n_slides'] <= 500))]
        reg_df['gan_exp'] = 'none'
        gan_df['gan_exp'] = gan_df['id'].str[-3:]
        gan_df = gan_df.loc[gan_df['uq']!='include']
        gan_df = gan_df.append(reg_df, ignore_index=True)
        experiment.display(gan_df, None, hue='gan_exp', relplot_uq_compare=False, prefix='gan_')

        # --- Show GAN results from rhubarb (n=500 with UQ) -------------------
        r_exp = experiment.config('{}', ['R'], 1, order='forward')
        r_exp.update(experiment.config('{}_R', ['R'], 1, order='reverse'))
        r_df, _ = experiment.results(r_exp)
        gan_df, _ = experiment.results(gan_exp)
        gan_df = gan_df.loc[gan_df['id'].str[0] == 'R']
        gan_df = gan_df.append(r_df, ignore_index=True)
        gan_df = gan_df.loc[gan_df['uq'].isin(['all', 'include'])]
        experiment.display(gan_df, None, boxplot_uq_compare=False, ttest_uq_groups=['all', 'include'], prefix='gan_uq_')

if __name__ == '__main__':
    show_results() # pylint: disable=no-value-for-parameter