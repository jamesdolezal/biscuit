import os
import click
import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from statistics import mean
from os.path import join, exists

import slideflow as sf
from biscuit import experiment, utils, threshold
from biscuit.errors import ModelNotFoundError
from biscuit.experiment import ALL_EXP, config


@click.command()
@click.option('--train_project', type=str, help='Override training project')
@click.option('--eval_project', type=str, help='Override eval project')
@click.option('--reg', type=bool, help='Regular models results', default=False)
@click.option('--ratio', type=bool, help='Ratio models results', default=False)
@click.option('--gan', type=bool, help='GAN models results', default=False)
@click.option('--umaps', type=bool, help='Generate UMAPs', default=False)
@click.option('--heatmap', type=bool, help='Generate heatmaps', default=False)
@click.option('--heatmap_slide', type=str, default='C3N-01417-23')
def show_results(train_project=None, eval_project=None, reg=False, ratio=False,
                 gan=False, umaps=False, heatmap=False,
                 heatmap_slide='C3N-01417-23'):
    '''Shows results from trained experiments.'''

    if not exists('results'):
        os.makedirs('results')

    # === Configure experiments ===============================================

    experiment.OUT = 'results'
    if train_project is None:
        experiment.TRAIN_PATH = join('projects', 'training')
    else:
        experiment.TRAIN_PATH = train_project
    if eval_project is None:
        experiment.EVAL_PATHS = [join('projects', 'evaluation')]
    else:
        experiment.EVAL_PATHS = [eval_project]

    # Configure regular experiments
    reg1 = config('{}', ALL_EXP, 1, order='f')
    reg1.update(config('{}_R', ALL_EXP, 1, order='r'))
    reg2 = config('{}2', ALL_EXP, 1, order='f', order_col='order2')
    reg2.update(config('{}_R2', ALL_EXP, 1, order='r', order_col='order2'))
    all_reg = copy.deepcopy(reg1)
    all_reg.update(reg2)

    # Configure 3:1 and 10:1 ratio experiments
    r_list = list('AMDPGZ')
    ratio_3 = config('{}_3', r_list, 3, order='f')
    ratio_3.update(config('{}_R_3', r_list, 3, order='r'))
    ratio_10 = config('{}_10', r_list, 10, order='f')
    ratio_10.update(config('{}_R_10', r_list, 10, order='r'))

    # GAN experiments
    g = list('RALMNDOPQGWY') + ['ZA', 'ZC']
    gan_exp = {}
    gan_exp.update(config('{}_g10', g, 1, gan=0.1, order='f'))
    gan_exp.update(config('{}_R_g10', g, 1, gan=0.1, order='r'))
    gan_exp.update(config('{}_g20', g, 1, gan=0.2, order='f'))
    gan_exp.update(config('{}_R_g20', g, 1, gan=0.2, order='r'))
    gan_exp.update(config('{}_g30', g, 1, gan=0.3, order='f'))
    gan_exp.update(config('{}_R_g30', g, 1, gan=0.3, order='r'))
    gan_exp.update(config('{}_g40', g, 1, gan=0.4, order='f'))
    gan_exp.update(config('{}_R_g40', g, 1, gan=0.4, order='r'))
    gan_exp.update(config('{}_g50', g, 1, gan=0.5, order='f'))
    gan_exp.update(config('{}_R_g50', g, 1, gan=0.5, order='r'))

    # === Show results ========================================================
    # --- Full experiment -----------------------------------------------------
    # Figures 1 and 3
    if reg:
        df, eval_dfs = experiment.results(all_reg, uq=True, plot=True)
        experiment.display(df, eval_dfs, prefix='reg_')

    # --- Cross-val ratio results ---------------------------------------------
    # Figure 2
    if ratio:
        r1_df, _ = experiment.results(reg1, uq=True, eval=False)
        r3_df, _ = experiment.results(ratio_3, uq=True, eval=False)
        r10_df, _ = experiment.results(ratio_10, uq=True, eval=False)

        r1_df['ratio'] = ['1' for _ in range(len(r1_df))]
        r3_df['ratio'] = ['3' for _ in range(len(r3_df))]
        r10_df['ratio'] = ['10' for _ in range(len(r10_df))]

        df = r1_df.append(r3_df, ignore_index=True)
        df = df.append(r10_df, ignore_index=True)

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

    if umaps or heatmap:
        # Load the external evaluation project and find the fully trained model
        P = sf.Project(experiment.TRAIN_PATH)
        cP = sf.Project(experiment.EVAL_PATHS[0])
        if not utils.model_exists(P, 'EXP_AA_FULL'):
            raise ModelNotFoundError("Couldn't find trained model EXP_AA_FULL")
        aa_model = utils.find_model(P, 'EXP_AA_FULL', epoch=1)

        # Get tile uncertainty threshold
        threshold_params = {
            'y_pred_header':        utils.y_pred_header,
            'y_true_header':        utils.y_true_header,
            'uncertainty_header':   utils.uncertainty_header,
            'patients':             P.dataset().patients()
        }
        all_tile_uq_thresh = []
        for k in range(1, 4):
            k_preds = [
                join(folder, 'tile_predictions_val_epoch1.csv')
                for folder in utils.find_cv(P, f'EXP_AA_UQ-k{k}', k=5)
            ]
            tile_thresh, *_ = threshold.from_cv(
                k_preds,
                tile_uq_thresh='detect',
                slide_uq_thresh=None,
                **threshold_params
            )
            all_tile_uq_thresh += [tile_thresh]
        aa_tile_uq_thresh = mean(all_tile_uq_thresh)

    # --- Heatmap -------------------------------------------------------------
    # Figure 4
    if heatmap:
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

        # --- Figure 4a -----
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
        # --- Figure 4b -----
        # Save the highest and lowest uncertainty tiles
        if not exists(join('results', 'uq_excl')):
            os.makedirs(join('results', 'uq_excl'))
        if not exists(join('results', 'uq_incl')):
            os.makedirs(join('results', 'uq_incl'))
        interface = sf.model.tensorflow.UncertaintyInterface(aa_model)
        wsi = sf.WSI(slide, 299, 302, roi_method='ignore')
        gen = wsi.build_generator(
            shuffle=False,
            include_loc='grid',
            show_progress=True
        )
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
            tilename = f"{u:.4f}-{tile['loc'][0]}-{tile['loc'][1]}.png"
            if uncertainty[0][0] > aa_tile_uq_thresh:
                img = Image.fromarray(tile['image'])
                img.save(join('results', 'uq_excl', tilename))
            else:
                img = Image.fromarray(tile['image'])
                img.save(join('results', 'uq_incl', tilename))

    # --- Plot UMAPs (Figure 5) -----------------------------------------------
    if umaps:

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
        mosaic.slide_map.label_by_uncertainty()
        mosaic.slide_map.save(
            join('results', 'umap_uncertainty.svg'), s=10, hue_norm=(0, 0.15)
        )

        # Figure 5e
        mosaic.slide_map.labels = mosaic.slide_map.labels < aa_tile_uq_thresh
        mosaic.slide_map.save(join('results', 'umap_confidence.svg'), s=10)

    # --- Analyze GAN (overview, non-UQ) (Figure 6)----------------------------
    if gan:
        gan_df, _ = experiment.results(gan_exp, uq=False, plot=False)
        reg_df, _ = experiment.results(reg1, uq=False, plot=False)
        reg_df = reg_df.loc[((reg_df['uq'] != 'include')
                             & (reg_df['n_slides'] <= 500))]
        reg_df['gan_exp'] = 'none'
        gan_df['gan_exp'] = gan_df['id'].str[-3:]
        gan_df = gan_df.loc[gan_df['uq'] != 'include']
        gan_df = gan_df.append(reg_df, ignore_index=True)
        experiment.display(
            gan_df,
            None,
            hue='gan_exp',
            relplot_uq_compare=False,
            prefix='gan_'
        )
        # --- Show GAN results (n=500 with UQ) --------------------------------
        r_exp = config('{}', ['R'], 1, order='f')
        r_exp.update(config('{}_R', ['R'], 1, order='r'))
        r_df, _ = experiment.results(r_exp)
        gan_df, _ = experiment.results(gan_exp)
        gan_df = gan_df.loc[gan_df['id'].str[0] == 'R']
        gan_df = gan_df.append(r_df, ignore_index=True)
        gan_df = gan_df.loc[gan_df['uq'].isin(['all', 'include'])]
        experiment.display(
            gan_df,
            None,
            boxplot_uq_compare=False,
            ttest_uq_groups=['all', 'include'],
            prefix='gan_uq_'
        )


if __name__ == '__main__':
    show_results()  # pylint: disable=no-value-for-parameter
