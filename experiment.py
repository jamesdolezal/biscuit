import shutil
import slideflow as sf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import utils
import threshold

from skmisc.loess import loess
from errors import *
from slideflow.util import log
from scipy import stats
from tqdm import tqdm
from statistics import mean
from os.path import join, exists

TRAIN_PATH = 'not_configured'
EVAL_PATHS = ['not_configured']
OUT = 'results'
EXP_NAME_MAP = {
    'AA': 'full',
    'U': 800,
    'T': 700,
    'S': 600,
    'R': 500,
    'A': 400,
    'L': 350,
    'M': 300,
    'N': 250,
    'D': 200,
    'O': 176,
    'P': 150,
    'Q': 126,
    'G': 100,
    'V': 90,
    'W': 80,
    'X': 70,
    'Y': 60,
    'Z': 50,
    'ZA': 40,
    'ZB': 30,
    'ZC': 20,
    'ZD': 10
}

# --- Training functions ----------------------------------------------------------------------------------------------

def train(P, hp, label, filters, save_predictions=False, save_model=False, **kwargs):
    '''Trains a model.

    Args:
        P (slideflow.Project):  Slideflow project.
        hp (slideflow.ModelParams): Hyperparameters object.
        label (str): Experimental label.
        filters (dict): Patient-level annotations filter.
        save_predictions (bool, optional): Save validation predictions to model folder in CSV format. Defaults to False.
        save_model (bool, optional): Save final model after training. Defaults to False.

    Returns:
        None
    '''

    P.train(
        utils.OUTCOME,
        exp_label=label,
        filters=filters,
        params=hp,
        save_predictions=save_predictions,
        save_model=save_model,
        multi_gpu=False,
        validate_on_batch=32,
        **kwargs
    )

def train_nested_cv(P, hp, label, **kwargs):
    '''Trains a model with nested cross-validation (outer_k=3, inner_k=5), skipping already-generated models.

    Args:
        P (slideflow.Project):  Slideflow project.
        hp (slideflow.ModelParams): Hyperparameters object.
        label (str): Experimental label.

    Returns:
        None
    '''

    k1, k2, k3 = utils.find_cv(P, label)

    # Nested crossval for k1
    val_k = [k for k in range(1, 6) if not utils.model_exists(P, f'{label}-k1', kfold=k)]
    if not len(val_k):
        print(f'Skipping Step 5-k1 for experiment {label}; already done.')
    else:
        if val_k != list(range(1,6)):
            print(f'Only running k-folds {val_k} for Step 5-k1 in experiment {label}; some k-folds already done.')
        train(P, hp, f"{label}-k1", {'slide': sf.util.get_slides_from_model_manifest(k1, dataset='training')}, val_k_fold=5, val_k=val_k, save_predictions=True, **kwargs)

    # Nested crossval for k2
    val_k = [k for k in range(1, 6) if not utils.model_exists(P, f'{label}-k2', kfold=k)]
    if not len(val_k):
        print(f'Skipping Step 5-k2 for experiment {label}; already done.')
    else:
        if val_k != list(range(1,6)):
            print(f'Only running k-folds {val_k} for Step 5-k2 in experiment {label}; some k-folds already done.')
        train(P, hp, f"{label}-k2", {'slide': sf.util.get_slides_from_model_manifest(k2, dataset='training')}, val_k_fold=5, val_k=val_k, save_predictions=True, **kwargs)

    # Nested crossval for k3
    val_k = [k for k in range(1, 6) if not utils.model_exists(P, f'{label}-k3', kfold=k)]
    if not len(val_k):
        print(f'Skipping Step 5-k3 for experiment {label}; already done.')
    else:
        if val_k != list(range(1,6)):
            print(f'Only running k-folds {val_k} for Step 5-k3 in experiment {label}; some k-folds already done.')
        train(P, hp, f"{label}-k3", {'slide': sf.util.get_slides_from_model_manifest(k3, dataset='training')}, val_k_fold=5, val_k=val_k, save_predictions=True, **kwargs)

# --- Plotting functions --------------------------------------------------------------------------------------------

def plot_uncertainty_calibration(project, exp, tile_thresh, slide_thresh, pred_thresh):
    '''Plots a graph of predictions vs. uncertainty.

    Args:
        project (slideflow.Project): Slideflow project.
        exp (str): Experiment ID/label.
        kfold (int): Validation k-fold.
        tile_thresh (float): Tile-level uncertainty threshold.
        slide_thresh (float): Slide-level uncertainty threshold.
        pred_thresh (float): Slide-level prediction threshold.

    Returns:
        None
    '''

    val_dfs = [pd.read_csv(join(utils.find_model(project, f'EXP_{exp}_UQ', kfold=k), 'tile_predictions_val_epoch1.csv'), dtype={'slide': str}) for k in range(1, 4)]
    for v in range(len(val_dfs)):
        val_dfs[v].rename(columns={utils.y_pred_header: 'y_pred', utils.y_true_header: 'y_true', utils.uncertainty_header: 'uncertainty'}, inplace=True)
    _df = val_dfs[0]
    _df = _df.append(val_dfs[1], ignore_index=True)
    _df = _df.append(val_dfs[2], ignore_index=True)

    # Plot tile-level uncertainty
    _df, _ = threshold.process_tile_predictions(_df, patients=project.dataset().patients())
    threshold.plot_uncertainty(_df, kind='tile', threshold=tile_thresh, title=f'CV UQ Calibration: Exp {exp}')

    # Plot slide-level uncertainty
    _df = _df[_df['uncertainty'] < tile_thresh]
    _s_df, _ = threshold.process_group_predictions(_df, pred_thresh=pred_thresh, level='slide')
    threshold.plot_uncertainty(_s_df, kind='slide', threshold=slide_thresh, title=f'CV UQ Calibration: Exp {exp}')

def plot_pancan(tile_thresh, slide_thresh, pred_thresh):
    '''Plots out-of-distribution, pan-cancer predictions.

    Args:
        tile_thresh (float): Tile-level uncertainty threshold.
        slide_thresh (float): Slide-level uncertainty threshold.
        pred_thresh (float): Slide-level prediction threshold.

    Returns:
        Pandas DataFrame of slide-level predictions.
    '''

    # Read pan-cancer histologic diagnoses
    hist_df = pd.read_csv('/mnt/data/projects/PANCAN/annotations.csv', dtype=str)
    diagnoses = hist_df['primary_diagnosis'].unique()
    squam = [d for d in diagnoses if 'squamous cell' in d.lower()]
    adeno = [d for d in diagnoses if 'adenocarcinoma' in d.lower()]
    hist = dict(zip(hist_df.slide, hist_df.primary_diagnosis))

    def hist_cat_fn(slide):
        if hist[slide] in squam: return 'Squamous'
        elif hist[slide] in adeno: return 'Adenocarcinoma'
        else: return 'Other'

    hist_cat = {s: hist_cat_fn(s) for s in hist}

    root = '/mnt/data/projects/TCGA_LUNG/rhubarb_pancan_preds/'
    annotations = pd.read_csv(join(root, 'annotations.csv'))
    preds = pd.read_csv(join(root, 'squam_tile_predictions.csv'), dtype={'slide': str})
    preds = preds.append(pd.read_csv(join(root, 'adeno_tile_predictions.csv'), dtype={'slide': str}), ignore_index=True)
    preds = preds.append(pd.read_csv(join(root, 'other_tile_predictions.csv'), dtype={'slide': str}), ignore_index=True)
    preds.rename(columns={f'{utils.OUTCOME}_y_pred1': 'y_pred', f'{utils.OUTCOME}_y_true0': 'y_true', f'{utils.OUTCOME}_uncertainty1': 'uncertainty'}, inplace=True)
    patient_labels = dict(zip(annotations['patient'], annotations['project_id']))
    preds['patient'] = preds['slide'].str[0:12]
    preds[utils.OUTCOME] = preds['patient'].map(patient_labels)
    preds['y_true'] = 0

    # Non-thresholed predictions
    slide_labels = dict(zip(preds['slide'], preds[utils.OUTCOME]))
    nouq_tile_preds, _ = threshold.process_tile_predictions(preds)
    nouq_slide_preds, _ = threshold.process_group_predictions(nouq_tile_preds, pred_thresh=pred_thresh, level='slide')

    # UQ thresholded tile-level predictions
    uq_tile_preds = nouq_tile_preds[nouq_tile_preds['uncertainty'] < tile_thresh]
    uq_slide_preds, _ = threshold.process_group_predictions(uq_tile_preds, pred_thresh=pred_thresh, level='slide')
    uq_slide_preds = uq_slide_preds[uq_slide_preds['uncertainty'] < slide_thresh]

    def proc_preds(s_df):
        s_df['primary_diagnosis'] = s_df['slide'].map(hist)
        s_df['histology_category'] = s_df['slide'].map(hist_cat)
        s_df['hist_pred'] = s_df['y_pred_bin'].map({0: "Adenocarcinoma", 1: "Squamous"})
        s_df[utils.OUTCOME] = s_df['slide'].map(slide_labels).str[5:]
        s_df = s_df.set_index('slide')
        return s_df

    nouq_slide_preds = proc_preds(nouq_slide_preds)
    nouq_slide_preds.rename(columns={'hist_pred':'nouq_hist_pred'}, inplace=True)
    uq_slide_preds = proc_preds(uq_slide_preds)
    uq_slide_preds.drop(columns=[utils.OUTCOME, 'histology_category', 'primary_diagnosis'], inplace=True)
    slide_preds = uq_slide_preds.merge(nouq_slide_preds[['primary_diagnosis', 'histology_category', 'nouq_hist_pred', utils.OUTCOME]], how='outer', left_index=True, right_index=True)

    f, axes = plt.subplots(1, 3)
    f.set_size_inches(20, 6)
    cmap = utils.truncate_colormap(plt.get_cmap('PRGn'), 0.15, 0.85)

    def plot_stacked(s_df, kind, ax):
        _df = s_df.loc[s_df['histology_category'] == kind]
        print(_df.cohort.value_counts())
        all_preds = _df.cohort.value_counts().to_dict()
        squam_preds = _df.loc[_df['hist_pred']=='Squamous'].cohort.value_counts().to_dict()
        adeno_preds = _df.loc[_df['hist_pred']=='Adenocarcinoma'].cohort.value_counts().to_dict()
        low_preds = _df.loc[_df['hist_pred'].isna()]
        low_squam_preds =  low_preds.loc[low_preds['nouq_hist_pred']=='Squamous'].cohort.value_counts().to_dict()
        low_adeno_preds =  low_preds.loc[low_preds['nouq_hist_pred']=='Adenocarcinoma'].cohort.value_counts().to_dict()
        cohorts = np.array(sorted(list(set(list(all_preds.keys())))))
        counts = np.array([all_preds[c] for c in cohorts])

        s_p = np.array([squam_preds[c] if c in squam_preds else 0 for c in cohorts])
        a_p = np.array([adeno_preds[c] if c in adeno_preds else 0 for c in cohorts])
        all_p = np.array([all_preds[c] if c in all_preds else 0 for c in cohorts])
        low_s_p = np.array([low_squam_preds[c] if c in low_squam_preds else 0 for c in cohorts])
        low_a_p = np.array([low_adeno_preds[c] if c in low_adeno_preds else 0 for c in cohorts])

        print('\n')
        print(f'Category: {kind}')
        print('All predictions:', all_p.sum())
        print('High-confidence squamous:', s_p.sum())
        print('High-confidence adeno:', a_p.sum())
        print('Low-confidence squamous:', low_s_p.sum())
        print('Low-confidence adeno:', low_a_p.sum())

        # Plots exclude cohorts with sample size <= 5
        s_p = (s_p / all_p)[counts > 5]
        a_p = (a_p / all_p)[counts > 5]
        low_s_p = (low_s_p / all_p)[counts > 5]
        low_a_p = (low_a_p / all_p)[counts > 5]

        ax.bar(cohorts[counts > 5], s_p, color=cmap(cmap.N))
        ax.bar(cohorts[counts > 5], a_p, bottom=s_p, color=cmap(0))
        ax.bar(cohorts[counts > 5], low_s_p, bottom=s_p+a_p, color='darkgray')
        ax.bar(cohorts[counts > 5], low_a_p, bottom=s_p+a_p+low_s_p, color='gray')
        ax.tick_params(labelrotation=90)

    plot_stacked(slide_preds, 'Squamous', axes[0])
    plot_stacked(slide_preds, 'Adenocarcinoma', axes[1])
    plot_stacked(slide_preds, 'Other', axes[2])

    plt.subplots_adjust(bottom=0.2)
    plt.savefig(join(OUT, 'pancan_preds.svg'))
    return slide_preds

# --- Experiment functions --------------------------------------------------------------------------------------------

def config(name_pattern, subset, ratio, **kwargs):
    '''Configures a set of experiments.

    Args:
        name_pattern (str): String pattern for experiment naming.
        subset (list(str)): List of experiment ID/labels.
        ratio (float): Float 0-1. n_out1 / n_out2 (or n_out2 / n_out1)
    '''

    assert isinstance(ratio, (int, float)) and ratio >= 1

    config = {}
    for exp in EXP_NAME_MAP:
        if exp not in subset: continue
        if exp == 'AA' and ratio != 1:
            raise ValueError("Cannot create AA experiment (full dataset) with ratio != 1")

        exp_name = name_pattern.format(exp)

        if ratio != 1:
            n1 = round(EXP_NAME_MAP[exp] / (1 + (1/ratio)))
            n2 = EXP_NAME_MAP[exp] - n1

            config.update({
                exp_name:     {'out1': n1, 'out2': n2, **kwargs},
                exp_name+'i': {'out1': n2, 'out2': n1, **kwargs}
            })

        else:
            if EXP_NAME_MAP[exp] == 'full':
                n_out1 = 467
                n_out2 = 474
            else:
                n_out1 = n_out2 = int(EXP_NAME_MAP[exp] / 2)
            config.update({
                exp_name: {'out1': n_out1, 'out2': n_out2, **kwargs},
            })


        config.update()
    return config

def add(path, label, out1, out2, outcome=utils.OUTCOME, order='forward', order_col='order', gan=0):
    '''Adds a sample size experiment to the given project annotations file.

    Args:
        path (str): Path to project annotations file.
        label (str): Experimental label.
        out1 (int): Number of lung adenocarcinomas (LUAD) to include in the experiment.
        out2 (int): Number of lung squamous cell carcinomas (LUSC) to include in the experiment.
        outcome (str, optional): Annotation header which indicates the outcome of interest. Defaults to 'cohort'.
        order (str, optional): 'forward' or 'reverse'. Indicates which direction to follow when sequentially
            adding slides. Defaults to 'forward'.
        order_col (str, optional): Annotation header column to use when sequentially adding slides. Defaults to 'order'.
        gan (int, optional): Number of GAN slides to include in experiment. Defaults to 0.

    Returns:
        None
    '''

    assert isinstance(out1, int)
    assert isinstance(out2, int)
    assert isinstance(gan, (int, float)) and 0 <= gan < 1
    assert order in ('forward', 'reverse')

    ann = pd.read_csv(path, dtype=str)
    print(f"Configuring experiment {label} with order {order} (sorted by {order_col})")
    ann[order_col] = pd.to_numeric(ann[order_col])
    ann.sort_values(['gan', utils.OUTCOME, order_col], ascending=[True, True, (order != 'reverse')], inplace=True)

    gan_out1 = round(gan * out1)
    gan_out2 = round(gan * out2)
    out1_indices = np.where((ann['site'].to_numpy() != 'GAN') & (ann[outcome] == utils.OUTCOME1))[0]
    out2_indices = np.where((ann['site'].to_numpy() != 'GAN') & (ann[outcome] == utils.OUTCOME2))[0]
    gan_out1_indices = np.where((ann['site'].to_numpy() == 'GAN') & (ann[outcome] == utils.OUTCOME1))[0]
    gan_out2_indices = np.where((ann['site'].to_numpy() == 'GAN') & (ann[outcome] == utils.OUTCOME2))[0]

    assert out1 <= out1_indices.shape[0]
    assert out2 <= out2_indices.shape[0]
    assert gan_out1 <= gan_out1_indices.shape[0]
    assert gan_out2 <= gan_out2_indices.shape[0]

    include = np.array(['exclude' for _ in range(len(ann))])
    include[out1_indices[:out1]] = 'include'
    include[out2_indices[:out2]] = 'include'
    include[gan_out1_indices[:gan_out1]] = 'include'
    include[gan_out2_indices[:gan_out2]] = 'include'
    ann[f'include_{label}'] = include
    ann.to_csv(path, index=False)

def run(all_exp, steps=None, hp='nature2022'):
    '''Trains the designated experiments.

    Args:
        all_exp (dict): Dict containing experiment configuration, as provided by config().
        steps (list(int)): Steps to run. Defaults to all steps, 1-6.
        hp (slideflow.ModelParams, optional): Hyperparameters object. Defaults to hyperparameters used for publication.

    Returns:
        None
    '''

    # === Initialize projects & prepare experiments ===========================
    print(sf.util.bold("Initializing experiments..."))
    P = sf.Project(TRAIN_PATH)
    eval_Ps = [(sf.Project(path) if path != 'not_configured' else None) for path in EVAL_PATHS]

    exp_annotations = join(P.root, 'experiments.csv')
    if P.annotations != exp_annotations:
        if not exists(exp_annotations):
            shutil.copy(P.annotations, exp_annotations)
        P.annotations = exp_annotations
    exp_to_add = [e for e in all_exp if f'include_{e}' not in pd.read_csv(exp_annotations).columns.tolist()]
    for exp in exp_to_add:
        add(exp_annotations, label=exp, **all_exp[exp])

    full_epoch_exp = [e for e in all_exp if e in ('AA', 'A', 'D', 'G')]

    if hp == 'nature2022':
        exp_hp = sf.model.ModelParams(
            model='xception',
            tile_px=299,
            tile_um=302,
            batch_size=128,
            epochs=[1],         # epochs 1, 3, 5, 10 used for initial sweep
            early_stop=True,
            early_stop_method='accuracy',
            dropout=0.1,
            uq=False,           # to be enabled in separate sub-experiments
            hidden_layer_width=1024,
            optimizer='Adam',
            learning_rate=0.0001,
            learning_rate_decay_steps=512,
            learning_rate_decay=0.98,
            loss='sparse_categorical_crossentropy',
            normalizer='reinhard_fast',
            normalizer_source='dataset',
            include_top=False,
            hidden_layers=2,
            pooling='avg',
            augment='xyrjb'
        )
    else:
        exp_hp = hp

    # Configure steps to run
    if steps is None:
        steps = range(7)

    # === Step 1: Initialize full-epochs experiments ==========================
    if 1 in steps:
        print(sf.util.bold("Running full-epoch experiments..."))
        exp_hp.epochs = [1, 3, 5, 10]
        for exp in full_epoch_exp:
            val_k = [k for k in range(1, 4) if not utils.model_exists(P, f'EXP_{exp}', kfold=k)]
            if not len(val_k):
                print(f'Skipping Step 1 for experiment {exp}; already done.')
                continue
            elif val_k != list(range(1,4)):
                print(f'Only running k-folds {val_k} for Step 1 in experiment {exp}; some k-folds already done.')
            train(P, exp_hp, f'EXP_{exp}', {f'include_{exp}': ['include']}, splits=f'splits_{exp}.json', val_k=val_k, val_strategy='k-fold')

    # === Step 2: Run the rest of the experiments at the designated epoch =====
    if 2 in steps:
        print(sf.util.bold("Running experiments at designated epoch..."))
        exp_hp.epochs = [1]
        for exp in all_exp:
            if exp in full_epoch_exp: continue # Already done in Step 2
            val_k = [k for k in range(1, 4) if not utils.model_exists(P, f'EXP_{exp}', kfold=k)]
            if not len(val_k):
                print(f'Skipping Step 2 for experiment {exp}; already done.')
                continue
            elif val_k != list(range(1,4)):
                print(f'Only running k-folds {val_k} for Step 2 in experiment {exp}; some k-folds already done.')
            train(P, exp_hp, f'EXP_{exp}', {f'include_{exp}': ['include']}, save_predictions=True, splits=f'splits_{exp}.json', val_k=val_k, val_strategy='k-fold')

    # === Step 3: Run experiments with UQ & save predictions ==================
    if 3 in steps:
        print(sf.util.bold("Running experiments with UQ..."))
        exp_hp.epochs = [1]
        exp_hp.uq = True
        for exp in all_exp:
            val_k = [k for k in range(1, 4) if not utils.model_exists(P, f'EXP_{exp}_UQ', kfold=k)]
            if not len(val_k):
                print(f'Skipping Step 3 for experiment {exp}; already done.')
                continue
            elif val_k != list(range(1,4)):
                print(f'Only running k-folds {val_k} for Step 4 in experiment {exp}; some k-folds already done.')
            train(P, exp_hp, f'EXP_{exp}_UQ', {f'include_{exp}': ['include']}, save_predictions=True, splits=f'splits_{exp}.json', val_k=val_k, val_strategy='k-fold')

    # === Step 4: Run nested UQ cross-validation ==============================
    if 4 in steps:
        print(sf.util.bold("Running nested UQ experiments..."))
        exp_hp.epochs = [1]
        exp_hp.uq = True
        for exp in all_exp:
            total_slides = all_exp[exp]['out2'] + all_exp[exp]['out1']
            if total_slides >= 50:
                train_nested_cv(P, exp_hp, f'EXP_{exp}_UQ', val_strategy='k-fold') # NO site-preservation for nested UQ
            else:
                print(f"Skipping nested UQ for exp {exp} (total slides: {total_slides}, need >= 50)")

    # === Step 5: Train models across full datasets ===========================
    if 5 in steps:
        print(sf.util.bold("Training across full datasets..."))
        exp_hp.epochs = [1]
        exp_hp.uq = True
        for exp in all_exp:
            if utils.model_exists(P, f'EXP_{exp}_FULL'):
                print(f'Skipping Step 5 for experiment {exp}; already done.')
            else:
                stop_batch = utils.find_cv_early_stop(P, f'EXP_{exp}', k=3)
                print(f"Using detected early stop batch {stop_batch}")
                train(P, exp_hp, f'EXP_{exp}_FULL', {f'include_{exp}': ['include']}, save_model=True, val_strategy='none', steps_per_epoch_override=stop_batch)

    # === Step 6: External validation  ========================================
    if 6 in steps:
        for val_P in eval_Ps:
            print(sf.util.bold(f"Running external evaluation ({val_P.name})..."))
            for exp in all_exp:
                full_model = utils.find_model(P, f'EXP_{exp}_FULL', epoch=1)
                if utils.eval_exists(val_P, f'EXP_{exp}_FULL', epoch=1):
                    print(f'Skipping evaluation for experiment {exp}; already done.')
                else:
                    val_P.evaluate(
                        full_model,
                        utils.OUTCOME,
                        filters={utils.OUTCOME: [utils.OUTCOME1, utils.OUTCOME2]},
                        save_predictions=True,
                )

def results(all_exp, uq=True, eval=True, plot=False):
    '''Assembles results from experiments, applies UQ thresholding, and returns pandas dataframes with metrics.

    Args:
        all_exp (list): List of experiment IDs to search for results.
        uq (bool, optional): Apply UQ thresholds. Defaults to True.
        eval (bool, optional): Calculate results of external evaluation models. Defaults to True.
        plot (bool, optional): Show plots. Defaults to False.

    Returns:
        Pandas DataFrame (cross-val results), Pandas DataFrame (external eval results)
    '''

    # === Initialize projects & prepare experiments ===========================

    P = sf.Project(TRAIN_PATH)
    eval_Ps = [sf.Project(path) for path in EVAL_PATHS if path != 'not_configured']
    df = pd.DataFrame()
    eval_dfs = {val_P.name: pd.DataFrame() for val_P in eval_Ps}
    prediction_thresholds = {}
    slide_uq_thresholds = {}
    tile_uq_thresholds = {}
    pred_uq_thresholds = {}

    # === Show results from designated epoch ==================================

    for exp in all_exp:
        try:
            models = utils.find_cv(P, f'EXP_{exp}')
        except MatchError:
            log.debug(f"Unable to find cross-val results for exp {exp}; skipping")
            continue
        for i, m in enumerate(models):
            try:
                patient_auc, patient_ap, slide_auc, slide_ap, tile_auc, tile_ap, _ = utils.get_model_results(m)
            except FileNotFoundError:
                print(f"Unable to open cross-val results for exp {exp}; skipping")
                continue
            df = df.append({
                'id': exp,
                'n_slides': len(sf.util.get_slides_from_model_manifest(m, dataset=None)),
                'fold': i+1,
                'uq': 'none',
                'patient_auc': patient_auc,
                'patient_ap': patient_ap,
                'slide_auc': slide_auc,
                'slide_ap': slide_ap,
                'tile_auc': tile_auc,
                'tile_ap': tile_ap,
            }, ignore_index=True)

    # === Add UQ Crossval results (non-thresholded) ===========================

    for exp in all_exp:
        try:
            skip = False
            models = utils.find_cv(P, f'EXP_{exp}_UQ')
        except MatchError:
            continue
        all_pred_thresh = []
        for i, m in enumerate(models):
            try:
                patient_auc, patient_ap, slide_auc, slide_ap, tile_auc, tile_ap, opt_thresh = utils.get_model_results(m)
                all_pred_thresh += [opt_thresh]
                df = df.append({
                    'id': exp,
                    'n_slides': len(sf.util.get_slides_from_model_manifest(m, dataset=None)),
                    'fold': i+1,
                    'uq': 'all',
                    'patient_auc': patient_auc,
                    'patient_ap': patient_ap,
                    'slide_auc': slide_auc,
                    'slide_ap': slide_ap,
                    'tile_auc': tile_auc,
                    'tile_ap': tile_ap,
                }, ignore_index=True)
            except FileNotFoundError:
                log.debug(f"Skipping UQ crossval (non-thresholded) results for {exp}; not found")
                skip = True
                break
        if not skip:
            prediction_thresholds[exp] = mean(all_pred_thresh)

    # === Get & Apply Nested UQ Threshold =====================================

    if uq:
        threshold_params = {
            'tile_pred_thresh':     'detect',
            'slide_pred_thresh':    'detect',
            'plot':                 False,
            'y_pred_header':        utils.y_pred_header,
            'y_true_header':        utils.y_true_header,
            'uncertainty_header':   utils.uncertainty_header,
            'patients':             P.dataset().patients()
        }

        pb = tqdm(all_exp)
        for exp in pb:
            # Skip UQ for experiments with n_slides < 100
            if exp in ('V', 'W', 'X', 'Y', 'Z', 'ZA', 'ZB', 'ZC', 'ZD'):
                continue
            pb.set_description(f"Calculating thresholds (exp {exp})...")
            all_tile_uq_thresh = []
            all_slide_uq_thresh = []
            all_slide_pred_thresh = []
            skip = False

            for k in range(1, 4):
                try:
                    k_preds = [join(folder, 'tile_predictions_val_epoch1.csv') for folder in utils.find_cv(P, f'EXP_{exp}_UQ-k{k}', k=5)]
                    val_path = join(utils.find_model(P, f'EXP_{exp}_UQ', kfold=k), 'tile_predictions_val_epoch1.csv')
                    if not exists(val_path): raise FileNotFoundError
                    tile_thresh, _, _, _ = threshold.from_cv(k_preds, tile_uq_thresh='detect', slide_uq_thresh=None, **threshold_params)
                    _, slide_thresh, tile_pred_thresh, slide_pred_thresh = threshold.from_cv(k_preds, tile_uq_thresh=tile_thresh, slide_uq_thresh='detect', **threshold_params)
                except (MatchError, FileNotFoundError, ModelNotFoundError) as e:
                    log.debug(str(e))
                    log.debug(f"Skipping UQ crossval thresholding results for {exp}; not found")
                    skip = True
                    break
                except ThresholdError as e:
                    log.debug(str(e))
                    log.debug(f'Skipping UQ crossval thresholding results for {exp}; could not find thresholds in cross-validation')
                    skip = True
                    break

                all_tile_uq_thresh += [tile_thresh]
                all_slide_uq_thresh += [slide_thresh]
                all_slide_pred_thresh += [slide_pred_thresh]
                tile_pred_df = pd.read_csv(val_path, dtype={'slide': str})
                tile_pred_df.rename(columns={utils.y_pred_header: 'y_pred', utils.y_true_header: 'y_true', utils.uncertainty_header: 'uncertainty'}, inplace=True)

                def get_auc_by_level(level):
                    auc, perc, _, _, _ = threshold.apply(
                        tile_pred_df,
                        thresh_tile=tile_thresh,
                        thresh_slide=slide_thresh,
                        tile_pred_thresh=tile_pred_thresh,
                        slide_pred_thresh=slide_pred_thresh,
                        plot=False,
                        patients=P.dataset().patients(),
                        level=level
                    )
                    return auc, perc

                pt_auc, pt_perc = get_auc_by_level('patient')
                slide_auc, slide_perc = get_auc_by_level('slide')

                df = df.append({
                    'id': exp,
                    'n_slides': len(sf.util.get_slides_from_model_manifest(utils.find_model(P, f'EXP_{exp}_UQ', kfold=k, epoch=1), dataset=None)),
                    'fold': k,
                    'uq': 'include',
                    'patient_auc': pt_auc,
                    'patient_uq_perc': pt_perc,
                    'slide_auc': slide_auc,
                    'slide_uq_perc': slide_perc
                }, ignore_index=True)

            if not skip:
                tile_uq_thresholds[exp] = mean(all_tile_uq_thresh)
                slide_uq_thresholds[exp] = mean(all_slide_uq_thresh)
                pred_uq_thresholds[exp] = mean(all_slide_pred_thresh)

            # Show CV uncertainty calibration
            if plot and exp == 'AA':
                plot_uncertainty_calibration(
                    project=P,
                    exp=exp,
                    tile_thresh=tile_uq_thresholds[exp],
                    slide_thresh=slide_uq_thresholds[exp],
                    pred_thresh=pred_uq_thresholds[exp]
                )

            # Show PANCAN OOD predictions
            if plot and exp == 'AA':
                plot_pancan(tile_thresh=tile_uq_thresholds[exp], slide_thresh=slide_uq_thresholds[exp], pred_thresh=pred_uq_thresholds[exp])

    # === Show external validation results ====================================

    if eval:
        # --- Step 7A: Show non-UQ external validation results ----------------

        for val_P in eval_Ps:
            name = val_P.name
            pb = tqdm(all_exp, ncols=80)
            for exp in pb:
                pb.set_description(f'Working on {name} eval (EXP {exp})...')

                # Read and prepare model results
                try:
                    eval_dir = utils.find_eval(val_P, f'EXP_{exp}_FULL')
                    patient_auc, patient_ap, slide_auc, slide_ap, tile_auc, tile_ap, _ = utils.get_model_results(eval_dir)
                except (FileNotFoundError, MatchError):
                    log.debug(f"Skipping eval for exp {exp}; eval not found")
                    continue
                if not utils.model_exists(P, f'EXP_{exp}_FULL', epoch=1):
                    log.debug(f'Skipping eval for exp {exp}; trained model not found')
                    continue
                if exp not in prediction_thresholds:
                    log.warn(f"No predictions threshold for experiment {exp}; using slide-level pred threshold of 0.5")
                    pred_thresh = 0.5
                else:
                    pred_thresh = prediction_thresholds[exp]

                # Patient-level and slide-level predictions & metrics
                patient_yt, patient_yp = utils.read_group_predictions(join(eval_dir, f'patient_predictions_{utils.OUTCOME}_eval.csv'))
                patient_metrics = utils.prediction_metrics(patient_yt, patient_yp, threshold=pred_thresh)
                patient_metrics = {f'patient_{m}': patient_metrics[m] for m in patient_metrics}
                slide_yt, slide_yp = utils.read_group_predictions(join(eval_dir, f'patient_predictions_{utils.OUTCOME}_eval.csv'))
                slide_metrics = utils.prediction_metrics(slide_yt, slide_yp, threshold=pred_thresh)
                slide_metrics = {f'slide_{m}': slide_metrics[m] for m in slide_metrics}

                eval_dfs[name] = eval_dfs[name].append({
                    'id': exp,
                    'n_slides': len(sf.util.get_slides_from_model_manifest(utils.find_model(P, f'EXP_{exp}_FULL', epoch=1), dataset=None)),
                    'uq': 'none',
                    'incl': 1,
                    'patient_auc': patient_auc,
                    'patient_ap': patient_ap,
                    'slide_auc': slide_auc,
                    'slide_ap': slide_ap,
                    **patient_metrics,
                    **slide_metrics,
                }, ignore_index=True)

                # --- [end patient-level predictions] -----------------------------------------------------------------

                if exp not in prediction_thresholds:
                    log.debug(f"Unable to calculate eval UQ performance; no prediction thresholds found for exp {exp}")
                    continue

                # --- Step 7B: Show UQ external validation results --------------------
                if uq:
                    if exp in tile_uq_thresholds:
                        for keep in ('high_confidence', 'low_confidence'):
                            tile_pred_df = pd.read_csv(join(eval_dir, 'tile_predictions_eval.csv'), dtype={'slide': str})
                            tile_pred_df.rename(columns={f'{utils.OUTCOME}_y_pred1': 'y_pred', f'{utils.OUTCOME}_y_true0': 'y_true', f'{utils.OUTCOME}_uncertainty1': 'uncertainty'}, inplace=True)

                            thresh_tile = tile_uq_thresholds[exp]
                            thresh_slide = slide_uq_thresholds[exp]

                            def get_metrics_by_level(level):
                                return threshold.apply(
                                    tile_pred_df,
                                    thresh_tile=thresh_tile,
                                    thresh_slide=thresh_slide,
                                    tile_pred_thresh=0.5,
                                    slide_pred_thresh=pred_uq_thresholds[exp],
                                    plot=(plot and keep=='high_confidence' and exp=='AA'),
                                    title=f'{name}: Exp. {exp} Uncertainty',
                                    keep=keep, # Keeps only LOW or HIGH-confidence slide predictions
                                    patients=val_P.dataset().patients(),
                                    level=level
                                )

                            s_uq_auc, s_uq_perc, s_uq_acc, s_uq_sens, s_uq_spec = get_metrics_by_level('slide')
                            p_uq_auc, p_uq_perc, p_uq_acc, p_uq_sens, p_uq_spec = get_metrics_by_level('patient')
                            if (plot and keep=='high_confidence' and exp=='AA'):
                                plt.savefig(join(OUT, f'{name}_uncertainty_v_preds.svg'))

                            full_model = utils.find_model(P, f'EXP_{exp}_FULL', epoch=1)
                            eval_dfs[name] = eval_dfs[name].append({
                                'id': exp,
                                'n_slides': len(sf.util.get_slides_from_model_manifest(full_model, dataset=None)),
                                'uq': ('include' if keep == 'high_confidence' else 'exclude'),
                                'slide_incl': s_uq_perc,
                                'slide_auc': s_uq_auc,
                                'slide_acc': s_uq_acc,
                                'slide_sens': s_uq_sens,
                                'slide_spec': s_uq_spec,
                                'slide_youden': s_uq_sens + s_uq_spec - 1,
                                'patient_incl': p_uq_perc,
                                'patient_auc': p_uq_auc,
                                'patient_acc': p_uq_acc,
                                'patient_sens': p_uq_sens,
                                'patient_spec': p_uq_spec,
                                'patient_youden': p_uq_sens + p_uq_spec - 1
                            }, ignore_index=True)
        for eval_name in eval_dfs:
            eval_dfs[eval_name].to_csv(join(OUT, f'{eval_name}_results.csv'), index=False)
    else:
        eval_dfs = None

    df.to_csv(join(OUT, 'crossval_results.csv'), index=False)
    return df, eval_dfs

def display(df, eval_dfs, hue='uq', palette='tab10', relplot_uq_compare=True, boxplot_uq_compare=True,
            ttest_uq_groups=['all', 'include'], prefix=''):
    '''Creates plots from assmebled results, exports results to CSV.

    Args:
        df (pandas.DataFrame): Cross-validation results metrics, as generated by results()
        eval_dfs (dict(pandas.DataFrame)): Dict of external eval dataset names (keys) mapped to pandas DataFrame of
            result metrics (values).
        hue (str, optional): Comparison to show with different hue on plots. Defaults to 'uq'.
        palette (str, optional): Seaborn color palette. Defaults to 'tab10'.
        relplot_uq_compare (bool, optional): For the Relplot display, ensure non-UQ and UQ results are generated from
            the same models/predictions.
        boxplot_uq_compare (bool, optional): For the boxplot display, ensure non-UQ and UQ results are generated from
            the same models/predictions.
        ttest_uq_groups (list(str)): UQ groups to compare via t-test. Defaults to ['all', 'include'].
        prefix (str, optional): Prefix to use when saving figures. Defaults to empty string.

    Returns:
        None
    '''

    if not len(df):
        log.error("No results to display")
        return

    # Filter out UQ results if n_slides < 100
    df = df.loc[~ ((df['n_slides'] < 100) & (df['uq'].isin(['include', 'exclude'])))]

    # --- Paired t-tests ---------------------------------------------------
    if ttest_uq_groups and len(ttest_uq_groups) != 2:
        raise ValueError("List of t-test group names (ttest_uq_groups) must be exactly 2")
    ttest_df = df.loc[df['uq'].isin(ttest_uq_groups)].copy()
    ttest_df = ttest_df.sort_values(['id', 'fold'])

    def perform_paired_testing(level):
        print(f"Paired t-tests ({level}-level):")
        for n in sorted(ttest_df['n_slides'].unique()):
            exp_df = ttest_df[ttest_df['n_slides']==n]
            try:
                ttest_result = stats.ttest_rel(
                    exp_df.loc[exp_df['uq']==ttest_uq_groups[0]][f'{level}_auc'],
                    exp_df.loc[exp_df['uq']==ttest_uq_groups[1]][f'{level}_auc'],
                    alternative='less')
                print(n, '\t', 'p =', ttest_result.pvalue)
            except ValueError:
                print(n, '\t', 'p = (error)')

    perform_paired_testing('patient')
    perform_paired_testing('slide')

    # --- Cross-validation plots -------------------------------------------

    if len(df):
        # AUC (relplot)
        if relplot_uq_compare:
            rel_df = df.loc[df['uq']!='none']
        else:
            rel_df = df
        sns.relplot(x='n_slides', y=f'slide_auc', data=rel_df, hue=hue, marker='o', kind='line', palette=palette)
        plt.title('Cross-val AUC')
        ax = plt.gca()
        ax.set_ylim([0.5, 1])
        ax.grid(visible=True, which='both', axis='both', color='white')
        ax.set_facecolor('#EAEAF2')
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(100))
        plt.subplots_adjust(top=0.9)
        plt.savefig(join(OUT, f'{prefix}relplot.svg'))

        f, axes = plt.subplots(1, 3)
        f.set_size_inches(18, 6)

        # AUC boxplot
        if boxplot_uq_compare:
            box_df = df.loc[df['uq']!='none']
        else:
            box_df = df
        sns.boxplot(x='n_slides', y=f'slide_auc', hue=hue, data=box_df, ax=axes[0], palette=palette)
        axes[0].title.set_text('Cross-val AUC')
        axes[0].set_ylabel('')
        axes[0].tick_params(labelrotation=90)

        # AUC scatter - LOESS & standard error
        df = df.sort_values(by=['n_slides'])
        x = df['n_slides'].to_numpy().astype(np.float32)
        y = df[f'slide_auc'].to_numpy()
        l = loess(x, y)
        try:
            l.fit()
            pred = l.predict(x, stderror=True)
            conf = pred.confidence()
            z = pred.values
            ll = conf.lower
            ul = conf.upper
            axes[1].plot(x, y, '+', ms=6)
            axes[1].plot(x, z)
            axes[1].fill_between(x,ll,ul,alpha=.33)
        except ValueError:
            pass

        axes[1].xaxis.set_minor_locator(plticker.MultipleLocator(20))
        axes[1].spines['bottom'].set_linewidth(0.5)
        axes[1].spines['bottom'].set_color('black')
        axes[1].tick_params(axis='x', colors='black')
        axes[1].grid(visible=True, which='both', axis='both', color='white')
        axes[1].set_facecolor('#EAEAF2')
        axes[1].set_xscale('log')
        axes[1].title.set_text('Cross-val AUC')

        # % slides included
        sns.lineplot(x='n_slides', y='patient_uq_perc', data=df, marker='o', ax=axes[2])
        axes[2].set_ylabel('')
        axes[2].title.set_text('% Patients Included with UQ (cross-val)')
        axes[2].xaxis.set_minor_locator(plticker.MultipleLocator(100))
        axes[2].tick_params(labelrotation=90)
        axes[2].grid(visible=True, which='both', axis='both', color='white')
        axes[2].set_facecolor('#EAEAF2')
        axes[2].set_xlim(100)

        plt.subplots_adjust(bottom=0.2)
        plt.savefig(join(OUT, f'{prefix}crossval.svg'))

    # --- Evaluation plots ----------------------------------------------------

    if eval_dfs:
        for eval_name, eval_df in eval_dfs.items():
            has_uq = len(eval_df.loc[eval_df['uq'].isin(['include', 'exclude'])])

            # Prepare figure
            sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
            f, axes = plt.subplots(1, (4 if has_uq else 3))
            f.suptitle(f'{eval_name} Evaluation Dataset')
            f.set_size_inches(16, 4)

            # AUC
            if not len(eval_df): continue
            eval_df = eval_df.loc[~ ((eval_df['n_slides'] < 100) & (eval_df['uq'].isin(['include', 'exclude'])))]
            sns.lineplot(x='n_slides', y=f'patient_auc', hue=hue, data=eval_df, marker="o", ax=axes[0])
            sns.scatterplot(x='n_slides', y=f'slide_auc', hue=hue, data=eval_df, marker="x", ax=axes[0])
            axes[0].get_legend().remove()
            axes[0].title.set_text(f'AUC')

            # Accuracy
            sns.lineplot(x='n_slides', y=f'patient_acc', hue=hue, data=eval_df, marker="o", ax=axes[1])
            sns.scatterplot(x='n_slides', y=f'slide_acc', hue=hue, data=eval_df, marker="x", ax=axes[1])
            axes[1].get_legend().remove()
            axes[1].title.set_text(f'Accuracy')

            # Youden's index
            sns.lineplot(x='n_slides', y=f'patient_youden', hue=hue, data=eval_df, marker="o", ax=axes[2])
            sns.scatterplot(x='n_slides', y=f'slide_youden', hue=hue, data=eval_df, marker="x", ax=axes[2])
            axes[2].title.set_text(f"Youden's J")
            axes[2].get_legend().remove()

            # % slides included
            if has_uq:
                sns.lineplot(x='n_slides', y=f'patient_incl', data=eval_df.loc[eval_df['uq'] == 'include'], marker='o')
                sns.scatterplot(x='n_slides', y=f'slide_incl', data=eval_df.loc[eval_df['uq'] == 'include'], marker='x')
                axes[3].title.set_text(f'% Included')

            for ax in axes:
                ax.set_ylabel('')
                ax.xaxis.set_major_locator(plticker.MultipleLocator(base=100))
                ax.tick_params(labelrotation=90)

            plt.subplots_adjust(top=0.8)
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(join(OUT, f'{prefix}eval.svg'))

