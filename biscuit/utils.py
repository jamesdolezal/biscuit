import os
import csv
import shutil
import pandas as pd
import numpy as np
import slideflow as sf
import matplotlib.colors as colors

from biscuit.errors import *
from biscuit.delong import delong_roc_variance
from sklearn import metrics
from scipy import stats
from statistics import mean, variance
from os.path import join

OUTCOME = 'cohort'
OUTCOME1 = 'LUAD'
OUTCOME2 = 'LUSC'

uncertainty_header = f'{OUTCOME}_uncertainty1'
y_true_header = f'{OUTCOME}_y_true0'
y_pred_header = f'{OUTCOME}_y_pred1'

# --- General utility functions ---------------------------------------------------------------------------------------

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_model_results(path, outcome=None):
    if outcome is None: outcome = OUTCOME
    csv = pd.read_csv(join(path, 'results_log.csv'))
    model_res = next(csv.iterrows())[1]
    pt_ap = mean(eval(model_res['patient_ap'])[outcome])
    pt_auc = eval(model_res['patient_auc'])[outcome][0]
    slide_ap = mean(eval(model_res['slide_ap'])[outcome])
    slide_auc = eval(model_res['slide_auc'])[outcome][0]
    tile_ap = mean(eval(model_res['tile_ap'])[outcome])
    tile_auc = eval(model_res['tile_auc'])[outcome][0]

    predictions_path = join(path, f'patient_predictions_{OUTCOME}_val_epoch1.csv')
    if os.path.exists(predictions_path):
        manual_auc, opt_thresh = auc_and_threshold(*read_group_predictions(predictions_path))
    else:
        manual_auc, opt_thresh = None, None

    return pt_auc, pt_ap, slide_auc, slide_ap, tile_auc, tile_ap, opt_thresh

def find_cv_early_stop(P, label, k=3):
    cv_folders = find_cv(P, label, k=k)
    early_stop_batch = []
    for cv_folder in cv_folders:
        csv = pd.read_csv(join(cv_folder, 'results_log.csv'))
        model_res = next(csv.iterrows())[1]
        if 'early_stop_batch' in model_res:
            early_stop_batch += [model_res['early_stop_batch']]
    if len(early_stop_batch) == len(cv_folders):
        # Only returns early stop if it was triggered in all crossfolds
        #TODO: consider if this is best approach or not
        return round(mean(early_stop_batch))
    else:
        return None

# --- Utility functions for finding experiment models -----------------------------------------------------------------

def find_model(P, label, epoch=None, kfold=None):
    tail = '' if kfold is None else f'-kfold{kfold}'
    model_name = f'{OUTCOME}-{label}-HP0{tail}'
    matching = [o for o in os.listdir(P.models_dir) if o[6:] == model_name]
    if len(matching) > 1:
        raise MultipleModelsFoundError(f"Multiple matching model experiments found matching {model_name}")
    elif not len(matching):
        raise ModelNotFoundError(f"No matching model found matching {model_name}.")
    elif epoch is not None:
        return join(P.models_dir, matching[0], f'{OUTCOME}-{label}-HP0{tail}_epoch{epoch}')
    else:
        return join(P.models_dir, matching[0])

def model_exists(P, label, epoch=None, kfold=None):
    try:
        find_model(P, label, epoch, kfold)
        return True
    except ModelNotFoundError:
        return False

def find_cv(P, label, epoch=None, k=3):
    return [find_model(P, label, epoch=epoch, kfold=_k) for _k in range(1, k+1)]

def find_eval(P, label, epoch=1):
    matching = [o for o in os.listdir(P.eval_dir) if o[11:] == f'{OUTCOME}-{label}-HP0_epoch{epoch}']
    if len(matching) > 1:
        raise MultipleModelsFoundError(f"Multiple matching eval experiments found for label {label}")
    elif not len(matching):
        raise ModelNotFoundError(f"No matching eval found for label {label}")
    else:
        return join(P.eval_dir, matching[0])

def eval_exists(P, label, epoch=1):
    try:
        find_eval(P, label, epoch)
        return True
    except ModelNotFoundError:
        return False

# --- Thresholding and metrics functions ------------------------------------------------------------------------------

def load_and_fix_patient_pred(path):
    '''Reads patient-level predictions CSV file, returning pandas dataframe'''

    # Fix file if necessary
    with open(path, 'r') as f:
        first_reader = csv.reader(f)
        header = next(first_reader)
        firstrow = next(first_reader)
    if len(header) == 6 and len(firstrow) == 7:
        print(f"Fixing predictions file at {path}")
        header[-1] += '0'
        header += ['uncertainty']
        shutil.move(path, path+'.backup')
        with open(path+'.backup', 'r') as backup:
            reader = csv.reader(backup)
            old_header = next(reader)
            with open(path, 'w') as fixed:
                writer = csv.writer(fixed)
                writer.writerow(header)
                for row in reader:
                    writer.writerow(row)

    return pd.read_csv(path)

def read_group_predictions(path):
    '''Reads patient- or slide-level predictions CSV file, returning y_true and y_pred'''

    # Read into pandas dataframe
    df = load_and_fix_patient_pred(path)
    level = 'patient' if 'patient' in df.columns else 'slide'
    y_true = df['y_true1'].to_numpy()
    y_pred = df['percent_tiles_positive1'].to_numpy()
    return y_true, y_pred

def prediction_metrics(y_true, y_pred, threshold):
    yt = y_true.astype(bool)
    yp = y_pred > threshold

    alpha = 0.05
    z = stats.norm.ppf((1 - alpha/2))
    tp = np.logical_and(yt, yp).sum()
    fp = np.logical_and(np.logical_not(yt), yp).sum()
    tn = np.logical_and(np.logical_not(yt), np.logical_not(yp)).sum()
    fn = np.logical_and(yt, np.logical_not(yp)).sum()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Youden's confidence interval, via BAC (bootstrap AC estimate)
    # Bootstrapping is performed with sample size n = 100 and iterations B = 500
    all_jac = []
    for _ in range(500):
        bootstrap_i = np.random.choice(np.arange(yt.shape[0]), size=(150,))
        _yt = yt[bootstrap_i]
        _yp = yp[bootstrap_i]
        _tp = np.logical_and(_yt, _yp).sum()
        _fp = np.logical_and(np.logical_not(_yt), _yp).sum()
        _tn = np.logical_and(np.logical_not(_yt), np.logical_not(_yp)).sum()
        _fn = np.logical_and(_yt, np.logical_not(_yp)).sum()
        _jac = ((_tn + 0.5 * z**2) / (_tn + _fp + z**2)) - ((_fn + 0.5 * z**2) / (_fn + _tp + z**2))
        all_jac += [_jac]
    jac = mean(all_jac)
    jac_var = variance(all_jac)
    jac_low = jac - z * np.sqrt(jac_var)
    jac_high = jac + z * np.sqrt(jac_var)

    # AUC confidence intervals
    if not np.array_equal(np.unique(y_true), [0, 1]):
        sf.util.log.warn(f"Unable to calculate CI; NaNs exist")
        ci = [None, None]
    else:
        delong_auc, auc_cov = delong_roc_variance(y_true, y_pred)
        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - alpha / 2)
        ci = stats.norm.ppf(lower_upper_q, loc=delong_auc, scale=auc_std)
        ci[ci > 1] = 1

    return {
        'auc_low': ci[0],
        'auc_high': ci[1],
        'acc': acc,
        'sens': sensitivity,
        'spec': specificity,
        'youden': sensitivity + specificity - 1,
        'youden_low': jac_low,
        'youden_high': jac_high,
    }

def auc_and_threshold(y_true, y_pred):
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_threshold = threshold[list(zip(tpr,fpr)).index(max(zip(tpr, fpr), key=lambda x: x[0]-x[1]))]
    return roc_auc, optimal_threshold

def auc(y_true, y_pred):
    try:
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
        return metrics.auc(fpr, tpr)
    except ValueError:
        sf.util.log.warn("Unable to calculate ROC")
        return np.nan
