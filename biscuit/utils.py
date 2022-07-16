import csv
import os
import shutil
from os.path import join
from statistics import mean, variance

import matplotlib.colors as colors
import numpy as np
import pandas as pd
import slideflow as sf
from scipy import stats
from sklearn import metrics

from biscuit.delong import delong_roc_variance
from biscuit.errors import ModelNotFoundError, MultipleModelsFoundError

OUTCOME = 'cohort'
OUTCOME1 = 'LUAD'
OUTCOME2 = 'LUSC'


def uncertainty_header(o=None):
    if o is None:
        o = OUTCOME
    return f'{o}_uncertainty1'


def y_true_header(o=None):
    if o is None:
        o = OUTCOME
    return f'{o}_y_true0'


def y_pred_header(o=None):
    if o is None:
        o = OUTCOME
    return f'{o}_y_pred1'


# --- General utility functions -----------------------------------------------

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncates matplotlib colormap."""

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def get_model_results(path, epoch, outcome=None):
    """Reads results/metrics from a trained model.

    Args:
        path (str): Path to model.
        outcome (str, optional): Outcome name. If none, uses default
            (biscuit.utils.OUTCOME). Defaults to None.

    Returns:
        Dict of results with the keys: pt_auc, pt_ap, slide_auc, slide_ap,
            tile_auc, tile_ap, opt_thresh
    """
    if outcome is None:
        outcome = OUTCOME
    csv = pd.read_csv(join(path, 'results_log.csv'))
    result_rows = {}
    for i, row in csv.iterrows():
        row_epoch = int(row['model_name'].split('epoch')[-1])
        result_rows.update({
            row_epoch: row
        })
    if epoch not in result_rows:
        raise ModelNotFoundError(f"Unable to find results for epoch {epoch}")
    model_res = result_rows[epoch]
    pt_ap = mean(eval(model_res['patient_ap'])[outcome])
    pt_auc = eval(model_res['patient_auc'])[outcome][0]
    slide_ap = mean(eval(model_res['slide_ap'])[outcome])
    slide_auc = eval(model_res['slide_auc'])[outcome][0]
    tile_ap = mean(eval(model_res['tile_ap'])[outcome])
    tile_auc = eval(model_res['tile_auc'])[outcome][0]
    pred_path = join(
        path,
        f'patient_predictions_{outcome}_val_epoch{epoch}.csv'
    )
    if os.path.exists(pred_path):
        _, opt_thresh = auc_and_threshold(*read_group_predictions(pred_path))
    else:
        opt_thresh = None
    return {
        'pt_auc': pt_auc,
        'pt_ap': pt_ap,
        'slide_auc': slide_auc,
        'slide_ap': slide_ap,
        'tile_auc': tile_auc,
        'tile_ap': tile_ap,
        'opt_thresh': opt_thresh
    }


def find_cv_early_stop(project, label, k=3, outcome=None):
    """Detects early stop batch from cross-val trained models.

    Args:
        project (slideflow.Project): Project.
        label (str): Experimental label.
        k (int, optional): Number of k-fold iterations. Defaults to 3.
        outcome (str, optional): Outcome name. If none, uses default
            (biscuit.utils.OUTCOME). Defaults to None.

    Returns:
        int: Early stop batch.
    """
    cv_folders = find_cv(project, label, k=k, outcome=outcome)
    early_stop_batch = []
    for cv_folder in cv_folders:
        csv = pd.read_csv(join(cv_folder, 'results_log.csv'))
        model_res = next(csv.iterrows())[1]
        if 'early_stop_batch' in model_res:
            early_stop_batch += [model_res['early_stop_batch']]
    if len(early_stop_batch) == len(cv_folders):
        # Only returns early stop if it was triggered in all crossfolds
        return round(mean(early_stop_batch))
    else:
        return None


# --- Utility functions for finding experiment models -------------------------

def find_model(project, label, epoch=None, kfold=None, outcome=None):
    """Searches for a model in a project model directory.

    Args:
        project (slideflow.Project): Project.
        label (str): Experimental label.
        epoch (int, optional): Epoch to search for. If not None, returns
            path to the saved model. If None, returns path to parent model
            folder. Defaults to None.
        kfold (int, optional): K-fold iteration. Defaults to None.
        outcome (str, optional): Outcome name. If none, uses default
            (biscuit.utils.OUTCOME). Defaults to None.

    Raises:
        MultipleModelsFoundError: If multiple potential matches are found.
        ModelNotFoundError: If no matching model is found.

    Returns:
        str: Path to matching model.
    """
    if outcome is None:
        outcome = OUTCOME
    tail = '' if kfold is None else f'-kfold{kfold}'
    model_name = f'{outcome}-{label}-HP0{tail}'
    matching = [
        o for o in os.listdir(project.models_dir)
        if o[6:] == model_name
    ]
    if len(matching) > 1:
        raise MultipleModelsFoundError("Multiple matching models found "
                                       f"matching {model_name}")
    elif not len(matching):
        raise ModelNotFoundError("No matching model found matching "
                                 f"{model_name}.")
    elif epoch is not None:
        return join(
            project.models_dir,
            matching[0],
            f'{outcome}-{label}-HP0{tail}_epoch{epoch}'
        )
    else:
        return join(project.models_dir, matching[0])


def model_exists(project, label, epoch=None, kfold=None, outcome=None):
    """Check if matching model exists.

    Args:
        project (slideflow.Project): Project.
        label (str): Experimental label.
        epoch (int, optional): Epoch number of saved model. Defaults to None.
        kfold (int, optional): K-fold iteration. Defaults to None.
        outcome (str, optional): Outcome name. If none, uses default
            (biscuit.utils.OUTCOME). Defaults to None.

    Returns:
        bool: If model exists
    """
    try:
        find_model(project, label, epoch, kfold=kfold, outcome=outcome)
        return True
    except ModelNotFoundError:
        return False


def find_cv(project, label, epoch=None, k=3, outcome=None):
    """Finds paths to cross-validation models.

    Args:
        project (slideflow.Project): Project.
        label (str): Experimental label.
        epoch (int, optional): Epoch number of saved model. Defaults to None.
        kfold (int, optional): K-fold iteration. Defaults to None.
        outcome (str, optional): Outcome name. If none, uses default
            (biscuit.utils.OUTCOME). Defaults to None.

    Returns:
        list(str): Paths to cross-validation models.
    """
    return [
        find_model(project, label, epoch=epoch, kfold=_k, outcome=outcome)
        for _k in range(1, k+1)
    ]


def find_eval(project, label, epoch=1, outcome=None):
    """Finds matching eval directory.

    Args:
        project (slideflow.Project): Project.
        label (str): Experimental label.
        epoch (int, optional): Epoch number of saved model. Defaults to None.
        outcome (str, optional): Outcome name. If none, uses default
            (biscuit.utils.OUTCOME). Defaults to None.

    Raises:
        MultipleModelsFoundError: If multiple matches are found.
        ModelNotFoundError: If no match is found.

    Returns:
        str: path to eval directory
    """
    if outcome is None:
        outcome = outcome
    matching = [
        o for o in os.listdir(project.eval_dir)
        if o[11:] == f'{OUTCOME}-{label}-HP0_epoch{epoch}'
    ]
    if len(matching) > 1:
        raise MultipleModelsFoundError("Multiple matching eval experiments "
                                       f"found for label {label}")
    elif not len(matching):
        raise ModelNotFoundError(f"No matching eval found for label {label}")
    else:
        return join(project.eval_dir, matching[0])


def eval_exists(project, label, epoch=1):
    """Check if matching eval exists.

    Args:
        project (slideflow.Project): Project.
        label (str): Experimental label.
        epoch (int, optional): Epoch number of saved model. Defaults to None.

    Returns:
        bool: If eval exists
    """
    try:
        find_eval(project, label, epoch)
        return True
    except ModelNotFoundError:
        return False


# --- Thresholding and metrics functions --------------------------------------

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
            _ = next(reader)
            with open(path, 'w') as fixed:
                writer = csv.writer(fixed)
                writer.writerow(header)
                for row in reader:
                    writer.writerow(row)
    return pd.read_csv(path)


def read_group_predictions(path):
    '''Reads patient- or slide-level predictions CSV file,
    returning y_true and y_pred
    '''
    df = load_and_fix_patient_pred(path)
    y_true = df['y_true1'].to_numpy()
    y_pred = df['percent_tiles_positive1'].to_numpy()
    return y_true, y_pred


def prediction_metrics(y_true, y_pred, threshold):
    """Calculate prediction metrics (AUC, sensitivity/specificity, etc)

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predictions.
        threshold (_type_): Prediction threshold.

    Returns:
        dict: Prediction metrics.
    """
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
    # Bootstrapping performed with sample size n = 100 and iterations B = 500
    all_jac = []
    for _ in range(500):
        bootstrap_i = np.random.choice(np.arange(yt.shape[0]), size=(150,))
        _yt = yt[bootstrap_i]
        _yp = yp[bootstrap_i]
        _tp = np.logical_and(_yt, _yp).sum()
        _fp = np.logical_and(np.logical_not(_yt), _yp).sum()
        _tn = np.logical_and(np.logical_not(_yt), np.logical_not(_yp)).sum()
        _fn = np.logical_and(_yt, np.logical_not(_yp)).sum()
        _jac = (((_tn + 0.5 * z**2) / (_tn + _fp + z**2))
                - ((_fn + 0.5 * z**2) / (_fn + _tp + z**2)))
        all_jac += [_jac]

    jac = mean(all_jac)
    jac_var = variance(all_jac)
    jac_low = jac - z * np.sqrt(jac_var)
    jac_high = jac + z * np.sqrt(jac_var)

    # AUC confidence intervals
    if not np.array_equal(np.unique(y_true), [0, 1]):
        sf.util.log.warn("Unable to calculate CI; NaNs exist")
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
    """Calculates AUC and optimal threshold (via Youden's J)

    Args:
        y_true (np.ndarray): Y true (labels).
        y_pred (np.ndarray): Y pred (predictions).

    Returns:
        float: AUC
        float: Optimal threshold
    """
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    max_j = max(zip(tpr, fpr), key=lambda x: x[0]-x[1])
    optimal_threshold = threshold[list(zip(tpr, fpr)).index(max_j)]
    return roc_auc, optimal_threshold


def auc(y_true, y_pred):
    """Calculate Area Under Receiver Operator Curve (AUC / AUROC)

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predictions.

    Returns:
        Float: AUC
    """
    try:
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
        return metrics.auc(fpr, tpr)
    except ValueError:
        sf.util.log.warn("Unable to calculate ROC")
        return np.nan
