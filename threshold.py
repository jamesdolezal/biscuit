import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import slideflow as sf

from slideflow.errors import *
from skmisc.loess import loess
from sklearn import metrics
from slideflow.util import log

color_palette = {
    'negative': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), # blue
    'positive': (1.0, 0.4980392156862745, 0.054901960784313725)				   # orange
}

def plot_uncertainty(df, kind, threshold=None, title=None):
    '''Plots figure of tile or slide-level predictions vs. uncertainty with matplotlib.

    Args:
        df (pandas.DataFrame): Processed dataframe containing columns 'uncertainty', 'correct', 'y_pred'.
        kind (str): Kind of plot. If 'tile', will subsample to only 1000 points. Included in title.
        threshold (float, optional): Uncertainty threshold. Defaults to None.
        title (str, optional): Title for plots. Defaults to None.

    Returns:
        None
    '''

    # Subsample tile-level predictions
    if kind == 'tile':
        df = df.sample(n=1000)

    f, axes = plt.subplots(1, 3)
    f.set_size_inches(15, 5)
    palette = sns.color_palette("Set2")
    tf_pal = {True: palette[0], False: palette[1]}

    # Left figure - KDE -------------------------------------------------------
    kde = sns.kdeplot(x='uncertainty', hue='correct', data=df, fill=True, palette=tf_pal, ax=axes[0])
    kde.set(xlabel='Uncertainty')
    axes[0].title.set_text(f'Uncertainty density ({kind}-level)')

    # Middle figure - Scatter --------------------------------------------------
    axes[1].axhline(y=threshold, color='r', linestyle='--')

    # - Above threshold
    if threshold is not None:
        at_df = df.loc[(df['uncertainty'] >= threshold)]
        c_a_df = at_df.loc[at_df['correct']]
        ic_a_df = at_df.loc[~at_df['correct']]
        axes[1].scatter(x=c_a_df['y_pred'], y=c_a_df['uncertainty'], marker='o', s=10, color='gray')
        axes[1].scatter(x=ic_a_df['y_pred'], y=ic_a_df['uncertainty'], marker='x', color='#FC6D77')

    # - Below threshold
    bt_df = df.loc[(df['uncertainty'] < threshold)] if threshold is not None else df
    c_df = bt_df.loc[bt_df['correct']]
    ic_df = bt_df.loc[~bt_df['correct']]
    axes[1].scatter(x=c_df['y_pred'], y=c_df['uncertainty'], marker='o', s=10)
    axes[1].scatter(x=ic_df['y_pred'], y=ic_df['uncertainty'], marker='x', color='red')
    if title is not None:
        axes[1].title.set_text(title)

    # Right figure - probability calibration ----------------------------------
    l_df = df[['uncertainty', 'correct']].sort_values(by=['uncertainty'])
    x = l_df['uncertainty'].to_numpy()
    y = l_df['correct'].astype(float).to_numpy()
    l = loess(x, y)
    l.fit()
    pred = l.predict(x, stderror=True)
    conf = pred.confidence()
    z = pred.values
    ll = conf.lower
    ul = conf.upper
    axes[2].plot(x, y, '+', ms=6)
    axes[2].plot(x, z)
    axes[2].fill_between(x,ll,ul,alpha=.2)
    axes[2].tick_params(labelrotation=90)
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].axvline(x=threshold, color='r', linestyle='--')

    # - Figure style
    for ax in (axes[1], axes[2]):
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['bottom'].set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.grid(visible=True, which='both', axis='both', color='white')
        ax.set_facecolor('#EAEAF2')

def process_tile_predictions(df, pred_thresh=0.5, patients=None):
    '''Load and process tile-level predictions from CSV.

    Args:
        df (pandas.DataFrame): Unprocess DataFrame from reading tile-level predictions.
        pred_thresh (float or str, optional): Tile-level prediction threshold. If 'detect', will auto-detect via
            Youden's J. Defaults to 0.5.
        patients (dict, optional): Dict mapping slides to patients, used for patient-level thresholding.
            Defaults to None.

    Returns:
        pandas.DataFrame, tile prediction threshold
    '''

    # Tile-level AUC
    fpr, tpr, thresh = metrics.roc_curve(df['y_true'].to_numpy(), df['y_pred'].to_numpy())
    tile_auc = metrics.auc(fpr, tpr)
    try:
        opt_pred = thresh[list(zip(tpr,fpr)).index(max(zip(tpr,fpr), key=lambda x: x[0]-x[1]))]
    except ValueError:
        log.debug(f"Unable to calculate tile-level prediction threshold; defaulting to 0.5")
        opt_pred = 0.5

    if pred_thresh == 'detect':
        log.debug(f"Using optimal, auto-detected prediction threshold (Youden's J): {opt_pred:.4f}")
        pred_thresh = opt_pred
    else:
        log.debug(sf.util.blue(f"Using tile prediction threshold: {pred_thresh:.4f}"))

    if patients is not None:
        df['patient'] = df['slide'].map(patients)
    else:
        log.warn('Patient dict not provided; assuming 1:1 mapping of slides to patients')

    log.debug(f'Tile AUC: {tile_auc:.4f}')

    # Calculate tile-level prediction accuracy
    df['error'] = abs(df['y_true'] - df['y_pred'])
    df['correct'] = ((df['y_pred'] < pred_thresh) & (df['y_true'] == 0)) | ((df['y_pred'] >= pred_thresh) & (df['y_true'] == 1))
    df['incorrect'] = (~df['correct']).astype(int)
    df['y_pred_bin'] = (df['y_pred'] >= pred_thresh).astype(int)

    return df, pred_thresh

def process_group_predictions(df, pred_thresh, level):
    '''From a given dataframe of tile-level predictions, calculate group-level predictions and uncertainty.'''

    # Calculate group-level predictions
    log.debug(f'Calculating {level}-level means')
    levels = pd.unique(df[level])
    grouped_mean = df[[level, 'y_pred', 'y_true', 'uncertainty']].groupby(level, as_index=False).mean()
    yp = np.array([grouped_mean.loc[grouped_mean[level]==l]['y_pred'].to_numpy()[0] for l in levels])
    yt = np.array([grouped_mean.loc[grouped_mean[level]==l]['y_true'].to_numpy()[0] for l in levels], dtype=np.uint8)
    u =  np.array([grouped_mean.loc[grouped_mean[level]==l]['uncertainty'].to_numpy()[0] for l in levels])

    # Slide-level AUC
    log.debug(f'Calculating {level}-level ROC')
    l_fpr, l_tpr, l_thresh = metrics.roc_curve(yt, yp)
    log.debug('Calculating AUC')
    level_auc = metrics.auc(l_fpr, l_tpr)
    log.debug('Calculating optimal threshold')

    if pred_thresh == 'detect':
        pred_thresh = l_thresh[list(zip(l_tpr,l_fpr)).index(max(zip(l_tpr,l_fpr), key=lambda x: x[0]-x[1]))]
        log.debug(f"Using optimal, auto-detected prediction threshold: {pred_thresh:.4f}")
    else:
        log.debug(sf.util.blue(f"Using {level} prediction threshold: {pred_thresh:.4f}"))

    log.debug(f'{level} AUC: {level_auc:.4f}')

    l_df = pd.DataFrame({
        level: pd.Series(levels),
        'error': pd.Series(abs(yt - yp)),
        'uncertainty': pd.Series(u),
        'correct': ((yp < pred_thresh) & (yt == 0)) | ((yp >= pred_thresh) & (yt == 1)), #pd.Series(abs(yt - yp) < 0.5),#
        'incorrect': pd.Series(((yp < pred_thresh) & (yt == 1)) | ((yp >= pred_thresh) & (yt == 0))).astype(int),#pd.Series(abs(yt - yp) >= 0.5).astype(int)
        'y_true': pd.Series(yt),
        'y_pred': pd.Series(yp),
        'y_pred_bin': pd.Series(yp >= pred_thresh).astype(int)
    })

    return l_df, pred_thresh

def apply(df, thresh_tile, thresh_slide, tile_pred_thresh=0.5, slide_pred_thresh=0.5, plot=False,
                    keep='high_confidence', title=None, patients=None, level='slide'):

    '''Apply pre-calculcated tile- and group-level uncertainty thresholds.

    Args:
        df (pandas.DataFrame): Must contain columns 'y_true', 'y_pred', and 'uncertainty'.
        thresh_tile (float): Tile-level uncertainty threshold.
        thresh_slide (float): Slide-level uncertainty threshold.
        tile_pred_thresh (float, optional): Tile-level prediction threshold. Defaults to 0.5.
        slide_pred_thresh (float, optional): Slide-level prediction threshold. Defaults to 0.5.
        plot (bool, optional): Plot slide-level uncertainty. Defaults to False.
        keep (str, optional): Either 'high_confidence' or 'low_confidence'. Cohort to keep after thresholding.
            Defaults to 'high_confidence'.
        title (str, optional): Title for uncertainty plot. Defaults to None.
        patients (dict, optional): Dictionary mapping slides to patients. Adds a 'patient' column in the tile prediction
            dataframe, enabling patient-level thresholding. Defaults to None.
        level (str, optional): Either 'slide' or 'patient'. Level at which to apply threshold. If 'patient', requires
            patient dict be supplied. Defaults to 'slide'.

    Returns:
        auc, percent_incl, accuracy, sensitivity, specificity
    '''

    assert keep in ('high_confidence', 'low_confidence')
    assert not (level == 'patient' and patients is None)

    log.debug(sf.util.purple(f"Using tile uncertainty threshold of {thresh_tile:.5f}"))
    df, _ = process_tile_predictions(df, pred_thresh=tile_pred_thresh, patients=patients)
    num_pre_filter = pd.unique(df[level]).shape[0]

    if thresh_tile:
        df = df[df['uncertainty'] < thresh_tile]

    log.debug(f"Number of {level} after filter: {pd.unique(df[level]).shape[0]}")
    log.debug(f"Number of tiles after filter: {len(df)}")

    # Build group-level predictions
    s_df, _ = process_group_predictions(df, pred_thresh=slide_pred_thresh, level=level)

    if plot:
        plot_uncertainty(s_df, threshold=thresh_slide, kind=level, title=title)

    # Apply slide-level thresholds
    if thresh_slide:
        log.debug(sf.util.purple(f"Using {level} uncertainty threshold of {thresh_slide:.5f}"))
        if keep == 'high_confidence':
            s_df = s_df[s_df['uncertainty'] < thresh_slide]
        elif keep == 'low_confidence':
            s_df = s_df[s_df['uncertainty'] >= thresh_slide]
        else:
            raise Exception(f"Unknown keep option {keep}")

    # Show post-filtering group-level predictions and AUC
    auc = sf.stats.auc(s_df['y_true'].to_numpy(), s_df['y_pred'].to_numpy())
    num_post_filter = len(s_df)
    percent_incl = num_post_filter / num_pre_filter
    log.debug(f"Percent {level} included: {percent_incl*100:.2f}%")

    # Calculate post-thresholded sensitivity/specificity
    y_true = s_df['y_true'].to_numpy().astype(bool)
    y_pred = s_df['y_pred'].to_numpy() > slide_pred_thresh

    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(np.logical_not(y_true), y_pred).sum()
    tn = np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)).sum()
    fn = np.logical_and(y_true, np.logical_not(y_pred)).sum()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    log.debug(f"Accuracy: {acc:.4f}")
    log.debug(f"Sensitivity: {sensitivity:.4f}")
    log.debug(f"Specificity: {specificity:.4f}")

    return auc, percent_incl, acc, sensitivity, specificity

def detect(df, tile_uq_thresh='detect', slide_uq_thresh='detect', tile_pred_thresh='detect',
                     slide_pred_thresh='detect', plot=False, patients=None):
    '''Detect optimal tile- and slide-level uncertainty thresholds.

    Args:
        df (pandas.DataFrame): Tile-level predictions. Must contain columns 'y_true', 'y_pred', and 'uncertainty'.
        tile_uq_thresh (str or float): Either 'detect' or float. If 'detect', will detect tile-level uncertainty
            threshold. If float, will use the specified tile-level uncertainty threshold.
        slide_uq_thresh (str or float): Either 'detect' or float. If 'detect', will detect slide-level uncertainty
            threshold. If float, will use the specified slide-level uncertainty threshold.
        tile_pred_thresh (str or float): Either 'detect' or float. If 'detect', will detect tile-level prediction
            threshold. If float, will use the specified tile-level prediction threshold.
        slide_pred_thresh (str or float): Either 'detect' or float. If 'detect', will detect slide-level prediction
            threshold. If float, will use the specified slide-level prediction threshold.
        plot (bool, optional): Plot slide-level uncertainty. Defaults to False.
        patients (dict, optional): Dict mapping slides to patients. Required for patient-level thresholding.

    Returns:
        Tile UQ threshold, Slide UQ threshold, AUC, Tile prediction threshold, Slide prediction threshold
    '''

    df, tile_pred_thresh = process_tile_predictions(df, pred_thresh=tile_pred_thresh, patients=patients)

    # Tile-level ROC and Youden's J
    if isinstance(tile_uq_thresh, float):
        thresh_tile = tile_uq_thresh
        df = df[df['uncertainty'] < thresh_tile]
    elif tile_uq_thresh != 'detect':
        log.debug("Not performing tile-level uncertainty thresholding.")
        thresh_tile = None
    else:
        t_fpr, t_tpr, t_thresh = metrics.roc_curve(df['incorrect'].to_numpy(), df['uncertainty'].to_numpy())
        thresh_tile = t_thresh[list(zip(t_tpr,t_fpr)).index(max(zip(t_tpr,t_fpr), key=lambda x: x[0] - x[1]))]
        log.debug(f"Tile-level optimal uncertainty threshold: {thresh_tile:.4f}")
        df = df[df['uncertainty'] < thresh_tile]

    slides = list(set(df['slide']))
    log.debug(f"Number of slides after filter: {len(slides)}")
    log.debug(f"Number of tiles after filter: {len(df)}")

    # Build slide-level predictions
    try:
        s_df, slide_pred_thresh = process_group_predictions(df, pred_thresh=slide_pred_thresh, level='slide')
    except ValueError:
        log.error(f"Unable to process slide predictions")
        return None, None, None, None, None

    # Slide-level thresholding
    if slide_uq_thresh == 'detect':
        if not s_df['incorrect'].to_numpy().sum():
            log.debug("Unable to calculate slide UQ threshold; no incorrect predictions made")
            thresh_slide = None
        else:
            s_fpr, s_tpr, s_thresh = metrics.roc_curve(s_df['incorrect'], s_df['uncertainty'].to_numpy())
            thresh_slide = s_thresh[list(zip(s_tpr,s_fpr)).index(max(zip(s_tpr,s_fpr), key=lambda x: x[0]-x[1]))]
            log.debug(f"Slide-level optimal uncertainty threshold: {thresh_slide:.4f}")
            if plot:
                plot_uncertainty(s_df, threshold=thresh_slide, kind='slide')
            s_df = s_df[s_df['uncertainty'] < thresh_slide]
    else:
        log.debug("Not performing slide-level uncertainty thresholding.")
        thresh_slide = 0.5
        if plot:
            plot_uncertainty(s_df, threshold=thresh_slide, kind='slide')

    # Show post-filtering slide predictions and AUC
    auc = sf.stats.auc(s_df['y_true'].to_numpy(), s_df['y_pred'].to_numpy())

    return thresh_tile, thresh_slide, auc, tile_pred_thresh, slide_pred_thresh

def from_cv(k_paths, y_pred_header='y_pred1', y_true_header='y_true0', uncertainty_header='uncertainty', **kwargs):
    '''Finds the optimal tile and slide-level thresholds from a set of nested cross-validation experiments.

    Args:
        k_paths (list(str)): List of paths to tile predictions in CSV format.
        y_pred_header (str, optional): Header indicating tile prediction. Defaults to 'y_pred1'.
        y_true_header (str, optional): Header indicating ground-truth label. Defaults to 'y_true0'.
        uncertainty_header (str, optional): Header indicating tile uncertainty. Defaults to 'uncertainty'.

    Keyword args:
        tile_uq_thresh (str or float): Either 'detect' or float. If 'detect', will detect tile-level uncertainty
            threshold. If float, will use the specified tile-level uncertainty threshold.
        slide_uq_thresh (str or float): Either 'detect' or float. If 'detect', will detect slide-level uncertainty
            threshold. If float, will use the specified slide-level uncertainty threshold.
        tile_pred_thresh (str or float): Either 'detect' or float. If 'detect', will detect tile-level prediction
            threshold. If float, will use the specified tile-level prediction threshold.
        slide_pred_thresh (str or float): Either 'detect' or float. If 'detect', will detect slide-level prediction
            threshold. If float, will use the specified slide-level prediction threshold.
        plot (bool, optional): Plot slide-level uncertainty. Defaults to False.
        patients (dict, optional): Dict mapping slides to patients. Required for patient-level thresholding.

    Returns:
        Tile UQ threshold, Slide UQ threshold, Tile prediction threshold, Slide prediction threshold
    '''

    k_tile_thresh, k_slide_thresh = [], []
    k_tile_pred_thresh, k_slide_pred_thresh = [], []
    k_auc = []
    skip_tile = 'tile_uq_thresh' in kwargs and kwargs['tile_uq_thresh'] is None
    skip_slide = 'slide_uq_thresh' in kwargs and kwargs['slide_uq_thresh'] is None

    for p, path in enumerate(k_paths):
        df = pd.read_csv(path)
        df.rename(columns={y_pred_header: 'y_pred', y_true_header: 'y_true', uncertainty_header: 'uncertainty'}, inplace=True)
        thresh_tile, thresh_slide, auc, slide_pred_thresh, tile_pred_thresh = detect(df, **kwargs)
        if thresh_tile is None or thresh_slide is None:
            log.debug(f"Skipping CV #{p}, unable to detect threshold")
            continue

        k_slide_pred_thresh += [slide_pred_thresh]
        k_tile_pred_thresh += [tile_pred_thresh]
        k_auc += [auc]

        if not skip_tile:
            k_tile_thresh += [thresh_tile]
        if not skip_slide:
            k_slide_thresh += [thresh_slide]

    if not skip_tile and not len(k_tile_thresh):
        raise ThresholdError('Unable to detect tile uncertainty threshold.')
    if not skip_slide and not len(k_slide_thresh):
        raise ThresholdError('Unable to detect slide uncertainty threshold.')

    k_slide_pred_thresh = np.mean(k_slide_pred_thresh)
    k_tile_pred_thresh = np.mean(k_tile_pred_thresh)

    if not skip_tile:
        k_tile_thresh = np.min(k_tile_thresh)
    if not skip_slide:
        k_slide_thresh = np.max(k_slide_thresh)

    return k_tile_thresh, k_slide_thresh, k_tile_pred_thresh, k_slide_pred_thresh