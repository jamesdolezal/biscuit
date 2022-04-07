import click
import multiprocessing
import experiment
import re
from typing import List
from os.path import join

from experiment import EXP_NAME_MAP

# Assumes a TCGA LUAD v LUSC project exists,
# with tiles already extracted.

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

@click.command()
@click.option('--steps', type=num_range, help='List of training steps to perform')
@click.option('--reg', type=bool, help='Train regular models', default=True, show_default=True)
@click.option('--ratio', type=bool, help='Train ratio experiments', default=True, show_default=True)
@click.option('--gan', type=bool, help='Train gan experiments', default=False, show_default=True)
@click.option('--train_project', type=str, help='Manually specify location of training project')
@click.option('--eval_project', type=str, help='Manually specify location of training project')
def train_models(steps=None, reg=True, ratio=True, gan=False, train_project=None, eval_project=None):

    # --- Configure experiments -----------------------------------------------

    experiment.TRAIN_PATH = join('projects', 'training') if train_project is None else train_project
    experiment.EVAL_PATHS = [join('projects', 'evaluation')] if eval_project is None else eval_project

    if steps is None: steps=range(7)
    to_run = []

    # Configure regular experiments
    if reg:
        reg_exp = experiment.config('{}', EXP_NAME_MAP, 1, order='forward')
        reg2_exp = experiment.config('{}2', EXP_NAME_MAP, 1, order='forward', order_col='order2')
        reverse_exp = experiment.config('{}_R', EXP_NAME_MAP, 1, order='reverse')
        reverse2_exp = experiment.config('{}_R2', EXP_NAME_MAP, 1, order='reverse', order_col='order2')
        to_run += [reg_exp, reg2_exp, reverse_exp, reverse2_exp]

    # Configure 3:1 and 10:1 ratio experiments
    if ratio:
        ratio_3 = experiment.config('{}_3', ['A', 'M', 'D', 'P', 'G', 'Z'], 3, order='forward')
        ratio_3_rev = experiment.config('{}_R_3', ['A', 'M', 'D', 'P', 'G', 'Z'], 3, order='reverse')
        ratio_10 = experiment.config('{}_10', ['A', 'M', 'D', 'P', 'G', 'Z'], 10, order='forward')
        ratio_10_rev = experiment.config('{}_R_10', ['A', 'M', 'D', 'P', 'G', 'Z'], 10, order='reverse')
        to_run += [ratio_3, ratio_3_rev, ratio_10, ratio_10_rev]

    # GAN experiments
    if gan:
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
        to_run += [gan_exp]

    # --- Train experiments ---------------------------------------------------

    for exp in to_run:
        experiment.run(exp, steps=steps)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_models() # pylint: disable=no-value-for-parameter
