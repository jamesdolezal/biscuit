'''Script to execute training. Assumes projects have already been created
and tiles have already been extracted from slides.
'''

import click
import multiprocessing
import re
from typing import List
from os.path import join

import biscuit
from biscuit.experiment import config, run
from biscuit.experiment import ALL_EXP


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a
    range 'a-c' and return as a list of ints.
    '''
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


@click.command()
@click.option('--steps', type=num_range, help='Training steps to perform')
@click.option('--reg', type=bool, help='Train regular models', default=True)
@click.option('--ratio', type=bool, help='Train ratio models', default=True)
@click.option('--gan', type=bool, help='Train gan models', default=False)
@click.option('--train_project', type=str, help='Override training project')
@click.option('--eval_project', type=str, help='Override eval project')
def train_models(steps=None, reg=True, ratio=True, gan=False,
                 train_project=None, eval_project=None):

    # --- Configure experiments -----------------------------------------------
    if train_project is None:
        biscuit.experiment.TRAIN_PATH = join('projects', 'training')
    else:
        biscuit.experiment.TRAIN_PATH = train_project
    if eval_project is None:
        biscuit.experiment.EVAL_PATHS = [join('projects', 'evaluation')]
    else:
        biscuit.experiment.EVAL_PATHS = [eval_project]
    if steps is None:
        steps = range(7)
    to_run = []

    # Configure regular experiments
    if reg:
        reg1 = config('{}', ALL_EXP, 1, order='f')
        reg2 = config('{}2', ALL_EXP, 1, order='f', order_col='order2')
        rev1 = config('{}_R', ALL_EXP, 1, order='r')
        rev2 = config('{}_R2', ALL_EXP, 1, order='r', order_col='order2')
        to_run += [reg1, reg2, rev1, rev2]

    # Configure 3:1 and 10:1 ratio experiments
    if ratio:
        ratio_exp = list('AMDPGZ')
        ratio_3 = config('{}_3', ratio_exp, 3, order='f')
        ratio_3_rev = config('{}_R_3', ratio_exp, 3, order='r')
        ratio_10 = config('{}_10', ratio_exp, 10, order='f')
        ratio_10_rev = config('{}_R_10', ratio_exp, 10, order='r')
        to_run += [ratio_3, ratio_3_rev, ratio_10, ratio_10_rev]

    # GAN experiments
    if gan:
        _g = list('RALMNDOPQGWY') + ['ZA', 'ZC']
        gan_exp = {}
        gan_exp.update(config('{}_g10', _g, 1, gan=0.1, order='f'))
        gan_exp.update(config('{}_R_g10', _g, 1, gan=0.1, order='r'))
        gan_exp.update(config('{}_g20', _g, 1, gan=0.2, order='f'))
        gan_exp.update(config('{}_R_g20', _g, 1, gan=0.2, order='r'))
        gan_exp.update(config('{}_g30', _g, 1, gan=0.3, order='f'))
        gan_exp.update(config('{}_R_g30', _g, 1, gan=0.3, order='r'))
        gan_exp.update(config('{}_g40', _g, 1, gan=0.4, order='f'))
        gan_exp.update(config('{}_R_g40', _g, 1, gan=0.4, order='r'))
        gan_exp.update(config('{}_g50', _g, 1, gan=0.5, order='f'))
        gan_exp.update(config('{}_R_g50', _g, 1, gan=0.5, order='r'))
        to_run += [gan_exp]

    # --- Train experiments ---------------------------------------------------
    for experiment in to_run:
        run(experiment, steps=steps)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_models()  # pylint: disable=no-value-for-parameter
