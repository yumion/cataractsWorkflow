#!/usr/bin/env python

"""
Evaluation script for the EndoVis / CATARACTS 2020 challenge.

Usage:
$ python evaluate.py --gt <truth directory> --predictions <prediction directory>

Where :
* the <truth directory> contains one CSV file with ground truth annotations for each video in the
  test set,
* the <prediction directory> contains one CSV file with automatic predictions for each video in the
  test set.
In both directories, CSV files should be named 'test<video index>.csv' (test01.csv, test02.csv, ...,
test20.csv).

The following dependencies must be installed: pandas, scikit-learn.
$ pip install pandas==0.23.4 scikit-learn==0.20.2
"""

__author__ = 'Hassan ALHAJJ (hassan.alhajj@univ-brest.fr)'
__copyright__ = 'Copyright (C) 2020 Hassan ALHAJJ'
__license__ = 'Apache Version 2.0'
__version__ = '0.1'

import os
import glob
import numpy as np
from argparse import ArgumentParser
from pandas import read_csv
from sklearn.metrics import f1_score

steps = ['Toric Marking', 'Implant Ejection', 'Incision', 'Viscodilatation', 'Capsulorhexis', 'Hydrodissetion',
         'Nucleus Breaking', 'Phacoemulsification', 'Vitrectomy', 'Irrigation/Aspiration', 'Preparing Implant',
         'Manual Aspiration', 'Implantation', 'Positioning', 'OVD Aspiration', 'Suturing', 'Sealing Control',
         'Wound Hydratation']


def fix_outliers(truth_data_steps, prediction_data_steps):

    steps_order = [truth_data_steps[0]]
    for step_index in range(len(truth_data_steps)):
        if step_index != 0 and truth_data_steps[step_index] != truth_data_steps[step_index - 1]:
            steps_order.append(truth_data_steps[step_index])

    k = 0
    for i in range(len(truth_data_steps)):

        if i != 0 and truth_data_steps[i] != truth_data_steps[i - 1]:
            k = k + 1

        if truth_data_steps[i] == 0:  # check for idle step
            if truth_data_steps[i] != prediction_data_steps[i]:
                list_values_to_check = [steps_order[k], steps_order[k + 1]] if k == 0 else ([steps_order[k - 1],
                                                                                             steps_order[k]] if k == len(steps_order) - 1 else
                                                                                            [steps_order[k - 1],
                                                                                             steps_order[k],
                                                                                             steps_order[k + 1]])
                if prediction_data_steps[i] in list_values_to_check:
                    prediction_data_steps[i] = truth_data_steps[i]

    return truth_data_steps, prediction_data_steps


def main():
    """
    Main function.
    """
    # parsing the command line
    parser = ArgumentParser(
        description='Evaluation script for the EndoVis / CATARACTS 2020 challenge.')
    parser.add_argument('-t', '--gt', required=True,
                        help='directory containing ground truth files')
    parser.add_argument('-p', '--predictions', required=True,
                        help='directory containing automatic predictions')

    num_steps = len(steps) + 1
    args = parser.parse_args()

    list_truth_files = sorted(glob.glob(os.path.join(args.gt, "*.csv")))

    assert len(list_truth_files) > 0, "No ground truth files were found!!"

    f1_scores = []
    try:
        for file in list_truth_files:
            file_name = os.path.basename(file)
            if os.path.exists(args.predictions + '/' + os.path.basename(file)):
                truth_filename = os.path.join(args.gt, file_name)
                prediction_filename = os.path.join(args.predictions, file_name)

                # parsing the right columns for the current tool
                truth_data = read_csv(truth_filename, header=0, skipinitialspace=True, squeeze=True, dtype='Int64')
                prediction_data = read_csv(prediction_filename, header=0, skipinitialspace=True, squeeze=True, dtype='Int64')

                if len(truth_data) != len(prediction_data):
                    raise ValueError('Files {} and {} have different row counts'.
                                     format(truth_filename, prediction_filename))

                truth_data_steps = truth_data["Steps"].tolist()
                prediction_data_steps = prediction_data["Steps"].tolist()

                # We don't take into account the idle frames
                # idle_frames = list(filter(lambda x: truth_data_steps[x] == 0, range(len(truth_data_steps))))

                # To reduce the impact of the idle step on the results.
                truth_data_steps, prediction_data_steps = fix_outliers(truth_data_steps, prediction_data_steps)

                truth_data_steps = np.eye(num_steps)[truth_data_steps]
                prediction_data_steps = np.eye(num_steps)[prediction_data_steps]

                # Remove idle frames
                # truth_data_steps = np.delete(truth_data_steps, idle_frames,0)
                # prediction_data_steps = np.delete(prediction_data_steps, idle_frames, 0)

                # Removing the idle class from the evaluation
                truth_data_steps = truth_data_steps[:, 1:]
                prediction_data_steps = prediction_data_steps[:, 1:]

                steps_f1_score = []
                ind = 0
                for step_gt, step_prediction in zip(truth_data_steps.T, prediction_data_steps.T):
                    count_nonzero = np.count_nonzero(step_gt)
                    if count_nonzero != 0:
                        f1_score_step = f1_score(step_gt, step_prediction, average='binary')
                        # print(str(ind + 1) + ' ' + steps[ind] + ' ' + "{:.4f}".format(f1_score_step))
                        steps_f1_score.append(f1_score_step)
                    ind = ind + 1

                # computing video-specific scores
                f1_score_video = np.mean(steps_f1_score)
                f1_scores.append(f1_score_video)
                print(file_name + ' : ' + "{:.4f}".format(f1_score_video))

            else:
                raise ValueError(file_name + ' could not be found in the predictions folder.')

    except Exception as e:
        print('Error: {} -> {}.'.format(file_name, e))
        f1_scores = []

    # computing the average score
    print('** Average: ' + "{:.4f}".format(np.mean(f1_scores)))


if __name__ == "__main__":
    main()
