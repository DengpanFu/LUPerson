# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import pprint
import sys
from collections import Mapping, OrderedDict

import numpy as np
from tabulate import tabulate
from termcolor import colored

logger = logging.getLogger(__name__)


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.
    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
    """
    assert isinstance(results, OrderedDict), results  # unordered results cannot be properly printed
    task = list(results.keys())[0]
    metrics = ["Datasets"] + [k for k in results[task]]

    csv_results = []
    for task, res in results.items():
        csv_results.append((task, *list(res.values())))

    # tabulate it
    table = tabulate(
        csv_results,
        tablefmt="pipe",
        floatfmt=".2%",
        headers=metrics,
        numalign="left",
    )

    logger.info("Evaluation results in csv format: \n" + colored(table, "cyan"))


def verify_results(cfg, results):
    """
    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
    Returns:
        bool: whether the verification succeeds or not
    """
    expected_results = cfg.TEST.EXPECTED_RESULTS
    if not len(expected_results):
        return True

    ok = True
    for task, metric, expected, tolerance in expected_results:
        actual = results[task][metric]
        if not np.isfinite(actual):
            ok = False
        diff = abs(actual - expected)
        if diff > tolerance:
            ok = False

    logger = logging.getLogger(__name__)
    if not ok:
        logger.error("Result verification failed!")
        logger.error("Expected Results: " + str(expected_results))
        logger.error("Actual Results: " + pprint.pformat(results))

        sys.exit(1)
    else:
        logger.info("Results verification passed.")
    return ok


def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.
    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r
