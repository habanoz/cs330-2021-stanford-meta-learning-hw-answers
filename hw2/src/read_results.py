import glob

import numpy as np
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import glob
import tensorboard as tb


def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    I = []
    X = []
    Y = []


    time0 = None
    for e in summary_iterator(file):
        if time0 is None:
            time0 = e.wall_time
        for v in e.summary.value:
            if v.tag == 'train_accuracy/support':
                X.append(v.simple_value)
            elif v.tag == 'train_accuracy/query':
                Y.append(v.simple_value)

    if X:
        print(f"Saved {file} train_accuracy_support")
        np.savetxt(logdir + "/train_accuracy_support.txt", np.array(X))
    if Y:
        print(f"Saved {file} train_accuracy_query")
        np.savetxt(logdir + "/train_accuracy_query.txt", np.array(Y))


def read_log_dir(logdir):
    eventfile = glob.glob(logdir + "/events.out.tfevents.*")[0]
    get_section_results(eventfile)


if __name__ == '__main__':
    log_dirs = glob.glob("logs/protonet/omniglot*")

    for logdir in log_dirs:
        read_log_dir(logdir)

    print(f"DONE")
