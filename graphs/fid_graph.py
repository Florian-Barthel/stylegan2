import matplotlib.pyplot as plt
import os
import numpy as np

graph_dir = ''
full_res_threshold = 5.4
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_data(metric, run_id):
    runs = os.listdir('../results')
    prefix = str(run_id).zfill(5)
    run = None
    for cur_run in runs:
        if cur_run.startswith(prefix):
            run = cur_run
            break
    assert run is not None
    metric_file_path = '../results/' + run + '/metric-' + metric + '.txt'
    with open(metric_file_path) as file:
        line = file.readline()
        kimg = []
        score = []
        while line:
            kimg.append(int(line[17:23]) / 1000)
            score.append(float(line[50 + len(metric):-3]))
            line = file.readline()
    return kimg, score


def simple_plot(x, y, xlabel, ylabel, title, filename, x_min, x_max, y_min, y_max, xticks, yticks):
    fig, ax = plt.subplots()
    ax.plot(x, y, 'orange')
    ax.xaxis.set_ticks(np.arange(x_min, x_max, xticks))
    ax.yaxis.set_ticks(np.arange(y_min, y_max, yticks))
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()

    fig.savefig(graph_dir + filename + '.png')
    plt.show()


def slice_before_full_res(x, y, number_before):
    index = 0
    for cur_x in x:
        if cur_x < full_res_threshold:
            index += 1
        else:
            break
    x = x[index - number_before:]
    y = y[index - number_before:]
    return x, y




run_id = 30
x, y = get_data('fid50k', run_id=run_id)
# x, y = slice_before_full_res(x, y, 0)
simple_plot(x, y, 'Million Images Seen by the Discriminator', 'Fréchet Inception Distance', 'Fréchet Inception Distance over the Training', 'fid_baseline' + str(run_id), 0, 10, 0, 20, 3, 1)

