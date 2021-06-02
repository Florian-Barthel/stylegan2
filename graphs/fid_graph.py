import matplotlib.pyplot as plt
import os
import numpy as np

colors = [
    'orange',
    'green',
    'royalblue',
    'purple',
    'yellow'
]


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


def plot_multiple_runs(
        run_ids,
        descriptions,
        xlabel,
        ylabel,
        title,
        filename,
        x_min=0,
        x_max=15,
        y_min=0,
        y_max=10,
        xticks=2,
        yticks=1,
        metric='fid50k'):
    fig, ax = plt.subplots()
    offset = 0
    for run_index in range(len(run_ids)):
        x, y = get_data(metric, run_ids[run_index])
        ax.plot(x, y, colors[run_index], label=descriptions[run_index])
        xmin = x[np.argmin(y)]
        ymin = min(y)
        text = "FID={:.3f}".format(ymin)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.52)
        arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=70")
        kw = dict(xycoords='data', textcoords="axes fraction", annotation_clip=False,
                  arrowprops=arrowprops, bbox=bbox_props, color=colors[run_index])
        ax.annotate(text, xy=(xmin, ymin), xytext=(0.83, 0.4 + offset), **kw)
        offset += 0.1
    ax.xaxis.set_ticks(np.arange(x_min, x_max, xticks))
    ax.yaxis.set_ticks(np.arange(y_min, y_max, yticks))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim([y_min, y_max])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    ax.legend()
    fig.savefig(filename, dpi=400)
    plt.show()


plot_multiple_runs(
    run_ids=[
        37,
        70,
        122
    ],
    descriptions=[
        'Baseline',
        'Label Mapping Network Add',
        'Label Mapping Network Concat Interpolate'
    ],
    xlabel='Million Images seen by the Discriminator',
    ylabel='Fréchet Inception Distance',
    title='Fréchet Inception Distance over the Training',
    filename='fid-label-mapping-add_vs_concat.png',
    y_min=2,
    x_max=8)
