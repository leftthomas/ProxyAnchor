import matplotlib.pyplot as plt

plt.style.use(['science', 'grid', 'scatter', 'no-latex'])

x = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
labels = ['GoogLeNet(512)', 'BN-Inception(512)', 'ResNet-50(512)',
          'GoogLeNet(1024)', 'BN-Inception(1024)', 'ResNet-50(1024)']

y = {'CUB-200': [[70.6, 72.1, 72.7, 72.8, 72.7, 72.7, 72.7], [76.8, 77.6, 78.1, 78.4, 78.6, 72.7, 72.7],
                 [78.5, 78.7, 79.5, 80.0, 80.4, 72.7, 72.7], [71.1, 72.5, 72.9, 73.0, 73.0, 72.7, 72.7],
                 [77.3, 78.0, 78.5, 78.7, 79.0, 72.7, 72.7], [78.2, 79.0, 79.7, 80.3, 80.6, 72.7, 72.7]],
     'Cars-196': [[70.6, 72.1, 72.7, 72.8, 72.7, 72.7, 72.7], [76.8, 77.6, 78.1, 78.4, 78.6, 72.7, 72.7],
                  [78.5, 78.7, 79.5, 80.0, 80.4, 72.7, 72.7], [71.1, 72.5, 72.9, 73.0, 73.0, 72.7, 72.7],
                  [77.3, 78.0, 78.5, 78.7, 79.0, 72.7, 72.7], [78.2, 79.0, 79.7, 80.3, 80.6, 72.7, 72.7]],
     'SOP': [[70.6, 72.1, 72.7, 72.8, 72.7, 72.7, 72.7], [76.8, 77.6, 78.1, 78.4, 78.6, 72.7, 72.7],
             [78.5, 78.7, 79.5, 80.0, 80.4, 72.7, 72.7], [71.1, 72.5, 72.9, 73.0, 73.0, 72.7, 72.7],
             [77.3, 78.0, 78.5, 78.7, 79.0, 72.7, 72.7], [78.2, 79.0, 79.7, 80.3, 80.6, 72.7, 72.7]],
     'In-Shop': [[70.6, 72.1, 72.7, 72.8, 72.7, 72.7, 72.7], [76.8, 77.6, 78.1, 78.4, 78.6, 72.7, 72.7],
                 [78.5, 78.7, 79.5, 80.0, 80.4, 72.7, 72.7], [71.1, 72.5, 72.9, 73.0, 73.0, 72.7, 72.7],
                 [77.3, 78.0, 78.5, 78.7, 79.0, 72.7, 72.7], [78.2, 79.0, 79.7, 80.3, 80.6, 72.7, 72.7]]}

fig = plt.figure(figsize=(20, 5))
for i, data_name in enumerate(['CUB-200', 'Cars-196', 'SOP', 'In-Shop']):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.set_title(data_name)
    ax.set(xlabel='Momentum')
    ax.set(ylabel='R@1')
    for index, label in enumerate(labels):
        ax.plot(x, y[data_name][index], label=label, linestyle='--')
lines, labels = fig.axes[-1].get_legend_handles_labels()
plt.legend(lines, labels, loc='center right', bbox_to_anchor=(0.1, -0.03, 0.8, 0.2), ncol=len(labels), mode='expand',
           bbox_transform=plt.gcf().transFigure)
fig.subplots_adjust(bottom=0.2)
fig.savefig('results/hyper.pdf')
