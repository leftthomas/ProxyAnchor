import matplotlib.pyplot as plt

plt.style.use(['science', 'grid', 'scatter', 'no-latex'])

x = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
labels = ['GoogLeNet(512)', 'BN-Inception(512)', 'ResNet-50(512)',
          'GoogLeNet(1024)', 'BN-Inception(1024)', 'ResNet-50(1024)']

y = {'CUB-200': [[63.3, 63.3, 62.8, 62.5, 63.0, 61.7, 62.4], [66.7, 65.7, 67.2, 66.7, 67.0, 66.7, 64.9],
                 [68.3, 69.2, 70.0, 70.0, 68.4, 68.7, 68.0], [63.7, 63.8, 64.7, 64.4, 64.0, 63.1, 62.2],
                 [67.0, 68.0, 68.0, 68.3, 67.4, 66.8, 66.4], [68.8, 70.1, 71.5, 70.2, 68.5, 70.8, 70.3]],
     'Cars-196': [[80.6, 80.4, 80.0, 78.8, 79.0, 76.2, 76.3], [6.8, 49.3, 83.6, 84.0, 83.8, 79.9, 79.3],
                  [74.9, 86.9, 85.8, 87.0, 86.9, 86.8, 86.4], [81.9, 82.2, 81.2, 81.0, 80.5, 78.5, 78.1],
                  [85.4, 3.7, 85.4, 85.0, 85.3, 81.4, 80.5], [4.0, 47.1, 87.7, 89.1, 89.3, 88.8, 88.5]],
     'SOP': [[70.6, 72.1, 72.7, 72.8, 72.7, 72.7, 72.7], [76.8, 77.6, 78.1, 78.4, 78.6, 72.7, 72.7],
             [78.5, 78.7, 79.5, 80.0, 80.4, 72.7, 72.7], [71.1, 72.5, 72.9, 73.0, 73.0, 72.7, 72.7],
             [77.3, 78.0, 78.5, 78.7, 79.0, 72.7, 72.7], [78.2, 79.0, 79.7, 80.3, 80.6, 72.7, 72.7]],
     'In-Shop': [[72.8, 75.1, 78.1, 79.1, 79.0, 78.9, 78.6], [83.0, 84.4, 86.1, 87.2, 87.6, 87.8, 87.7],
                 [85.8, 86.9, 88.7, 89.7, 90.0, 90.2, 90.2], [73.8, 76.6, 78.7, 79.7, 80.2, 79.5, 79.4],
                 [84.0, 85.3, 86.7, 87.6, 87.8, 88.1, 88.3], [2.9, 86.9, 88.5, 89.9, 90.5, 90.7, 90.8]]}

fig = plt.figure(figsize=(20, 5))
for i, data_name in enumerate(['CUB-200', 'Cars-196', 'SOP', 'In-Shop']):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.set_title(data_name)
    ax.set(xlabel='Momentum')
    ax.set(ylabel='R@1')
    for index, label in enumerate(labels):
        ax.plot(x, y[data_name][index], label=label, linestyle='--')
lines, labels = fig.axes[-1].get_legend_handles_labels()
plt.legend(lines, labels, loc='center right', bbox_to_anchor=(0.1, -0.03, 0.81, 0.2), ncol=len(labels), mode='expand',
           bbox_transform=plt.gcf().transFigure)
fig.subplots_adjust(bottom=0.2)
fig.savefig('results/hyper.pdf')
