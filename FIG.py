import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.pyplot import MultipleLocator

with open('./samples_for_scatter/samples_for_scatter_8.pkl','rb')as file:
    train_points = pickle.load(file)
with open('./vecs_to_save/vecs_to_save_8.pkl','rb')as file:
    array = pickle.load(file)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (10,10)
colors = ['red', 'orange', 'darkblue', 'saddlebrown', 'darkgreen','purple','lightseagreen','slateblue']
for i,level in enumerate(array):
    # print(level.detach().cpu())
    # print(i)
    # print(level)
    x = np.array([elem[0] for elem in level.detach().cpu()])
    y = np.array([elem[1] for elem in level.detach().cpu()])
    # name =  str(2*pow(2,i)) + 'Bits Vectors'
    if i == 0:
        name = 'Codewords of $Q_1$'
    if i == 1:
        name = 'Codewords of $Q_2$'
    if i == 2:
        name = 'Codewords of $Q_3$'
    if i == 3:
        name = 'Codewords of $Q_4$'
    if i == 4:
        name = 'Codewords of $Q_5$'
    if i == 5:
        name = 'Codewords of $Q_6$'
    if i == 6:
        name = 'Codewords of $Q_7$'
    if i == 7:
        name = 'Codewords of $Q_8$'
    train_points_level = train_points[-1]
    train_points_level = train_points_level[:40000]
    train_x_vals = np.array([elem[0] for elem in train_points_level.detach().cpu()])
    train_y_vals = np.array([elem[1] for elem in train_points_level.detach().cpu()])
    plt.scatter(train_x_vals, train_y_vals,s=20, alpha=0.3,label = "Encoder's Output Vectors", c='#808080') # '#5F9EA0','#00008B','#008B8B','#556B2F','#808080'
    plt.scatter(x, y,s=100, alpha=1,label = name,c=colors[i % 8])

    x_major_locator = MultipleLocator(1.2)
    y_major_locator = MultipleLocator(1.2)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # Add axis labels and a title
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Received Vectors')
    plt.grid(linestyle = '-.', linewidth = 1.8)
    plt.legend(loc='best')
    plt.xlim([-4.8, 4.8])
    plt.ylim([-4.8, 6])
    # Show the plot
    plt.show()