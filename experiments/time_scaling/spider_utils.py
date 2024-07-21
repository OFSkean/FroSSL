import matplotlib.pyplot as plt
from math import pi
import numpy as np


def make_spider(df, row, title, color):

    # number of variable
    categories=list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(2,6,row+1, polar=True) 

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    # MEMORY
    #ax.set_rlabel_position(0)
    ax.set_rgrids([0.25, 0.5, 0.75, 1], labels=list('abcd'), angle=angles[1], fontsize=12)
    ax.set_rgrids([0.5, 0.5, 0.75, 1], labels=list('efgh'), angle=angles[5], fontsize=12)

    #plt.yticks([0.25, 0.5, 0.75, 1], ["2GB", "4GB", "6GB", "8GB"], color="black", size=8)
    
    
    plt.ylim(0,1)

    # Ind1
    values=df.loc[row].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=25, color='black', y=1.1)
