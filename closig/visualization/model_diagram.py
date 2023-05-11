from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects

# Create an empty canvas to draw a diagram on with matplotlib


def illustrate_model(layers, ax=None):
    """
    Illustrate the model with a diagram
    :param
    layers: a dictionary with the layers and their sub tiles and weights
    ax: the axis to plot on
    """

    if ax is None:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)

    # remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')

    # for each layer make a rectangle and print its name, plot each rectangle below the previous one
    for i, layer in enumerate(layers):
        # get the number of sub tiles
        num_sub_tiles = len(layers[layer])
        color = 'forestgreen'

        # Compute four corners but add space between the previous rectangle
        padding = 0.1
        xpadd = 0.005
        offset = i * (1+padding)
        ystart = 0.01
        y = [ystart * (offset), ystart * offset, ystart *
             (offset + 1), ystart * (offset + 1)]
        x = [0, 0, 0, 0]

        # for each tile in the layer
        for j in range(len(layers[layer]['covmodels'])):

            fraction = layers[layer]['fractions'][j]
            name = layers[layer]['covmodels'][j]
            if name is None:
                name = layer

            length = 0.1 * fraction
            x[1] += length
            x[2] += length

            center = ((x[0] + x[1]) / 2, (y[0] + y[2]) / 2)

            # plot the rectangle
            ax.fill(x, y, color=color, alpha=0.5,
                    edgecolor='black', linewidth=2, joinstyle='round')

            # Annotate the layer title
            pe = [path_effects.Stroke(linewidth=3, foreground='white'),
                  path_effects.Normal()]
            ax.annotate(name, xy=(center[0], center[1]), xytext=(center[0], center[1]),
                        color='k', weight='bold', ha='center', va='center', path_effects=pe)

            x[0] += length + xpadd
            x[3] += length + xpadd
    if ax is None:
        plt.show()
