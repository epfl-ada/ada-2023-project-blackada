import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def embeddings(
    embeddings: np.ndarray,
    hue: list,
    color_palette: dict,
    subset: list[int] | None = None,
    plot_type="scatter",
    figsize: tuple[int, int] = (5, 5),
    title: str = "Embeddings",
    plot_legend: bool = True,
    ax: plt.Axes | None = None,
    sns_config: dict | None = None,
    **kwargs,
):
    """
    Visualises the embeddings of beer reviews according with a specified
    hue (e.g. beer style, beer name, etc.). A subset of reviews can be
    plotted by passing a list of indices to be plotted. The figure size,
    title and legend can be customised. Custom plotting arguments can be
    passsed to the underlying seaborn plotting function.

    The plot type can be specified by passing the plot_type argument. Useful
    ones are `scatter` and `kde`.

    Args:
        embeddings: Embeddings to be plotted (must be 2d)
        hue: List of values to be used for the hue.
        subset: List of indices to be plotted.
        plot_type: Type of plot to be used.
        figsize: Figure size.
        title: Title of the plot.
        plot_legend: Whether to plot the legend.
        ax: Axes object to be used for plotting.
        kwargs: Additional arguments to be passed to the underlying

    Returns:
        None
    """
    kwargs["palette"] = color_palette

    if not ax:
        _, ax = plt.subplots(figsize)

    if subset is not None:
        embeddings = embeddings[subset]
        hue = hue[subset]

    func = getattr(sns, plot_type + "plot")
    sns.set(**sns_config)
    func(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        hue=hue,
        ax=ax,
        legend="full",
        **kwargs,
    )

    ax.set(
        title=title,
        xlabel="T-SNE 1",
        ylabel="T-SNE 2",
    )

    # Disable ticks
    ax.set_axis_off()

    # Create custom legend at bottom
    if plot_legend:
        handles, labels = ax.get_legend().legendHandles, ax.get_legend().get_texts()
        labels = [label.get_text() for label in labels]
        ax.legend(
            handles,
            labels,
            loc="lower center",
            fontsize="small",
            fancybox=True,  # Rounded corners
            framealpha=0.0,  # Transparency of the frame
            bbox_to_anchor=(0.5, -0.12),
            # Use circle as marker
            handler_map={plt.Line2D: plt.Circle},
        )
    else:
        ax.get_legend().remove()


def interpretation_barplot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    image_path: str,
    image_name: str,
    seaborn_cfgs: dict,
    plot: bool = True,
    save: bool = True,
):
    """
    Plots a barplot with the given data and saves it to the given path. Intended to be used for plotting interpretation values of TFIDF embeddings e.g. top 10 words used within a set or reviews.

    Args:
        x: x values for the barplot.
        y: y values for the barplot.
        title: Title of the plot.
        xlabel: Label of the x axis.
        ylabel: Label of the y axis.
        image_path: Path to save the image to.
        image_name: Name of the image.
        seaborn_cfgs: List of seaborn configs to be used for the plot.
        plot: Whether to plot the figure.
        save: Whether to save the figure.

    Returns:
        None
    """

    # Create the image path if it doesn't exist
    os.makedirs(image_path, exist_ok=True)

    # Loop through the seaborn configs
    for mode, seaborn_cfgs in seaborn_cfgs:
        # Set the config for given mode light / dark
        sns.set(**seaborn_cfgs)

        # Plot the figure
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=x, y=y, ax=ax, orient="h")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()

        # Save the figure
        if save:
            # Remove the file if it already exists
            file_path = os.path.join(image_path, f"{image_name}_{mode}.png")
            if os.path.isfile(file_path):
                os.remove(file_path)

            fig.savefig(file_path, dpi=300)

        # Plot the figure
        if plot:
            plt.show()
        else:
            plt.close()
