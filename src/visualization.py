import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
import numpy as np

def display_rating_intime(data):

    def visualize_rating_intime(rating, yearx):

        # Get the data
        years = data.review.date.dt.year
        months = data.review.date.dt.month
        if rating != "# of new reviews":
            agg = data.review[rating]

        if rating == "# of new reviews":
            joined = pd.concat([years, months], axis=1)
            joined.columns = ["year", "month"]
        else:
            joined = pd.concat([years, months, agg], axis=1)
            joined.columns = ["year", "month", "rating"]

        # Subset the data to only include users who data in yearX or later
        joined = joined[joined.year >= yearx]

        # Map the month number to the month name
        joined.month = joined.month.map({
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
            5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        })

        if rating == "# of new reviews":
            joined = joined.groupby(["year", "month"]).size().reset_index()
        else:
            joined = joined.groupby(["year", "month"]).mean().reset_index()
        joined.columns = ["year", "month", "agg"]

        g = sns.relplot(
            data=joined,
            x="month", y="agg", col="year", hue="year",
            kind="line", linewidth=4, zorder=5,
            col_wrap=3, height=2, aspect=1.5, legend=False,
        )

        # Iterate over each subplot to customize further
        for year, ax in g.axes_dict.items():

            # Add the title as an annotation within the plot
            ax.text(.8, .85, int(year), transform=ax.transAxes, fontweight="bold")

            # Plot every year's time series in the background
            sns.lineplot(
                data=joined, x="month", y="agg", units="year",
                estimator=None, color=".4", linewidth=1, ax=ax,
            )

        # Reduce the frequency of the x axis ticks
        ax.set_xticks(ax.get_xticks()[::2])

        # Tweak the supporting aspects of the plot
        g.set_titles("")
        g.set_axis_labels("", f"")
        g.tight_layout();

        # Set the title
        if rating == "# of new reviews":
            g.fig.suptitle(f"Number of new reviews per month since {yearx}", fontsize=16);
        else:
            g.fig.suptitle(f"Mean {rating} Rating per month since {yearx}", fontsize=16);

        # Tight layout
        plt.tight_layout()

    rating_selector = widgets.Dropdown(
        options=["# of new reviews", "appearance", "aroma", "palate", "taste", "overall"],
        value="appearance",
        description='Agg:',
    )

    year_selector = widgets.IntSlider(
        value=2010,
        min=2000,
        max=2015,
        step=1,
        description='Year:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

    return visualize_rating_intime, rating_selector, year_selector