
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Import data
df = pd.read_csv("medical_examination.csv")

# Calculate BMI
bmi = df["weight"] / ((df["height"] / 100) ** 2)

# Create overweight column
# 0 = Not overweight
# 1 = Overweight
df["overweight"] = (bmi > 25).astype(int)

# Normalize cholesterol and glucose
# 0 = Good
# 1 = Bad
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)
df["gluc"] = (df["gluc"] > 1).astype(int)


def draw_cat_plot():
    """
    Draws a categorical plot showing the counts
    of different health and lifestyle features
    separated by cardiovascular disease status.
    """

    # Create a DataFrame for the categorical plot
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "overweight"
        ]
    )

    # Group and reformat the data
    df_cat = (
        df_cat
        .groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # Create the categorical plot
    fig = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar"
    )

    # Do not modify the next two lines
    fig = fig.fig
    fig.savefig("catplot.png")

    return fig


def draw_heat_map():
    """
    Cleans the medical data and draws a correlation heatmap.
    """

    # Clean the data by removing incorrect and extreme values
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"])
        & (
            df["height"]
            >= df["height"].quantile(0.025)
        )
        & (
            df["height"]
            <= df["height"].quantile(0.975)
        )
        & (
            df["weight"]
            >= df["weight"].quantile(0.025)
        )
        & (
            df["weight"]
            <= df["weight"].quantile(0.975)
        )
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(
        np.ones_like(corr, dtype=bool)
    )

    # Set up the matplotlib figure
    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    # Draw the heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={
            "shrink": 0.5
        },
        ax=ax
    )

    # Do not modify the next two lines
    fig.savefig("heatmap.png")

    return fig