import pandas as pd
import numpy as np
from src.diffusivity import calc_k_quantile, calc_tilda_c
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pymc as pm

import seaborn as sns

sns.set_context("notebook")
sns.set_style("whitegrid")


def make_quantile_contour_plot(fig_size=(10, 5), max_quantile=0.95):
    """Reproduce the quantile contour plot"""
    ks = np.arange(1e-3, 1000, 1)
    ps = np.arange(1, 50, 1)
    kv, pv = np.meshgrid(ks, ps, sparse=False)
    quantiles = calc_k_quantile(kv, pv, max_quantile)

    levels = np.round(np.linspace(0, max_quantile, 10 + 1), 2)
    fig, ax = plt.subplots(figsize=fig_size)
    cs = ax.contourf(kv, pv, quantiles, levels=levels)
    fig.colorbar(cs, ax=ax)
    ax.set_xlabel("Permeability Compliance (GPa)")
    ax.set_ylabel("Net Injection Pressure (MPa)")
    return fig


def make_basic_diffusivity_plot(fig_size=(10, 5)):
    """Reproduce the diffusivity plot"""
    t = np.linspace(0, 5000, 100)
    Do = 1  # matrix diffusivity (m2/s)
    Po = 10  # pressure (MPa)

    k_dict = {1: "black", 10: "#e41a1c", 100: "#4daf4a"}

    fig, ax = plt.subplots(figsize=fig_size)

    for k in k_dict.keys():
        k_tilda = k * Po / 1000
        D = Do * np.exp(k_tilda)
        r_tf = np.sqrt(6 * D * t)
        tilda_c = calc_tilda_c(k, Po)
        r_fd = tilda_c * np.sqrt(Do * t)
        sns.lineplot(x=t / 3600, y=r_fd, color=k_dict[k], linestyle="--", ax=ax)
        sns.lineplot(x=t / 3600, y=r_tf, color=k_dict[k], linestyle="-", ax=ax)

    ax.set_ylim([0, 300])
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Distance (m)")
    return fig

def make_combined_basic_diff_quantile_plot(fig_size=(12,4), max_quantile=0.95):
    """Make a combined plot for the paper"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [0.9, 1]}, figsize=fig_size)
    max_quantile = 0.95

    # basic diff plot
    t = np.linspace(0, 5000, 100)
    Do = 1  # matrix diffusivity (m2/s)
    Po = 10  # pressure (MPa)
    k_dict = {1: "black", 10: "#e41a1c", 100: "#4daf4a"}

    for k in k_dict.keys():
        k_tilda = k * Po / 1000
        D = Do * np.exp(k_tilda)
        r_tf = np.sqrt(6 * D * t)
        tilda_c = calc_tilda_c(k, Po)
        r_fd = tilda_c * np.sqrt(Do * t)
        sns.lineplot(x=t / 3600, y=r_fd, color=k_dict[k], linestyle="--", ax=axes[0])
        sns.lineplot(x=t / 3600, y=r_tf, color=k_dict[k], linestyle="-", ax=axes[0])

    axes[0].set_ylim([0, 300])
    axes[0].set_xlabel("Time (hr)")
    axes[0].set_ylabel("Distance (m)")
    axes[0].set_title("a)", y=1.0, x=-0.12)

    # contour plot
    ks = np.arange(1e-3, 1000, 1)
    ps = np.arange(1, 50, 1)
    kv, pv = np.meshgrid(ks, ps, sparse=False)
    quantiles = calc_k_quantile(kv, pv, max_quantile)

    levels = np.round(np.linspace(0, max_quantile, 10 + 1), 2)
    cs = axes[1].contourf(kv, pv, quantiles, levels=levels)
    fig.colorbar(cs, ax=axes[1])
    axes[1].set_xlabel("Permeability Compliance (GPa)")
    axes[1].set_ylabel("Net Injection Pressure (MPa)")
    axes[1].set_title("b)", y=1.0, x=-0.14)
    
    plt.tight_layout()
    return fig

def make_stage_plot(distances, params, well, stage, figsize=(10, 3), Do=2):
    stg_dist = distances.query("WellID == @well").query("Stage == @stage").copy()
    stg_dist["in_frac_ellipsoid"] = ~stg_dist["in_frac_ellipsoid"]
    stg_dist["dx_strike_m"] = stg_dist["dx_strike_m"].abs()
    sns.set_palette("colorblind")
    stg_dist = stg_dist.rename(
        columns={
            "dx_cart_m": "Easting (m)",
            "dy_cart_m": "Northing (m)",
            "dz_cart_m": "Vertical Offset (m)",
            "dx_strike_m": "Lateral Distance (m)",
            "t_start_s": "Time from Stage Start (s)",
            "in_frac_ellipsoid": "In Plausible Ellipsoid",
        }
    )
    length = params["plausible_ellipsoid_lwh"][0]
    width = params["plausible_ellipsoid_lwh"][1]
    height = params["plausible_ellipsoid_lwh"][2]

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=figsize, ncols=3)

    sns.scatterplot(
        data=stg_dist,
        x="Time from Stage Start (s)",
        y="Lateral Distance (m)",
        hue="In Plausible Ellipsoid",
        ax=ax1,
    )
    sns.scatterplot(x=[0], y=[0], s=250, color=".2", ax=ax1, marker="*")
    t = np.linspace(10, stg_dist["Time from Stage Start (s)"].max(), 1000)
    sns.lineplot(x=t, y=np.sqrt(6 * Do * t), ax=ax1, color="black")
    ax1.get_legend().remove()
    ax1.set_ylim(0, length * 2)

    sns.scatterplot(
        data=stg_dist,
        x="Easting (m)",
        y="Northing (m)",
        hue="In Plausible Ellipsoid",
        ax=ax2,
    )
    sns.scatterplot(x=[0], y=[0], s=250, color=".2", ax=ax2, marker="*")
    elps = Ellipse(
        (0, 0),
        length * 2,
        width * 2,
        angle=params["onstrike_angle_deg"],
        edgecolor="b",
        facecolor="none",
    )
    ax2.add_artist(elps)
    ax2.get_legend().remove()
    ax2.set_xlim(-length * 2, length * 2)
    ax2.set_ylim(-width * 2, width * 2)

    sns.scatterplot(
        data=stg_dist,
        x="Easting (m)",
        y="Vertical Offset (m)",
        hue="In Plausible Ellipsoid",
        ax=ax3,
    )
    sns.scatterplot(x=[0], y=[0], s=250, color=".2", ax=ax3, marker="*")
    elps = Ellipse(
        (0, 0), length * 2, height * 2, angle=0, edgecolor="b", facecolor="none"
    )
    ax3.add_artist(elps)
    ax3.get_legend().remove()
    ax3.set_xlim(-length * 2, length * 2)
    ax3.set_ylim(-height * 2, height * 2)

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.suptitle(f"Well {well} Stage {stage}", x=0.15, y=1.02, ha="right")
    return fig


def make_stage_w_posterior_predictive_plot(
    distances,
    params,
    well,
    stage,
    nonlinear_trace,
    fig_size=(10, 3),
    title_prefix="",
    Do=2,
    samples=100,
):
    stg_dist = distances.query("WellID == @well").query("Stage == @stage").copy()
    stg_dist["in_frac_ellipsoid"] = ~stg_dist["in_frac_ellipsoid"]
    stg_dist["dx_strike_m"] = stg_dist["dx_strike_m"].abs()
    sns.set_palette("colorblind")
    stg_dist = stg_dist.rename(
        columns={
            "dx_cart_m": "Easting (m)",
            "dy_cart_m": "Northing (m)",
            "dz_cart_m": "Vertical Offset (m)",
            "dx_strike_m": "Lateral Distance (m)",
            "t_start_s": "Time from Stage Start (s)",
            "in_frac_ellipsoid": "In Plausible Ellipsoid",
        }
    )
    length = params["plausible_ellipsoid_lwh"][0]
    width = params["plausible_ellipsoid_lwh"][1]
    ms_y_max = int(np.ceil(stg_dist["Lateral Distance (m)"].max() / 100.0)) * 100

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=fig_size, ncols=3)

    sns.scatterplot(
        data=stg_dist,
        x="Time from Stage Start (s)",
        y="Lateral Distance (m)",
        hue="In Plausible Ellipsoid",
        ax=ax1,
    )
    sns.scatterplot(x=[0], y=[0], s=250, color=".2", ax=ax1, marker="*")
    t = np.linspace(10, stg_dist["Time from Stage Start (s)"].max(), 1000)
    sns.lineplot(x=t, y=np.sqrt(6 * Do * t), ax=ax1, color="black")
    ax1.get_legend().remove()
    ax1.set_ylim(0, ms_y_max)

    sns.scatterplot(
        data=stg_dist,
        x="Easting (m)",
        y="Northing (m)",
        hue="In Plausible Ellipsoid",
        ax=ax2,
    )
    sns.scatterplot(x=[0], y=[0], s=250, color=".2", ax=ax2, marker="*")
    elps = Ellipse(
        (0, 0),
        length * 2,
        width * 2,
        angle=params["onstrike_angle_deg"],
        edgecolor="b",
        facecolor="none",
    )
    ax2.add_artist(elps)
    ax2.get_legend().remove()
    ax2.set_xlim(-length * 2, length * 2)
    ax2.set_ylim(-width * 2, width * 2)

    sns.set_style("whitegrid")
    t = np.linspace(1, stg_dist["Time from Stage Start (s)"].max(), 1000)
    sns.scatterplot(
        x=stg_dist["Time from Stage Start (s)"],
        y=stg_dist["Lateral Distance (m)"].abs(),
        ax=ax3,
    )
    for chain in nonlinear_trace.posterior.chain.values:
        draws = np.random.choice(
            nonlinear_trace.posterior.sel({"chain": chain}).draw.values, samples
        )
        for draw in draws:
            k = nonlinear_trace.posterior.sel({"chain": chain, "draw": draw}).k.values
            Po = nonlinear_trace.posterior.sel({"chain": chain, "draw": draw}).Po.values
            Do = nonlinear_trace.posterior.sel(
                {"chain": chain, "draw": draw}
            ).Do_x.values
            sns.lineplot(x=t, y=np.sqrt(6 * Do * t), color="red", alpha=0.1, ax=ax3)
            sns.lineplot(
                x=t,
                y=calc_tilda_c(k, Po) * np.sqrt(Do * t),
                color="green",
                alpha=0.1,
                ax=ax3,
            )
    ax3.set_xlabel("Time from Stage Start (s)")
    ax3.set_ylabel("Lateral Distance (m)")
    ax3.set_ylim(0, ms_y_max)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.suptitle(
        f"{title_prefix} Well {well} Stage {stage}", x=0.15, y=0.98, ha="right"
    )
    return fig


def make_panel_stage_ms_plot(
    ms_data: pd.DataFrame, stagedata: pd.DataFrame, fig_size=(10, 4)
) -> plt.figure:
    """Make a three panel (plan, gun barrel, and time) plot to show
    microseismic data

    Args:
        ms_data (pd.DataFrame): microseismic data
        stagedata (pd.DataFrame): stage data

    Returns:
        plt.figure: output figure
    """
    cmap = sns.color_palette("viridis", as_cmap=True)
    fig, axes = plt.subplots(ncols=3, figsize=fig_size)

    # plan view
    sns.scatterplot(
        data=ms_data,
        x="Easting",
        y="Northing",
        size="Magnitude",
        hue="Event Depth",
        legend=True,
        sizes=(1, 50),
        palette=cmap,
        ax=axes[0],
        rasterized=True,
    )
    sns.scatterplot(
        data=stagedata.sort_values(["uwi", "md_mid_m"]),
        x="Easting",
        y="Northing",
        color="black",
        ax=axes[0],
    )
    axes[0].set_xlim(-1200, 1200)
    axes[0].set_ylim(-2000, 2000)
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")
    axes[0].set_title("a)", y=1.0, x=-0.2)

    # all data - gun barrel
    cmap = sns.color_palette("viridis", as_cmap=True)
    sns.scatterplot(
        data=ms_data,
        x="Easting",
        y="Event Depth",
        size="Magnitude",
        hue="Event Depth",
        legend=False,
        sizes=(1, 50),
        palette=cmap,
        ax=axes[1],
        rasterized=True,
    )
    sns.scatterplot(
        data=stagedata.sort_values(["uwi", "md_mid_m"]),
        x="Easting",
        y="Depth",
        color="black",
        ax=axes[1],
    )
    axes[1].set_xlim(-1200, 1200)
    axes[1].set_ylim(2700, 2100)
    axes[1].set_xlabel("Easting (m)")
    axes[1].set_ylabel("Event Depth (mbsl)")
    axes[1].set_title("b)", y=1.0, x=-0.18)

    # make time series plot (without stages)
    cmap = sns.color_palette("viridis", as_cmap=True)
    sns.scatterplot(
        data=ms_data,
        x="Days",
        y="Magnitude",
        size="Magnitude",
        hue="Event Depth",
        legend=False,
        sizes=(1, 50),
        palette=cmap,
        ax=axes[2],
        rasterized=True,
    )
    axes[2].set_title("c)", y=1.0, x=-0.16)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0.02),
        loc="lower center",
        borderaxespad=0,
        ncol=12,
    )
    axes[0].get_legend().remove()
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def make_linear_trace_analysis_plot(
    stg_dist, linear_model, linear_trace, fig_size=(12, 4), samples=10
):
    sns.set_style("whitegrid")
    sns.set_palette("Blues")

    fig = plt.figure(figsize=fig_size)
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    # prior predictive check
    with linear_model:
        prior_pred_check = pm.sample_prior_predictive(samples=samples)
    sns.scatterplot(
        x=stg_dist.t_start_s, y=stg_dist.dx_strike_m.abs(), color="black", ax=ax1
    )
    t = np.linspace(1, stg_dist.t_start_s.max(), 1000)
    for draw in prior_pred_check.prior.draw.values:
        Do = prior_pred_check.prior.sel({"draw": draw}).Do_x.values[0]
        sns.lineplot(x=t, y=np.sqrt(6 * Do * t), color="#4878d0", alpha=0.2, ax=ax1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Distance (m)")
    ax1.set_ylim(0, 1000)
    ax1.set_title("a)", y=1.0, x=-0.2)

    # posterior predictive check
    sns.scatterplot(
        x=stg_dist.t_start_s, y=stg_dist.dx_strike_m.abs(), color="black", ax=ax2
    )
    for chain in linear_trace.posterior.chain.values:
        draws = np.random.choice(
            linear_trace.posterior.sel({"chain": chain}).draw.values, samples
        )
        for draw in draws:
            Do = linear_trace.posterior.sel({"chain": chain, "draw": draw}).Do_x.values
            sns.lineplot(x=t, y=np.sqrt(6 * Do * t), color="#4878d0", alpha=0.1, ax=ax2)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Distance (m)")
    ax2.set_ylim(0, 1000)
    ax2.set_title("c)", y=1.0, x=-0.18)

    # trace plot
    idx = linear_trace["posterior"]["draw"].values
    dox_trace = linear_trace["posterior"]["Do_x"].values

    ls_dict = {0: "-", 1: ":", 2: "--", 3: "-."}

    for trace in range(dox_trace.shape[0]):
        sns.lineplot(x=idx, y=dox_trace[trace, :], alpha=0.5, ls=ls_dict[trace], ax=ax3)
    ax3.set_ylabel("D$_{0x}$")
    ax3.set_ylim(np.round(dox_trace.min() - 0.1, 1), np.round(dox_trace.max() + 0.1, 1))
    ax3.set_xlabel("Draw")
    ax3.set_title("e)", y=1.0, x=-0.16)

    plt.tight_layout()
    return fig


def make_nonlinear_trace_analysis_plot(
    stg_dist, nonlinear_model, nonlinear_trace, fig_size=(12, 4), samples=10
):
    sns.set_style("whitegrid")
    sns.set_palette("Greens_r")

    fig = plt.figure(figsize=fig_size)
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    # prior predictive check
    with nonlinear_model:
        prior_pred_check = pm.sample_prior_predictive(samples=samples)

    sns.scatterplot(
        x=stg_dist.t_start_s, y=stg_dist.dx_strike_m.abs(), color="black", ax=ax1
    )
    t = np.linspace(1, stg_dist.t_start_s.max(), 1000)
    for draw in prior_pred_check.prior.draw.values:
        Do = prior_pred_check.prior.sel({"draw": draw}).Do_x.values[0]
        sns.lineplot(x=t, y=np.sqrt(6 * Do * t), color="green", alpha=0.2, ax=ax1)
        if "k" in prior_pred_check.prior.keys():
            k = prior_pred_check.prior.sel({"draw": draw}).k.values[0]
            Po = prior_pred_check.prior.sel({"draw": draw}).Po.values[0]
            sns.lineplot(
                x=t,
                y=calc_tilda_c(k, Po) * np.sqrt(Do * t),
                color="red",
                alpha=0.05,
                ax=ax1,
            )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Distance (m)")
    ax1.set_ylim(0, 1000)
    ax1.set_title("b)", y=1.0, x=-0.2)

    # posterior predictive check
    t = np.linspace(1, stg_dist.t_start_s.max(), 1000)
    sns.scatterplot(
        x=stg_dist.t_start_s, y=stg_dist.dx_strike_m.abs(), color="black", ax=ax2
    )
    for chain in nonlinear_trace.posterior.chain.values:
        draws = np.random.choice(
            nonlinear_trace.posterior.sel({"chain": chain}).draw.values, samples
        )
        for draw in draws:
            k = nonlinear_trace.posterior.sel({"chain": chain, "draw": draw}).k.values
            Po = nonlinear_trace.posterior.sel({"chain": chain, "draw": draw}).Po.values
            Do = nonlinear_trace.posterior.sel(
                {"chain": chain, "draw": draw}
            ).Do_x.values
            sns.lineplot(x=t, y=np.sqrt(6 * Do * t), color="red", alpha=0.1, ax=ax2)
            sns.lineplot(
                x=t,
                y=calc_tilda_c(k, Po) * np.sqrt(Do * t),
                color="green",
                alpha=0.1,
                ax=ax2,
            )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Lateral Distance (m)")
    ax2.set_ylim(0, 1000)
    ax2.set_title("d)", y=1.0, x=-0.18)

    # trace plot
    idx = nonlinear_trace["posterior"]["draw"].values
    dox_trace = nonlinear_trace["posterior"]["Do_x"].values

    ls_dict = {0: "-", 1: ":", 2: "--", 3: "-."}

    for trace in range(dox_trace.shape[0]):
        sns.lineplot(x=idx, y=dox_trace[trace, :], alpha=0.5, ls=ls_dict[trace], ax=ax3)
    ax3.set_ylabel("D$_{0x}$")
    ax3.set_ylim(np.round(dox_trace.min() - 0.1, 1), np.round(dox_trace.max() + 0.1, 1))
    ax3.set_xlabel("Draw")
    ax3.set_title("f)", y=1.0, x=-0.16)

    plt.tight_layout()
    return fig


def make_diffusivity_comparison_plot(
    combined_diff_results, nonlinear_bayes_ms_data, stage_data, fig_size=(13.5, 8)
):

    nonlinear_hf_data = (
        nonlinear_bayes_ms_data.query("in_frac_ellipsoid")
        .query("in_diff_ellipsoid")
        .copy()
    )
    nonlinear_is_data = nonlinear_bayes_ms_data.query(
        "(in_diff_ellipsoid == False) | (in_frac_ellipsoid == False)"
    ).copy()

    sns.set_style("whitegrid")
    sns.set_palette("Set1")

    fig, axes = plt.subplots(
        2, 3, gridspec_kw={"height_ratios": [1, 2.5]}, figsize=fig_size
    )

    sns.histplot(
        data=combined_diff_results,
        x="mean_Do_x",
        kde=True,
        hue="model",
        ax=axes[0][0],
        bins=50,
    )
    axes[0][0].set_xlabel("Lateral Diffusivity (m$^2$/s)")
    axes[0][0].get_legend().remove()
    axes[0][0].set_xlim(0, 20)
    axes[0][0].set_title("a)", y=1.0, x=-0.2)

    sns.histplot(
        data=combined_diff_results,
        x="mean_Do_z",
        kde=True,
        hue="model",
        ax=axes[0][1],
        bins=50,
    )
    axes[0][1].set_xlabel("Vertical Diffusivity (m$^2$/s)")
    axes[0][1].get_legend().remove()
    axes[0][1].set_xlim(0, 10)
    axes[0][1].set_title("b)", y=1.0, x=-0.20)

    sns.histplot(
        data=combined_diff_results,
        x="mean_Do_y",
        kde=True,
        hue="model",
        ax=axes[0][2],
        bins=50,
    )
    axes[0][2].set_xlabel("Perpendicular Diffusivity (m$^2$/s)")
    axes[0][2].set_xlim(0, 5)
    axes[0][2].set_title("c)", y=1.0, x=-0.20)

    sns.set_palette("tab10")
    sns.scatterplot(
        data=nonlinear_bayes_ms_data,
        x="dx_cart_m",
        y="dy_cart_m",
        hue="Category",
        color=None,
        s=3,
        ax=axes[1][0],
    )
    axes[1][0].set_ylim(-1600, 1600)
    axes[1][0].set_xlim(-1600, 1600)
    axes[1][0].set_ylabel("Northing (m)")
    axes[1][0].set_xlabel("Easting (m)")
    axes[1][0].set_title("d)", y=1.0, x=-0.20)

    cmap = sns.color_palette("viridis", as_cmap=True)

    sns.scatterplot(
        data=nonlinear_hf_data,
        x="Easting",
        y="Northing",
        size="Magnitude",
        hue="Event Depth",
        legend=True,
        sizes=(1, 50),
        palette=cmap,
        ax=axes[1][1],
        rasterized=True,
    )
    sns.scatterplot(
        data=stage_data.sort_values(["uwi", "md_mid_m"]),
        x="Easting",
        y="Northing",
        color="black",
        ax=axes[1][1],
    )
    axes[1][1].set_xlim(-1200, 1200)
    axes[1][1].set_ylim(-1800, 1500)
    axes[1][1].set_xlabel("Easting (m)")
    axes[1][1].set_ylabel("Northing (m)")
    axes[1][1].get_legend().remove()
    axes[1][1].set_title("e)", y=1.0, x=-0.20)

    sns.scatterplot(
        data=nonlinear_is_data,
        x="Easting",
        y="Northing",
        size="Magnitude",
        hue="Event Depth",
        legend=True,
        sizes=(1, 50),
        palette=cmap,
        ax=axes[1][2],
        rasterized=True,
    )
    sns.scatterplot(
        data=stage_data.sort_values(["uwi", "md_mid_m"]),
        x="Easting",
        y="Northing",
        color="black",
        ax=axes[1][2],
    )
    axes[1][2].set_xlim(-1200, 1200)
    axes[1][2].set_ylim(-1800, 1500)
    axes[1][2].set_xlabel("Easting (m)")
    axes[1][2].set_ylabel("Northing (m)")
    axes[1][2].get_legend().remove()
    axes[1][2].set_title("f)", y=1.0, x=-0.20)

    plt.tight_layout()
    return fig
