import pandas as pd
import numpy as np
from src.diffusivity import calc_k_quantile, calc_tilda_c
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import seaborn as sns
sns.set_context("notebook")
sns.set_style("whitegrid")

def make_quantile_contour_plot(fig_size=(10,5), max_quantile=0.95):
    """Reproduce the quantile contour plot"""
    ks = np.arange(1E-3,1000,1)
    ps = np.arange(1,50,1)
    kv, pv = np.meshgrid(ks, ps, sparse=False)
    quantiles = calc_k_quantile(kv,pv, max_quantile)

    levels = np.round(np.linspace(0, max_quantile, 10+1),2)
    fig, ax = plt.subplots(figsize=fig_size)
    cs = ax.contourf(kv, pv, quantiles, levels=levels)
    fig.colorbar(cs, ax=ax)
    ax.set_xlabel('Permeability Compliance (GPa)')
    ax.set_ylabel('Net Injection Pressure (MPa)')
    return fig

def make_basic_diffusivity_plot(fig_size=(10,5)):
    """Reproduce the diffusivity plot"""
    t = np.linspace(0,5000,100)
    Do = 1 # matrix diffusivity (m2/s)
    Po = 10 # pressure (MPa)

    k_dict = {
        1:'black',
        10: '#e41a1c',
        100: '#4daf4a'
    }

    fig, ax = plt.subplots(figsize=fig_size)

    for k in k_dict.keys():
        k_tilda = k * Po / 1000
        D = Do * np.exp(k_tilda)
        r_tf = np.sqrt(6*D*t)
        tilda_c = calc_tilda_c(k, Po)
        r_fd = tilda_c * np.sqrt(Do * t)
        sns.lineplot(x=t/3600, y=r_fd, color=k_dict[k], linestyle='--', ax=ax)
        sns.lineplot(x=t/3600, y=r_tf, color=k_dict[k], linestyle='-', ax=ax)

    ax.set_ylim([0,300])
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Distance (m)')
    return fig


def make_stage_plot(distances, params, well, stage, figsize=(10,3), Do=2):    
    stg_dist = distances.query("WellID == @well").query("Stage == @stage").copy()
    stg_dist['in_frac_ellipsoid'] = ~stg_dist['in_frac_ellipsoid']
    stg_dist['dx_strike_m'] = stg_dist['dx_strike_m'].abs()
    sns.set_palette('colorblind')
    stg_dist = stg_dist.rename(columns={
        'dx_cart_m':'Easting (m)', 
        'dy_cart_m':'Northing (m)', 
        'dz_cart_m':'Vertical Offset (m)', 
        'dx_strike_m':'Lateral Distance (m)',
        't_start_s':'Time from Stage Start (s)', 
        'in_frac_ellipsoid': 'In Plausible Ellipsoid'
    })
    length =params['plausible_ellipsoid_lwh'][0] 
    width = params['plausible_ellipsoid_lwh'][1]
    height = params['plausible_ellipsoid_lwh'][2]

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=figsize, ncols=3)

    sns.scatterplot(data=stg_dist, x='Time from Stage Start (s)', y='Lateral Distance (m)', hue='In Plausible Ellipsoid', ax=ax1)
    sns.scatterplot(x=[0], y=[0], s=250, color=".2", ax=ax1, marker="*")
    t = np.linspace(10, stg_dist['Time from Stage Start (s)'].max(), 1000)
    sns.lineplot(x=t, y=np.sqrt(6*Do*t), ax=ax1, color='black')
    ax1.get_legend().remove()
    ax1.set_ylim(0,length*2)

    sns.scatterplot(data=stg_dist, x='Easting (m)', y='Northing (m)', hue='In Plausible Ellipsoid', ax=ax2)
    sns.scatterplot(x=[0], y=[0], s=250, color=".2", ax=ax2, marker="*")
    elps = Ellipse((0, 0), length*2, width*2, angle=params['onstrike_angle_deg'], edgecolor='b',facecolor='none')
    ax2.add_artist(elps)
    ax2.get_legend().remove()
    ax2.set_xlim(-length*2,length*2)
    ax2.set_ylim(-width*2,width*2)

    sns.scatterplot(data=stg_dist, x='Easting (m)', y='Vertical Offset (m)', hue='In Plausible Ellipsoid', ax=ax3)
    sns.scatterplot(x=[0], y=[0], s=250, color=".2", ax=ax3, marker="*")
    elps = Ellipse((0, 0), length*2, height*2, angle=0, edgecolor='b',facecolor='none')
    ax3.add_artist(elps)
    ax3.get_legend().remove()
    ax3.set_xlim(-length*2,length*2)
    ax3.set_ylim(-height*2,height*2)

    plt.tight_layout()
    fig.suptitle(f"Well {well} Stage {stage}", x=0.15, y=1.02, ha='right')
    return fig

def make_segregation_stage_overlay(bayes_filtered_distances: pd.DataFrame, figsize=(10,8)):
    """Make an overlay of all stages in plan view, differentiating the different
    segregation categories (induced seismicity from plausible ellipsoid or diffusivity 
    vs hydraulic fracture). Adds a category column for the plot.

    Args:
        bayes_filtered_distances (pd.DataFrame): Dataframe of distances with in_frac_ellipsoid, in_diff_ellipsoid columns
        figsize (tuple, optional): _description_. Defaults to (10,8).
    """
    sns.set_style('whitegrid')
    sns.set_palette('Set1')
    
    bayes_filtered_distances['Category'] = np.where(
        bayes_filtered_distances.in_frac_ellipsoid == False, 'Induced Seismicity - Ellipsoid',
        np.where(bayes_filtered_distances.in_diff_ellipsoid == False, 'Induced Seismicity - Diffusivity',
        'Hydraulic Fracture')
        )

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=bayes_filtered_distances, x='dx_cart_m', y='dy_cart_m', hue='Category', color=None, alpha=0.1)
    ax.set_aspect('equal')
    ax.set_ylim(-2000,2000)
    ax.set_ylabel('Northing')
    ax.set_xlabel('Easting')
    plt.show()