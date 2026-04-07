# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# **Hyperactivation is a distinct change in sperm motility** characterized by:
#
# * High-amplitude, asymmetrical flagellar beating.
#
# * Increased head movement.
#
# * Non-linear, often erratic or circular trajectories.

# %% slideshow={"slide_type": "skip"}
import numpy as np
import pandas as pd
import glob
import os
import plotly.express as px

import scipy.signal
from scipy.signal import savgol_filter
from scipy.stats import entropy as shannon_entropy
import itertools


from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
import plotly.io as pio
import csv
import json
from IPython.display import Video


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # My functions

# %% slideshow={"slide_type": "skip"}
def interpolFFT(frecs,A,t):
    n=len(A)
    acum=0
    for k in range(n):
        acum=acum+A[k]*np.exp(2*np.pi*1j*t*frecs[k])
    return acum/n

def Dfourfit(recf,reca,t):# Primera derivada de la función dada
    acum=0
    n=len(reca)
    for i in range(n):
        acum=acum+reca[i]*recf[i]*np.exp(2*1j*np.pi*recf[i]*t)
    return 2*np.pi*1j*acum/(90*n)

def D2fourfit(recf,reca,t):# Segunda derivada de la función dada
  acum=0
  for i in range(len(recf)):
    acum=acum+reca[i]*recf[i]**2*np.exp(2*1j*np.pi*recf[i]*t)
  return -4*np.pi**2*acum/(90**2*len(recf))

def dist2D(x1,y1,x2,y2):
    xdifsq=(x2-x1)**2
    ydifsq=(y2-y1)**2
    return(np.sqrt(xdifsq+ydifsq))

def dist3D(x1,y1,z1,x2,y2,z2):
    xdifsq=(x2-x1)**2
    ydifsq=(y2-y1)**2
    zdifsq=(z2-z1)**2
    return(np.sqrt(xdifsq+ydifsq+zdifsq))

def curvature_torsion(x, y, z):
    r = np.vstack((x, y, z)).T
    dr = np.gradient(r, axis=0)
    d2r = np.gradient(dr, axis=0)
    d3r = np.gradient(d2r, axis=0)

    cross = np.cross(dr, d2r)
    norm_cross = np.linalg.norm(cross, axis=1)
    norm_dr = np.linalg.norm(dr, axis=1)
    curvature = norm_cross / (norm_dr**3 + 1e-8)

    torsion = np.einsum('ij,ij->i', cross, d3r) / (norm_cross**2 + 1e-8)
    return curvature, torsion

def curvDist3D(curva,nptos=400):
    lcl=0.0
    for pto in range(nptos-1):                               
        xi=curva.x.iloc[pto]
        yi=curva.y.iloc[pto]
        zi=curva.z.iloc[pto]
        xf=curva.x.iloc[pto+1]
        yf=curva.y.iloc[pto+1]
        zf=curva.z.iloc[pto+1]
        lcl=lcl+dist3D(xi,yi,zi,xf,yf,zf)
    return lcl

def conjuntoDiametro3D(curva, nptos=400):
    lstLongitudes=[]
    xi=curva.x.iloc[0]
    yi=curva.y.iloc[0]
    zi=curva.z.iloc[0]

    for pto in range(nptos-1):
        xf=curva.x.iloc[pto]
        yf=curva.y.iloc[pto]
        zf=curva.z.iloc[pto]
        lstLongitudes.append(dist3D(xi,yi,zi,xf,yf,zf))
    d=np.max(lstLongitudes)
    return d

def homogenizarRangos(curva1,curva2):
    xmax1=curva1.x.max()
    xmin1=curva1.x.min()
    ymax1=curva1.y.max()
    ymin1=curva1.y.min()
    zmax1=curva1.z.max()
    zmin1=curva1.z.min()
    xmax2=curva2.x.max()
    xmin2=curva2.x.min()
    ymax2=curva2.y.max()
    ymin2=curva2.y.min()
    zmax2=curva2.z.max()
    zmin2=curva2.z.min()

    xmax=np.max([xmax1,xmax2])
    ymax=np.max([ymax1,ymax2])
    zmax=np.max([zmax1,zmax2])
    xmin=np.min([xmin1,xmin2])
    ymin=np.min([ymin1,ymin2])
    zmin=np.min([zmin1,zmin2])

    
    deltax=xmax-xmin
    deltay=ymax-ymin
    deltaz=zmax-zmin
    maxdelta=np.max([deltax,deltay,deltaz])
    xadd=maxdelta-deltax
    yadd=maxdelta-deltay
    zadd=maxdelta-deltaz
    xadd=np.ceil(xadd/2)
    yadd=np.ceil(yadd/2)
    zadd=np.ceil(zadd/2)
    xmax=xmax+xadd+5
    xmin=xmin-xadd-5
    ymax=ymax+yadd+5
    ymin=ymin-yadd-5
    zmax=zmax+zadd+5
    zmin=zmin-zadd-5
    return [xmin,xmax,ymin,ymax,zmin,zmax]

def dimFrac3D(curva, nptos=400):
    lcl=curvDist3D(curva)
    d=conjuntoDiametro3D(curva)
    curveDimFractal=np.log10(nptos-1)/(np.log10(nptos-1)+np.log10(d/lcl))
    return curveDimFractal

def reflejarCurva3D(curva):
    micurva=curva.copy()
    x0=micurva.x.iloc[0]
    y0=micurva.y.iloc[0]
    z0=micurva.z.iloc[0]
    micurva.x=micurva.x-x0
    micurva.y=micurva.y-y0
    micurva.z=micurva.z-z0

    micurva.y=-micurva.y         # se refleja 

    micurva.x=micurva.x+x0
    micurva.y=micurva.y+y0
    micurva.z=micurva.z+z0
    return micurva

def buscarSwcidx(cel):
    nswcs=identificadoresArchivosSwcs.shape[0]
    fecha=cel.split('_')[-2]
    exp=cel.split('_')[-1]
    fileid='No encontrada'
    print('Buscando: ',fecha,exp)
    for j in range(nswcs):
        if(fecha in identificadoresArchivosSwcs.swc.iloc[j]):
            xp=identificadoresArchivosSwcs.swc.iloc[j].split('_')[-2]
            if(exp == xp):
                fileid=identificadoresArchivosSwcs.fileID.iloc[j].strip("'")
                print('Encontrado: ',identificadoresArchivosSwcs.swc.iloc[j].strip("'"))
                print(cel, ' corresponde a ')
                print('FileID= ', fileid,'\n')
    # return fileid
    encontrada='a'
    norigs=len(curvasFlagelares)
    for i in range(norigs):
        if(fileid in curvasFlagelares[i][0].nom.iloc[0]):
            encontrada=curvasFlagelares[i][0].nom.iloc[0]
            print('Orig: ',i,encontrada)
    if( encontrada != 'a'):
        nzenodo=len(curvasFlagelaresZenodo)
        for i in range(nzenodo):
            if(cel in curvasFlagelaresZenodo[i][0]):
                print('Zen: ',i,curvasFlagelaresZenodo[i][0])
    else:
        print('No encontrada')

def reflejarCurva3D(curva):
    micurva=curva.copy()
    x0=micurva.x.iloc[0]
    y0=micurva.y.iloc[0]
    z0=micurva.z.iloc[0]
    micurva.x=micurva.x-x0
    micurva.y=micurva.y-y0
    micurva.z=micurva.z-z0

    micurva.y=-micurva.y         # se refleja 

    micurva.x=micurva.x+x0
    micurva.y=micurva.y+y0
    micurva.z=micurva.z+z0
    return micurva

# Preprocessing

def preprocess(X, smooth=True):
    """
    X: (T, D) multivariate time-series
    """
    X = np.asarray(X, dtype=float)
    
    # z-score normalization
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    
    if smooth:
        X = savgol_filter(X, window_length=11, polyorder=2, axis=0)
    
    return X


# Kinematics

def kinematics(X, dt=1/90):
    """
    Returns velocity and acceleration
    """
    V = np.gradient(X, dt, axis=0)
    A = np.gradient(V, dt, axis=0)
    return V, A


# Dynamical Descriptors

def curvature(V, A):
    """
    Curvature for vector-valued motion
    """
    cross = np.cross(V, A)
    num = np.linalg.norm(cross, axis=1)
    den = (np.linalg.norm(V, axis=1) ** 3) + 1e-9
    return num / den


def jerk(A, dt=1/90):
    """
    Time derivative of acceleration magnitude
    """
    a_mag = np.linalg.norm(A, axis=1)
    J = np.gradient(a_mag, dt)
    return J


def signal_entropy(signal, bins=30):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    return shannon_entropy(hist + 1e-12)


def nonlinearity_measure(signal):
    """
    Measures deviation from linear dynamics
    """
    return np.mean(np.abs(np.diff(signal, n=2)))


def asymmetry_measure(X):
    """
    Generic asymmetry between dimensions
    """
    D = X.shape[1]
    diffs = []
    for i in range(D):
        for j in range(i+1, D):
            diffs.append(np.mean(np.abs(X[:,i] - X[:,j])))
    return np.mean(diffs)


def periodicity_measure(signal):
    """
    Harmonicity / periodic structure indicator
    """
    fft = np.fft.rfft(signal)
    power = np.abs(fft)**2
    power /= power.sum() + 1e-12
    return power.max()   # high → periodic, low → chaotic


# Main Descriptor Function

def compute_dynamical_descriptors(X, dt=1/90):
    """
    X shape: (T, D)
    returns: feature vector + dictionary
    """
    X = preprocess(X)
    V, A = kinematics(X, dt)

    speed = np.linalg.norm(V, axis=1)
    accel = np.linalg.norm(A, axis=1)

    kappa = curvature(V, A)
    J = jerk(A, dt)

    features = {
        # Kinematics
        "mean_speed": speed.mean(),
        "std_speed": speed.std(),
        "max_speed": speed.max(),

        "mean_accel": accel.mean(),
        "std_accel": accel.std(),
        "max_accel": accel.max(),

        # Geometry
        "mean_curvature": np.nanmean(kappa),
        "std_curvature": np.nanstd(kappa),
        "max_curvature": np.nanmax(kappa),

        # Dynamics
        "mean_jerk": J.mean(),
        "std_jerk": J.std(),

        # Complexity
        "speed_entropy": signal_entropy(speed),
        "accel_entropy": signal_entropy(accel),

        "speed_nonlinearity": nonlinearity_measure(speed),
        "accel_nonlinearity": nonlinearity_measure(accel),

        # Structure
        "asymmetry": asymmetry_measure(X),
        "periodicity": periodicity_measure(speed),

        # Energy-like
        "kinetic_energy": np.mean(speed**2)
    }

    feature_vector = np.array(list(features.values()))
    return feature_vector, features


# Batch Processing

def build_feature_matrix(dataset, dt=1/90):
    """
    dataset: list or array of shape (N, T, D)
    """
    F = []
    dicts = []
    for Xi in dataset:
        fv, fd = compute_dynamical_descriptors(Xi, dt)
        F.append(fv)
        dicts.append(fd)
    return np.array(F), dicts

def descri(arreglo):
    mini=np.argmin(arreglo)
    maxi=np.argmax(arreglo)
    print('Min= ',np.min(arreglo),', idx= ',mini)
    print('Max= ',np.max(arreglo),', idx= ',maxi)
    temp=[]
    mu=np.mean(arreglo)
    for i in range(len(arreglo)):
        temp.append(np.abs(arreglo[i]-mu))
    meanClosest=np.argmin(temp)
    print('Mean= ',mu,', idxClosest= ',meanClosest)

    return mini,maxi,meanClosest

def plot_single_cell_flagella_with_head(
    flagella_data,
    cell_index=0,
    x_col="x",
    y_col="y",
    z_col="z",
    title=None,
    line_width=4,
    marker_size=3,
    head_marker_size=10,
    show_markers=False
):
    """
    Animate a single cell flagellum with a larger point
    marking the initial point of the curve.

    Parameters
    ----------
    head_marker_size : int
        Size of the initial point marker
    """

    # ----------------------------------------
    # Detect input type
    # ----------------------------------------

    if isinstance(flagella_data[0], list):

        cell_data = flagella_data[cell_index]

    else:

        cell_data = flagella_data

    n_times = len(cell_data)

    if title is None:
        title = f"3D Flagella Animation — Cell {cell_index}"

    # ----------------------------------------
    # Compute global axis limits
    # ----------------------------------------

    all_x, all_y, all_z = [], [], []

    for df in cell_data:

        all_x.extend(df[x_col].values)
        all_y.extend(df[y_col].values)
        all_z.extend(df[z_col].values)

    x_range = [min(all_x), max(all_x)]
    y_range = [min(all_y), max(all_y)]
    z_range = [min(all_z), max(all_z)]

    # ----------------------------------------
    # Initial frame
    # ----------------------------------------

    fig = go.Figure()
    fig.update_layout(autosize=False,
        width=1500,
        height=1000,
        scene_aspectmode='cube'
        # margin=dict(l=10, r=10, t=10, b=10, pad=10)
        )
    df0 = cell_data[0]

    # Flagellum line
    fig.add_trace(
        go.Scatter3d(
            x=df0[x_col],
            y=df0[y_col],
            z=df0[z_col],
            mode="lines+markers" if show_markers else "lines",
            line=dict(width=line_width),
            marker=dict(size=marker_size),
            name="Flagellum"
        )
    )

    # Initial point marker (HEAD / BASE)
    fig.add_trace(
        go.Scatter3d(
            x=[df0[x_col].iloc[0]],
            y=[df0[y_col].iloc[0]],
            z=[df0[z_col].iloc[0]],
            mode="markers",
            marker=dict(
                size=head_marker_size,
                symbol="circle"
            ),
            name="Initial point"
        )
    )

    # ----------------------------------------
    # Frames
    # ----------------------------------------

    frames = []

    for t in range(n_times):

        df = cell_data[t]

        frames.append(
            go.Frame(
                data=[

                    # Flagellum
                    go.Scatter3d(
                        x=df[x_col],
                        y=df[y_col],
                        z=df[z_col],
                        mode="lines+markers" if show_markers else "lines",
                        line=dict(width=line_width),
                        marker=dict(size=marker_size),
                    ),

                    # Initial point
                    go.Scatter3d(
                        x=[df[x_col].iloc[0]],
                        y=[df[y_col].iloc[0]],
                        z=[df[z_col].iloc[0]],
                        mode="markers",
                        marker=dict(
                            size=head_marker_size
                        ),
                    )

                ],
                name=str(t)
            )
        )

    fig.frames = frames

    # ----------------------------------------
    # Slider
    # ----------------------------------------

    slider_steps = []

    for t in range(n_times):

        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [str(t)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=40, redraw=True),
                        transition=dict(duration=0)
                    )
                ],
                label=str(t)
            )
        )

    sliders = [
        dict(
            active=0,
            pad={"t": 50},
            steps=slider_steps
        )
    ]

    # ----------------------------------------
    # Layout
    # ----------------------------------------

    fig.update_layout(
        title=title,

        scene=dict(
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            aspectmode="data"
        ),

        sliders=sliders,

        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[

                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=40, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True
                            )
                        ]
                    ),

                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0),
                                transition=dict(duration=0),
                                mode="immediate"
                            )
                        ]
                    )

                ]
            )
        ]
    )

    return fig



def animate_flagella_with_feature_vertical(
    cell_data,
    feature,
    time=None,
    x_col="x",
    y_col="y",
    z_col="z",
    feature_name="Feature",
    head_marker_size=10,
    frame_duration=40,
    flagellarTitle="3D Flagellum"
):
    """
    Synchronized animation:

        TOP    -> 3D flagellum
        BOTTOM -> time series feature

    Shared slider and animation timing.
    """

    feature = np.asarray(feature)

    T = len(cell_data)

    if time is None:
        time = np.arange(T)

    time = np.asarray(time)

    # ----------------------------------
    # Compute axis limits for 3D
    # ----------------------------------

    all_x, all_y, all_z = [], [], []

    for df in cell_data:

        all_x.extend(df[x_col].values)
        all_y.extend(df[y_col].values)
        all_z.extend(df[z_col].values)

    x_range = [min(all_x), max(all_x)]
    y_range = [min(all_y), max(all_y)]
    z_range = [min(all_z), max(all_z)]

    # ----------------------------------
    # Create vertical subplots
    # ----------------------------------

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        specs=[
            [{"type": "scene"}],
            [{"type": "xy"}]
        ],
        subplot_titles=(
            flagellarTitle,
            feature_name
        )
    )

    # ----------------------------------
    # Initial frame
    # ----------------------------------

    df0 = cell_data[0]

    # 3D flagellum

    fig.add_trace(
        go.Scatter3d(
            x=df0[x_col],
            y=df0[y_col],
            z=df0[z_col],
            mode="lines",
            line=dict(width=4),
            name="Flagellum"
        ),
        row=1,
        col=1
    )

    # Head marker

    fig.add_trace(
        go.Scatter3d(
            x=[df0[x_col].iloc[0]],
            y=[df0[y_col].iloc[0]],
            z=[df0[z_col].iloc[0]],
            mode="markers",
            marker=dict(size=head_marker_size),
            name="Head"
        ),
        row=1,
        col=1
    )

    # Feature line

    fig.add_trace(
        go.Scatter(
            x=time,
            y=feature,
            mode="lines+markers",
            name=feature_name
        ),
        row=2,
        col=1
    )

    # Moving point

    fig.add_trace(
        go.Scatter(
            x=[time[0]],
            y=[feature[0]],
            mode="markers",
            marker=dict(size=12),
            name="Current time"
        ),
        row=2,
        col=1
    )

    # ----------------------------------
    # Frames
    # ----------------------------------

    frames = []

    for t in range(T):

        df = cell_data[t]

        frames.append(
            go.Frame(
                data=[

                    go.Scatter3d(
                        x=df[x_col],
                        y=df[y_col],
                        z=df[z_col],
                        mode="lines",
                        line=dict(width=4)
                    ),

                    go.Scatter3d(
                        x=[df[x_col].iloc[0]],
                        y=[df[y_col].iloc[0]],
                        z=[df[z_col].iloc[0]],
                        mode="markers",
                        marker=dict(size=head_marker_size)
                    ),

                    go.Scatter(
                        x=time,
                        y=feature,
                        mode="lines+markers"
                    ),

                    go.Scatter(
                        x=[time[t]],
                        y=[feature[t]],
                        mode="markers",
                        marker=dict(size=12)
                    )

                ],
                name=str(t)
            )
        )

    fig.frames = frames

    # ----------------------------------
    # Slider
    # ----------------------------------

    steps = []

    for t in range(T):

        steps.append(
            dict(
                method="animate",
                args=[
                    [str(t)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=frame_duration, redraw=True),
                        transition=dict(duration=0)
                    )
                ],
                label=str(t)
            )
        )

    sliders = [
        dict(
            active=0,
            pad={"t": 50},
            steps=steps
        )
    ]

    # ----------------------------------
    # Layout
    # ----------------------------------

    fig.update_layout(

        height=800,

        scene=dict(
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            aspectmode="data"
        ),

        xaxis=dict(
            title="Time"
        ),

        yaxis=dict(
            title="Fractal Dimension",#feature_name,
            range=[feature.min(), feature.max()]
        ),

        sliders=sliders,

        updatemenus=[
            dict(
                type="buttons",
                showactive=False,

                buttons=[

                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_duration, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True
                            )
                        ]
                    ),

                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0),
                                transition=dict(duration=0),
                                mode="immediate"
                            )
                        ]
                    )

                ]
            )
        ]
    )

    return fig

def animate_time_series_point(
    feature,
    time=None,
    title="Time Series Animation",
    y_label="Feature",
    point_size=12,
    frame_duration=40
):
    """
    Animate a moving point along a time series with slider.

    Parameters
    ----------
    feature : array-like
        1D time series

    time : array-like, optional
        Time vector. If None, uses indices.

    point_size : int
        Size of moving point

    frame_duration : int
        milliseconds between frames
    """

    feature = np.asarray(feature)

    T = len(feature)

    if time is None:
        time = np.arange(T)

    time = np.asarray(time)

    # ----------------------------------
    # Axis limits (fixed for stability)
    # ----------------------------------

    x_range = [time.min(), time.max()]
    y_range = [feature.min(), feature.max()]

    # ----------------------------------
    # Initial figure
    # ----------------------------------

    fig = go.Figure()

    # Full time series line
    fig.add_trace(
        go.Scatter(
            x=time,
            y=feature,
            mode="lines+markers",
            name="Feature"
        )
    )

    # Moving point
    fig.add_trace(
        go.Scatter(
            x=[time[0]],
            y=[feature[0]],
            mode="markers",
            marker=dict(size=point_size),
            name="Current time"
        )
    )

    # ----------------------------------
    # Frames
    # ----------------------------------

    frames = []

    for t in range(T):

        frames.append(
            go.Frame(
                data=[

                    go.Scatter(
                        x=time,
                        y=feature,
                        mode="lines+markers"
                    ),

                    go.Scatter(
                        x=[time[t]],
                        y=[feature[t]],
                        mode="markers",
                        marker=dict(size=point_size)
                    )

                ],
                name=str(t)
            )
        )

    fig.frames = frames

    # ----------------------------------
    # Slider
    # ----------------------------------

    steps = []

    for t in range(T):

        steps.append(
            dict(
                method="animate",
                args=[
                    [str(t)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=frame_duration, redraw=True),
                        transition=dict(duration=0)
                    )
                ],
                label=str(t)
            )
        )

    sliders = [
        dict(
            active=0,
            pad={"t": 50},
            steps=steps
        )
    ]

    # ----------------------------------
    # Layout
    # ----------------------------------

    fig.update_layout(

        title=title,

        xaxis=dict(title="Time", range=x_range),

        yaxis=dict(title=y_label, range=y_range),
        # yaxis=dict(title="Fractal Dimension", range=y_range),

        sliders=sliders,

        updatemenus=[
            dict(
                type="buttons",
                showactive=False,

                buttons=[

                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_duration, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True
                            )
                        ]
                    ),

                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0),
                                transition=dict(duration=0),
                                mode="immediate"
                            )
                        ]
                    )

                ]
            )
        ]
    )

    return fig



# ----------------------------
# 0. Preprocessing
# ----------------------------
def normalize_curve(r):
    r = np.asarray(r)
    r = r - r.mean(axis=0)
    r = r / (np.std(r) + 1e-8)
    return r


def match_length(r1, r2):
    """Trim to same length (simple version)"""
    T = min(len(r1), len(r2))
    return r1[:T], r2[:T]


# ----------------------------
# 1. Position Cross-Correlation
# ----------------------------
def position_cross_correlation(r1, r2, max_lag):
    r1, r2 = match_length(normalize_curve(r1), normalize_curve(r2))
    
    C = []
    for lag in range(max_lag):
        if lag == 0:
            val = np.mean(np.sum(r1 * r2, axis=1))
        else:
            val = np.mean(np.sum(r1[:-lag] * r2[lag:], axis=1))
        C.append(val)
    
    return np.array(C)


# ----------------------------
# 2. Velocity Cross-Correlation
# ----------------------------
def velocity_cross_correlation(r1, r2, max_lag):
    r1, r2 = match_length(normalize_curve(r1), normalize_curve(r2))
    
    v1 = np.diff(r1, axis=0)
    v2 = np.diff(r2, axis=0)
    
    C = []
    for lag in range(max_lag):
        if lag == 0:
            val = np.mean(np.sum(v1 * v2, axis=1))
        else:
            val = np.mean(np.sum(v1[:-lag] * v2[lag:], axis=1))
        C.append(val)
    
    C = np.array(C)
    return C / (C[0] + 1e-8)


# ----------------------------
# 3. Tangent Cross-Correlation
# ----------------------------
def tangent_cross_correlation(r1, r2, max_lag):
    r1, r2 = match_length(normalize_curve(r1), normalize_curve(r2))
    
    v1 = np.diff(r1, axis=0)
    v2 = np.diff(r2, axis=0)
    
    t1 = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8)
    t2 = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8)
    
    C = []
    for lag in range(max_lag):
        if lag == 0:
            val = np.mean(np.sum(t1 * t2, axis=1))
        else:
            val = np.mean(np.sum(t1[:-lag] * t2[lag:], axis=1))
        C.append(val)
    
    return np.array(C)


# ----------------------------
# 4. Displacement Cross-Correlation
# ----------------------------
def displacement_cross_correlation(r1, r2, max_lag):
    r1, r2 = match_length(normalize_curve(r1), normalize_curve(r2))
    
    D = []
    for lag in range(1, max_lag + 1):
        d1 = r1[lag:] - r1[:-lag]
        d2 = r2[lag:] - r2[:-lag]
        
        val = np.mean(np.sum(d1 * d2, axis=1))
        D.append(val)
    
    D = np.array(D)
    return D / (np.max(np.abs(D)) + 1e-8)


# ----------------------------
# 5. Spectral Cross-Correlation
# ----------------------------
def spectral_cross_correlation(r1, r2, n_freq):
    r1, r2 = match_length(normalize_curve(r1), normalize_curve(r2))
    
    S1 = np.abs(fft(r1, axis=0))**2
    S2 = np.abs(fft(r2, axis=0))**2
    
    S1 = S1.mean(axis=1)[:n_freq]
    S2 = S2.mean(axis=1)[:n_freq]
    
    # cosine similarity in frequency space
    num = np.dot(S1, S2)
    den = np.linalg.norm(S1) * np.linalg.norm(S2) + 1e-8
    
    return num / den


# ----------------------------
# 6. Unified Cross-Descriptor
# ----------------------------
def cross_descriptor(r1, r2, max_lag=50, n_freq=30):
    Cp = position_cross_correlation(r1, r2, max_lag)
    Cv = velocity_cross_correlation(r1, r2, max_lag)
    Ct = tangent_cross_correlation(r1, r2, max_lag)
    D  = displacement_cross_correlation(r1, r2, max_lag)
    S  = np.array([spectral_cross_correlation(r1, r2, n_freq)])
    
    return np.concatenate([Cp, Cv, Ct, D, S])


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    T = 300
    t = np.linspace(0, 10, T)
    
    r1 = np.stack([np.sin(t), np.cos(t), np.sin(2*t)], axis=1)
    r2 = np.stack([np.sin(t+0.5), np.cos(t+0.5), np.sin(2*t+0.5)], axis=1)
    
    descriptor = cross_descriptor(r1, r2)
    
    print("Cross-descriptor shape:", descriptor.shape)

def features(curva):
    X=preprocess([curva.x,curva.y,curva.z], smooth=False)
    X=np.array(X)
    X=X.T
    V, A = kinematics(X, 1)
    speed = np.linalg.norm(V, axis=1)
    accel = np.linalg.norm(A, axis=1)
    J = jerk(A, 1)
    normJerk=np.linalg.norm(J)
    per=periodicity_measure(X)
    ent=signal_entropy(X)
    kappa=curvature(V,A)
    print('periodicidad: ',periodicity_measure(X))#,periodicity_measure(Y),periodicity_measure(Z))
    print('entropia: ',signal_entropy(X))#,signal_entropy(Y),signal_entropy(Z))
    print('Primera Derivada: ', speed.mean(),speed.std(),speed.max())
    print('Segunda Derivada: ', accel.mean(),accel.std(),accel.max())
    print('Tercera Derivada: ', normJerk.mean(),normJerk.std(),normJerk.max())
    print('curvatura: ', np.nanmean(kappa), np.nanstd(kappa), np.nanmax(kappa))

    # return speed,accel,J,per,ent,k


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Main

# %% [markdown]
# First, the files with the space-curves coordinates are read relating the name of the cell with the list of dataframes containing its coordinates.

# %%
# Se leen las coordenadas tridimensionales de las células en Z e n o d o
# homedir=os.path.expanduser('~')
homedir=os.path.expanduser('~')
root_dir = homedir+"/lastestxZenodo/correccion-2026/traces_micrometers/"
#root_dir = "/home/sidney/lastestxZenodo/trace_microns_smooth/"
curvasFlagelaresZenodo=[]
# Iterate over all folders and files
for foldername, subfolders, filenames in os.walk(root_dir):
    # print(f"Current folder: {foldername}")
    # if(not filenames):
    #     temp=foldername.split('/')
    #     fecha=temp[-1]
    #     print('------>',fecha)
    if(filenames):
        temp=foldername.split('/')
        celula=temp[-1]
        # print('------>',celula)
        fileX=os.path.join(foldername,'X.csv')
        fileY=os.path.join(foldername,'Y.csv')
        fileZ=os.path.join(foldername,'Z.csv')
        xdf = pd.read_csv(fileX, sep=' ', header=None)
        ydf = pd.read_csv(fileY, sep=' ', header=None)
        zdf = pd.read_csv(fileZ, sep=' ', header=None)
        ntiempos=xdf.shape[1]
        tiempos=[]
        for i in range(ntiempos):
            tempdf=pd.concat([xdf[i],ydf[i],zdf[i]],axis=1)
            tempdf.columns=['x','y','z']
            cleanTempdf=tempdf.dropna()
            tiempos.append(cleanTempdf)
        curvasFlagelaresZenodo.append([celula,tiempos])

# %% [markdown]
# The following variable contains the space-curves coordinates without the names of the cells.

# %%
flagellar_data = [item[1] for item in curvasFlagelaresZenodo]   # list of list of DataFrames, without the names

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Dimensión fractal flagelar 
#
# <!--
# Esta dimensión fractal flagelar es diferente al "flagellar curvature ratio" definido por Suarez en: Movement characteristics and acrosomal status of rabbit spermatozoa recovered at the site and time of fertilization Suarez 1983. Suarez lo define como *la distancia en línea recta desde la unión de la cabeza con la pieza media al primer punto de inflección de la cola, dividido por la distancia curvilinea entre esos mismo dos puntos*. 
#
# El objetivo principal es al análisis de la **dimensión fractal flagelar D**, 
# -->
#
# This flagellar fractal dimension is different from the "flagellar curvature ratio" defined by Suarez in: Movement characteristics and acrosomal status of rabbit spermatozoa recovered at the site and time of fertilization, Suarez 1983. Suarez defines it as *the rectilinear distance from the head-middle pice junction to the first inflection point of the tail divided by the curvilinear distance among those two points*.
#
# ---
#
# The main goal is the analysis of the Katz **space-curve's fractal dimension D**:
#
# $$
# D=\frac{\log(n)}{\log(n)+\log(d/L)}.\nonumber
# $$
#
# ---

# %%
len(curvasFlagelaresZenodo),len(curvasFlagelaresZenodo[0][1]),curvasFlagelaresZenodo[0][1][0].shape[0]

# %% [markdown]
# First, we compute the **Katz Fractal Dimension** <u>for each flagellar curve of every sperm cell</u>.

# %% jupyter={"source_hidden": true}
''' Computation
# Computation of the Katz fractal dimension for each space curve in the dataset

ncels=len(curvasFlagelaresZenodo)
celsZenodoDimFrac=[]
for celula in range(ncels):
    ntiempos=len(curvasFlagelaresZenodo[celula][1])
    fracDimDistro=[]
    for tiempo in range(ntiempos):
        nptos=curvasFlagelaresZenodo[celula][1][tiempo].shape[0]   #400
# The computation is only for those space-curves with at least 342 points
        if( nptos >= 342 ):
            nptos=342
            lcl=curvDist3D(curvasFlagelaresZenodo[celula][1][tiempo],nptos)
            d=conjuntoDiametro3D(curvasFlagelaresZenodo[celula][1][tiempo],nptos)
            flglrDimFractal=np.log10(nptos-1)/(np.log10(nptos-1)+np.log10(d/lcl))
            fracDimDistro.append(flglrDimFractal)
    celname=curvasFlagelaresZenodo[celula][0]   #.nom.iloc[0].split('_stac')[0]
    celsZenodoDimFrac.append([celname,fracDimDistro])

# dfCelsZenodoDimFrac=pd.DataFrame(celsZenodoDimFrac,columns=['cel','dimFracDistro'])

with open('celsZenodoDimFrac.json', 'w') as f:
    json.dump(celsZenodoDimFrac, f)
    '''

# %%
# the fractal dimensions previously computed are readed from a json file
with open("celsZenodoDimFrac.json") as f:
    lst = json.load(f)

dfCelsZenodoDimFrac=pd.DataFrame(lst,columns=['cel','dimFracDistro'])

# %%
dfCelsZenodoDimFrac.head()

# %% [markdown]
# Now, the next step is to explore wich kind of **analysis** could get the best statistical classfication. Of course, if the <u>nature of the data implies certain requirements</u> these should be considered over the purely statistical point of view.

# %% [markdown]
# # Mean flagellar form
#
# In general, it could be considered that the dynamic mean is the best way to describe the behavior of several individual elements. As such, the first attempt is to use the mean flagellar Katz Fractal Dimension to analyze the motion of the space curves.

# %%
ncels=dfCelsZenodoDimFrac.shape[0]
FDmeanDist=[]
for i in range(ncels):
    nom=dfCelsZenodoDimFrac.cel.iloc[i]
    fdmean=np.mean(dfCelsZenodoDimFrac.dimFracDistro.iloc[i])-1
    if( 'No' in dfCelsZenodoDimFrac.cel.iloc[i] ):
        cond='NoCap'
    else:
        cond='Cap'
    FDmeanDist.append([nom,cond,fdmean])

# %%
dfZenodoFDmeanDist=pd.DataFrame(FDmeanDist, columns=['cel','Condition','FDmean'])

# %%
fig=px.violin(dfZenodoFDmeanDist, x='FDmean', color='Condition', box=True, points='all',
             category_orders={"Condition":["NoCap","Cap"]})
fig.update_traces(meanline_visible=True)
fig.update_layout(title_text='Distribution of the cellular mean for the flagellar fractal dimension', 
                  title_x=0.5,font=dict(size=20),xaxis_title="Fractal dimension mean")
# fig.update_layout(title_text='Distribution of the mean for the flagellar fractal dimension of each cell of the Zenodo dataset', 
#                   title_x=0.5,font=dict(size=20))
fig.show()

# %%
dfZenodoFDmeanDist[dfZenodoFDmeanDist['FDmean']>0.04]

# %% [markdown]
# ## Sperm-10-NoCap_210702_Exp21

# %%
indiceDistro=20
print(np.max(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro])-np.min(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]))
fig = px.line(y=dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro], markers=True)
fig.update_layout(title_text='Fractal dimension series for : '+dfCelsZenodoDimFrac.cel.iloc[indiceDistro], 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
minidx,maxidx,muidx=descri(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro])

# %%
for i in range(len(curvasFlagelaresZenodo)):
    if( curvasFlagelaresZenodo[i][0] == dfCelsZenodoDimFrac.cel.iloc[indiceDistro] ):
        print(i)
        indiceFlagelar=i
print(dfCelsZenodoDimFrac.cel.iloc[indiceDistro],curvasFlagelaresZenodo[indiceFlagelar][0])

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=96
tiempo2=84
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1250,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name=' min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name=' max, Tiempo = '+str(tiempo2), line=dict(
        color='blue', # Optional: set line color
        # width=10       # Set the line thickness in pixels
    )))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)
# ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    # scene=dict(
    #     xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
    #     yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]], title='Y'),   # Set Y-axis range and title
    #     zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    # ),
    title=curvasFlagelaresZenodo[celula2][0],
)
fig.show()

# %%
kappa1,tau1=curvature_torsion(curvasFlagelaresZenodo[celula1][1][tiempo1].x, curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                                                                                               curvasFlagelaresZenodo[celula1][1][tiempo1].z)

kappa2,tau2=curvature_torsion(curvasFlagelaresZenodo[celula2][1][tiempo2].x, curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                                                                                               curvasFlagelaresZenodo[celula2][1][tiempo2].z)
fig = px.line(y=kappa1, markers=True)
fig.update_layout(title_text='Curvature for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMin): '+str(tiempo1), 
                  title_x=0.5,font=dict(size=15))
fig.show()

fig = px.line(y=kappa2, markers=True)
fig.update_layout(title_text='Curvature for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMax): '+str(tiempo2), 
                  title_x=0.5,font=dict(size=15))
fig.show()
fig = px.line(y=tau1, markers=True)
fig.update_layout(title_text='Torsion for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMin): '+str(tiempo1), 
                  title_x=0.5,font=dict(size=15))
fig.show()

fig = px.line(y=tau2, markers=True)
fig.update_layout(title_text='Torsion for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMax): '+str(tiempo2), 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
np.mean(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro])#,np.argmean(dfCelsZenodoDimFrac.dimFracDistro.iloc[mindice])

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=minidx
tiempo2=muidx
tiempo3=maxidx
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1250,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name=' min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name=' mean, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)

fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo3].x, y=curvasFlagelaresZenodo[celula2][1][tiempo3].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo3].z, mode='lines',marker=dict(color='green', size=5), 
                           name=' max, Tiempo = '+str(tiempo3)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo3].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo3].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo3].z.iloc[0]],
                 mode='markers',marker=dict(color='green'))
)

# ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    # scene=dict(
    #     xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
    #     yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]], title='Y'),   # Set Y-axis range and title
    #     zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    # ),
    title=curvasFlagelaresZenodo[celula1][0],
)
fig.show()

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=15
tiempo2=16
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name=curvasFlagelaresZenodo[celula1][0]+' min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)
# ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    # scene=dict(
    #     xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
    #     yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]], title='Y'),   # Set Y-axis range and title
    #     zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    # ),
    title='Un cambio muy drástico de un tiempo al siguiente.',
)
fig.show()

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Sperm-1-NoCap_210702_Exp4_cell-1

# %%
indiceDistro=27
fig = px.line(y=dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro][:], markers=True)
# fig.update_layout(title_text='Serie de tiempo de la dimensión fractal para una célula control', 
#                   title_x=0.5,font=dict(size=15))
fig.update_layout(
    xaxis_title="curve number",
    yaxis_title="DimFrac",
    # legend_title="Legend Title"
)
fig.update_layout(title_text='Fractal time series for : '+dfCelsZenodoDimFrac.cel.iloc[indiceDistro], 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
print(np.min(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]),np.argmin(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]),
      np.max(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]),np.argmax(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]))

# %%
for i in range(len(curvasFlagelaresZenodo)):
    if( curvasFlagelaresZenodo[i][0] == dfCelsZenodoDimFrac.cel.iloc[indiceDistro] ):
        print(i)
        indiceFlagelar=i
print( dfCelsZenodoDimFrac.cel.iloc[indiceDistro],curvasFlagelaresZenodo[indiceFlagelar][0] )

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=334
tiempo2=211
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name=curvasFlagelaresZenodo[celula1][0]+' min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)
ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula1][1][tiempo1],curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
        yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]], title='Y'),   # Set Y-axis range and title
        zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    ),
    title='Comparación de curvas flagelares para la misma célula con DF mínima y máxima',
)
fig.show()

# %%
kappa1,tau1=curvature_torsion(curvasFlagelaresZenodo[celula1][1][tiempo1].x, curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                                                                                               curvasFlagelaresZenodo[celula1][1][tiempo1].z)

kappa2,tau2=curvature_torsion(curvasFlagelaresZenodo[celula2][1][tiempo2].x, curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                                                                                               curvasFlagelaresZenodo[celula2][1][tiempo2].z)
fig = px.line(y=kappa1, markers=True)
fig.update_layout(title_text='Curvature for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMin): '+str(tiempo1), 
                  title_x=0.5,font=dict(size=15))
fig.show()

fig = px.line(y=kappa2, markers=True)
fig.update_layout(title_text='Curvature for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMax): '+str(tiempo2), 
                  title_x=0.5,font=dict(size=15))
fig.show()
fig = px.line(y=tau1, markers=True)
fig.update_layout(title_text='Torsion for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMin): '+str(tiempo1), 
                  title_x=0.5,font=dict(size=15))
fig.show()

fig = px.line(y=tau2, markers=True)
fig.update_layout(title_text='Torsion for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMax): '+str(tiempo2), 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
curvasFlagelaresZenodo[celula2][1][tiempo1].shape[0],curvasFlagelaresZenodo[celula2][1][tiempo2].shape[0]

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=15
tiempo2=16
tiempo3=17
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name=curvasFlagelaresZenodo[celula1][0]+' min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)

fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo3].x, y=curvasFlagelaresZenodo[celula2][1][tiempo3].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo3].z, mode='lines',marker=dict(color='green', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo3)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo3].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo3].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo3].z.iloc[0]],
                 mode='markers',marker=dict(color='green'))
)

ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula1][1][tiempo1],curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    # scene=dict(
    #     xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
    #     yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]], title='Y'),   # Set Y-axis range and title
    #     zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    # ),
    title='Comparación de curvas flagelares para la misma célula con DF mínima y máxima',
)
fig.show()

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Sperm-13-Cap_170601_Exp10-T0

# %%
indiceDistro=11
fig = px.line(y=dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro], markers=True)
fig.update_layout(title_text='Fractal dimension series for : '+dfCelsZenodoDimFrac.cel.iloc[indiceDistro], 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
print(np.min(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]),np.argmin(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]),
      np.max(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]),np.argmax(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]))

# %%
for i in range(len(curvasFlagelaresZenodo)):
    if( curvasFlagelaresZenodo[i][0] in dfCelsZenodoDimFrac.cel.iloc[indiceDistro] ):
        print(i)
        indiceFlagelar=i

print(dfCelsZenodoDimFrac.cel.iloc[indiceDistro], curvasFlagelaresZenodo[indiceFlagelar][0])

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=68
tiempo2=96
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name=curvasFlagelaresZenodo[celula1][0]+' min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)
ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula1][1][tiempo1],curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
        yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]+20], title='Y'),   # Set Y-axis range and title
        zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    ),
    title='Comparación de curvas flagelares para la misma célula con DF mínima y máxima',
)
fig.show()

# %%
kappa1,tau1=curvature_torsion(curvasFlagelaresZenodo[celula1][1][tiempo1].x, curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                                                                                               curvasFlagelaresZenodo[celula1][1][tiempo1].z)

kappa2,tau2=curvature_torsion(curvasFlagelaresZenodo[celula2][1][tiempo2].x, curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                                                                                               curvasFlagelaresZenodo[celula2][1][tiempo2].z)
fig = px.line(y=kappa1, markers=True)
fig.update_layout(title_text='Curvature for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMin): '+str(tiempo1), 
                  title_x=0.5,font=dict(size=15))
fig.show()

fig = px.line(y=kappa2, markers=True)
fig.update_layout(title_text='Curvature for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMax): '+str(tiempo2), 
                  title_x=0.5,font=dict(size=15))
fig.show()
fig = px.line(y=tau1, markers=True)
fig.update_layout(title_text='Torsion for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMin): '+str(tiempo1), 
                  title_x=0.5,font=dict(size=15))
fig.show()

fig = px.line(y=tau2, markers=True)
fig.update_layout(title_text='Torsion for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMax): '+str(tiempo2), 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=15
tiempo2=16
tiempo3=17
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name=curvasFlagelaresZenodo[celula1][0]+' min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)

fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo3].x, y=curvasFlagelaresZenodo[celula2][1][tiempo3].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo3].z, mode='lines',marker=dict(color='green', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo3)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo3].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo3].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo3].z.iloc[0]],
                 mode='markers',marker=dict(color='green'))
)

# ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    # scene=dict(
    #     xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
    #     yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]], title='Y'),   # Set Y-axis range and title
    #     zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    # ),
    title='Comparación de curvas flagelares para la misma célula con DF mínima y máxima',
)
fig.show()

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Sperm-1-Cap_171108_Exp9

# %%
indiceDistro=116
fig = px.line(y=dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro], markers=True)
fig.update_layout(title_text='Fractal dimension series for : '+dfCelsZenodoDimFrac.cel.iloc[indiceDistro], 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
print(np.min(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]),np.argmin(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]),
      np.max(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]),np.argmax(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]))

# %%
for i in range(len(curvasFlagelaresZenodo)):
    if( curvasFlagelaresZenodo[i][0] in dfCelsZenodoDimFrac.cel.iloc[indiceDistro] ):
        print(i)
        indiceFlagelar=i

print(dfCelsZenodoDimFrac.cel.iloc[indiceDistro], curvasFlagelaresZenodo[indiceFlagelar][0])

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=85
tiempo2=51
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name=curvasFlagelaresZenodo[celula1][0]+' min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)
ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula1][1][tiempo1],curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
        yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]+20], title='Y'),   # Set Y-axis range and title
        zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    ),
    title='Comparación de curvas flagelares para la misma célula con DF mínima y máxima',
)
fig.show()

# %%
kappa1,tau1=curvature_torsion(curvasFlagelaresZenodo[celula1][1][tiempo1].x, curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                                                                                               curvasFlagelaresZenodo[celula1][1][tiempo1].z)

kappa2,tau2=curvature_torsion(curvasFlagelaresZenodo[celula2][1][tiempo2].x, curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                                                                                               curvasFlagelaresZenodo[celula2][1][tiempo2].z)
fig = px.line(y=kappa1, markers=True)
fig.update_layout(title_text='Curvature for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMin): '+str(tiempo1), 
                  title_x=0.5,font=dict(size=15))
fig.show()

fig = px.line(y=kappa2, markers=True)
fig.update_layout(title_text='Curvature for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMax): '+str(tiempo2), 
                  title_x=0.5,font=dict(size=15))
fig.show()
fig = px.line(y=tau1, markers=True)
fig.update_layout(title_text='Torsion for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMin): '+str(tiempo1), 
                  title_x=0.5,font=dict(size=15))
fig.show()

fig = px.line(y=tau2, markers=True)
fig.update_layout(title_text='Torsion for : '+curvasFlagelaresZenodo[celula1][0]+' tiempo (DimFracMax): '+str(tiempo2), 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=80
tiempo2=51
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name=curvasFlagelaresZenodo[celula1][0]+' min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)
ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula1][1][tiempo1],curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
        yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]+20], title='Y'),   # Set Y-axis range and title
        zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    ),
    title='Comparación de curvas flagelares para la misma célula con DF máxima y la submáxima',
)
fig.show()

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=15
tiempo2=16
tiempo3=17
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name=curvasFlagelaresZenodo[celula1][0]+' min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)

fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo3].x, y=curvasFlagelaresZenodo[celula2][1][tiempo3].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo3].z, mode='lines',marker=dict(color='green', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo3)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo3].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo3].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo3].z.iloc[0]],
                 mode='markers',marker=dict(color='green'))
)

# ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    # scene=dict(
    #     xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
    #     yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]], title='Y'),   # Set Y-axis range and title
    #     zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    # ),
    title='Comparación de curvas flagelares para la misma célula con DF mínima y máxima',
)
fig.show()

# %% jupyter={"source_hidden": true}
# celula1=indiceFlagelar
# celula2=indiceFlagelar
# tiempo1=85
# tiempo2=51
# fig=go.Figure()
# fig.update_layout(
#     autosize=False,
#     width=1500,
#     height=1000,
#     scene_aspectmode='cube'
#     # margin=dict(l=10, r=10, t=10, b=10, pad=10)
# )
# fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[50:150], y=curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[50:150], 
#                            z=curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[50:150],marker=dict(color='red', size=5), 
#                            name=curvasFlagelaresZenodo[celula1][0]+' min, Tiempo = '+str(tiempo1)))
# # ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula2][1][tiempo2])
# fig.update_layout(
#     # scene=dict(
#     #     xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
#     #     yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]+20], title='Y'),   # Set Y-axis range and title
#     #     zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
#     # ),
#     title='Comparación de curvas flagelares para la misma célula con DF mínima y máxima',
# )
# fig.show()

# %%
import numpy as np
from scipy.fft import fft


# ----------------------------
# 0. Preprocessing
# ----------------------------
def normalize_curve(r):
    r = np.asarray(r)
    r = r - r.mean(axis=0)
    r = r / (np.std(r) + 1e-8)
    return r


def match_length(r1, r2):
    """Trim to same length (simple version)"""
    T = min(len(r1), len(r2))
    return r1[:T], r2[:T]


# ----------------------------
# 1. Position Cross-Correlation
# ----------------------------
def position_cross_correlation(r1, r2, max_lag):
    r1, r2 = match_length(normalize_curve(r1), normalize_curve(r2))
    
    C = []
    for lag in range(max_lag):
        if lag == 0:
            val = np.mean(np.sum(r1 * r2, axis=1))
        else:
            val = np.mean(np.sum(r1[:-lag] * r2[lag:], axis=1))
        C.append(val)
    
    return np.array(C)


# ----------------------------
# 2. Velocity Cross-Correlation
# ----------------------------
def velocity_cross_correlation(r1, r2, max_lag):
    r1, r2 = match_length(normalize_curve(r1), normalize_curve(r2))
    
    v1 = np.diff(r1, axis=0)
    v2 = np.diff(r2, axis=0)
    
    C = []
    for lag in range(max_lag):
        if lag == 0:
            val = np.mean(np.sum(v1 * v2, axis=1))
        else:
            val = np.mean(np.sum(v1[:-lag] * v2[lag:], axis=1))
        C.append(val)
    
    C = np.array(C)
    return C / (C[0] + 1e-8)


# ----------------------------
# 3. Tangent Cross-Correlation
# ----------------------------
def tangent_cross_correlation(r1, r2, max_lag):
    r1, r2 = match_length(normalize_curve(r1), normalize_curve(r2))
    
    v1 = np.diff(r1, axis=0)
    v2 = np.diff(r2, axis=0)
    
    t1 = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8)
    t2 = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8)
    
    C = []
    for lag in range(max_lag):
        if lag == 0:
            val = np.mean(np.sum(t1 * t2, axis=1))
        else:
            val = np.mean(np.sum(t1[:-lag] * t2[lag:], axis=1))
        C.append(val)
    
    return np.array(C)


# ----------------------------
# 4. Displacement Cross-Correlation
# ----------------------------
def displacement_cross_correlation(r1, r2, max_lag):
    r1, r2 = match_length(normalize_curve(r1), normalize_curve(r2))
    
    D = []
    for lag in range(1, max_lag + 1):
        d1 = r1[lag:] - r1[:-lag]
        d2 = r2[lag:] - r2[:-lag]
        
        val = np.mean(np.sum(d1 * d2, axis=1))
        D.append(val)
    
    D = np.array(D)
    return D / (np.max(np.abs(D)) + 1e-8)


# ----------------------------
# 5. Spectral Cross-Correlation
# ----------------------------
def spectral_cross_correlation(r1, r2, n_freq):
    r1, r2 = match_length(normalize_curve(r1), normalize_curve(r2))
    
    S1 = np.abs(fft(r1, axis=0))**2
    S2 = np.abs(fft(r2, axis=0))**2
    
    S1 = S1.mean(axis=1)[:n_freq]
    S2 = S2.mean(axis=1)[:n_freq]
    
    # cosine similarity in frequency space
    num = np.dot(S1, S2)
    den = np.linalg.norm(S1) * np.linalg.norm(S2) + 1e-8
    
    return num / den


# ----------------------------
# 6. Unified Cross-Descriptor
# ----------------------------
def cross_descriptor(r1, r2, max_lag=50, n_freq=30):
    Cp = position_cross_correlation(r1, r2, max_lag)
    Cv = velocity_cross_correlation(r1, r2, max_lag)
    Ct = tangent_cross_correlation(r1, r2, max_lag)
    D  = displacement_cross_correlation(r1, r2, max_lag)
    S  = np.array([spectral_cross_correlation(r1, r2, n_freq)])
    
    return np.concatenate([Cp, Cv, Ct, D, S])


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    T = 300
    t = np.linspace(0, 10, T)
    
    r1 = np.stack([np.sin(t), np.cos(t), np.sin(2*t)], axis=1)
    r2 = np.stack([np.sin(t+0.5), np.cos(t+0.5), np.sin(2*t+0.5)], axis=1)
    
    descriptor = cross_descriptor(r1, r2)
    
    print("Cross-descriptor shape:", descriptor.shape)

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=85
tiempo2=51
cf1=np.stack([curvasFlagelaresZenodo[celula1][1][tiempo1].x,curvasFlagelaresZenodo[celula1][1][tiempo1].y,curvasFlagelaresZenodo[celula1][1][tiempo1].z], axis=1)
cf2=np.stack([curvasFlagelaresZenodo[celula1][1][tiempo2].x,curvasFlagelaresZenodo[celula1][1][tiempo2].y,curvasFlagelaresZenodo[celula1][1][tiempo2].z],axis=1)
tccCf85_51=tangent_cross_correlation(cf1, cf2, 50)
print(tccCf85_51.mean())

celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=80
tiempo2=51
cf1=np.stack([curvasFlagelaresZenodo[celula1][1][tiempo1].x,curvasFlagelaresZenodo[celula1][1][tiempo1].y,curvasFlagelaresZenodo[celula1][1][tiempo1].z], axis=1)
cf2=np.stack([curvasFlagelaresZenodo[celula1][1][tiempo2].x,curvasFlagelaresZenodo[celula1][1][tiempo2].y,curvasFlagelaresZenodo[celula1][1][tiempo2].z],axis=1)
tccCf80_51=tangent_cross_correlation(cf1, cf2, 50)
print(tccCf80_51.mean())

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Sperm-3-Cap_181026_Exp13 Promedio

# %%
indiceDistro=126
fig = px.line(y=dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro], markers=True)
fig.update_layout(title_text='Fractal dimension series for : '+dfCelsZenodoDimFrac.cel.iloc[indiceDistro], 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
print(np.min(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]),np.argmin(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]),
      np.max(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]),np.argmax(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]))

# %%
for i in range(len(curvasFlagelaresZenodo)):
    if( curvasFlagelaresZenodo[i][0] in dfCelsZenodoDimFrac.cel.iloc[indiceDistro] ):
        print(i)
        indiceFlagelar=i

print(dfCelsZenodoDimFrac.cel.iloc[indiceDistro], curvasFlagelaresZenodo[indiceFlagelar][0])

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=85
tiempo2=51
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[100:150], y=curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[100:150], 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[100:150],marker=dict(color='red', size=5), 
                           name=curvasFlagelaresZenodo[celula1][0]+' min, Tiempo = '+str(tiempo1)))
# fig.add_trace(
#     go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
#                  y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
#                  z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
#                  mode='markers',marker=dict(color='red'))
# )
# fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
#                            z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
#                            name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo2)))
# fig.add_trace(
#     go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
#                  y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
#                  z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
#                  mode='markers',marker=dict(color='blue'))
# )
# ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    # scene=dict(
    #     xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
    #     yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]+20], title='Y'),   # Set Y-axis range and title
    #     zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    # ),
    title='Comparación de curvas flagelares para la misma célula con DF mínima y máxima',
)
fig.show()

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=46
tiempo2=84
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name=curvasFlagelaresZenodo[celula1][0]+' min, Tiempo = '+str(tiempo1), line=dict(
        color='red', # Optional: set line color
        width=10       # Set the line thickness in pixels
    )))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo2), line=dict(
        color='blue', # Optional: set line color
        width=10       # Set the line thickness in pixels
    )))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)
# ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    # scene=dict(
    #     xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
    #     yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]+20], title='Y'),   # Set Y-axis range and title
    #     zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    # ),
    title='Comparación de curvas flagelares para la misma célula con DF mínima y máxima',
)
fig.show()

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=15
tiempo2=16
tiempo3=17
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name=curvasFlagelaresZenodo[celula1][0]+' min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)

fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo3].x, y=curvasFlagelaresZenodo[celula2][1][tiempo3].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo3].z, mode='lines',marker=dict(color='green', size=5), 
                           name=curvasFlagelaresZenodo[celula2][0]+' max, Tiempo = '+str(tiempo3)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo3].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo3].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo3].z.iloc[0]],
                 mode='markers',marker=dict(color='green'))
)

# ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    # scene=dict(
    #     xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
    #     yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]], title='Y'),   # Set Y-axis range and title
    #     zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    # ),
    title='Comparación de curvas flagelares para la misma célula con DF mínima y máxima',
)
fig.show()

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Median

# %%
ncels=dfCelsZenodoDimFrac.shape[0]
FDmedianDist=[]
for i in range(ncels):
    nom=dfCelsZenodoDimFrac.cel.iloc[i]
    fdmedian=np.median(dfCelsZenodoDimFrac.dimFracDistro.iloc[i])
    if( 'No' in dfCelsZenodoDimFrac.cel.iloc[i] ):
        cond='NoCap'
    else:
        cond='Cap'
    FDmedianDist.append([nom,cond,fdmedian])

# %%
dfZenodoFDmedianDist=pd.DataFrame(FDmedianDist, columns=['cel','Condition','FDmedian'])

# %%
fig=px.violin(dfZenodoFDmedianDist, x='FDmedian', color='Condition', box=True, points='all',
             category_orders={"Condition":["NoCap","Cap"]})
fig.update_traces(meanline_visible=True)
fig.update_layout(title_text='Distribution of the cellular median for the flagellar fractal dimension', 
                  title_x=0.5,font=dict(size=20),xaxis_title="Fractal dimension median")
# fig.update_layout(title_text='Distribution of the mean for the flagellar fractal dimension of each cell of the Zenodo dataset', 
#                   title_x=0.5,font=dict(size=20))
fig.show()

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Maximmum flagellar form
#
# The maximmum of the Katz Fractal Dimension corresponds to the flagellar curve with the more "irregular" form, thus this descriptor is briefly explored.

# %%
ncels=dfCelsZenodoDimFrac.shape[0]
FDmaxDist=[]
for i in range(ncels):
    nom=dfCelsZenodoDimFrac.cel.iloc[i]
    fdmax=np.max(dfCelsZenodoDimFrac.dimFracDistro.iloc[i])
    if( 'No' in dfCelsZenodoDimFrac.cel.iloc[i] ):
        cond='NoCap'
    else:
        cond='Cap'
    FDmaxDist.append([nom,cond,fdmax])

# %%
dfZenodoFDmaxDist=pd.DataFrame(FDmaxDist, columns=['cel','Condition','FDmax'])

# %%
fig=px.violin(dfZenodoFDmaxDist, x='FDmax', color='Condition', box=True, points='all',
             category_orders={"Condition":["NoCap","Cap"]})
fig.update_traces(meanline_visible=True)
# fig.update_layout(title_text='Distribution of the max for the flagellar fractal dimension of each cell of the Zenodo dataset', 
fig.update_layout(title_text='Distribution of the cellular max for the flagellar fractal dimension', 
                  title_x=0.5,font=dict(size=20),xaxis_title="Fractal dimension max")
fig.show()

# %%
dfZenodoFDmaxDist[dfZenodoFDmaxDist['FDmax']>1.05]

# %%
dfZenodoFDmaxDist[dfZenodoFDmaxDist['FDmax']>1.1]

# %%
celsDFmaxis=dfZenodoFDmaxDist[dfZenodoFDmaxDist['FDmax']>1.1]

# %%
celsDFmaxis.head()

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Weighted max form

# %%
ncels=dfCelsZenodoDimFrac.shape[0]
FDSidMaxDist=[]
for i in range(ncels):
    nom=dfCelsZenodoDimFrac.cel.iloc[i]
    fouFD = np.fft.fft(dfCelsZenodoDimFrac.dimFracDistro.iloc[i])
    frequency_amplitudes = np.abs(fouFD)
    freqContribution=np.max(frequency_amplitudes[1:])
    fdRpCV=np.max(dfCelsZenodoDimFrac.dimFracDistro.iloc[i])*freqContribution
    if( 'No' in dfCelsZenodoDimFrac.cel.iloc[i] ):
        cond='NoCap'
    else:
        cond='Cap'
    FDSidMaxDist.append([nom,cond,fdRpCV])

# %%
dfZenodoFDSidMaxDist=pd.DataFrame(FDSidMaxDist, columns=['cel','Condition','fdRpCV'])

# %%
fig=px.violin(dfZenodoFDSidMaxDist, x='fdRpCV', color='Condition', box=True, points='all', #log_x=True,
             category_orders={"Condition":["NoCap","Cap"]})
fig.update_traces(meanline_visible=True)
fig.update_layout(title_text='Distribution of the cellular weighted max for the flagellar fractal dimension', 
                  title_x=0.5,font=dict(size=20),xaxis_title="Fractal dimension weighted max")
fig.show()

# %% [markdown]
# # Weighted mean flagellar form

# %%
ncels=dfCelsZenodoDimFrac.shape[0]
FDSidDist=[]
for i in range(ncels):
    nom=dfCelsZenodoDimFrac.cel.iloc[i]
    fouFD = np.fft.fft(dfCelsZenodoDimFrac.dimFracDistro.iloc[i])
    frequency_amplitudes = np.abs(fouFD)
    freqContribution=np.max(frequency_amplitudes[1:])
    fdRpCV=freqContribution*(np.mean(dfCelsZenodoDimFrac.dimFracDistro.iloc[i]))
    if( 'No' in dfCelsZenodoDimFrac.cel.iloc[i] ):
        cond='NoCap'
    else:
        cond='Cap'
    FDSidDist.append([nom,cond,fdRpCV])

# %%
dfZenodoFDSidDist=pd.DataFrame(FDSidDist, columns=['cel','Condition','dfPonderada'])

# %%
fig=px.violin(dfZenodoFDSidDist, x='dfPonderada', color='Condition', box=True, points='all', #log_x=True,
             category_orders={"Condition":["NoCap","Cap"]})
fig.update_traces(meanline_visible=True)
fig.update_layout(title_text='Distribution of the cellular weighted mean for the flagellar fractal dimension', 
                  title_x=0.5,font=dict(size=20),xaxis_title="Fractal dimension weighted mean")
# fig.update_layout(title_text='Distribution of the fdRpCV for the flagellar fractal dimension of each cell of the Zenodo dataset', 
#                   title_x=0.5,font=dict(size=20))
fig.show()

# %%
ncels=dfCelsZenodoDimFrac.shape[0]
FDMediaPonderadaTrasDist=[]
for i in range(ncels):
    nom=dfCelsZenodoDimFrac.cel.iloc[i]
    fouFD = np.fft.fft(dfCelsZenodoDimFrac.dimFracDistro.iloc[i])
    frequency_amplitudes = np.abs(fouFD)
    freqContribution=np.max(frequency_amplitudes[1:])
    fdRpCV=freqContribution*(np.mean(dfCelsZenodoDimFrac.dimFracDistro.iloc[i])-1)
    if( 'No' in dfCelsZenodoDimFrac.cel.iloc[i] ):
        cond='NoCap'
    else:
        cond='Cap'
    FDMediaPonderadaTrasDist.append([nom,cond,fdRpCV])

# %%
dfZenodoFDMediaPonderadaTrasDist=pd.DataFrame(FDMediaPonderadaTrasDist, columns=['cel','Condition','dfPondMenos1'])

# %% jupyter={"source_hidden": true}
# fig=px.violin(dfZenodoFDMediaPonderadaTrasDist, x='dfPondMenos1', color='Condition', box=True, points='all', #log_x=True,
#              category_orders={"Condition":["NoCap","Cap"]})
# fig.update_traces(meanline_visible=True)
# fig.update_layout(title_text='Distribución de la media de dimensión fractal, menos 1, para cada célula ponderada por la magnitud de la frecuencia principal', 
#                   title_x=0.5,font=dict(size=18))
# # fig.update_layout(title_text='Distribution of the fdRpCV for the flagellar fractal dimension of each cell of the Zenodo dataset', 
# #                   title_x=0.5,font=dict(size=20))
# fig.show()

# %%
dfZenodoFDSidDist[dfZenodoFDSidDist['dfPonderada']>1]

# %% [markdown]
# ## Sperm-12-NoCap_210702_Exp27

# %%
mindice=13
fig=px.violin(x=dfCelsZenodoDimFrac.dimFracDistro.iloc[mindice], box=True, points='all') #log_x=True,
             # category_orders={"Condition":["NoCap","Cap"]})
fig.update_traces(meanline_visible=True)
# fig.update_layout(title_text='Distribution of the fdRpCV for the flagellar fractal dimension of each cell of the Zenodo dataset', 
#                   title_x=0.5,font=dict(size=20))
fig.update_layout(title_text='Fractal dimension distribution for : '+dfCelsZenodoDimFrac.cel.iloc[mindice], 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
indiceDistro=13
print(np.max(dfCelsZenodoDimFrac.dimFracDistro.iloc[mindice])-np.min(dfCelsZenodoDimFrac.dimFracDistro.iloc[mindice]))
fig = px.line(y=dfCelsZenodoDimFrac.dimFracDistro.iloc[mindice], markers=True)
fig.update_layout(title_text='Serie de tiempo de la dimensión fractal para una célula inducida a capacitación', 
                  title_x=0.5,font=dict(size=15))
fig.update_layout(
    xaxis_title="número de curva",
    yaxis_title="DimFrac",
    # legend_title="Legend Title"
)
# fig.update_layout(title_text='Fractal dimension series for : '+dfCelsZenodoDimFrac.cel.iloc[mindice], 
#                   title_x=0.5,font=dict(size=15))
fig.show()

# %%
minidx,maxidx,muidx=descri(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro])

# %%
for i in range(len(curvasFlagelaresZenodo)):
    if( curvasFlagelaresZenodo[i][0] == dfCelsZenodoDimFrac.cel.iloc[indiceDistro] ):
        print(i)
        indiceFlagelar=i
print(dfCelsZenodoDimFrac.cel.iloc[indiceDistro],curvasFlagelaresZenodo[indiceFlagelar][0])

# %%

# %%
dfCelsZenodoDimFrac.cel.iloc[indiceDistro], curvasFlagelaresZenodo[indiceFlagelar][0]

# %%
feature=dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]
cell_data=flagellar_data[indiceFlagelar]
dt = 1/90

time = np.arange(len(feature)) * dt

fig = animate_flagella_with_feature_vertical(
    cell_data,
    feature,
    time=time,
    feature_name="Fractal dimension",
    flagellarTitle=curvasFlagelaresZenodo[indiceFlagelar][0],
    frame_duration=200
)

fig.show()

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Sperm-12-NoCap_210730_Exp18

# %%
mindice=39
fig=px.violin(x=dfCelsZenodoDimFrac.dimFracDistro.iloc[mindice], box=True, points='all') #log_x=True,
             # category_orders={"Condition":["NoCap","Cap"]})
fig.update_traces(meanline_visible=True)
# fig.update_layout(title_text='Distribution of the fdRpCV for the flagellar fractal dimension of each cell of the Zenodo dataset', 
#                   title_x=0.5,font=dict(size=20))
fig.update_layout(title_text='Fractal dimension distribution for : '+dfCelsZenodoDimFrac.cel.iloc[mindice], 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
indiceDistro=39
print(np.max(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro])-np.min(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]))
fig = px.line(y=dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro], markers=True)
fig.update_layout(title_text='Fractal dimension series for : '+dfCelsZenodoDimFrac.cel.iloc[indiceDistro], 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
minidx,maxidx,muidx=descri(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro])

# %%
for i in range(len(curvasFlagelaresZenodo)):
    if( curvasFlagelaresZenodo[i][0] == dfCelsZenodoDimFrac.cel.iloc[indiceDistro] ):
        print(i)
        indiceFlagelar=i
print(dfCelsZenodoDimFrac.cel.iloc[indiceDistro],curvasFlagelaresZenodo[indiceFlagelar][0])

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=minidx
tiempo2=muidx
tiempo3=maxidx
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name='Min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name='Mu, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)

fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo3].x, y=curvasFlagelaresZenodo[celula2][1][tiempo3].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo3].z, mode='lines',marker=dict(color='green', size=5), 
                           name='Max, Tiempo = '+str(tiempo3)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo3].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo3].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo3].z.iloc[0]],
                 mode='markers',marker=dict(color='green'))
)

ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula1][1][tiempo1],curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
        yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]], title='Y'),   # Set Y-axis range and title
        zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    ),
    title='Curvas para '+curvasFlagelaresZenodo[celula2][0]+' con DF mínima, media y máxima',
)
fig.show()

# %%
dfCelsZenodoDimFrac.cel.iloc[indiceDistro], curvasFlagelaresZenodo[indiceFlagelar][0]

# %%
feature=dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]
cell_data=flagellar_data[indiceFlagelar]
dt = 1/90

time = np.arange(len(feature)) * dt

fig = animate_flagella_with_feature_vertical(
    cell_data,
    feature,
    time=time,
    feature_name="Fractal dimension",
    flagellarTitle=curvasFlagelaresZenodo[indiceFlagelar][0]
)

fig.show()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Sperm-1-NoCap_210702_Exp4_cell-1

# %%
mindice=27
fig=px.violin(x=dfCelsZenodoDimFrac.dimFracDistro.iloc[mindice], box=True, points='all') #log_x=True,
             # category_orders={"Condition":["NoCap","Cap"]})
fig.update_traces(meanline_visible=True)
# fig.update_layout(title_text='Distribution of the fdRpCV for the flagellar fractal dimension of each cell of the Zenodo dataset', 
#                   title_x=0.5,font=dict(size=20))
fig.update_layout(title_text='Fractal dimension distribution for : '+dfCelsZenodoDimFrac.cel.iloc[mindice], 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
indiceDistro=27
print(np.max(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro])-np.min(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]))
fig = px.line(y=dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro], markers=True)
fig.update_layout(title_text='Fractal dimension series for : '+dfCelsZenodoDimFrac.cel.iloc[indiceDistro], 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
minidx,maxidx,muidx=descri(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro])
for i in range(len(curvasFlagelaresZenodo)):
    if( curvasFlagelaresZenodo[i][0] == dfCelsZenodoDimFrac.cel.iloc[indiceDistro] ):
        print(i)
        indiceFlagelar=i
print(dfCelsZenodoDimFrac.cel.iloc[indiceDistro],curvasFlagelaresZenodo[indiceFlagelar][0])

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=minidx
tiempo2=muidx
tiempo3=maxidx
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name='Min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name='Mu, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)

fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo3].x, y=curvasFlagelaresZenodo[celula2][1][tiempo3].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo3].z, mode='lines',marker=dict(color='green', size=5), 
                           name='Max, Tiempo = '+str(tiempo3)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo3].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo3].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo3].z.iloc[0]],
                 mode='markers',marker=dict(color='green'))
)

ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula1][1][tiempo1],curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
        yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]], title='Y'),   # Set Y-axis range and title
        zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    ),
    title='Curvas para '+curvasFlagelaresZenodo[celula2][0]+' con DF mínima, media y máxima',
)
fig.show()

# %%
# flagellar_data = [item[1] for item in curvasFlagelaresZenodo]   # list of list of DataFrames

# %%
# fig = plot_single_cell_flagella_with_head(
#     flagellar_data,
#     cell_index=83,
#     head_marker_size=6
# )

# fig.show()

# %%
# fig = animate_time_series_point(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro])
# fig.show()

# %%
dfCelsZenodoDimFrac.cel.iloc[indiceDistro]

# %%
feature=dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]
cell_data=flagellar_data[indiceFlagelar]
dt = 1/90

time = np.arange(len(feature)) * dt

fig = animate_flagella_with_feature_vertical(
    cell_data,
    feature,
    time=time,
    feature_name="Fractal dimension",
    flagellarTitle=curvasFlagelaresZenodo[indiceFlagelar][0]
)

fig.show()

# %%

# %% [markdown]
# ## Sperm-1-Cap_171108_Exp9

# %%
mindice=116
fig=px.violin(x=dfCelsZenodoDimFrac.dimFracDistro.iloc[mindice], box=True, points='all') #log_x=True,
             # category_orders={"Condition":["NoCap","Cap"]})
fig.update_traces(meanline_visible=True)
# fig.update_layout(title_text='Distribution of the fdRpCV for the flagellar fractal dimension of each cell of the Zenodo dataset', 
#                   title_x=0.5,font=dict(size=20))
fig.update_layout(title_text='Fractal dimension distribution for : '+dfCelsZenodoDimFrac.cel.iloc[mindice], 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
indiceDistro=116
print(np.max(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro])-np.min(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]))
fig = px.line(y=dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro], markers=True)
fig.update_layout(title_text='Fractal dimension series for : '+dfCelsZenodoDimFrac.cel.iloc[indiceDistro], 
                  title_x=0.5,font=dict(size=15))
fig.show()

# %%
minidx,maxidx,muidx=descri(dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro])
for i in range(len(curvasFlagelaresZenodo)):
    if( curvasFlagelaresZenodo[i][0] == dfCelsZenodoDimFrac.cel.iloc[indiceDistro] ):
        print(i)
        indiceFlagelar=i
print(dfCelsZenodoDimFrac.cel.iloc[indiceDistro],curvasFlagelaresZenodo[indiceFlagelar][0])

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=minidx
tiempo2=muidx
tiempo3=maxidx
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name='Min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name='Mu, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)

fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo3].x, y=curvasFlagelaresZenodo[celula2][1][tiempo3].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo3].z, mode='lines',marker=dict(color='green', size=5), 
                           name='Max, Tiempo = '+str(tiempo3)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo3].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo3].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo3].z.iloc[0]],
                 mode='markers',marker=dict(color='green'))
)

ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula1][1][tiempo1],curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
        yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]], title='Y'),   # Set Y-axis range and title
        zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    ),
    title='Curvas para '+curvasFlagelaresZenodo[celula2][0]+' con DF mínima, media y máxima',
)
fig.show()

# %%
feature=dfCelsZenodoDimFrac.dimFracDistro.iloc[indiceDistro]
cell_data=flagellar_data[indiceFlagelar]
dt = 1/90

time = np.arange(len(feature)) * dt

fig = animate_flagella_with_feature_vertical(
    cell_data,
    feature,
    time=time,
    feature_name="Fractal dimension of "+dfCelsZenodoDimFrac.cel.iloc[indiceDistro],
    flagellarTitle=curvasFlagelaresZenodo[indiceFlagelar][0],
    frame_duration=150
)

fig.show()

# %%
celula1=indiceFlagelar
celula2=indiceFlagelar
tiempo1=4
tiempo2=13
tiempo3=42
fig=go.Figure()
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000,
    scene_aspectmode='cube'
    # margin=dict(l=10, r=10, t=10, b=10, pad=10)
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula1][1][tiempo1].x, y=curvasFlagelaresZenodo[celula1][1][tiempo1].y, 
                           z=curvasFlagelaresZenodo[celula1][1][tiempo1].z, mode='lines',marker=dict(color='red', size=5), 
                           name='Min, Tiempo = '+str(tiempo1)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula1][1][tiempo1].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula1][1][tiempo1].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula1][1][tiempo1].z.iloc[0]],
                 mode='markers',marker=dict(color='red'))
)
fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo2].x, y=curvasFlagelaresZenodo[celula2][1][tiempo2].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo2].z, mode='lines',marker=dict(color='blue', size=5), 
                           name='Mu, Tiempo = '+str(tiempo2)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo2].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo2].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo2].z.iloc[0]],
                 mode='markers',marker=dict(color='blue'))
)

fig.add_trace(go.Scatter3d(x=curvasFlagelaresZenodo[celula2][1][tiempo3].x, y=curvasFlagelaresZenodo[celula2][1][tiempo3].y, 
                           z=curvasFlagelaresZenodo[celula2][1][tiempo3].z, mode='lines',marker=dict(color='green', size=5), 
                           name='Max, Tiempo = '+str(tiempo3)))
fig.add_trace(
    go.Scatter3d(x=[curvasFlagelaresZenodo[celula2][1][tiempo3].x.iloc[0]],
                 y=[curvasFlagelaresZenodo[celula2][1][tiempo3].y.iloc[0]],
                 z=[curvasFlagelaresZenodo[celula2][1][tiempo3].z.iloc[0]],
                 mode='markers',marker=dict(color='green'))
)

ejeshomogeneos=homogenizarRangos(curvasFlagelaresZenodo[celula1][1][tiempo1],curvasFlagelaresZenodo[celula2][1][tiempo2])
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[ejeshomogeneos[0], ejeshomogeneos[1]], title='X'),  # Set X-axis range and title
        yaxis=dict(range=[ejeshomogeneos[2], ejeshomogeneos[3]], title='Y'),   # Set Y-axis range and title
        zaxis=dict(range=[ejeshomogeneos[4], ejeshomogeneos[5]], title='Z'),   # Set Z-axis range and title
    ),
    title='Curvas para '+curvasFlagelaresZenodo[celula2][0]+' con DF mínima, media y máxima',
)
fig.show()

# %%

# %%

# %%

# %% [markdown]
# ###### 
