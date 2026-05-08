# -*- coding: utf-8 -*-
import os
import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# ==============================
# CONFIGURACION
# ==============================
_parser = argparse.ArgumentParser()
_parser.add_argument("--base_folder", required=True,
                     help="Path to the system analysis folder")
_parser.add_argument("--magnet_info", required=True,
                     help="System name shown in plot titles")
_parser.add_argument("--include_lids", action="store_true",
                     help="Include per-lid analysis and plots")
_args = _parser.parse_args()

base_folder = _args.base_folder
Magnet_info = _args.magnet_info
include_lids = _args.include_lids

# ==============================
# DETECTAR STEP MÁS ALTO
# ==============================

step_folders = [
    os.path.join(base_folder, f)
    for f in os.listdir(base_folder)
    if os.path.isdir(os.path.join(base_folder, f)) and f.startswith("Step")
]

def extract_step_number(folder_name):
    match = re.search(r"Step(\d+)", folder_name)
    return int(match.group(1)) if match else -1

step_folders_sorted = sorted(
    step_folders,
    key=lambda f: extract_step_number(os.path.basename(f))
)

if not step_folders_sorted:
    raise RuntimeError("No se encontraron carpetas Step")

reference_folder = step_folders_sorted[-1]
print(f"\nReferencia de escala global: {reference_folder}")

ref_data = np.loadtxt(os.path.join(reference_folder,'Fuerzas.txt'), skiprows=1)

Fx_ref, Fy_ref, Fz_ref = ref_data[:,1], ref_data[:,2], ref_data[:,3]
Tx_ref, Ty_ref, Tz_ref = ref_data[:,4], ref_data[:,5], ref_data[:,6]
x_ref,  y_ref,  z_ref  = ref_data[:,7], ref_data[:,8], ref_data[:,9]

xlim_global = (x_ref.min(), x_ref.max())
ylim_global = (y_ref.min(), y_ref.max())
zlim_global = (z_ref.min(), z_ref.max())

# Escala global fija por componente (signed) en todos los steps
global_component_absmax = {
    "Fx": 0.0, "Fy": 0.0, "Fz": 0.0,
    "Tx": 0.0, "Ty": 0.0, "Tz": 0.0
}

for step_folder in step_folders_sorted:
    input_path = os.path.join(step_folder, 'Fuerzas.txt')
    if not os.path.exists(input_path):
        continue

    data = np.loadtxt(input_path, skiprows=1)

    global_component_absmax["Fx"] = max(global_component_absmax["Fx"], np.max(np.abs(data[:, 1])))
    global_component_absmax["Fy"] = max(global_component_absmax["Fy"], np.max(np.abs(data[:, 2])))
    global_component_absmax["Fz"] = max(global_component_absmax["Fz"], np.max(np.abs(data[:, 3])))
    global_component_absmax["Tx"] = max(global_component_absmax["Tx"], np.max(np.abs(data[:, 4])))
    global_component_absmax["Ty"] = max(global_component_absmax["Ty"], np.max(np.abs(data[:, 5])))
    global_component_absmax["Tz"] = max(global_component_absmax["Tz"], np.max(np.abs(data[:, 6])))

# ============================================================
# FUNCIONES DE PLOT
# ============================================================

    
def plot_component_separate(x,y,z,ring,F,label,title,savepath,component_name=None,is_torque=False):

    from mpl_toolkits.mplot3d import Axes3D

    is_norm = "norm" in label.lower() or "|" in label

    if is_norm:
        vmin = np.min(F)
        vmax = np.max(F)
        cmap = 'hot_r'
    else:
        if component_name in global_component_absmax:
            vmax = global_component_absmax[component_name]
        else:
            vmax = np.max(np.abs(F))
        if vmax == 0:
            vmax = 1.0
        vmin = -vmax
        cmap = 'seismic'

    imax = np.argmax(F)
    imin = np.argmin(F)

    fig = plt.figure(figsize=(20,4.8))

    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1.35,1.2,1.2,1,0.05],
        wspace=0.55
    )

    # ===== 3D =====
    ax3d = fig.add_subplot(gs[0,0],projection='3d')
    sc = ax3d.scatter(x,y,z,c=F,cmap=cmap,vmin=vmin,vmax=vmax,s=40)

    ax3d.scatter(x[imax],y[imax],z[imax],s=180,marker='^',
                 facecolor='maroon',edgecolor='black',linewidth=1.5,zorder=10)

    ax3d.scatter(x[imin],y[imin],z[imin],s=180,marker='v',
                 facecolor='navy',edgecolor='black',linewidth=1.5,zorder=10)

    ax3d.set_box_aspect([1,1,1])
    ax3d.set_xlim(xlim_global)
    ax3d.set_ylim(ylim_global)
    ax3d.set_zlim(zlim_global)
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")

    leg = ax3d.legend(handles=[
        Line2D([0],[0],marker='^',color='w',
               markerfacecolor='maroon',markeredgecolor='black',
               markersize=9,linestyle='None',
               label=f"MAX: {F[imax]:.2f} (Ring {ring[imax]})"),
        Line2D([0],[0],marker='v',color='w',
               markerfacecolor='navy',markeredgecolor='black',
               markersize=9,linestyle='None',
               label=f"MIN: {F[imin]:.2f} (Ring {ring[imin]})")
    ],loc='upper right',frameon=True)

    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_edgecolor('black')

    # ===== XY =====
    ax = fig.add_subplot(gs[0,1])
    ax.scatter(x,y,c=F,cmap=cmap,vmin=vmin,vmax=vmax,s=40)
    ax.scatter(x[imax],y[imax],s=130,marker='^',
               facecolor='maroon',edgecolor='black')
    ax.scatter(x[imin],y[imin],s=130,marker='v',
               facecolor='navy',edgecolor='black')
    ax.set_aspect('equal')
    ax.set_xlim(xlim_global)
    ax.set_ylim(ylim_global)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # ===== XZ =====
    ax = fig.add_subplot(gs[0,2])
    ax.scatter(x,z,c=F,cmap=cmap,vmin=vmin,vmax=vmax,s=40)
    ax.scatter(x[imax],z[imax],s=130,marker='^',
               facecolor='maroon',edgecolor='black')
    ax.scatter(x[imin],z[imin],s=130,marker='v',
               facecolor='navy',edgecolor='black')
    ax.set_aspect('equal')
    ax.set_xlim(xlim_global)
    ax.set_ylim(zlim_global)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")

    # ===== YZ =====
    ax = fig.add_subplot(gs[0,3])
    ax.scatter(y,z,c=F,cmap=cmap,vmin=vmin,vmax=vmax,s=40)
    ax.scatter(y[imax],z[imax],s=130,marker='^',
               facecolor='maroon',edgecolor='black')
    ax.scatter(y[imin],z[imin],s=130,marker='v',
               facecolor='navy',edgecolor='black')
    ax.set_aspect('equal')
    ax.set_xlim(ylim_global)
    ax.set_ylim(zlim_global)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")

    fig.colorbar(sc,cax=fig.add_subplot(gs[0,4])).set_label(label)

    fig.suptitle(
        title+"\n"+f"$\\it{{Magnet:\\ {Magnet_info}}}$",
        fontsize=15,y=0.975
    )
    
    plt.tight_layout(rect=[0, 0, 0.97, 0.94])
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig) 
# ============================================================
# BUCLE PRINCIPAL
# ============================================================

for step_folder in step_folders_sorted:

    print(f"\nProcesando: {step_folder}")

    input_path = os.path.join(step_folder, 'Fuerzas.txt')
    if not os.path.exists(input_path):
        continue

    data = np.loadtxt(input_path, skiprows=1)

    base_name = os.path.basename(step_folder)
    base_output = os.path.join(step_folder, base_name)

    Fx, Fy, Fz = data[:,1], data[:,2], data[:,3]
    normF = np.sqrt(Fx**2 + Fy**2 + Fz**2)

    Tx, Ty, Tz = data[:,4], data[:,5], data[:,6]
    normT = np.sqrt(Tx**2 + Ty**2 + Tz**2)

    x, y, z = data[:,7], data[:,8], data[:,9]

    tol = 1e-6
    x_round = np.round(x / tol) * tol
    unique_x = np.unique(x_round)
    ring = np.array([np.where(unique_x == xi)[0][0] + 1 for xi in x_round])

    # ==============================
    # TABLA SUMATORIO POR RING
    # ==============================

    output_ring_path = base_output + "_TableForceTorqueSum_perRing.txt"

    with open(output_ring_path, "w", encoding="utf-8") as f:

        f.write("Ring\tFx_sum\tFy_sum\tFz_sum\tnormF\tTx_sum\tTy_sum\tTz_sum\tnormT\n")

        for r in np.unique(ring):

            idx = ring == r

            Fx_r = Fx[idx].sum()
            Fy_r = Fy[idx].sum()
            Fz_r = Fz[idx].sum()

            # Cálculo de torques usando posiciones y fuerzas por cubo
            Tx_r = (Fy[idx] * z[idx] + Fz[idx] * y[idx]).sum()
            Ty_r = (Fx[idx] * z[idx]).sum()
            Tz_r = (Fx[idx] * y[idx]).sum()

            normF_r = np.sqrt(Fx_r**2 + Fy_r**2 + Fz_r**2)
            normT_r = np.sqrt(Tx_r**2 + Ty_r**2 + Tz_r**2)

            row = [
                r,
                Fx_r, Fy_r, Fz_r, normF_r,
                Tx_r, Ty_r, Tz_r, normT_r
            ]

            row_str = "\t".join(
                f"{val:g}".replace(".", ",")
                if isinstance(val, float)
                else str(val)
                for val in row
            )

            f.write(row_str + "\n")

    print(f"Tabla guardada: {output_ring_path}")
    
    # ==============================
    # PLOTS SEPARADOS POR COMPONENTE
    # ==============================
    
    plot_component_separate(x,y,z,ring,Fx,"Fx [N]","Fx per magnet",
                            base_output + "_Fx.png",component_name="Fx")
    
    plot_component_separate(x,y,z,ring,Fy,"Fy [N]","Fy per magnet",
                            base_output + "_Fy.png",component_name="Fy")
    
    plot_component_separate(x,y,z,ring,Fz,"Fz [N]","Fz per magnet",
                            base_output + "_Fz.png",component_name="Fz")
    
    plot_component_separate(x,y,z,ring,normF,"|F| [N]","NormF per magnet",
                            base_output + "_NormF.png",component_name="normF")
    
    plot_component_separate(x,y,z,ring,Tx,"Tx [N·m]","Tx per magnet",
                            base_output + "_Tx.png",component_name="Tx",is_torque=True)
    
    plot_component_separate(x,y,z,ring,Ty,"Ty [N·m]","Ty per magnet",
                            base_output + "_Ty.png",component_name="Ty",is_torque=True)
    
    plot_component_separate(x,y,z,ring,Tz,"Tz [N·m]","Tz per magnet",
                            base_output + "_Tz.png",component_name="Tz",is_torque=True)
    
    plot_component_separate(x,y,z,ring,normT,"|T| [N·m]","NormT per magnet",
                            base_output + "_NormT.png",component_name="normT",is_torque=True)

# ============================================================
# WORST CASE GLOBAL - MIN/MAX PER RING Y PER CUBE
# ============================================================

print("\nCalculando Min/Max globales...")

components = ["Fx","Fy","Fz","normF","Tx","Ty","Tz","normT"]
worst_results = []

for component in components:

    # PerRing: track min/max values and their locations
    ring_min_value = np.inf
    ring_min_step = None
    ring_min_ring = None
    
    ring_max_value = -np.inf
    ring_max_step = None
    ring_max_ring = None

    # PerCube: track min/max values and their locations
    cube_min_value = np.inf
    cube_min_step = None
    cube_min_ring = None
    
    cube_max_value = -np.inf
    cube_max_step = None
    cube_max_ring = None

    for step_folder in step_folders_sorted:

        input_path = os.path.join(step_folder, 'Fuerzas.txt')
        base_name = os.path.basename(step_folder)

        if not os.path.exists(input_path):
            continue

        data = np.loadtxt(input_path, skiprows=1)

        Fx, Fy, Fz = data[:,1], data[:,2], data[:,3]
        Tx, Ty, Tz = data[:,4], data[:,5], data[:,6]
        x = data[:,7]

        if component == "Fx": values = Fx
        elif component == "Fy": values = Fy
        elif component == "Fz": values = Fz
        elif component == "Tx": values = Tx
        elif component == "Ty": values = Ty
        elif component == "Tz": values = Tz
        elif component == "normF":
            values = np.sqrt(Fx**2 + Fy**2 + Fz**2)
        elif component == "normT":
            values = np.sqrt(Tx**2 + Ty**2 + Tz**2)

        # =========================
        # PER CUBE (individual magnet values)
        # =========================
        abs_values = np.abs(values)
        
        cube_min = abs_values.min()
        cube_max = abs_values.max()
        
        if cube_min < cube_min_value:
            cube_min_value = cube_min
            cube_min_step = base_name
            cube_min_ring = np.argmin(abs_values) + 1  # magnet index
        
        if cube_max > cube_max_value:
            cube_max_value = cube_max
            cube_max_step = base_name
            cube_max_ring = np.argmax(abs_values) + 1  # magnet index

        # =========================
        # SUM PER RING
        # =========================
        tol = 1e-6
        x_round = np.round(x / tol) * tol
        unique_x = np.unique(x_round)
        ring = np.array([np.where(unique_x == xi)[0][0] + 1 for xi in x_round])

        for r in np.unique(ring):
        
            idx = ring == r
        
            Fx_r = Fx[idx].sum()
            Fy_r = Fy[idx].sum()
            Fz_r = Fz[idx].sum()
        
            Tx_r = Tx[idx].sum()
            Ty_r = Ty[idx].sum()
            Tz_r = Tz[idx].sum()
        
            if component == "Fx":
                sum_val = abs(Fx_r)
        
            elif component == "Fy":
                sum_val = abs(Fy_r)
        
            elif component == "Fz":
                sum_val = abs(Fz_r)
        
            elif component == "Tx":
                sum_val = abs(Tx_r)
        
            elif component == "Ty":
                sum_val = abs(Ty_r)
        
            elif component == "Tz":
                sum_val = abs(Tz_r)
        
            elif component == "normF":
                sum_val = np.sqrt(Fx_r**2 + Fy_r**2 + Fz_r**2)
        
            elif component == "normT":
                sum_val = np.sqrt(Tx_r**2 + Ty_r**2 + Tz_r**2)
        
            # Track min/max
            if sum_val < ring_min_value:
                ring_min_value = sum_val
                ring_min_step = base_name
                ring_min_ring = r
            
            if sum_val > ring_max_value:
                ring_max_value = sum_val
                ring_max_step = base_name
                ring_max_ring = r

    worst_results.append([
        component,
        ring_min_step,
        ring_min_ring if ring_min_ring is not None else "",
        f"{ring_min_value:g}",
        ring_max_step,
        ring_max_ring if ring_max_ring is not None else "",
        f"{ring_max_value:g}",
        cube_min_step,
        cube_min_ring if cube_min_ring is not None else "",
        f"{cube_min_value:g}",
        cube_max_step,
        cube_max_ring if cube_max_ring is not None else "",
        f"{cube_max_value:g}"
    ])

# =============================
# GUARDAR TABLA GLOBAL
# =============================

global_worst_path = os.path.join(base_folder, "Worst_casesglobal.txt")

with open(global_worst_path, "w", encoding="utf-8") as f:

    f.write(
        "Component\t"
        "Step_PerRing\tRing\tMin_PerRing\tMax_PerRing\t"
        "Step_PerCube\tRing\tMin_PerCube\tMax_PerCube\n"
    )

    for row in worst_results:
        # Extract values from row
        component = row[0]
        ring_min_step = row[1]
        ring_min_ring = row[2]
        ring_min_value = row[3]
        ring_max_step = row[4]
        ring_max_ring = row[5]
        ring_max_value = row[6]
        cube_min_step = row[7]
        cube_min_ring = row[8]
        cube_min_value = row[9]
        cube_max_step = row[10]
        cube_max_ring = row[11]
        cube_max_value = row[12]
        
        # Create output row with desired column structure
        # For PerRing: use max step (you can modify to use min if preferred)
        # For PerCube: use max step (you can modify to use min if preferred)
        output_row = [
            component,
            ring_max_step,
            ring_max_ring,
            ring_min_value,
            ring_max_value,
            cube_max_step,
            cube_max_ring,
            cube_min_value,
            cube_max_value
        ]
        
        f.write("\t".join(str(x) for x in output_row) + "\n")

print(f"Tabla global guardada en: {global_worst_path}")
print("\nAnalisis completado correctamente.")

if not include_lids:
    print("\nAnalisis por Lids omitido (--include_lids no activado).")
    sys.exit(0)


# ============================================================
#%% ===================== ANALISIS POR LIDS =====================
# ============================================================


print("\nCalculando analisis por Lids...")

# =========================
# LIMITES GEOMETRICOS
# =========================

limit_z = 0.164
limit_y = 0.114


def classify_lids(x, y, z):

    lid_labels = []

    for xi, yi, zi in zip(x, y, z):

        if zi > limit_z and abs(yi) < limit_y:
            lid_labels.append("Lid+Z")

        elif zi < -limit_z and abs(yi) < limit_y:
            lid_labels.append("Lid-Z")

        elif abs(zi) < limit_z and yi > limit_y:
            lid_labels.append("Lid+Y")

        elif abs(zi) < limit_z and yi < -limit_y:
            lid_labels.append("Lid-Y")

        else:
            lid_labels.append(None)

    return np.array(lid_labels)


# Escala global fija por componente (signed) para plots perLid en todos los steps
global_lids_component_absmax = {
    "Fx": 0.0, "Fy": 0.0, "Fz": 0.0,
    "Tx": 0.0, "Ty": 0.0, "Tz": 0.0
}

for step_folder in step_folders_sorted:
    input_path = os.path.join(step_folder, 'Fuerzas.txt')
    if not os.path.exists(input_path):
        continue

    data = np.loadtxt(input_path, skiprows=1)

    Fx, Fy, Fz = data[:,1], data[:,2], data[:,3]
    Tx, Ty, Tz = data[:,4], data[:,5], data[:,6]
    x, y, z = data[:,7], data[:,8], data[:,9]

    tol = 1e-6
    x_round = np.round(x / tol) * tol
    unique_x = np.unique(x_round)
    ring = np.array([np.where(unique_x == xi)[0][0] + 1 for xi in x_round])
    lids = classify_lids(x, y, z)

    for r in np.unique(ring):
        for lid_name in ["Lid+Z","Lid-Z","Lid+Y","Lid-Y"]:
            idx = (ring == r) & (lids == lid_name)
            if not np.any(idx):
                continue

            global_lids_component_absmax["Fx"] = max(global_lids_component_absmax["Fx"], abs(Fx[idx].sum()))
            global_lids_component_absmax["Fy"] = max(global_lids_component_absmax["Fy"], abs(Fy[idx].sum()))
            global_lids_component_absmax["Fz"] = max(global_lids_component_absmax["Fz"], abs(Fz[idx].sum()))
            global_lids_component_absmax["Tx"] = max(global_lids_component_absmax["Tx"], abs(Tx[idx].sum()))
            global_lids_component_absmax["Ty"] = max(global_lids_component_absmax["Ty"], abs(Ty[idx].sum()))
            global_lids_component_absmax["Tz"] = max(global_lids_component_absmax["Tz"], abs(Tz[idx].sum()))


def plot_lids_3d(x_positions, F_values, labels, title, savepath, component_name=None, is_torque=False):

    from mpl_toolkits.mplot3d import Axes3D

    is_norm = "norm" in title.lower() or "|" in title

    if is_norm:
        vmin = np.min(F_values)
        vmax = np.max(F_values)
        cmap = 'hot_r'
    else:
        if component_name in global_lids_component_absmax:
            vmax = global_lids_component_absmax[component_name]
        else:
            vmax = np.max(np.abs(F_values))
        if vmax == 0:
            vmax = 1.0
        vmin = -vmax
        cmap = 'seismic'

    imax = np.argmax(F_values)
    imin = np.argmin(F_values)

    fig = plt.figure(figsize=(20,4.8))

    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1.35,1.2,1.2,1,0.05],
        wspace=0.55
    )
    # ===== 3D =====
    ax3d = fig.add_subplot(gs[0,0], projection='3d')
    sc = ax3d.scatter(
        x_positions[:,0],
        x_positions[:,1],
        x_positions[:,2],
        c=F_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=100
    )

    ax3d.scatter(*x_positions[imax], s=100, marker='^',
                 facecolor='maroon', edgecolor='black', linewidth=1.5)

    ax3d.scatter(*x_positions[imin], s=100, marker='v',
                 facecolor='navy', edgecolor='black', linewidth=1.5)

    ax3d.set_box_aspect([1,1,1])
    ax3d.set_xlim(xlim_global)
    ax3d.set_ylim(ylim_global)
    ax3d.set_zlim(zlim_global)
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")

    leg = ax3d.legend(handles=[
        Line2D([0],[0],marker='^',color='w',
               markerfacecolor='maroon',
               markeredgecolor='black',
               markersize=9,linestyle='None',
               label=f"MAX: {F_values[imax]:.2f} ({labels[imax]})"),
        Line2D([0],[0],marker='v',color='w',
               markerfacecolor='navy',
               markeredgecolor='black',
               markersize=9,linestyle='None',
               label=f"MIN: {F_values[imin]:.2f} ({labels[imin]})")
    ],loc='upper right',frameon=True)

    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_edgecolor('black')

    # ===== XY =====
    ax = fig.add_subplot(gs[0,1])
    ax.scatter(x_positions[:,0], x_positions[:,1],
               c=F_values, cmap=cmap, vmin=vmin, vmax=vmax, s=100)

    ax.scatter(x_positions[imax,0], x_positions[imax,1], s=100, marker='^',
               facecolor='maroon', edgecolor='black')

    ax.scatter(x_positions[imin,0], x_positions[imin,1], s=100, marker='v',
               facecolor='navy', edgecolor='black')

    ax.set_aspect('equal')
    ax.set_xlim(xlim_global)
    ax.set_ylim(ylim_global)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # ===== XZ =====
    ax = fig.add_subplot(gs[0,2])
    ax.scatter(x_positions[:,0], x_positions[:,2],
               c=F_values, cmap=cmap, vmin=vmin, vmax=vmax, s=100)

    ax.scatter(x_positions[imax,0], x_positions[imax,2], s=100, marker='^',
               facecolor='maroon', edgecolor='black')

    ax.scatter(x_positions[imin,0], x_positions[imin,2], s=100, marker='v',
               facecolor='navy', edgecolor='black')

    ax.set_aspect('equal')
    ax.set_xlim(xlim_global)
    ax.set_ylim(zlim_global)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")

    # ===== YZ =====
    ax = fig.add_subplot(gs[0,3])
    ax.scatter(x_positions[:,1], x_positions[:,2],
               c=F_values, cmap=cmap, vmin=vmin, vmax=vmax, s=100)

    ax.scatter(x_positions[imax,1], x_positions[imax,2], s=100, marker='^',
               facecolor='maroon', edgecolor='black')

    ax.scatter(x_positions[imin,1], x_positions[imin,2], s=100, marker='v',
               facecolor='navy', edgecolor='black')

    ax.set_aspect('equal')
    ax.set_xlim(ylim_global)
    ax.set_ylim(zlim_global)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")

    fig.colorbar(sc, cax=fig.add_subplot(gs[0,4])).set_label(title)

    fig.suptitle(
        title+"\n"+f"$\\it{{Magnet:\\ {Magnet_info}}}$",
        fontsize=15,y=0.975
    )
    
    plt.tight_layout(rect=[0, 0, 0.97, 0.94])
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# BUCLE POR STEPS
# ============================================================

for step_folder in step_folders_sorted:

    input_path = os.path.join(step_folder, 'Fuerzas.txt')
    if not os.path.exists(input_path):
        continue

    data = np.loadtxt(input_path, skiprows=1)

    base_name = os.path.basename(step_folder)
    base_output = os.path.join(step_folder, base_name)

    Fx, Fy, Fz = data[:,1], data[:,2], data[:,3]
    normF = np.sqrt(Fx**2 + Fy**2 + Fz**2)

    Tx, Ty, Tz = data[:,4], data[:,5], data[:,6]
    normT = np.sqrt(Tx**2 + Ty**2 + Tz**2)

    x, y, z = data[:,7], data[:,8], data[:,9]

    tol = 1e-6
    x_round = np.round(x / tol) * tol
    unique_x = np.unique(x_round)
    ring = np.array([np.where(unique_x == xi)[0][0] + 1 for xi in x_round])

    lids = classify_lids(x, y, z)

    output_lid_path = base_output + "_TableForceTorqueSum_perLid.txt"

    positions = []
    values_dict = {k: [] for k in ["Fx","Fy","Fz","normF","Tx","Ty","Tz","normT"]}
    labels = []

    with open(output_lid_path, "w", encoding="utf-8") as f:

        f.write("Ring_Lid\tN_cubes\tFx_sum\tFy_sum\tFz_sum\tnormF\tTx_sum\tTy_sum\tTz_sum\tnormT\n")

        for r in np.unique(ring):

            for lid_name in ["Lid+Z","Lid-Z","Lid+Y","Lid-Y"]:

                idx = (ring == r) & (lids == lid_name)

                if not np.any(idx):
                    continue

                n_cubes = np.sum(idx)

                Fx_r = Fx[idx].sum()
                Fy_r = Fy[idx].sum()
                Fz_r = Fz[idx].sum()

                Tx_r = Tx[idx].sum()
                Ty_r = Ty[idx].sum()
                Tz_r = Tz[idx].sum()

                normF_r = np.sqrt(Fx_r**2 + Fy_r**2 + Fz_r**2)
                normT_r = np.sqrt(Tx_r**2 + Ty_r**2 + Tz_r**2)

                row_name = f"Ring{r}_{lid_name}"

                print(f"{base_name} - {row_name}: {n_cubes} cubos")

                row = [
                    row_name,
                    n_cubes,
                    Fx_r, Fy_r, Fz_r, normF_r,
                    Tx_r, Ty_r, Tz_r, normT_r
                ]

                f.write("\t".join(map(str,row)) + "\n")

                # Posiciones ficticias
                if lid_name == "Lid+Z":
                    pos = [unique_x[r-1], 0, limit_z]
                elif lid_name == "Lid-Z":
                    pos = [unique_x[r-1], 0, -limit_z]
                elif lid_name == "Lid+Y":
                    pos = [unique_x[r-1], limit_y, 0]
                elif lid_name == "Lid-Y":
                    pos = [unique_x[r-1], -limit_y, 0]

                positions.append(pos)
                labels.append(row_name)

                values_dict["Fx"].append(Fx_r)
                values_dict["Fy"].append(Fy_r)
                values_dict["Fz"].append(Fz_r)
                values_dict["normF"].append(normF_r)
                values_dict["Tx"].append(Tx_r)
                values_dict["Ty"].append(Ty_r)
                values_dict["Tz"].append(Tz_r)
                values_dict["normT"].append(normT_r)

    print(f"Tabla perLid guardada: {output_lid_path}")

    if not positions:
        continue

    positions = np.array(positions)

    for comp in values_dict.keys():

        plot_lids_3d(
            positions,
            np.array(values_dict[comp]),
            labels,
            f"{comp} Lids",
            base_output + f"_{comp}_Lids.png",
            component_name=comp,
            is_torque=("T" in comp)
        )

# ============================================================
# ===================== WORST CASE PER LID ====================
# ============================================================

print("\nCalculando Worst Cases PerLid...")

components = ["Fx","Fy","Fz","normF","Tx","Ty","Tz","normT"]
lids_list = ["Lid+Y","Lid-Y","Lid+Z","Lid-Z"]

worst_perlid = {}
worst_perlid_by_lid = {lid:{} for lid in lids_list}

for component in components:

    worst_value = -np.inf
    worst_step = None

    # Inicializar por lid
    for lid in lids_list:
        worst_perlid_by_lid[lid][component] = (None, -np.inf)

    for step_folder in step_folders_sorted:

        base_name = os.path.basename(step_folder)

        table_path = os.path.join(
            step_folder,
            base_name + "_TableForceTorqueSum_perLid.txt"
        )

        if not os.path.exists(table_path):
            continue

        df = pd.read_csv(table_path, sep="\t")

        col_map = {
            "Fx": "Fx_sum",
            "Fy": "Fy_sum",
            "Fz": "Fz_sum",
            "normF": "normF",
            "Tx": "Tx_sum",
            "Ty": "Ty_sum",
            "Tz": "Tz_sum",
            "normT": "normT"
        }

        # ---------------------------
        # GLOBAL PER LID (como ya estaba)
        # ---------------------------

        values = df[col_map[component]].abs().values
        max_val = values.max()

        if max_val > worst_value:
            worst_value = max_val
            worst_step = base_name

        # ---------------------------
        # PER LID INDIVIDUAL
        # ---------------------------

        for lid in lids_list:

            df_lid = df[df["Ring_Lid"].str.endswith(lid)]

            if df_lid.empty:
                continue

            lid_values = df_lid[col_map[component]].abs().values
            lid_max = lid_values.max()

            current_step, current_val = worst_perlid_by_lid[lid][component]

            if lid_max > current_val:
                worst_perlid_by_lid[lid][component] = (base_name, lid_max)

    worst_perlid[component] = (worst_step, worst_value)

print("Worst PerLid calculado.")


# ============================================================
# ACTUALIZAR TABLA GLOBAL 
# ============================================================

print("Actualizando WorstCases_Global.txt con columnas PerLid...")

global_worst_path = os.path.join(base_folder, "WorstCases_Global.txt")

df_global = pd.read_csv(global_worst_path, sep="\t")

# ---------------------------
# Columnas nuevas (si no existen)
# ---------------------------

new_columns = [
    "PerLid_Step", "PerLid_Value",
    "PerLid+Y_Step","PerLid+Y_Value",
    "PerLid-Y_Step","PerLid-Y_Value",
    "PerLid+Z_Step","PerLid+Z_Value",
    "PerLid-Z_Step","PerLid-Z_Value"
]

for col in new_columns:
    if col not in df_global.columns:
        df_global[col] = ""

# ---------------------------
# Rellenar valores
# ---------------------------

for i, row in df_global.iterrows():

    comp = row["Component"]

    # Global PerLid
    if comp in worst_perlid:
        step, value = worst_perlid[comp]
        df_global.at[i, "PerLid_Step"] = step
        df_global.at[i, "PerLid_Value"] = value

    # Por cada lid
    for lid in lids_list:

        if comp in worst_perlid_by_lid[lid]:

            step, value = worst_perlid_by_lid[lid][comp]

            df_global.at[i, f"Per{lid}_Step"] = step
            df_global.at[i, f"Per{lid}_Value"] = value

df_global.to_csv(global_worst_path, sep="\t", index=False)

print("WorstCases_Global.txt actualizado con PerLid + subdivisiones.")
print("\nAnalisis por Lids completado.")

