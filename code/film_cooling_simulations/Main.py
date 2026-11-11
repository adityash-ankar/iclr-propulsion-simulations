# %% ------------------------- Helper functions and bookkeeping -------------------------

import numpy as np
import math
import matplotlib.pyplot as plt
# Enable LaTeX rendering
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'  # This works well with LaTeX
plt.rcParams['text.latex.preamble'] = r''  # Empty or no override
from rocketcea.cea_obj import CEA_Obj
from pyfluids import Fluid, FluidsList, Input, Phases
from CoolProp.CoolProp import PropsSI
import scipy as sp
from scipy.interpolate import CubicSpline


def ethanol_Tsat_from_P(P_pa):
    P_mmhg = np.asarray(P_pa) / 133.322368
    A1,B1,C1 = 8.20417, 1642.89, 230.3
    A2,B2,C2 = 7.68117, 1332.04, 199.200
    logP = np.log10(P_mmhg)
    T1_C = B1/(A1 - logP) - C1
    T2_C = B2/(A2 - logP) - C2
    T_C  = np.where(T1_C <= 78.3, T1_C, T2_C)
    return T_C + 273.15

# %% ------------------------- Unit conversions (RocketCEA -> SI) ------------------------
_degR_to_K         = 5.0/9.0          # K per degR
_ftps_to_mps       = 0.3048           # m/s per ft/s
_BTUlbmR_to_JkgK   = 4186.8           # J/(kg·K) per (BTU/lbm·°R)
_millipoise_to_Pas = 1e-4             # Pa·s per millipoise
_mcalcmKs_to_WmK   = 418.4            # W/(m·K) per mcal/(cm·K·s)

# %% ================================= User inputs ==================================
ox               = 'N2O'
fuel             = 'Ethanol'
Material         = 'AlSi10Mg'  # 'Inconel', 'AlSi10Mg', 'ABD900'

Chamber_Pressure = 30 # bar
Feed_Pressure    = 40.0   # bar
Thrust_req       = 5000.0 # N
OF               = 3.25
CR               = 6    # contraction ratio Ac/At
Lstar_target_mm  = 750.0  # mm (chamber only)
tw               = 0.7e-3 # wall thickness [m]
channel_height   = 1.2e-3 # [m]
arc_angle        = 3.8    # degrees
n                = 50     # channels
fcp              = 0.275   # film-cooling percentage 

#Graphs
Complete_Thermal_Analysis = True
Thermal_Diagnostic_3x3    = False
Film_Cooling_vs_No_FC     = False
SF_and_Temps_3D           = False #Looks Great, Takes like 5 Minutes to run, proceed with caution
Stress_Plot               = True
# Efficiencies
eta_cstar = 0.94
eta_Cf    = 0.98

# %% ========================= CEA / ideal performance =========================
ispObj = CEA_Obj(oxName=ox, fuelName=fuel)

Pc_SI   = Chamber_Pressure * 1e5       # Pa
Pc_psia = Chamber_Pressure * 14.5038   # psia
g0      = 9.80665

# Area ratio ~ Pe = 1 atm
Area_Ratio = ispObj.get_eps_at_PcOvPe(Pc=Pc_psia, MR=OF,
                                      PcOvPe=Chamber_Pressure/1.013,
                                      frozen=0, frozenAtThroat=0)

# Ideal Isp & c*
ISP_vac_ideal = ispObj.get_Isp(Pc=Pc_psia, MR=OF, eps=Area_Ratio, frozen=0, frozenAtThroat=0)     # s
cstar_ideal   = ispObj.get_Cstar(Pc=Pc_psia, MR=OF) / 3.28                                        # m/s

# Ideal Cf at SL
Cf_list_SL   = ispObj.get_PambCf(Pamb=14.7, Pc=Pc_psia, MR=OF, eps=Area_Ratio)
Cf_SL_ideal  = float(Cf_list_SL[0])

# Apply efficiencies
cstar_act = eta_cstar * cstar_ideal
Cf_SL_act = eta_Cf    * Cf_SL_ideal

# Throat area from thrust target (SL)
At_SI = Thrust_req / (Cf_SL_act * Pc_SI)   # m²
Rt_SI = math.sqrt(At_SI / math.pi)         # m

# Mass flow & check
mdot_act   = Pc_SI * At_SI / cstar_act
ISP_SL_act = Cf_SL_act * cstar_act / g0
F_check_SL = Cf_SL_act * Pc_SI * At_SI

# %% ============================== Geometry (mm) ==============================
def rao_nozzle(Rt, epsilon):
    """
    Generate a Rao bell nozzle contour with high resolution.
    Total segments: ~800
    """

    nozzle_length = 0.8 * ((math.sqrt(epsilon) - 1) * Rt / math.tan(math.radians(15)))
    theta_n = math.radians(24)
    theta_e = math.radians(12)
    Rn = 0.382*Rt

    # 1) Circular arc at throat - Increased to 50 points
    x_change = 0.382 * Rt * math.sin(theta_n)
    x_circ = np.linspace(0, x_change, 50)
    theta_circ = np.arcsin(x_circ / (0.382 * Rt))
    y_circ = Rt * (1.382 - 0.382 * np.cos(theta_circ))

    # 2) Parabolic section
    Nx = 0.382 * Rt * math.cos(theta_n - math.radians(90))
    Ny = 0.382 * Rt * math.sin(theta_n - math.radians(90)) + 0.382 * Rt + Rt
    Ex = nozzle_length
    Ey = np.sqrt(epsilon) * Rt

    m1, m2 = math.tan(theta_n), math.tan(theta_e)
    C1, C2 = Ny - m1 * Nx, Ey - m2 * Ex
    Qx = (C2 - C1) / (m1 - m2)
    Qy = (m1 * C2 - m2 * C1) / (m1 - m2)

    # Quadratic Bezier curve parameter t - Increased to 751 points
    t = np.linspace(0, 1, 751)
    x_para = (1 - t)**2 * Nx + 2 * (1 - t) * t * Qx + t**2 * Ex
    y_para = (1 - t)**2 * Ny + 2 * (1 - t) * t * Qy + t**2 * Ey

    x_array = np.concatenate((x_circ, x_para[1:]))
    y_array = np.concatenate((y_circ, y_para[1:]))

    volume = 0
    for i in range(len(x_array) - 1):
        volume += math.pi * y_array[i]**2 * (x_array[i+1] - x_array[i])

    return x_array, y_array, volume, Rn, theta_n, theta_e

def convergent_sizing(CR, Rt):
    """
    Generate a convergent section contour with high resolution.
    Total segments: ~700
    """

    R1 = 1.5 * Rt
    b = math.radians(30)
    Rc = np.sqrt(CR) * Rt
    R2max = (Rc - Rt) / (1 - math.cos(b)) - R1
    R2 = R2max * 0.5

    # 1) R2 arc - Increased to 250 points
    theta2 = np.linspace(0, b, 250)
    x2_array = R2 * np.sin(theta2)
    y2_array = Rc - R2 + R2 * np.cos(theta2)

    # 2) R1 arc - Increased to 250 points
    theta1 = np.linspace(b, 0, 250)
    x1_array = R1 * math.sin(b) - R1 * np.sin(theta1)
    y1_array = Rt + R1 - R1 * np.cos(theta1)

    # 3) Straight cone segment - Increased to 200 points
    dely = y2_array[-1] - y1_array[0]
    delx = dely / math.tan(b)
    xl_array = np.linspace(np.max(x2_array), np.max(x2_array) + delx, 200)
    yl_array = np.linspace(np.min(y2_array), np.min(y2_array) - dely, 200)

    x1_array += np.max(xl_array)

    x_array = np.concatenate((x2_array, xl_array[1:-1], x1_array))
    y_array = np.concatenate((y2_array, yl_array[1:-1], y1_array))

    volume = 0
    for i in range(len(x_array) - 1):
        volume += math.pi * y_array[i]**2 * (x_array[i+1] - x_array[i])

    return x_array, y_array, volume, R1, R2, b

def cylinder_sizing(Lstar, other_volume, Rt, CR):
    """
    Size the cylindrical chamber with high resolution.
    Total segments: 500
    """

    required_volume = Lstar * math.pi * Rt**2
    Volume = required_volume - other_volume
    Length = Volume / (CR * math.pi * Rt**2)

    # Increased to 501 points (500 segments)
    x_array = np.linspace(0, Length, 501)
    y_array = np.ones_like(x_array) * Rt * math.sqrt(CR)

    return x_array, y_array, Length

# --- Execution and Stitching ---

Rt_mm = Rt_SI * 1e3
Re_mm = math.sqrt(Area_Ratio) * Rt_mm
Rc_mm = math.sqrt(CR) * Rt_mm

# Bell (Rao) nozzle (~800 points)
xn, yn, vn_mm3, Rn_mm, theta_n, theta_e = rao_nozzle(Rt_mm, Area_Ratio)

# Convergent (arc–cone–arc) (~700 points)
xc, yc, vc_mm3, R1_mm, R2_mm, b_rad = convergent_sizing(CR, Rt_mm)

# Cylinder sized by L* (501 points)
xcyl, ycyl, Lcyl_mm = cylinder_sizing(Lstar_target_mm, vc_mm3, Rt_mm, CR)

# Stitch sections axially
xc   = xc + np.max(xcyl)
xn   = xn + np.max(xc)


# 1. Stitch raw coordinates together
# Note: xc and xn are already shifted by their previous section maxes
x_raw = np.concatenate((xcyl, xc, xn))
y_raw = np.concatenate((ycyl, yc, yn))

# 2. Robust Duplicate Removal and Sorting
# We must ensure x is strictly increasing for CubicSpline.
# np.unique with return_index=True finds the first occurrence of each unique x.
_, unique_indices = np.unique(x_raw, return_index=True)
x_clean = x_raw[unique_indices]
y_clean = y_raw[unique_indices]

# 3. Create the Spline
# This forces C2 continuity (smooth curvature) across section joints
cs = CubicSpline(x_clean, y_clean)

# 4. Resample at High Resolution (1001 nodes for 1000 uniform segments)
x_nodes_smooth = np.linspace(x_clean.min(), x_clean.max(), 1001)
y_nodes_smooth = cs(x_nodes_smooth)

# 5. Redefine segment midpoints for the thermal loops
x_seg = 0.5 * (x_nodes_smooth[:-1] + x_nodes_smooth[1:])
y_seg = 0.5 * (y_nodes_smooth[:-1] + y_nodes_smooth[1:])

# Re-identify throat index on the smooth curve
throat_node_fwd = np.argmin(y_nodes_smooth)
# ---------------- [NEW: Curvature Smoothing] ----------------
# Precompute and smooth the second derivative to avoid "steps" in the Dean number
ddy_raw = cs.derivative(2)(x_nodes_smooth)
# Apply a Gaussian filter to smooth out the sharp transitions at geometric seams
ddy_smooth = sp.ndimage.gaussian_filter1d(ddy_raw, sigma=30)
ddy_func = CubicSpline(x_nodes_smooth, ddy_smooth)
x_nodes = x_nodes_smooth
y_nodes = y_nodes_smooth
# %% ========================== Material properties ==========================
if Material == 'Inconel718':
    Surf_roughness      = np.array([54, 38, 12, 6, 5]) * 1e-6 * 10
    manufacturing_angle = np.array([30, 45, 60, 75, 90]) * math.pi/180
    v                   = 0.28
    conductivity        = 12
    cte                 = 16e-6
    modulus_temps       = np.array([21, 93, 204, 316, 427, 538, 649, 760, 871, 954]) + 273.15
    modulus             = np.array([208, 205, 202, 194, 186, 179, 172, 162, 127, 78]) * 1e9
    yield_temps         = np.array([93, 204, 316, 427, 538, 649, 760]) + 273.15
    yield_stress        = np.array([1172, 1124, 1096, 1076, 1069, 1027, 758]) * 1e6
    emissivity          = 0.3
elif Material == 'AlSi10Mg':
    Surf_roughness      = np.array([66, 56, 21, 12, 8]) * 1e-6 * 10
    manufacturing_angle = np.array([30, 45, 60, 75, 90]) * math.pi/180
    modulus_temps       = np.array([25, 50, 100, 150, 200, 250, 300, 350, 400]) + 273.15
    modulus             = np.array([77.6, 75.5, 72.8, 63.2, 60, 55, 45, 37, 28]) * 1e9
    yield_temps         = np.array([298, 323, 373, 423, 473, 523, 573, 623, 673])
    yield_stress        = np.array([204, 198, 181, 182, 158, 132, 70, 30, 12]) * 1e6
    conductivity        = 140
    cte                 = 27e-6
    v                   = 0.33
    emissivity          = 0.650
elif Material == 'ABD900':
    Surf_roughness      = np.array([54, 38, 12, 6, 5]) * 1e-6 * 0.7 * 10
    manufacturing_angle = np.array([30, 45, 60, 75, 90]) * math.pi/180
    modulus_temps       = np.array([25, 700, 800, 900]) + 273.15
    modulus             = np.array([192, 157, 131, 103]) * 1e9
    yield_temps         = np.array([25, 427, 538, 649, 732, 760, 788, 816, 843, 871, 927]) + 273.15
    yield_stress        = np.array([978, 931, 903, 903, 758, 680, 600, 503, 434, 352, 221]) * 1e6
    cte                 = np.array([11.4, 12.9, 13.7, 14.4, 15.5, 17.5, 19.2])*1e-6
    cte_temps           = np.array([50, 200, 400, 600, 800, 1000, 1200]) + 273.15
    conductivity        = np.array([11, 12.6, 15.7, 18.8, 23.2, 26.4, 30.1])
    conductivity_temps  = np.array([25, 200, 400, 600, 800, 1000, 1200]) + 273.15
    v                   = 0.28
    emissivity          = 0.3
else:
    raise ValueError("Unknown Material")

# %% ============================ Regen preparation ============================
fuel_mdot   = mdot_act/(OF + 1)
ox_mdot = mdot_act * OF / (OF + 1)
feed_press  = Feed_Pressure*1e5

# Channel width from printing arc
channel_width = (y_seg + (tw*1e3) + 0.5*(channel_height*1e3)) * math.sin(math.radians(arc_angle)) * 1e-3  # [m]
w = channel_width
h = channel_height
Dh = 2*h*w/(h+w)

# Wall angle (radians) at nodes (forward)
angles_seg = np.zeros_like(y_seg, dtype=float)
for i in range(len(y_seg)):
    if i == 0 or i == len(y_seg)-1:
        angles_seg[i] = 0.0
    else:
        dy = y_nodes[i+1] - y_nodes[i]
        dx = x_nodes[i+1] - x_nodes[i]
        angles_seg[i] = math.atan2(dy, dx)
# Roughness from build angle; clip to table range for continuity
angles_abs = np.abs(angles_seg)
angles_clipped = np.clip(angles_abs, manufacturing_angle.min(), manufacturing_angle.max())
roughness_node = np.interp(angles_clipped, manufacturing_angle, Surf_roughness)

# Reverse for coolant counterflow
rev = {
    'x': x_seg[::-1],          # mm
    'x_nodes': x_nodes[::-1],
    'r': y_seg[::-1],
    'r_nodes': y_nodes[::-1],        # mm
    'Dh': Dh[::-1],             # m
    'w': w[::-1],               # m
    'eps': roughness_node[::-1] # m (as k_s)
}
throat_idx_rev = np.argmin(rev['r'])

# %% ============================ Thermal arrays ============================
Nseg = len(rev['r'])           # segments in reversed march
qg_array       = np.zeros(Nseg, dtype=float)
alpha_t_array  = np.zeros(Nseg, dtype=float)
twg_array      = np.zeros(Nseg, dtype=float)
twc_array      = np.zeros(Nseg, dtype=float)
te_array       = np.zeros(Nseg, dtype=float)
Mach_array     = np.zeros(Nseg, dtype=float)
regen_pressures= np.zeros(Nseg, dtype=float)
Nu_array       = np.zeros(Nseg, dtype=float)
Re_array       = np.zeros(Nseg, dtype=float)

tc_array       = np.zeros(Nseg+1, dtype=float)
tc_array[0]    = 300.0  # K


regen_pressures = np.zeros_like(qg_array)

# =========================[CEA Gas props (held constant axially)]=========================
throat_idx_fwd   = np.argmin(y_seg)
T0_chamber       = ispObj.get_Tcomb(Pc=Pc_psia, MR=OF)                           # degR
cp_chamber, mu_chamber, _, Pr_chamber = ispObj.get_Chamber_Transport(Pc=Pc_psia, MR=OF, frozen=0)
cp_throat , mu_throat, _, Pr_throat   = ispObj.get_Throat_Transport(Pc=Pc_psia, MR=OF, frozen=0)
cp_exit   , mu_exit, _,   Pr_exit     = ispObj.get_Exit_Transport(Pc=Pc_psia, MR=OF, frozen=0)
_, gamma_chamber = ispObj.get_Chamber_MolWt_gamma(Pc=Pc_psia, MR=OF, eps=Area_Ratio)
_, gamma_throat  = ispObj.get_Throat_MolWt_gamma(Pc=Pc_psia, MR=OF, eps=Area_Ratio)
_, gamma_exit    = ispObj.get_exit_MolWt_gamma(Pc=Pc_psia, MR=OF, eps=Area_Ratio)

c_star_chamber   = ispObj.get_Cstar(Pc=Pc_psia, MR=OF)                           # ft/s


cp_array = np.interp(x_seg,(0, x_seg[throat_idx_fwd], np.max(x_seg)) , (cp_chamber, cp_throat, cp_exit)) * _BTUlbmR_to_JkgK
mu_array = np.interp(x_seg,(0, x_seg[throat_idx_fwd], np.max(x_seg)) , (mu_chamber, mu_throat, mu_exit)) * _millipoise_to_Pas
Pr_array = np.interp(x_seg,(0, x_seg[throat_idx_fwd], np.max(x_seg)) , (Pr_chamber, Pr_throat, Pr_exit))
gamma_array = np.interp(x_seg,(0, x_seg[throat_idx_fwd], np.max(x_seg)) , (gamma_chamber, gamma_throat, gamma_exit))

cp_rev_array, mu_rev_array, Pr_rev_array, gamma_rev_array = cp_array[::-1],mu_array[::-1],Pr_array[::-1],gamma_array[::-1]

T0    = T0_chamber * _degR_to_K
cstar = c_star_chamber * _ftps_to_mps


Rt = Rt_SI
Dt = 2.0 * Rt
At = At_SI

# %% ================= Main Bartz loop (reversed coolant direction) =================

# Ethanol critical point (CoolProp values)
_ETH_Tc = 514.0        # K (approx)
_ETH_Pc = 6.38e6       # Pa (approx)

def fuel_phase(Tc_loc_K, regen_press, fuel_st, *, pressure_in_pa=True):
    """
    Robust ethanol state chooser for PyFluids:
    - Lets EOS decide from (P,T) first.
    - If that fails, picks phase by Tsat(P), then retries.
    - Adds tiny nudges near saturation to avoid singularities.
    - Expects Tc_loc_K in Kelvin. 'regen_press' in Pa if pressure_in_pa=True, else bar.
    Returns: a PyFluids Fluid(FluidsList.Ethanol) at the requested state.
    """

    # --- Units ---
    P = float(regen_press if pressure_in_pa else regen_press * 1e5)  # Pa
    T_K = float(Tc_loc_K)                                            # K
    T_C = T_K - 273.15                                               # °C (PyFluids expects °C)
    if not (np.isfinite(P) and np.isfinite(T_K)) or P <= 0.0:
        raise ValueError(f"Bad inputs to fuel_phase: P={P} Pa, T={T_K} K")

    # Use a FRESH instance so we don't carry a previously specified phase
    fs = Fluid(FluidsList.Ethanol)

    # Quick supercritical check: if clearly supercritical, don't force a phase
    if (P > 0.99*_ETH_Pc and T_K > 0.99*_ETH_Tc):
        try:
            return fs.with_state(Input.pressure(P), Input.temperature(T_C))
        except ValueError:
            # tiny nudge in T to get off critical manifold
            return fs.with_state(Input.pressure(P), Input.temperature((T_K + 0.2) - 273.15))

    # First try: let backend pick the region automatically
    try:
        return fs.with_state(Input.pressure(P), Input.temperature(T_C))
    except ValueError:
        # Decide side of saturation line
        try:
            Tsat_K = PropsSI('T', 'P', P, 'Q', 0, 'Ethanol')  # sat. liquid T at this P
        except Exception as e:
            # If saturation query fails (rare), nudge T and retry auto
            return fs.with_state(Input.pressure(P), Input.temperature((T_K + 0.2) - 273.15))

        # Stay a hair away from the coexistence curve to avoid singularities
        eps = 0.15  # K
        if T_K < Tsat_K - eps:
            fs.specify_phase(Phases.Liquid)
            T_try = T_K + eps
        elif T_K > Tsat_K + eps:
            fs.specify_phase(Phases.Gas)
            T_try = T_K - eps
        else:
            # Within epsilon of Tsat: treat as liquid side for regen coolant
            fs.specify_phase(Phases.Liquid)
            T_try = Tsat_K - eps

        # Retry with explicit phase
        try:
            return fs.with_state(Input.pressure(P), Input.temperature(T_try - 273.15))
        except ValueError:
            # Final fallback: tiny additional nudge
            return fs.with_state(Input.pressure(P), Input.temperature((T_try + 0.3) - 273.15))


def get_ethanol_state(T_K, P_pa, phase_hint=None, pressure_floor_pa=1e5, temp_floor_K=200.0, temp_cap_K=None):
    """
    Return a robust ethanol state while guarding against unphysical marching values.
    """
    P_use = max(float(P_pa), pressure_floor_pa)
    T_use = max(float(T_K), temp_floor_K)
    if temp_cap_K is not None:
        T_use = min(T_use, float(temp_cap_K))

    fuel = Fluid(FluidsList.Ethanol)
    if phase_hint == "gas":
        try:
            fuel.specify_phase(Phases.Gas)
            return fuel.with_state(Input.pressure(P_use), Input.temperature(T_use - 273.15))
        except ValueError:
            return fuel_phase(T_use, P_use, fuel)
    if phase_hint == "liquid":
        try:
            fuel.specify_phase(Phases.Liquid)
            return fuel.with_state(Input.pressure(P_use), Input.temperature(T_use - 273.15))
        except ValueError:
            return fuel_phase(T_use, P_use, fuel)

    return fuel_phase(T_use, P_use, fuel)

def Nu_func(Re, f, Pr, c_height, c_width, Dh, L, mu_c, mu_tw, x_loc, cs_obj, ddy_func):
    """
    Returns local Nusselt number for a regen channel with spline-based curvature enhancement.
    Inputs:
      Re, f, Pr        : Reynolds (local), Darcy friction factor (local), Prandtl (bulk)
      c_height, c_width: channel rectangle
      Dh, L            : hydraulic diameter, local distance from coolant inlet
      mu_c, mu_tw      : dynamic viscosity at bulk and at coolant-side wall
      x_loc            : current axial position [mm]
      cs_obj           : the CubicSpline object for the nozzle geometry
      ddy_func         : a function to evaluate the smoothed second derivative
    """
    # ---------------- Analytical Radius of Curvature (via Spline) ----------------
    # Formula: Rc = (1 + y'^2)^1.5 / |y''|
    dy = cs_obj.derivative(1)(x_loc)
    ddy = ddy_func(x_loc) # Use the pre-smoothed derivative
    Rc = np.abs((1 + dy**2)**1.5 / max(abs(ddy), 1e-12))

    # Dean number (curvature enhancement); safe if Rc is large
    De = 1
    # ---------------- Helpers ----------------
    # Finite-length correction uses local x=L; cap near inlet
    L_eff = max(L, 3.0 * Dh)

    # Thermal entrance length (laminar estimate)
    xth = 0.05 * Re * Pr * Dh

    # Aspect ratio handling for fully developed laminar (Shah & London, H1 ~ uniform q'')
    a, b = (c_height, c_width) if c_height >= c_width else (c_width, c_height)
    ar_grid = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)
    Nu_fd_grid = np.array([3.608, 4.123, 5.331, 6.490], dtype=float)
    Nu_fd_rect = np.interp(a / max(b, 1e-12), ar_grid, Nu_fd_grid)

    # Property corrections
    wall_corr_turb = (mu_c / max(mu_tw, 1e-12))**0.11
    wall_corr_lam  = (mu_c / max(mu_tw, 1e-12))**0.14

    # Curvature multipliers
    C_turb, m_turb = 0.06, 0.20
    C_lam,  m_lam  = 0.03, 0.50

    # ---------------- Base turbulent Gnielinski (single-phase) ----------------
    def Nu_gnielinski(Re_, Pr_, f_):
        num = (f_ / 8.0) * (Re_ - 1000.0) * Pr_
        den = 1.0 + 12.7 * np.sqrt(f_ / 8.0) * (Pr_**(2.0/3.0) - 1.0)
        return num / max(den, 1e-12)

    # ---------------- Laminar developing (Hausen) ----------------
    def Nu_hausen(Re_, Pr_, Dh_, L_):
        Gz = Re_ * Pr_ * Dh_ / max(L_, 1e-12)
        return 3.66 + 0.0668 * Gz / (1.0 + 0.04 * (Gz**(2.0/3.0)))

    # ---------------- Regime logic ----------------
    if Re >= 4000.0:
        # Turbulent branch
        Nu_base = Nu_gnielinski(Re, Pr, f)
        len_corr = 1.0 + (Dh / L_eff)**(2.0/3.0)
        Nu_turb  = Nu_base * len_corr * wall_corr_turb
        Nu = Nu_turb * (1.0 + C_turb * (De**m_turb))

    elif 2300.0 < Re < 4000.0:
        # Transitional blend
        Nu_base = Nu_gnielinski(Re, Pr, f)
        len_corr = 1.0 + (Dh / L_eff)**(2.0/3.0)
        Nu_turb  = Nu_base * len_corr * wall_corr_turb
        Nu_turb  = Nu_turb * (1.0 + C_turb * (De**m_turb))

        if L_eff > xth:
            Nu_lam = Nu_fd_rect * wall_corr_lam
        else:
            Nu_lam = Nu_hausen(Re, Pr, Dh, L_eff) * wall_corr_lam
        Nu_lam  = Nu_lam * (1.0 + C_lam * (De**m_lam))

        # Smooth blend
        gamma = (Re - 2300.0) / (4000.0 - 2300.0)
        gamma = min(max(gamma, 0.0), 1.0)
        Nu = (1.0 - gamma) * Nu_lam + gamma * Nu_turb

    else:
        # Re <= 2300: Laminar branch
        if L_eff > xth:
            Nu_lam = Nu_fd_rect * wall_corr_lam
        else:
            Nu_lam = Nu_hausen(Re, Pr, Dh, L_eff) * wall_corr_lam
        Nu = Nu_lam * (1.0 + C_lam * (De**m_lam))

    return float(Nu)

dr_array = abs(np.diff(rev['r_nodes'])) * 1e-3
dx_array = abs(np.diff(rev['x_nodes'])) * 1e-3
seg_len_array = np.sqrt(dr_array**2 + dx_array**2)
for i in range(Nseg):  # segment i is between nodes i and i+1 in REVERSED arrays

    Pr_g, cp_g, mu_g, gamma = Pr_rev_array[i], cp_rev_array[i], mu_rev_array[i], gamma_rev_array[i]
    # --- Segment midpoint geometry (used everywhere below) ---
    A_seg     = math.pi * rev['r'][i]**2 * 1e-6                         # m^2

    # Segment lengths
    seg_len = seg_len_array[i]

    # --- Coolant hydraulics on this segment ---
    mass_flow_rate = fuel_mdot / n
    Dh_i  = rev['Dh'][i]
    Aflow = channel_height * rev['w'][i]

    # pressure reference
    Press_ref = float(feed_press) if i == 0 else float(regen_pressures[i-1])
    Tc_loc    = float(tc_array[i])

    # 1) provisional pressure drop then one light iterate
    Press_ref = max(float(Press_ref), 1e5)
    Tc_loc = max(float(Tc_loc), 200.0)
    fuel_ref = get_ethanol_state(Tc_loc, Press_ref, phase_hint="liquid")
    rho_ref  = fuel_ref.density
    mu_f_ref = fuel_ref.dynamic_viscosity
    u1       = mass_flow_rate/(rho_ref*Aflow)
    Re_c     = rho_ref * u1 * Dh_i / mu_f_ref
    # Darcy with roughness (k_s ~ eps)
    fD = (-1.8*math.log10(rev['eps'][i]/(3.7*Dh_i) + (6.9/Re_c)**1.1))**-2
    dyn_head   = 0.5*rho_ref*u1**2
    dP_segment = fD * seg_len/Dh_i * dyn_head
    if i == 0 or i == (Nseg-1):
        dP_segment += 0.8*dyn_head
    raw_regen_press = float(Press_ref - dP_segment)
    if raw_regen_press <= 0.0:
        print(
            f"Warning: negative coolant pressure at segment {i}: "
            f"Press_ref={Press_ref:.3f} Pa, dP_segment={dP_segment:.3f} Pa, "
            f"Tc_loc={Tc_loc:.3f} K"
        )
    regen_press = max(raw_regen_press, 1e5)         # guard: avoid ≤ 0 Pa
    regen_pressures[i] = regen_press

    # refine once
    if abs(regen_press - Press_ref)/Press_ref > 5e-2:
        Press_ref = float(regen_press)
        Press_ref = max(float(Press_ref), 1e5)
        Tc_loc = max(float(Tc_loc), 200.0)
        fuel_ref = get_ethanol_state(Tc_loc, Press_ref, phase_hint="liquid")
        rho_ref  = fuel_ref.density
        mu_f_ref = fuel_ref.dynamic_viscosity
        u1       = mass_flow_rate/(rho_ref*Aflow)
        Re_c     = rho_ref*u1*Dh_i/mu_f_ref
        fD       = (-1.8*math.log10(rev['eps'][i]/(3.7*Dh_i) + (6.9/Re_c)**1.1))**-2
        dyn_head = 0.5*rho_ref*u1**2
        dP_segment = fD * seg_len/Dh_i * dyn_head
        if i == 0 or i == (Nseg-1):
            dP_segment += 0.8*dyn_head
        regen_press = max(Press_ref - dP_segment, 1e5)

    regen_pressures[i] = regen_press

    # coolant props at regen_press
    regen_press = max(float(regen_press), 1e5)
    Tc_loc = max(float(Tc_loc), 200.0)
    fuel_st = get_ethanol_state(Tc_loc, regen_press, phase_hint="liquid")
    rho     = fuel_st.density
    mu_f    = fuel_st.dynamic_viscosity
    lam_c   = fuel_st.conductivity
    cp_c    = fuel_st.specific_heat
    Pr_c    = cp_c * mu_f / lam_c

    # --- Mach number from area ratio at segment MIDPOINT ---
    if (i + 0.5) > throat_idx_rev:
        # upstream of throat (subsonic)
        M = ispObj.get_Chamber_MachNumber(Pc=Pc_psia, MR=OF, fac_CR=A_seg/At)
    elif (i + 0.5) < throat_idx_rev:
        # downstream (supersonic)
        M = ispObj.get_MachNumber(Pc=Pc_psia, MR=OF, eps=A_seg/At,
                                  frozen=0, frozenAtThroat=0)
    else:
        M = 1.0
    Mach_array[i] = M

    # --- Edge temperature (Bartz recovery) ---
    Pr_use = float(Pr_g) if (np.isfinite(Pr_g) and Pr_g > 0.0) else 0.7
    r      = Pr_use ** (1.0/3.0)
    den    = 1.0 + 0.5*(gamma - 1.0)*M*M
    den    = max(den, 1e-8)  # guard
    Te     = T0 * (1.0 + r*0.5*(gamma - 1.0)*M*M) / den
    te_array[i] = Te

    # --- Coolant-side HTC (Dittus-Boelter) ---
    u1   = mass_flow_rate / (rho * Aflow)
    Re_c = rho * u1 * Dh_i / mu_f


    # work in meters once, then reuse
    r_m = 1e-3 * np.asarray(rev['r'], dtype=float)
    x_m = 1e-3 * np.asarray(rev['x'], dtype=float)

    if i == 0:
        y1, y2, y3 = r_m[i],   r_m[i],   r_m[i+1]
        x1, x2, x3 = x_m[i],   x_m[i],   x_m[i+1]
    elif i == Nseg - 1:
        y1, y2, y3 = r_m[i], r_m[i],   r_m[i]
        x1, x2, x3 = x_m[i-1], x_m[i],   x_m[i]
    else:
        y1, y2, y3 = r_m[i-1], r_m[i],   r_m[i+1]
        x1, x2, x3 = x_m[i-1], x_m[i],   x_m[i+1]


    L    = np.sum(seg_len_array[:i+1])
    Nu   = Nu_func(Re_c, fD, Pr_c, channel_height, rev['w'][i], Dh_i, L, mu_f, mu_f, rev['x'][i], cs, ddy_func)
    Nu_array[i] = Nu
    Re_array[i] = Re_c
    alpha_c = Nu * lam_c / Dh_i

   # --- Bartz alpha_T using MIDPOINT geometry ratios ---
    At_over_A = At / A_seg
    Dt_over_R = (2.0*Rt) / (rev['r'][i] *1e-3)

    # iterative wall solve (with cap + under-relax)
    Twg_0   = Te - 50.0
    tol     = 1e-4
    max_it  = 500
    relax   = 0.1

    # guard values
    alpha_c_eff = max(alpha_c, 50.0)
    lambda_w_min = 1.0
    tw_eff = max(tw, 1e-4)
    base2 = max(1.0 + 0.5*(gamma - 1.0)*M*M, 1e-9)

    def bartz_alpha_T(Twg_local):
        base1 = 0.5*(Twg_local/max(T0,1e-6))*base2 + 0.5
        base1 = max(base1, 1e-9)
        sigma = (base1 ** -0.68) * (base2 ** -0.12)
        return 0.026 * (Dt**-0.2) * (mu_g**0.2) * cp_g * (Pr_g**-0.6) \
               * ((Pc_SI/cstar)**0.8) * (Dt_over_R**0.1) \
               * (At_over_A**0.9) * sigma

    for k in range(max_it):
        alpha_T = bartz_alpha_T(Twg_0)
        qg  = alpha_T * max(Te - Twg_0, 0.0)
        Twc = qg / alpha_c_eff + Tc_loc

        if isinstance(conductivity, np.ndarray):
            lambda_w = np.interp((Twg_0+Twc)/2.0, conductivity_temps, conductivity)
        else:
            lambda_w = conductivity
        lambda_w = max(lambda_w, lambda_w_min)

        Twg_new = Twc + qg * tw_eff / lambda_w

        Twg_1 = (1.0 - relax)*Twg_0 + relax*Twg_new
        if abs(Twg_1 - Twg_0)/max(abs(Twg_1), 1.0) < tol:
            Twg_0 = Twg_1
            break
        Twg_0 = Twg_1
    else:
        # didn’t converge — keep last Twg_0
        pass

    # final values after iteration
    alpha_T = bartz_alpha_T(Twg_0)
    qg  = alpha_T * max(Te - Twg_0, 0.0)
    Twc = qg / alpha_c_eff + Tc_loc
    if isinstance(conductivity, np.ndarray):
        lambda_w = np.interp((Twg_0+Twc)/2.0, conductivity_temps, conductivity)
    else:
        lambda_w = conductivity
    lambda_w = max(lambda_w, lambda_w_min)

    Twg_1 = Twc + qg * tw_eff / lambda_w

    # # ================= IEVLEV BOUNDARY-LAYER COOLING CORRECTION =================
    # # Uses q''_film = q''_nofilm * (S_surf / S_core), with S per Ievlev.
    # # Assumes the following vars already exist in your loop:
    # # T0, M, gamma, Pr_g, mu_g, cp_g, Pc_SI, rev['r'][i], Twg_1, qg, alpha_c_eff,
    # # Tc_loc, tw_eff, conductivity, conductivity_temps, lambda_w_min

    # # --- knobs / safe defaults (change if you have better sources) ---
    # try:
    #     blc_strength = max(0.05, min(0.60, 0.50 * fcp))  # if 'fcp' exists
    # except NameError:
    #     blc_strength = 0.30                               # otherwise use 30%
    # T_film_inj    = max(200.0, min(Tc_loc, 700.0))        # K, proxy for injected layer temp
    # mu_T_exponent = 0.7                                    # mu ~ T^0.7 for surface layer

    # # --- core (edge) static state & Re (isentropic relations) ---
    # den_isent     = 1.0 + 0.5*(gamma - 1.0)*M*M
    # Te_core       = T0 / den_isent                                         # static K
    # Rg            = cp_g * (gamma - 1.0) / gamma                           # J/(kg.K)
    # P_core        = Pc_SI / (den_isent ** (gamma/(gamma - 1.0)))           # static Pa
    # a_core        = (gamma * Rg * Te_core) ** 0.5
    # u_core        = M * a_core
    # rho_core      = P_core / max(Rg * Te_core, 1e-9)
    # D_char        = 2.0 * (rev['r'][i] * 1e-3)                              # m
    # Re_core       = rho_core * u_core * D_char / max(mu_g, 1e-12)
    # r_recovery    = max(Pr_g, 1e-9) ** (1.0/3.0)
    # Taw_core      = T0 * (1.0 + r_recovery * 0.5*(gamma - 1.0)*M*M) / den_isent

    # # --- surface layer proxy (cooled near-wall mixture) ---
    # Te_surf       = (1.0 - blc_strength) * Te_core + blc_strength * T_film_inj
    # Te_surf       = max(200.0, min(Te_surf, Te_core))                       # clamp
    # T0_surf       = Te_surf * den_isent
    # Taw_surf      = T0_surf * (1.0 + r_recovery * 0.5*(gamma - 1.0)*M*M) / den_isent
    # mu_surf       = mu_g * (Te_surf / max(Te_core, 1e-9)) ** mu_T_exponent
    # a_surf        = (gamma * Rg * Te_surf) ** 0.5
    # u_surf        = M * a_surf
    # rho_surf      = P_core / max(Rg * Te_surf, 1e-9)
    # Re_surf       = rho_surf * u_surf * D_char / max(mu_surf, 1e-12)

    # # --- Ievlev S-factors (core vs surface layer) ---
    # num_core = max(Taw_core - Twg_1, 0.0) * (Te_core ** 0.425) * (mu_g ** 0.15)
    # den_core = (Re_core ** 0.425) * ((Te_core + Twg_1) ** 0.595) * ((3.0*Te_core + Twg_1) ** 0.15)
    # S_core   = num_core / max(den_core, 1e-30)

    # num_surf = max(Taw_surf - Twg_1, 0.0) * (Te_surf ** 0.425) * (mu_surf ** 0.15)
    # den_surf = (Re_surf ** 0.425) * ((Te_surf + Twg_1) ** 0.595) * ((3.0*Te_surf + Twg_1) ** 0.15)
    # S_surf   = num_surf / max(den_surf, 1e-30)

    # # --- apply correction to convective heat flux; keep within sane bounds ---
    # ratio   = max(0.2, min(1.0, S_surf / max(S_core, 1e-30)))  # 0.2–1.0 clamp for stability
    # qg_corr = qg * ratio

    # # --- recompute wall temperatures with corrected convective load ---
    # Twc = qg_corr / alpha_c_eff + Tc_loc
    # if isinstance(conductivity, np.ndarray):
    #     lambda_w = np.interp((Twg_1 + Twc)/2.0, conductivity_temps, conductivity)
    # else:
    #     lambda_w = conductivity
    # lambda_w = max(lambda_w, lambda_w_min)
    # Twg_1 = Twc + qg_corr * tw_eff / lambda_w

    # # overwrite with film-cooled result
    # qg = qg_corr
    # # ================= END IEVLEV CORRECTION =================

    # store segment results (still reversed order)
    qg_array[i]       = qg
    alpha_t_array[i]  = alpha_T
    twg_array[i]      = Twg_1
    twc_array[i]      = Twc

    Ahot_seg = 2.0 * math.pi * rev['r'][i] * seg_len *1e-3          # m^2
    Q_seg    = qg * Ahot_seg                               # W (J/s)
    tc_array[i+1] = tc_array[i] + Q_seg / (fuel_mdot * cp_c)

# =========================[FLIP to forward gas-flow order]=========================
def flip_seg(arr):
    return arr[-1:0:-1]  # reverse and drop the first element to get length Nseg-1? No:
# Safer: our arrays are length Nseg; we want forward order with same length as forward segments
# Use simple reverse:
def flip_full(arr):
    return arr[::-1]

qg_fwd       = flip_full(qg_array).squeeze()
alphaT_fwd   = flip_full(alpha_t_array).squeeze()
Twg_fwd      = flip_full(twg_array).squeeze()
Twc_fwd      = flip_full(twc_array).squeeze()
Mach_fwd     = flip_full(Mach_array).squeeze()
Tc_fwd_nodes = flip_full(tc_array).squeeze()   # node-based (reversed march had Nseg+1 nodes)
Te_fwd       = flip_full(te_array).squeeze()
Pcool_fwd    = flip_full(regen_pressures).squeeze()
Taw_core     = Twg_fwd + qg_fwd / np.maximum(alphaT_fwd, 1e-9)

# Build forward segment midpoints for plotting (length = len(x_nodes)-1)
x_nodes_mm = np.asarray(x_nodes).ravel()
r_nodes_mm = np.asarray(y_nodes).ravel()
x_seg_mm   = 0.5*(x_nodes_mm[:-1] + x_nodes_mm[1:])
r_seg_mm   = 0.5*(r_nodes_mm[:-1] + r_nodes_mm[1:])
x_throat   = x_nodes_mm[throat_node_fwd]

# tc bulk per segment (avg of adjacent nodes)
Tc_seg = 0.5*(Tc_fwd_nodes[:-1] + Tc_fwd_nodes[1:])


# %% ============================== Film cooling ==============================
cp_g_array     = np.zeros_like(x_seg_mm) #Specific heat capacity of freestream gas in chamber
Pr_g_array     = np.zeros_like(x_seg_mm) #Prandtl number of freestream gas in chamber
mu_g_array     = np.zeros_like(x_seg_mm) #Viscosity of freestream gas in chamber
gamma_array    = np.zeros_like(x_seg_mm) #Ratio of specific heat capacities for freestream gas in chamber
Mg_array       = np.zeros_like(x_seg_mm) #Molecular weight of freestream gas in chamber
pressure_array = np.zeros_like(x_seg_mm)
T_fcool_array  = np.zeros_like(x_seg_mm)
Gamma_array    = np.zeros_like(x_seg_mm)
h_corr_array   = np.zeros_like(x_seg_mm)
q_corr_array   = np.zeros_like(x_seg_mm)
throat_idx     = np.argmin(r_seg_mm)
Mv             = 46 #g/mol for Ethanol
Km_array       = np.zeros_like(x_seg_mm) #Blowing factor correction term
eps_array      = r_seg_mm**2 / np.min(r_seg_mm)**2
ethanol_latent_heat_vap = 2.5e5 #J/kg

dx_seg_m = np.diff(x_nodes_mm) * 1e-3  # length = Nseg
def dx_at(i):
    # safe pick even at the last usable index
    return float(dx_seg_m[min(i, dx_seg_m.size-1)])


for i in range(len(x_seg_mm)):
    cp_g_array[i], mu_g_array[i], _, Pr_g_array[i] = ispObj.get_Chamber_Transport(Pc=Pc_psia, MR=OF, frozen=0, eps = eps_array[i])
    cp_g_array[i]*=_BTUlbmR_to_JkgK
    Mg_array[i],gamma_array[i] = ispObj.get_Chamber_MolWt_gamma(Pc=Pc_psia, MR=OF, eps = eps_array[i])
    Km_array[i] = (Mg_array[i]/Mv)**0.6 if Mg_array[i]>Mv else (Mg_array[i]/Mv)**0.35
    g = float(gamma_array[i])
    M = float(Mach_fwd[i])
    pressure_array[i] = Pc_SI / (1.0 + 0.5*(g-1.0)*M*M)**(g/(g-1.0))

#-----------------------------------[LIQUID FILM]--------------------------------------------------
#Initially, since film cooling is injected at x=0
Gamma_init  = (fuel_mdot*fcp)/(2*np.pi*r_seg_mm[0]*1e-3) #Initial Mass flow of coolant per unit circumference
T_fcoolant  = np.max(Tc_seg)
dT_liq      = 0
Gamma       = Gamma_init
burnout_idx = None

# --- CORRECTED LIQUID FILM LOOP WITH FLASH CHECK ---
P_entry = pressure_array[0]
Tsat_entry = ethanol_Tsat_from_P(P_entry)

if T_fcoolant >= Tsat_entry:
    burnout_idx = 0
    # Ensure array is filled with the gas temp for the transition logic
    T_fcool_array[:] = T_fcoolant
else:
    Gamma = Gamma_init
    burnout_idx = throat_idx
    for i in range(throat_idx):
        Tsat             = ethanol_Tsat_from_P(pressure_array[i])
        T_fcool_array[i] = T_fcoolant
        coolant_ref      = get_ethanol_state(T_fcoolant, pressure_array[i], phase_hint="liquid")
        cp_cool_liq      = coolant_ref.specific_heat
        h0               = alphaT_fwd[i]
        q_conv0          = h0 * (Te_fwd[i] - T_fcoolant)
        mdotvap          = q_conv0/max(ethanol_latent_heat_vap,1.0)
        wet_P            = 2 *np.pi * r_seg_mm[i] *1e-3

        DeltaT = max(Te_fwd[i] - T_fcoolant, 0.0)
        H      = (cp_g_array[i] * Km_array[i] / ethanol_latent_heat_vap) * DeltaT
        h      = h0 * np.log1p(H) / max(H, 1e-12)
        h_corr_array[i] = h
        Gamma_array[i] = Gamma
        q_to_film   = qg_fwd[i] * wet_P * dx_at(i)
        dT_liq      = q_to_film / max(Gamma*cp_cool_liq, 1e-12)
        T_fcoolant += dT_liq

        if T_fcoolant >= Tsat - 1e-9:
            T_fcoolant = Tsat
            dGamma = - mdotvap * wet_P * dx_at(i)
            Gamma  = Gamma + dGamma
            if Gamma < 1e-8:
                burnout_idx = i
                break
    if burnout_idx is None: burnout_idx = throat_idx

# If never burned out in chamber region
if burnout_idx is None:
    burnout_idx = throat_idx


# ------------------------------[GAS FILM — per-span m′]------------------------------
L = len(twg_array)

# Per-span film mass flow m′ (kg/(m·s))
mbl        = np.zeros(L - burnout_idx, dtype=float)
P0         = 2*np.pi*max(r_seg_mm[burnout_idx]*1e-3, 1e-9)
mbl[0]   = max(Gamma_init, 1e-8)

# Per-span bookkeeping terms (entrainment, geometry, etc.)
dmb1       = np.zeros_like(mbl)  # predictor (entrainment only), per-span
dmb2       = np.zeros_like(mbl)  # corrector (entrainment only), per-span
dmb_e      = np.zeros_like(mbl)  # entrained increment this step, per-span
dmb_geom   = np.zeros_like(mbl)  # geometry (area-change) increment, per-span

# Temperature bookkeeping
dTe        = np.zeros_like(mbl)
Taw_fc_g   = np.zeros_like(mbl)
Tsat_burn = ethanol_Tsat_from_P(pressure_array[burnout_idx])
Taw_seed  = np.clip(Tsat_burn, T_fcoolant, Taw_core[burnout_idx])
Taw_fc_g[0] = Taw_seed


for i in range(burnout_idx, L-1):
    j = i - burnout_idx  # zero-index for the reduced arrays

    Kt = 1.0
    Km = (Mv / max(Mg_array[i], 1e-12))**0.14   # (kept as you have it)

    P_i             = 2 * np.pi  * r_seg_mm[i] * 1e-3
    # Core external mass flux (kg/m^2/s). If you separate streams elsewhere, keep this as-is.
    mdotgas         = (ox_mdot + fuel_mdot*(1 - fcp))
    A_g             = np.pi * (max(r_seg_mm[i]*1e-3, 1e-9))**2
    mdot_total_core = (ox_mdot + fuel_mdot*(1 - fcp))
    mdot_film       = mbl[j] * P_i                   # per-span * perimeter -> kg/s
    mdot_core_free  = max(mdot_total_core - mdot_film, 1e-9)
    G               = mdot_core_free / max(A_g, 1e-9)

    # Film properties at the local film temperature
    ethanol_gas = get_ethanol_state(Taw_fc_g[j], pressure_array[i], phase_hint="gas")
    cp_c = ethanol_gas.specific_heat

    # Geometry terms
    D   = 2.0 * max(r_seg_mm[i]*1e-3, 1e-9)                  # local diameter [m]
    dD  = 2.0 * (r_seg_mm[i+1] - r_seg_mm[i]) * 1e-3         # finite diff ∆D over segment [m]
    dx  = max(dx_at(i), 1e-12)                             # segment length

    # Local perimeters (start/end of segment)
    P_i   = 2*np.pi*max(r_seg_mm[i]*1e-3,   1e-9)
    P_ip1 = 2*np.pi*max(r_seg_mm[i+1]*1e-3, 1e-9)

    # ---------------- Entrainment growth (per-span) with predictor–corrector ---------------
    # Predictor at station i
    mu_g_i = mu_g_array[i]
    m_i    = max(mbl[j], 1e-12)
    dmb1[j]= 0.1963 * Kt * G * (mu_g_i / m_i)**0.25 * dx

    # Corrector using provisional m′ at i+1
    m_p    = max(mbl[j] + dmb1[j], 1e-12)
    mu_g_ip1 = mu_g_array[i+1]
    # Use same dx for the step (safer than indexing dx_array[i+1] at the end of domain)
    dmb2[j]= 0.1963 * Kt * G * (mu_g_ip1 / m_p)**0.25 * dx

    # Trapezoidal average entrainment increment (per-span)
    dmb_e[j] = 0.5 * (dmb1[j] + dmb2[j])

    # ---------------- Geometry (area-change) term (per-span) ----------------
    # dm'/dx|geom ≈ - m' * (1/D) * dD/dx  -> finite-diff over the segment:
    dmb_geom[j] = - m_i * (dD / max(D, 1e-12))

    # ---------------- Update film per-span mass flow ----------------
    m_next = mbl[j] + dmb_e[j] + dmb_geom[j]
    mbl[j+1] = max(m_next, 1e-12)

    # ---------------- Temperature update (use TOTALS for heat capacities) ----------------
    # Convert per-span to totals at station i using P_i
    m_f_tot = mbl[j]        * P_i           # film already in the layer
    m_e_tot = max(dmb_e[j], 0.0) * P_i      # entrained this step (only the positive part)

    # External reference temperature (kept as your Te_fwd; if you prefer recovery, swap here)
    Tr = Taw_core[i]

    C_f   = m_f_tot * max(cp_c, 1e-6)                   # film capacity rate
    C_e   = m_e_tot * Km * max(cp_g_array[i], 1e-6)     # external (weighted) capacity rate
    denom = max(C_f + C_e, 1e-9)

    dTe[j]        = (Tr - Taw_fc_g[j]) * (C_e / denom)
    Taw_fc_g[j+1] = Taw_fc_g[j] + dTe[j]


#-------------------------------------[REDOING BARTZ WITH FILM COOLING]--------------------------------------------

#Initialise reversed arrays for film cooling thermals
Twg_fc_rev        = np.zeros_like(Twg_fwd)
Twc_fc_rev        = np.zeros_like(Twg_fwd)
Tc_fc_rev         = np.zeros_like(Twg_fwd)
Pc_fc_rev         = np.zeros_like(Twg_fwd)
qg_fc_rev         = np.zeros_like(Twg_fwd)
Tc_fc_rev[0]      = 300.0

#Assign new h for liquid region and keep old h for gaseous region:
h_fc_array        = np.concatenate((h_corr_array[:burnout_idx+1], alphaT_fwd[burnout_idx+1:]))
h_fc_array_rev    = flip_full(h_fc_array)
burnout_idx_rev   = len(h_fc_array_rev) - burnout_idx

#Assign new Taw for gaseous region and keep old Taw for liquid region:
Taw_fc_array        = np.concatenate((Taw_core[:burnout_idx+1], Taw_fc_g[1:]))
Taw_fc_array_rev    = flip_full(Taw_fc_array)

for i in range(len(Twg_fc_rev)-1):
    # --- Segment midpoint geometry (used everywhere below) ---
    R_mid_m   = 0.5 * (rev['r'][i] + rev['r'][i+1]) * 1e-3   # m
    A_mid     = math.pi * R_mid_m**2                         # m^2
    x_mid_mm  = 0.5 * (rev['x'][i] + rev['x'][i+1])          # mm

    # Segment lengths
    dr = abs(rev['r'][i+1] - rev['r'][i]) * 1e-3             # m
    dx = abs(rev['x'][i+1] - rev['x'][i]) * 1e-3             # m
    seg_len = math.hypot(dx, dr)

    # --- Coolant hydraulics on this segment ---
    mass_flow_rate = fuel_mdot / n
    Dh_i  = rev['Dh'][i]
    Aflow = channel_height * rev['w'][i]

    # pressure reference
    Press_ref = float(feed_press) if i == 0 else float(regen_pressures[i-1])
    Tc_loc    = float(Tc_fc_rev[i])

    # 1) provisional pressure drop then one light iterate
    Press_ref = max(float(Press_ref), 1e5)
    Tc_loc = max(float(Tc_loc), 200.0)
    fuel_ref = get_ethanol_state(Tc_loc, Press_ref, phase_hint="liquid")
    rho_ref  = fuel_ref.density
    mu_f_ref = fuel_ref.dynamic_viscosity
    u1       = mass_flow_rate/(rho_ref*Aflow)
    Re_c     = rho_ref * u1 * Dh_i / mu_f_ref
    # Darcy with roughness (k_s ~ eps)
    fD = (-1.8*math.log10(rev['eps'][i]/(3.7*Dh_i) + (6.9/Re_c)**1.1))**-2
    dyn_head   = 0.5*rho_ref*u1**2
    dP_segment = fD * seg_len/Dh_i * dyn_head
    if i == 0 or i == (Nseg-1):
        dP_segment += 0.8*dyn_head
    raw_regen_press = float(Press_ref - dP_segment)
    if raw_regen_press <= 0.0:
        print(
            f"Warning: negative coolant pressure at segment {i}: "
            f"Press_ref={Press_ref:.3f} Pa, dP_segment={dP_segment:.3f} Pa, "
            f"Tc_loc={Tc_loc:.3f} K"
        )
    regen_press = max(raw_regen_press, 1e5)         # guard: avoid ≤ 0 Pa
    regen_pressures[i] = regen_press

    # refine once
    if abs(regen_press - Press_ref)/Press_ref > 5e-2:
        Press_ref = float(regen_press)
        Press_ref = max(float(Press_ref), 1e5)
        Tc_loc = max(float(Tc_loc), 200.0)
        fuel_ref = get_ethanol_state(Tc_loc, Press_ref, phase_hint="liquid")
        rho_ref  = fuel_ref.density
        mu_f_ref = fuel_ref.dynamic_viscosity
        u1       = mass_flow_rate/(rho_ref*Aflow)
        Re_c     = rho_ref*u1*Dh_i/mu_f_ref
        fD       = (-1.8*math.log10(rev['eps'][i]/(3.7*Dh_i) + (6.9/Re_c)**1.1))**-2
        dyn_head = 0.5*rho_ref*u1**2
        dP_segment = fD * seg_len/Dh_i * dyn_head
        if i == 0 or i == (Nseg-1):
            dP_segment += 0.8*dyn_head
        regen_press = max(Press_ref - dP_segment, 1e5)

    regen_pressures[i] = regen_press

    # coolant props at regen_press
    fuel_st = Fluid(FluidsList.Ethanol)
    fuel_st = fuel_phase(Tc_loc, regen_press, fuel_st)
    rho     = fuel_st.density
    mu_f    = fuel_st.dynamic_viscosity
    lam_c   = fuel_st.conductivity
    cp_c    = fuel_st.specific_heat
    Pr_c    = cp_c * mu_f / lam_c

    # --- Mach number from area ratio at segment MIDPOINT ---
    if (i + 0.5) > throat_idx_rev:
        # upstream of throat (subsonic)
        M = ispObj.get_Chamber_MachNumber(Pc=Pc_psia, MR=OF, fac_CR=A_mid/At)
    elif (i + 0.5) < throat_idx_rev:
        # downstream (supersonic)
        M = ispObj.get_MachNumber(Pc=Pc_psia, MR=OF, eps=A_mid/At,
                                  frozen=0, frozenAtThroat=0)
    else:
        M = 1.0
    Mach_array[i] = M

    # --- Coolant-side HTC (Dittus-Boelter) ---
    u1   = mass_flow_rate / (rho * Aflow)
    Re_c = rho * u1 * Dh_i / mu_f
    # work in meters once, then reuse
    r_m = 1e-3 * np.asarray(rev['r'], dtype=float)
    x_m = 1e-3 * np.asarray(rev['x'], dtype=float)

    if i == 0:
        y1, y2, y3 = r_m[i],   r_m[i],   r_m[i+1]
        x1, x2, x3 = x_m[i],   x_m[i],   x_m[i+1]
    elif i == Nseg - 1:
        y1, y2, y3 = r_m[i], r_m[i],   r_m[i]
        x1, x2, x3 = x_m[i-1], x_m[i],   x_m[i]
    else:
        y1, y2, y3 = r_m[i-1], r_m[i],   r_m[i+1]
        x1, x2, x3 = x_m[i-1], x_m[i],   x_m[i+1]


    L    = np.sum(seg_len_array[:i+1])
    Nu   = Nu_func(Re_c, fD, Pr_c, channel_height, rev['w'][i], Dh_i, L, mu_f, mu_f, rev['x'][i], cs, ddy_func)
    alpha_c = Nu * lam_c / Dh_i

    # iterative wall solve (with cap + under-relax)
    Taw     = Taw_fc_array_rev[i]
    Twg_0   = Taw - 50.0
    tol     = 1e-4
    max_it  = 200

    # guard values
    alpha_c_eff = max(alpha_c, 1e-9)
    lambda_w_min = 1.0
    tw_eff = max(tw, 1e-4)
    base2 = max(1.0 + 0.5*(gamma - 1.0)*M*M, 1e-9)

    if isinstance(conductivity, np.ndarray):
        lambda_w = np.interp((Twg_0+Twc)/2.0, conductivity_temps, conductivity)
    else:
        lambda_w = conductivity
    lambda_w = max(lambda_w, lambda_w_min)

    h   = h_fc_array_rev[i]
    qg_i  = (Taw_fc_array_rev[i] - Tc_loc)/(1/h + 1/alpha_c_eff + tw/lambda_w)
    Twc = qg_i / alpha_c_eff + Tc_loc
    Twg_1 = Taw_fc_array_rev[i] - qg_i/h

    # store segment results (still reversed order)
    qg_fc_rev[i]       = qg_i
    Twg_fc_rev[i]      = Twg_1
    Twc_fc_rev[i]      = Twc

    Ahot_seg = 2.0 * math.pi * R_mid_m * seg_len          # m^2
    Q_seg    = qg_i * Ahot_seg                               # W (J/s)
    Tc_fc_rev[i+1] = Tc_fc_rev[i] + Q_seg / (fuel_mdot * cp_c)

Tc_max = np.max(Tc_fc_rev)
# After the for-loop that fills qg_fc_rev, Twg_fc_rev, Twc_fc_rev
Twg_fc_rev[-1] = Twg_fc_rev[-2]
Twc_fc_rev[-1] = Twc_fc_rev[-2]
qg_fc_rev[-1]  = qg_fc_rev[-2]




# %% -------------------- Iterate until coolant temperature converges --------------------
iter_count = 0
while abs(Tc_fc_rev[-1] - T_fcool_array[0])/T_fcool_array[0] >1e-4 and iter_count < 25:
    #-----------------------------------[LIQUID FILM]--------------------------------------------------
    #Initially, since film cooling is injected at x=0
    Gamma_init  = (fuel_mdot*fcp)/(2*np.pi*r_seg_mm[0]*1e-3) #Initial Mass flow of coolant per unit circumference
    T_fcoolant  = Tc_fc_rev[-1]
    dT_liq      = 0
    Gamma       = Gamma_init
    burnout_idx = None
    # --- CORRECTED LIQUID FILM LOOP WITH FLASH CHECK ---
    P_entry = pressure_array[0]
    Tsat_entry = ethanol_Tsat_from_P(P_entry)

    if T_fcoolant >= Tsat_entry:
        burnout_idx = 0
        # Ensure array is filled with the gas temp for the transition logic
        T_fcool_array[:] = T_fcoolant
    else:
        Gamma = Gamma_init
        burnout_idx = throat_idx
        for i in range(throat_idx):
            Tsat             = ethanol_Tsat_from_P(pressure_array[i])
            T_fcool_array[i] = T_fcoolant
            coolant_ref      = get_ethanol_state(T_fcoolant, pressure_array[i], phase_hint="liquid")
            cp_cool_liq      = coolant_ref.specific_heat
            h0               = alphaT_fwd[i]
            q_conv0          = h0 * (Te_fwd[i] - T_fcoolant)
            mdotvap          = q_conv0/max(ethanol_latent_heat_vap,1.0)
            wet_P            = 2 *np.pi * r_seg_mm[i] *1e-3

            DeltaT = max(Te_fwd[i] - T_fcoolant, 0.0)
            H      = (cp_g_array[i] * Km_array[i] / ethanol_latent_heat_vap) * DeltaT
            h      = h0 * np.log1p(H) / max(H, 1e-12)
            h_corr_array[i] = h
            Gamma_array[i] = Gamma
            q_to_film   = qg_fwd[i] * wet_P * dx_at(i)
            dT_liq      = q_to_film / max(Gamma*cp_cool_liq, 1e-12)
            T_fcoolant += dT_liq

            if T_fcoolant >= Tsat - 1e-9:
                T_fcoolant = Tsat
                dGamma = - mdotvap * wet_P * dx_at(i)
                Gamma  = Gamma + dGamma
                if Gamma < 1e-8:
                    burnout_idx = i
                    break
        if burnout_idx is None: burnout_idx = throat_idx


    # ------------------------------[GAS FILM — per-span m′]------------------------------
    L = len(twg_array)

    # Per-span film mass flow m′ (kg/(m·s))
    mbl        = np.zeros(L - burnout_idx, dtype=float)
    P0         = 2*np.pi*max(r_seg_mm[burnout_idx]*1e-3, 1e-9)
    mbl[0]   = max(Gamma_init, 1e-8)

    # Per-span bookkeeping terms (entrainment, geometry, etc.)
    dmb1       = np.zeros_like(mbl)  # predictor (entrainment only), per-span
    dmb2       = np.zeros_like(mbl)  # corrector (entrainment only), per-span
    dmb_e      = np.zeros_like(mbl)  # entrained increment this step, per-span
    dmb_geom   = np.zeros_like(mbl)  # geometry (area-change) increment, per-span

    # Temperature bookkeeping
    dTe        = np.zeros_like(mbl)
    Taw_fc_g   = np.zeros_like(mbl)
    Tsat_burn = ethanol_Tsat_from_P(pressure_array[burnout_idx])
    Taw_seed  = np.clip(Tsat_burn, T_fcoolant, Taw_core[burnout_idx])
    Taw_fc_g[0] = Taw_seed

    for i in range(burnout_idx, L-1):
        j = i - burnout_idx  # zero-index for the reduced arrays

        Kt = 1.0
        Km = (Mv / max(Mg_array[i], 1e-12))**0.14   # (kept as you have it)

        P_i             = 2 * np.pi  * r_seg_mm[i] * 1e-3
        # Core external mass flux (kg/m^2/s). If you separate streams elsewhere, keep this as-is.
        mdotgas         = (ox_mdot + fuel_mdot*(1 - fcp))
        A_g             = np.pi * (max(r_seg_mm[i]*1e-3, 1e-9))**2
        mdot_total_core = (ox_mdot + fuel_mdot*(1 - fcp))
        mdot_film       = mbl[j] * P_i                   # per-span * perimeter -> kg/s
        mdot_core_free  = max(mdot_total_core - mdot_film, 1e-9)
        G               = mdot_core_free / max(A_g, 1e-9)

        # Film properties at the local film temperature
        ethanol_gas = get_ethanol_state(Taw_fc_g[j], pressure_array[i], phase_hint="gas")
        cp_c = ethanol_gas.specific_heat

        # Geometry terms
        D   = 2.0 * max(r_seg_mm[i]*1e-3, 1e-9)                  # local diameter [m]
        dD  = 2.0 * (r_seg_mm[i+1] - r_seg_mm[i]) * 1e-3         # finite diff ∆D over segment [m]
        dx  = max(dx_at(i), 1e-12)                             # segment length

        # Local perimeters (start/end of segment)
        P_i   = 2*np.pi*max(r_seg_mm[i]*1e-3,   1e-9)
        P_ip1 = 2*np.pi*max(r_seg_mm[i+1]*1e-3, 1e-9)

        # ---------------- Entrainment growth (per-span) with predictor–corrector ---------------
        # Predictor at station i
        mu_g_i = mu_g_array[i]
        m_i    = max(mbl[j], 1e-12)
        dmb1[j]= 0.1963 * Kt * G * (mu_g_i / m_i)**0.25 * dx

        # Corrector using provisional m′ at i+1
        m_p    = max(mbl[j] + dmb1[j], 1e-12)
        mu_g_ip1 = mu_g_array[i+1]
        # Use same dx for the step (safer than indexing dx_array[i+1] at the end of domain)
        dmb2[j]= 0.1963 * Kt * G * (mu_g_ip1 / m_p)**0.25 * dx

        # Trapezoidal average entrainment increment (per-span)
        dmb_e[j] = 0.5 * (dmb1[j] + dmb2[j])

        # ---------------- Geometry (area-change) term (per-span) ----------------
        # dm'/dx|geom ≈ - m' * (1/D) * dD/dx  -> finite-diff over the segment:
        dmb_geom[j] = - m_i * (dD / max(D, 1e-12))

        # ---------------- Update film per-span mass flow ----------------
        m_next = mbl[j] + dmb_e[j] + dmb_geom[j]
        mbl[j+1] = max(m_next, 1e-12)

        # ---------------- Temperature update (use TOTALS for heat capacities) ----------------
        # Convert per-span to totals at station i using P_i
        m_f_tot = mbl[j]        * P_i           # film already in the layer
        m_e_tot = max(dmb_e[j], 0.0) * P_i      # entrained this step (only the positive part)

        # External reference temperature (kept as your Te_fwd; if you prefer recovery, swap here)
        Tr = Taw_core[i]

        C_f   = m_f_tot * max(cp_c, 1e-6)                   # film capacity rate
        C_e   = m_e_tot * Km * max(cp_g_array[i], 1e-6)     # external (weighted) capacity rate
        denom = max(C_f + C_e, 1e-9)

        dTe[j]        = (Tr - Taw_fc_g[j]) * (C_e / denom)
        Taw_fc_g[j+1] = Taw_fc_g[j] + dTe[j]

  # --- CORRECTED BLENDING BLOCK WITH INDEX OFFSET ---
    blend_width = 15
    h_fc_smooth = np.copy(alphaT_fwd)
    Taw_fc_smooth = np.copy(Taw_core)

    for i in range(len(x_seg_mm)):
        if burnout_idx == 0:
            # Entirely gaseous case
            h_fc_smooth[i] = alphaT_fwd[i]
            Taw_fc_smooth[i] = Taw_fc_g[i]
        elif i < burnout_idx - blend_width//2:
            # Pure liquid region
            h_fc_smooth[i] = h_corr_array[i]
            Taw_fc_smooth[i] = T_fcool_array[i]
        elif i > burnout_idx + blend_width//2:
            # Pure gas region: Use the offset [i - burnout_idx]
            h_fc_smooth[i] = alphaT_fwd[i]
            Taw_fc_smooth[i] = Taw_fc_g[min(i - burnout_idx, len(Taw_fc_g)-1)]
        else:
            # Transition blend: Use the offset here too
            w_b = (i - (burnout_idx - blend_width//2)) / blend_width
            idx_g = min(max(0, i - burnout_idx), len(Taw_fc_g)-1)

            h_start = h_corr_array[max(0, burnout_idx-1)]
            h_fc_smooth[i] = (1-w_b)*h_start + w_b*alphaT_fwd[i]

            T_liq_at_start = T_fcool_array[max(0, burnout_idx-1)]
            T_gas_now = Taw_fc_g[idx_g]
            Taw_fc_smooth[i] = (1-w_b)*T_liq_at_start + w_b*T_gas_now

    #-------------------------------------[REDOING BARTZ WITH FILM COOLING]--------------------------------------------

    #Initialise reversed arrays for film cooling thermals
    Twg_fc_rev        = np.zeros_like(Twg_fwd)
    Twc_fc_rev        = np.zeros_like(Twg_fwd)
    Tc_fc_rev         = np.zeros_like(Twg_fwd)
    Pc_fc_rev         = np.zeros_like(Twg_fwd)
    qg_fc_rev         = np.zeros_like(Twg_fwd)
    Tc_fc_rev[0]      = 300.0

    #Assign new h for liquid region and keep old h for gaseous region:
    h_fc_array        = np.concatenate((h_corr_array[:burnout_idx], alphaT_fwd[burnout_idx:]))
    h_fc_array_rev    = flip_full(h_fc_array)
    burnout_idx_rev   = len(h_fc_array_rev) - burnout_idx

    #Assign new Taw for gaseous region and keep old Taw for liquid region:
    Taw_fc_array = np.concatenate((Taw_core[:burnout_idx], Taw_fc_g))
    Taw_fc_array_rev    = flip_full(Taw_fc_array)

    for i in range(len(Twg_fc_rev)-1):
        # --- Segment midpoint geometry (used everywhere below) ---
        R_mid_m   = 0.5 * (rev['r'][i] + rev['r'][i+1]) * 1e-3   # m
        A_mid     = math.pi * R_mid_m**2                         # m^2
        x_mid_mm  = 0.5 * (rev['x'][i] + rev['x'][i+1])          # mm

        # Segment lengths
        dr = abs(rev['r'][i+1] - rev['r'][i]) * 1e-3             # m
        dx = abs(rev['x'][i+1] - rev['x'][i]) * 1e-3             # m
        seg_len = math.hypot(dx, dr)

        # --- Coolant hydraulics on this segment ---
        mass_flow_rate = fuel_mdot / n
        Dh_i  = rev['Dh'][i]
        Aflow = channel_height * rev['w'][i]

        # pressure reference
        Press_ref = float(feed_press) if i == 0 else float(regen_pressures[i-1])
        Tc_loc    = float(Tc_fc_rev[i])

        # 1) provisional pressure drop then one light iterate
        Press_ref = max(float(Press_ref), 1e5)
        Tc_loc = max(float(Tc_loc), 200.0)
        fuel_ref = get_ethanol_state(Tc_loc, Press_ref, phase_hint="liquid")
        rho_ref  = fuel_ref.density
        mu_f_ref = fuel_ref.dynamic_viscosity
        u1       = mass_flow_rate/(rho_ref*Aflow)
        Re_c     = rho_ref * u1 * Dh_i / mu_f_ref
        # Darcy with roughness (k_s ~ eps)
        fD = (-1.8*math.log10(rev['eps'][i]/(3.7*Dh_i) + (6.9/Re_c)**1.1))**-2
        dyn_head   = 0.5*rho_ref*u1**2
        dP_segment = fD * seg_len/Dh_i * dyn_head
        if i == 0 or i == (Nseg-1):
            dP_segment += 0.8*dyn_head
        regen_press = float(Press_ref - dP_segment)     # Pa
        regen_press = max(regen_press, 1e5)             # guard: avoid ≤ 0 Pa
        regen_pressures[i] = regen_press

        # refine once
        if abs(regen_press - Press_ref)/Press_ref > 5e-2:
            Press_ref = float(regen_press)
            Press_ref = max(float(Press_ref), 1e5)
            Tc_loc = max(float(Tc_loc), 200.0)
            fuel_ref = get_ethanol_state(Tc_loc, Press_ref, phase_hint="liquid")
            rho_ref  = fuel_ref.density
            mu_f_ref = fuel_ref.dynamic_viscosity
            u1       = mass_flow_rate/(rho_ref*Aflow)
            Re_c     = rho_ref*u1*Dh_i/mu_f_ref
            fD       = (-1.8*math.log10(rev['eps'][i]/(3.7*Dh_i) + (6.9/Re_c)**1.1))**-2
            dyn_head = 0.5*rho_ref*u1**2
            dP_segment = fD * seg_len/Dh_i * dyn_head
            if i == 0 or i == (Nseg-1):
                dP_segment += 0.8*dyn_head
            regen_press = Press_ref - dP_segment

        regen_pressures[i] = regen_press

        # coolant props at regen_press
        fuel_st = Fluid(FluidsList.Ethanol)
        fuel_st = fuel_phase(Tc_loc, regen_press, fuel_st)
        rho     = fuel_st.density
        mu_f    = fuel_st.dynamic_viscosity
        lam_c   = fuel_st.conductivity
        cp_c    = fuel_st.specific_heat
        Pr_c    = cp_c * mu_f / lam_c

        # --- Mach number from area ratio at segment MIDPOINT ---
        if (i + 0.5) > throat_idx_rev:
            # upstream of throat (subsonic)
            M = ispObj.get_Chamber_MachNumber(Pc=Pc_psia, MR=OF, fac_CR=A_mid/At)
        elif (i + 0.5) < throat_idx_rev:
            # downstream (supersonic)
            M = ispObj.get_MachNumber(Pc=Pc_psia, MR=OF, eps=A_mid/At,
                                      frozen=0, frozenAtThroat=0)
        else:
            M = 1.0
        Mach_array[i] = M

        # --- Coolant-side HTC (Dittus-Boelter) ---
        u1   = mass_flow_rate / (rho * Aflow)
        Re_c = rho * u1 * Dh_i / mu_f
        # work in meters once, then reuse
        r_m = 1e-3 * np.asarray(rev['r'], dtype=float)
        x_m = 1e-3 * np.asarray(rev['x'], dtype=float)

        if i == 0:
            y1, y2, y3 = r_m[i],   r_m[i],   r_m[i+1]
            x1, x2, x3 = x_m[i],   x_m[i],   x_m[i+1]
        elif i == Nseg - 1:
            y1, y2, y3 = r_m[i], r_m[i],   r_m[i]
            x1, x2, x3 = x_m[i-1], x_m[i],   x_m[i]
        else:
            y1, y2, y3 = r_m[i-1], r_m[i],   r_m[i+1]
            x1, x2, x3 = x_m[i-1], x_m[i],   x_m[i+1]


        L    = np.sum(seg_len_array[:i+1])
        Nu   = Nu_func(Re_c, fD, Pr_c, channel_height, rev['w'][i], Dh_i, L, mu_f, mu_f, rev['x'][i], cs, ddy_func)
        alpha_c = Nu * lam_c / Dh_i

        # iterative wall solve (with cap + under-relax)
        Taw     = Taw_fc_array_rev[i]
        Twg_0   = Taw - 50.0
        tol     = 1e-4
        max_it  = 200

        # guard values
        alpha_c_eff = max(alpha_c, 1e-9)
        lambda_w_min = 1.0
        tw_eff = max(tw, 1e-4)
        base2 = max(1.0 + 0.5*(gamma - 1.0)*M*M, 1e-9)

        if isinstance(conductivity, np.ndarray):
            lambda_w = np.interp((Twg_0+Twc)/2.0, conductivity_temps, conductivity)
        else:
            lambda_w = conductivity
        lambda_w = max(lambda_w, lambda_w_min)

        h   = h_fc_array_rev[i]
        qg_i  = (Taw_fc_array_rev[i] - Tc_loc)/(1/h + 1/alpha_c_eff + tw/lambda_w)
        Twc = qg_i / alpha_c_eff + Tc_loc
        Twg_1 = Taw_fc_array_rev[i] - qg_i/h

        # store segment results (still reversed order)
        qg_fc_rev[i]       = qg_i
        Twg_fc_rev[i]      = Twg_1
        Twc_fc_rev[i]      = Twc

        Ahot_seg = 2.0 * math.pi * R_mid_m * seg_len          # m^2
        Q_seg    = qg_i * Ahot_seg                               # W (J/s)
        Tc_fc_rev[i+1] = Tc_fc_rev[i] + Q_seg / (fuel_mdot * cp_c)

    Tc_max = np.max(Tc_fc_rev)
    # After the for-loop that fills qg_fc_rev, Twg_fc_rev, Twc_fc_rev
    Twg_fc_rev[-1] = Twg_fc_rev[-2]
    Twc_fc_rev[-1] = Twc_fc_rev[-2]
    qg_fc_rev[-1]  = qg_fc_rev[-2]

    iter_count+=1

#---------------------------[FILTER OUT SPIKES DUE TO NANS/ZEROS]----------------------
# Tc_fc_array    = flip_full(median_filter(Tc_fc_rev,3, mode='reflect'))
# Twg_fc_array   = flip_full(median_filter(Twg_fc_rev, 3, mode='reflect'))
# Twc_fc_array   = flip_full(median_filter(Twc_fc_rev, 3, mode='reflect'))
# qg_fc_array    = flip_full(median_filter(qg_fc_rev, 3, mode='reflect'))
Tc_fc_array    = flip_full(Tc_fc_rev)
Twg_fc_array   = flip_full(Twg_fc_rev)
Twc_fc_array   = flip_full(Twc_fc_rev)
qg_fc_array    = flip_full(qg_fc_rev)
#%%==========================[PLOTTING]=========================

# x-axis (segment midpoints) + throat
x = x_seg_mm                              # length = len(x_full) - 1
x_throat = x_nodes_mm[throat_node_fwd]

# Forward (gas-flow) arrays
Twg   = np.asarray(Twg_fc_array).squeeze()
Twc   = np.asarray(Twc_fc_array).squeeze()
qg    = np.asarray(qg_fc_array).squeeze()
aT    = np.asarray(alphaT_fwd).squeeze()
Mach  = np.asarray(Mach_fwd).squeeze()
Tc_s  = np.asarray(Tc_fc_array).squeeze()
Pcool = np.asarray(Pcool_fwd).squeeze()

# Safety trim to the same length
N = x.size
def trim(y): return y[:N]
Twg, Twc, qg, aT, Mach, Tc_s, Pcool = map(trim, [Twg, Twc, qg, aT, Mach, Tc_s, Pcool])


## ======= Combined Temps (+Tsat) & Heat Flux with Cleaner Colours =======


# Colors (RGB -> matplotlib normalized 0–1)
RED    = (1.0, 0.0, 0.0)       # (255,0,0)
GREEN  = (0.0, 1.0, 0.0)       # (0,255,0)
TEAL   = (0.0, 0.7, 0.7)        # teal
PINK   = (1.0, 0.0, 1.0)       # (255,0,255)
BLUE   = (0.0, 0.0, 1.0)       # (0,0,255)
GREY   = (0.4, 0.4, 0.4)
ORANGE = (1.0, 0.5, 0.0)

# Boiling curve (pressure-dependent if valid; else 1 atm)
tsat_label = r'Ethanol $T_\mathrm{sat}(P)$'
if np.all(np.isfinite(Pcool)) and Pcool.size == x.size:
    Tsat_curve = ethanol_Tsat_from_P(Pcool)
else:
    Tsat_curve = np.full_like(x, 351.52)  # K
    tsat_label = r'Ethanol $T_\mathrm{sat}$ @1 atm'

# --- CORRECTED PHASE CHANGE DETECTION ---

# 1. Regenerative Channel Phase Change (Coolant Boiling)
regen_boiling_idx = None
boiling_mask = Tc_s >= Tsat_curve
if np.any(boiling_mask):
    # We want the LAST index in the forward array where boiling occurs.
    # This represents the point furthest from the injector where the transition happens.
    regen_boiling_idx = np.where(boiling_mask)[0][-1]

# 2. Film Cooling Phase Change (Film Burnout)
# This is usually the index where the liquid film mass (Gamma) hits zero.
film_phase_x = x_seg_mm[burnout_idx] if burnout_idx is not None else None

if Complete_Thermal_Analysis == True:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # --- Left axis (Temps + Tsat) ---
    l_wg,  = ax1.plot(x, Twg,  label=r'$T_{w,g}$',  color=RED,   linewidth=1.6)
    l_wc,  = ax1.plot(x, Twc,  label=r'$T_{w,c}$',  color=TEAL, linewidth=1.4, linestyle='--')
    l_tc,  = ax1.plot(x, Tc_s, label=r'$T_c$ (Regen)', color=BLUE,  linewidth=1.4, linestyle='-.')
    l_sat, = ax1.plot(x, Tsat_curve, label=tsat_label, color=PINK, linewidth=1.4, alpha=0.6, linestyle=':')

    # --- Right axis (Heat flux) ---
    r_qg,  = ax2.plot(x, qg,   label=r'$q_g$',       color=ORANGE, linewidth=1.6, alpha=0.8)

    # --- Phase Change Markers ---

    # Mark Regen Boiling
    if regen_boiling_idx is not None:
        x_boil = x[regen_boiling_idx]
        ax1.axvline(x=x_boil, color='darkblue', linestyle='--', linewidth=2,
                    label='Regen Phase Change (Boiling)')


    # Mark Film Burnout
    if film_phase_x is not None:
        ax1.axvline(x=film_phase_x, color='magenta', linestyle='-', linewidth=2,
                    label='Film Phase Change (Burnout)')


    # Throat marker
    ax1.axvline(x=x_throat, color='k', linestyle=':', linewidth=1.2, label='Throat')

    # Labels and Formatting
    ax1.set_xlabel('Axial position [mm]')
    ax1.set_ylabel('Temperature [K]')
    ax2.set_ylabel(r'Heat flux [W/m$^2$]')
    ax1.set_title('Custom Simulation Thermal Analysis')
    ax1.grid(True, alpha=0.2)

    # # Combined legend logic
    # handles, labels = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # ax1.legend(handles + h2, labels + l2, loc='upper left', frameon=True, fontsize='small')

    # plt.tight_layout()
    # plt.show()

    # Combined legend — placed outside the plot area (below)
    handles, labels = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + h2, labels + l2,
            loc='upper center', bbox_to_anchor=(0.5, -0.12),
            ncol=3, frameon=True, fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)   # make room for the legend
    plt.show()


# ===== Diagnostic 3×3 with Distinct Colours =====
Te = T0 * (1.0 + 0.5*(gamma-1.0)*Mach**2 * (Pr_g**(1/3))) / (1.0 + 0.5*(gamma-1.0)*Mach**2)
dT = Te - Twg
sigma = (0.5*(Twg/T0)*(1 + 0.5*(gamma-1)*Mach**2) + 0.5)**(-0.68) * (1 + 0.5*(gamma-1)*Mach**2)**(-0.12)
At_over_A_09 = (At / (np.pi*(r_seg_mm*1e-3)**2))**0.9
Dt_over_R_01 = ((2.0*Rt_SI) / (r_seg_mm*1e-3))**0.1
At_over_A_09 = trim(At_over_A_09); Dt_over_R_01 = trim(Dt_over_R_01)

# Define a palette of 9 distinct colours
colors = [
    (1.0, 0.0, 0.0),    # red
    (0.0, 0.6, 0.0),    # green
    (0.0, 0.0, 1.0),    # blue
    (1.0, 0.0, 1.0),    # magenta
    (1.0, 0.5, 0.0),    # orange
    (0.0, 0.7, 0.7),    # teal
    (0.6, 0.0, 0.6),    # purple
    (0.3, 0.3, 0.3),    # grey
    (0.6, 0.4, 0.0)     # brownish gold
]

if Thermal_Diagnostic_3x3 == True:
    fig, axs = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
    (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs

    ax1.plot(x, qg, color=colors[0]);      ax1.set_title(r'$q_g$  [W/m$^2$]')
    ax2.plot(x, aT, color=colors[1]);      ax2.set_title(r'$\alpha_T$  [W/m$^2$K]')
    ax3.plot(x, Mach, color=colors[2]);    ax3.set_title('Mach')
    ax4.plot(x, Te, color=colors[3]);      ax4.set_title(r'$T_e$  [K]')
    ax5.plot(x, Twg, color=colors[4]);     ax5.set_title(r'$T_{w,g}$  [K]')
    ax6.plot(x, dT, color=colors[5]);      ax6.set_title(r'$\Delta T$  [K]')
    ax7.plot(x, sigma, color=colors[6]);   ax7.set_title(r'$\sigma$  [–]')
    ax8.plot(x, At_over_A_09, color=colors[7]); ax8.set_title(r'$(A_t/A)^{0.9}$')
    ax9.plot(x, Dt_over_R_01, color=colors[8]); ax9.set_title(r'$(D_t/R)^{0.1}$')

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
        ax.axvline(x=x_throat, color='k', linestyle=':', linewidth=1.0)

    axs[-1,0].set_xlabel('Axial position [mm]')
    axs[-1,1].set_xlabel('Axial position [mm]')
    axs[-1,2].set_xlabel('Axial position [mm]')

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()


if Film_Cooling_vs_No_FC== True:
    # ===== Film Cooling Comparison Graphs =====
    def get_color(name, idx):
        if name in globals():
            return globals()[name]
        if 'colors' in globals() and len(colors) > idx:
            return colors[idx]  # your 9-color palette
        return None  # let mpl cycle

    c_Twg_no   = get_color('RED',    0)
    c_Twg_film = get_color('BLUE',   2)
    c_Twc_no   = get_color('GREY',   7)
    c_Twc_film = get_color('TEAL',   5)
    c_Tc_no    = get_color('GREEN',  1)
    c_Tc_film  = get_color('PINK',   3)
    c_qg_no    = get_color('ORANGE', 4)
    c_qg_film  = get_color('PURPLE', 6) or (0.6, 0.0, 0.6)

    # ---- optional: show qg in MW/m² (set scale=1 for W/m²) ----
    q_scale = 1e-6
    q_label = 'qg [MW/m²]' if q_scale == 1e-6 else 'qg [W/m²]'

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # (1) Twg: film vs no-film
    axs[0, 0].plot(x_seg_mm, Twg_fwd,       label='Twg (no film)',
                   color=c_Twg_no,   linewidth=1.8)
    axs[0, 0].plot(x_seg_mm, Twg_fc_array,  label='Twg (film)',
                   color=c_Twg_film, linewidth=1.8, linestyle='--')
    axs[0, 0].set_ylabel('Temperature [K]')
    axs[0, 0].set_title('Gas-side wall temperature (Twg)')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # (2) Twc: film vs no-film
    axs[0, 1].plot(x_seg_mm, Twc_fwd,       label='Twc (no film)',
                   color=c_Twc_no,   linewidth=1.8)
    axs[0, 1].plot(x_seg_mm, Twc_fc_array,  label='Twc (film)',
                   color=c_Twc_film, linewidth=1.8, linestyle='--')
    axs[0, 1].set_ylabel('Temperature [K]')
    axs[0, 1].set_title('Coolant-side wall temperature (Twc)')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # (3) Tc: film vs no-film
    axs[1, 0].plot(x_seg_mm, Tc_fwd_nodes[:-1], label='Tc (no film)',
                   color=c_Tc_no,   linewidth=1.8)
    axs[1, 0].plot(x_seg_mm, Tc_fc_array,       label='Tc (film)',
                   color=c_Tc_film, linewidth=1.8, linestyle='--')
    axs[1, 0].set_ylabel('Temperature [K]')
    axs[1, 0].set_title('Coolant bulk temperature (Tc)')
    axs[1, 0].set_xlabel('Axial position x [mm]')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # (4) qg: film vs no-film
    axs[1, 1].plot(x_seg_mm, qg_fwd * q_scale,      label='qg (no film)',
                   color=c_qg_no,   linewidth=1.8)
    axs[1, 1].plot(x_seg_mm, qg_fc_array * q_scale, label='qg (film)',
                   color=c_qg_film, linewidth=1.8, linestyle='--')
    axs[1, 1].set_ylabel(q_label)
    axs[1, 1].set_title('Hot-gas heat flux')
    axs[1, 1].set_xlabel('Axial position x [mm]')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    # throat marker if you have x_throat defined
    try:
        for ax in axs.flat:
            ax.axvline(x=x_throat, color='k', linestyle=':', linewidth=1.0)
    except Exception:
        pass

    plt.tight_layout()
    plt.show()


#%%==========================[STRESS CALCULATIONS AND PLOTTING]=====================================


contour   = r_seg_mm * 1e-3
ax_pos    = x_seg_mm * 1e-3
fuel_temp = Tc_fc_rev
regen_press = flip_full(regen_pressures)

#=========================== Conduct Temperature and Pressure Calculations ==============================================
stagnation_pressure = Pc_SI
radii = r_seg_mm * 1e-3
throat_radius = np.min(radii)
area_ratios = (radii/throat_radius)**2
mach_numbers = np.zeros_like(area_ratios)
for i in range(len(area_ratios)):
    gamma = 1.1674
    area_ratio = area_ratios[i]
    if i==0:
        machfunc = lambda M, area_ratio = area_ratio: (1/area_ratio)*(1 + (gamma-1)*M**2/2)**((gamma+1)/(2*(gamma-1))) - M * (1 + (gamma - 1)/2)**((gamma+1)/(2*(gamma-1)))
        sol = sp.optimize.root_scalar(machfunc, method = 'brentq',bracket=[0,1])
        mach_numbers[i] = sol.root if sol.converged else np.nan

    elif radii[i] <= radii[i-1]:
        machfunc = lambda M, area_ratio = area_ratio: (1/area_ratio)*(1 + (gamma-1)*M**2/2)**((gamma+1)/(2*(gamma-1))) - M * (1 + (gamma - 1)/2)**((gamma+1)/(2*(gamma-1)))
        sol = sp.optimize.root_scalar(machfunc, method = 'brentq',bracket=[0,1])
        mach_numbers[i] = sol.root if sol.converged else np.nan

    elif radii[i] >= radii[i-1]:
        machfunc = lambda M, area_ratio = area_ratio: (1/area_ratio)*(1 + (gamma-1)*M**2/2)**((gamma+1)/(2*(gamma-1))) - M * (1 + (gamma - 1)/2)**((gamma+1)/(2*(gamma-1)))
        sol = sp.optimize.root_scalar(machfunc, method = 'brentq',bracket=[1.00001,20])
        mach_numbers[i] = sol.root if sol.converged else np.nan

stagnation_pressure = stagnation_pressure*np.ones_like(mach_numbers)
pressures = stagnation_pressure*(1+(gamma-1)/2 * mach_numbers**2)**((gamma-1)/gamma)
pressures = np.divide(stagnation_pressure, (1+(gamma-1)*mach_numbers**2/2)**(gamma/(gamma-1)))

del area_ratio, area_ratios, i, sol, stagnation_pressure

#=========================== Conduct Stress Calculations =================================================================
longitudinal_thermal_stress = np.zeros_like(pressures)
avg_innerwall_temp = (Twg_fc_array + Twc_fc_array)/2
E = np.interp(avg_innerwall_temp, modulus_temps, modulus)
yield_strength = np.interp(avg_innerwall_temp, yield_temps, yield_stress)
dT = (Twg_fc_array - Twc_fc_array)

channel_dP = regen_press - pressures
if isinstance(modulus, np.ndarray):
    E = np.interp(avg_innerwall_temp, modulus_temps, modulus)
else:
    E = modulus
if isinstance(yield_stress, np.ndarray):
    yield_strength = np.interp(avg_innerwall_temp, yield_temps, yield_stress)
else:
    yield_strength = yield_stress
if isinstance(cte, np.ndarray):
    coeff_thermal_expansion = np.interp(avg_innerwall_temp, cte_temps, cte)
else:
    coeff_thermal_expansion = cte

if isinstance(conductivity, np.ndarray):
    thermal_conductivity = np.interp(avg_innerwall_temp, conductivity_temps, conductivity)
else:
    thermal_conductivity = conductivity


longitudinal_thermal_stress = E*coeff_thermal_expansion*dT
tangential_thermal_stress = (E*coeff_thermal_expansion*qg_fc_array*tw)/(2*(1-v)*thermal_conductivity)
tangential_pressure_stress = channel_dP*(channel_width/tw)**2 * 0.5
crit_longitudinal_buckling_stress = E*tw/(np.sqrt(3*(1-v**2))*(radii))
tot_tangential_stress = tangential_pressure_stress + tangential_thermal_stress
von_mises_stress = np.sqrt(tot_tangential_stress**2 + longitudinal_thermal_stress**2  - tot_tangential_stress*longitudinal_thermal_stress)
yield_sf = yield_strength/von_mises_stress

# --- PHASE CHANGE DETECTION ---

# 1. Regenerative Channel Phase Change (Coolant Boiling)
# Coolant flows exit -> injector, so we look for the first index (from the exit)
# where Tc exceeds Tsat.
regen_boiling_idx = None
boiling_mask = Tc_s >= Tsat_curve
if np.any(boiling_mask):
    # Find the first occurrence of boiling
    regen_boiling_idx = np.where(boiling_mask)[0][0]

# 2. Film Cooling Phase Change (Film Burnout)
# This is already stored in 'burnout_idx' from your liquid film loop.
film_phase_x = x_seg_mm[burnout_idx] if burnout_idx is not None else None
#%% ============================ PLOTTING (FAST, NO-LaTeX) ===============================
import matplotlib as mpl

# Hard-disable any LaTeX usage (must be before plotting)
mpl.rcParams.update({
    "text.usetex": False,
    "text.latex.preamble": "",
    "font.family": "serif",   # pick any
})

# -------- Helpers --------
def setup_3d_subplot(fig, position, title, xlabel='X (mm)', ylabel='Y (mm)', zlabel='Axial Position (mm)'):
    ax = fig.add_subplot(position, projection='3d')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    ax.set_title(title)
    return ax

def set_equal_axes(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    eps = 1e-9 if max_range == 0 else 0
    ax.set_xlim(mid_x - max_range - eps, mid_x + max_range + eps)
    ax.set_ylim(mid_y - max_range - eps, mid_y + max_range + eps)
    ax.set_zlim(mid_z - max_range - eps, mid_z + max_range + eps)

def rotate_and_plot_surfaces_fast(ax, X, Y, Z, facecolors, num_channels_to_plot):
    phis = np.linspace(0, 2*np.pi, num_channels_to_plot, endpoint=False)
    c = np.cos(phis); s = np.sin(phis)
    for ci, si in zip(c, s):
        Xr = X*ci - Y*si
        Yr = X*si + Y*ci
        ax.plot_surface(Xr, Yr, Z, facecolors=facecolors, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False, alpha=1.0)

def annotate_extreme_point_fast(ax, X, Y, Z, target_array, label_coords=None,
                                text_prefix="Value", is_max=True):
    if np.all(np.isnan(target_array)):  # nothing to show
        return
    idx = np.unravel_index((np.nanargmax if is_max else np.nanargmin)(target_array), target_array.shape)
    x0, y0, z0 = float(X[idx]), float(Y[idx]), float(Z[idx])

    if label_coords is None:
        r_plot = float(np.hypot(X, Y).max())
        lx = 1.15 * r_plot
        ly = -0.60 * r_plot
        lz = float(Z.min() + 0.35 * (Z.max() - Z.min()))
        label_coords = (lx, ly, lz)

    lx, ly, lz = label_coords
    ang_point = np.arctan2(y0, x0)
    ang_label = np.arctan2(ly, lx)
    phi = ang_label - ang_point
    c, s = np.cos(phi), np.sin(phi)
    xt = x0*c - y0*s
    yt = x0*s + y0*c

    val = float(target_array[idx])
    ax.text(lx, ly, lz, f"{text_prefix}: {val:.2f}", color='black',
            fontsize=12, fontweight='bold')

if SF_and_Temps_3D == True:
    # -------- Common mesh (make shapes consistent and light) --------
    theta_min = 0.0
    theta_max = np.deg2rad(3.8)
    n_theta   = 30
    thetas    = np.linspace(theta_min, theta_max, n_theta)

    ax_pos = np.asarray(ax_pos)
    radii  = np.asarray(radii)
    Nr = radii.size; Nt = n_theta
    assert ax_pos.size == Nr, "ax_pos must have same length as radii"

    Theta, Radius = np.meshgrid(thetas, radii)      # (Nr, Nt)
    X = Radius * np.cos(Theta)
    Y = Radius * np.sin(Theta)
    Z = np.tile(ax_pos[:, None], (1, Nt))

    # Downsample for rendering speed (tune as needed)
    DS_R = max(1, Nr // 220)     # keep ~≤220 radial samples
    DS_T = max(1, Nt // 48)      # keep ≤48 theta samples
    Xs = X[::DS_R, ::DS_T]; Ys = Y[::DS_R, ::DS_T]; Zs = Z[::DS_R, ::DS_T]

    # -------- Data fields (ensure 1D length Nr, then tile and downsample) --------
    yield_sf = np.asarray(yield_sf);           assert yield_sf.size == Nr
    Yield_SF = np.tile(yield_sf[:, None], (1, Nt))
    Yield_SF_s = Yield_SF[::DS_R, ::DS_T]
    Yield_SF_capped_s = np.clip(Yield_SF_s, 0, 12)

    firewall_temp = np.asarray(Twg_fc_array);  assert firewall_temp.size == Nr
    Firewall_Temp = np.tile(firewall_temp[:, None], (1, Nt))
    Firewall_Temp_s = Firewall_Temp[::DS_R, ::DS_T]

    # Colormaps on downsampled grids
    cmap_sf  = plt.get_cmap('jet_r')
    norm_sf  = plt.Normalize(vmin=0, vmax=12)
    faces_sf = cmap_sf(norm_sf(Yield_SF_capped_s))

    vmin_t = float(np.nanmin(Firewall_Temp_s))
    vmax_t = float(np.nanmax(Firewall_Temp_s))
    cmap_t  = plt.get_cmap('hot_r')
    norm_t  = plt.Normalize(vmin=vmin_t, vmax=vmax_t)
    faces_t = cmap_t(norm_t(Firewall_Temp_s))

    # How many channel copies to draw (visual, not physical)
    n_channels = int(n) if "n" in globals() or "n" in locals() else 8
    # num_channels_to_plot = max(4, min(n_channels, 12))   # 4–12 looks good & fast
    num_channels_to_plot = n
    # -------- Figure --------
    fig = plt.figure(figsize=(16, 8))

    # Plot 1: Yield Safety Factor
    ax1 = setup_3d_subplot(fig, 121, "Yield Safety Factor Contours")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_sf, norm=norm_sf),
                 ax=ax1, shrink=0.6, label='Yield Safety Factor (capped at 12)')
    set_equal_axes(ax1, Xs, Ys, Zs)
    ax1.invert_zaxis()     # flip vertical orientation
    rotate_and_plot_surfaces_fast(ax1, Xs, Ys, Zs, faces_sf, num_channels_to_plot)
    annotate_extreme_point_fast(ax1, Xs, Ys, Zs, Yield_SF_s, text_prefix="Minimum SF", is_max=False)
    ax1.view_init(elev=10, azim=210)

    # Plot 2: Firewall Temperature
    ax2 = setup_3d_subplot(fig, 122, "Firewall Temperature Contours")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_t, norm=norm_t),
                 ax=ax2, shrink=0.6, label='Firewall Temperature (K)')
    set_equal_axes(ax2, Xs, Ys, Zs)
    ax2.invert_zaxis()

    rotate_and_plot_surfaces_fast(ax2, Xs, Ys, Zs, faces_t, num_channels_to_plot)
    annotate_extreme_point_fast(ax2, Xs, Ys, Zs, Firewall_Temp_s, text_prefix="Max Temp. (K)", is_max=True)
    ax2.view_init(elev=10, azim=210)

    plt.tight_layout()
    plt.show()


if Stress_Plot == True:
    # -------- Other (2D) plots --------
    plt.figure()
    plt.plot(ax_pos, tangential_pressure_stress, label='Tangential Pressure Stress')
    plt.plot(ax_pos, tangential_thermal_stress,  label='Tangential Thermal Stress')
    plt.plot(ax_pos, radii * 0.5e9,linestyle='--', label='Contour')
    plt.legend()
    plt.xlabel("Axial Position (mm)")
    plt.ylabel("Value (units)")
    plt.title("Stress & Contour")
    plt.grid(alpha=0.3)
    plt.show()


plt.plot(x_seg, Nu_array)

# %% ========================== Export to RPA Format ==========================

output_filename = "Sim_Thermals.txt"

# We need to explicitly calculate the final coolant velocity (wc) and density (rho)
wc_export = np.zeros(N)
rho_export = np.zeros(N)

# Calculate forward flow areas
# 'w' is already the forward channel width array, 'channel_height' is constant
Aflow_fwd = channel_height * w[:N] 
mass_flow_rate_ch = fuel_mdot / n

for i in range(N):
    try:
        # Pcool is in Pa, Tc_s is in K
        state = get_ethanol_state(Tc_s[i], Pcool[i], phase_hint="liquid")
        rho_export[i] = state.density
        wc_export[i] = mass_flow_rate_ch / (rho_export[i] * Aflow_fwd[i])
    except Exception:
        # Fallback if state derivation fails at boundaries
        rho_export[i] = 0.0
        wc_export[i] = 0.0

with open(output_filename, 'w') as f:
    # Write exact RPA headers
    f.write("#" + "-"*158 + "\n")
    f.write("# Results of thermal analysis\n")
    f.write("#" + "-"*158 + "\n")
    f.write("#Location, \t Radius, mm\t Conv. heat\t Conv. heat\t Rad. heat \t Total heat\t     Twg, K\t     Twi, K\t     Twc, K\t      Tc, K\t    pc, MPa\t    wc, m/s\t rho, kg/m^3\t                       Comments\n")
    f.write("#        mm\t           \t  coeff., k\t  flux, kW/\t flux, kW/m\t  flux, kW/\t           \t           \t           \t           \t           \t           \t            \n")
    f.write("#          \t           \t    W/m^2-K\t        m^2\t         ^2\t        m^2\t           \t           \t           \t           \t           \t           \t            \n")
    f.write("#" + "-"*158 + "\n")
    
    # Write data rows
    for i in range(N):
        loc = x[i]
        rad = r_seg_mm[i]
        
        # Convert units to match RPA kW scale
        h_conv = aT[i] / 1000.0
        q_conv = qg[i] / 1000.0
        q_rad = 0.0  # Not explicitly tracked in the main script, defaulting to 0
        q_tot = q_conv + q_rad
        
        t_wg = Twg[i]
        t_wi = Twg[i]  # Inner wall temp (same as gas wall since no TBC is defined)
        t_wc = Twc[i]
        t_c = Tc_s[i]
        
        p_c = Pcool[i] / 1e6  # Convert Pa back to MPa for RPA standard
        v_c = wc_export[i]
        rho_c = rho_export[i]
        
        # Determine comment string based on film cooling presence
        comment = "regenerative cooling, opposite flow"
        if film_phase_x is not None and loc <= film_phase_x:
            comment += "; film coolant injection"
             
        # Format columns with explicit spacing to match RPA text output
        row_str = f"{loc:10.4f}\t{rad:11.4f}\t{h_conv:11.4f}\t{q_conv:11.4f}\t{q_rad:11.4f}\t{q_tot:11.4f}\t{t_wg:11.4f}\t{t_wi:11.4f}\t{t_wc:11.4f}\t{t_c:11.4f}\t{p_c:11.4f}\t{v_c:11.4f}\t{rho_c:11.4f}\t {comment}\n"
        f.write(row_str)



# %% ============================ Results summary ============================
def _section(title):
    print(f"\n{'=' * 8} {title} {'=' * (60 - len(title))}")

def _loc(arr, fn):
    """Index and axial position [mm] of the min/max of arr."""
    idx = int(fn(arr))
    return idx, x_seg_mm[idx]

# --- Performance ---
_section("PERFORMANCE")
print(f"Propellants          : {ox} / {fuel}, O/F = {OF:.2f}")
print(f"Chamber pressure     : {Chamber_Pressure:.1f} bar  (feed {Feed_Pressure:.1f} bar)")
print(f"Thrust (SL, actual)  : {F_check_SL:.1f} N  (target {Thrust_req:.0f} N)")
print(f"Isp SL / vac (ideal) : {ISP_SL_act:.1f} s  /  {ISP_vac_ideal:.1f} s")
print(f"c* ideal / actual    : {cstar_ideal:.1f} / {cstar_act:.1f} m/s")
print(f"Cf SL ideal / actual : {Cf_SL_ideal:.3f} / {Cf_SL_act:.3f}")
print(f"Chamber temp (T0)    : {T0:.0f} K")
print(f"Mass flow            : {mdot_act:.3f} kg/s total "
      f"({ox_mdot:.3f} ox, {fuel_mdot:.3f} fuel, {fuel_mdot*fcp:.3f} film)")

# --- Geometry ---
_section("GEOMETRY")
print(f"Throat radius / dia  : {Rt_mm:.2f} mm / {2*Rt_mm:.2f} mm")
print(f"Chamber radius       : {Rc_mm:.2f} mm  (CR = {CR:.1f})")
print(f"Exit radius          : {Re_mm:.2f} mm  (area ratio = {Area_Ratio:.2f})")
print(f"Cylinder length      : {Lcyl_mm:.1f} mm  (L* = {Lstar_target_mm:.0f} mm)")
print(f"Total length         : {x_nodes_mm.max() - x_nodes_mm.min():.1f} mm  "
      f"(throat at x = {x_throat:.1f} mm)")
print(f"Material / wall      : {Material}, {tw*1e3:.2f} mm wall")

# --- Cooling channels ---
_section("COOLING CHANNELS")
print(f"Channels             : {n} x {channel_height*1e3:.2f} mm tall")
print(f"Channel width        : {np.min(channel_width)*1e3:.2f} mm (min) to "
      f"{np.max(channel_width)*1e3:.2f} mm (max)")
print(f"Hydraulic diameter   : {np.min(Dh)*1e3:.2f} mm (min) to {np.max(Dh)*1e3:.2f} mm (max)")
print(f"Coolant Reynolds no. : {np.min(Re_array):.3g} to {np.max(Re_array):.3g}")

# --- Coolant ---
_section("COOLANT (ETHANOL)")
dP_regen = feed_press - regen_pressures[-2]
print(f"Inlet / outlet temp  : {tc_array[0]:.1f} K / {Tc_fc_rev[-1]:.1f} K")
print(f"Max bulk temp        : {np.max(Tc_s):.1f} K")
print(f"Regen pressure drop  : {dP_regen/1e5:.2f} bar "
      f"({100*dP_regen/feed_press:.1f}% of feed)")
print(f"Injector dP margin   : {(feed_press - dP_regen - Pc_SI)/1e5:.2f} bar "
      f"(coolant outlet P minus chamber P)")
if boiling_mask.any():
    print(f"Bulk boiling         : YES, x = {x[boiling_mask].min():.1f} to {x[boiling_mask].max():.1f} mm")
else:
    print("Bulk boiling         : none")

# --- Film cooling ---
_section("FILM COOLING")
final_resid = abs(Tc_fc_rev[-1] - T_fcool_array[0]) / T_fcool_array[0]
print(f"Film fraction        : {fcp*100:.1f}% of fuel")
if burnout_idx == 0:
    print("Film status          : flashes to vapour at injector face")
elif burnout_idx >= throat_idx:
    print(f"Film status          : survives to throat (x = {x_seg_mm[throat_idx]:.1f} mm)")
else:
    print(f"Film burnout         : x = {film_phase_x:.1f} mm "
          f"({film_phase_x / x_seg_mm[throat_idx] * 100:.0f}% of the way to the throat)")
print(f"Coolant T iteration  : {iter_count} iterations, residual {final_resid:.2e}")

# --- Thermal ---
_section("THERMAL")
i_twg, x_twg = _loc(Twg_fc_array, np.argmax)
i_qg,  x_qg  = _loc(qg_fc_array,  np.argmax)
i_twc, x_twc = _loc(Twc_fc_array, np.argmax)
print(f"Max gas-side wall T  : {Twg_fc_array[i_twg]:.0f} K at x = {x_twg:.1f} mm "
      f"(no film: {np.max(Twg_fwd):.0f} K)")
print(f"Max coolant-side T   : {Twc_fc_array[i_twc]:.0f} K at x = {x_twc:.1f} mm")
print(f"Peak heat flux       : {qg_fc_array[i_qg]/1e6:.2f} MW/m^2 at x = {x_qg:.1f} mm "
      f"(no film: {np.max(qg_fwd)/1e6:.2f})")
print(f"Total heat to coolant: {np.sum(qg_fc_array * 2*np.pi*radii * np.gradient(ax_pos))/1e3:.1f} kW")

# --- Structural ---
_section("STRUCTURAL")
i_sf, x_sf = _loc(yield_sf, np.nanargmin)
i_vm, x_vm = _loc(von_mises_stress, np.nanargmax)
print(f"Min yield SF         : {yield_sf[i_sf]:.2f} at x = {x_sf:.1f} mm "
      f"(Twg there {Twg_fc_array[i_sf]:.0f} K)")
print(f"Max von Mises        : {von_mises_stress[i_vm]/1e6:.1f} MPa at x = {x_vm:.1f} mm")
print(f"Max thermal / pressure tangential stress: "
      f"{np.nanmax(tangential_thermal_stress)/1e6:.1f} / "
      f"{np.nanmax(tangential_pressure_stress)/1e6:.1f} MPa")

# --- Warnings ---
warnings_list = []
if np.nanmin(yield_sf) < 1.0:
    warnings_list.append(f"Yield SF below 1 ({np.nanmin(yield_sf):.2f}) at x = {x_sf:.1f} mm")
if boiling_mask.any():
    warnings_list.append("Coolant bulk temperature exceeds Tsat somewhere in the channels")
floor_hits = int(np.sum(regen_pressures[:-1] <= 1.0001e5))
if floor_hits:
    warnings_list.append(f"Coolant pressure hit the 1 bar floor in {floor_hits} segments "
                         f"(feed pressure too low / dP too high)")
if final_resid > 1e-4:
    warnings_list.append(f"Coolant temperature loop did not converge (residual {final_resid:.2e})")
if F_check_SL and abs(F_check_SL - Thrust_req) / Thrust_req > 0.01:
    warnings_list.append("Thrust deviates from target by more than 1%")

_section("WARNINGS")
if warnings_list:
    for w_msg in warnings_list:
        print(f"! {w_msg}")
else:
    print("None")
print("=" * 70)

print(f"Exported thermal results to {output_filename}")