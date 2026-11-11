#%% -------------------------[HELPER FUNCTIONS AND BOOKKEEPING]------------------------

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
from scipy.ndimage import median_filter
import scipy as sp

def ethanol_Tsat_from_P(P_pa):
    P_mmhg = np.asarray(P_pa) / 133.322368
    A1,B1,C1 = 8.20417, 1642.89, 230.3
    A2,B2,C2 = 7.68117, 1332.04, 199.200
    logP = np.log10(P_mmhg)
    T1_C = B1/(A1 - logP) - C1
    T2_C = B2/(A2 - logP) - C2
    T_C  = np.where(T1_C <= 78.3, T1_C, T2_C)
    return T_C + 273.15

#%% -------------------------[UNITS: RocketCEA -> SI multipliers]------------------------
_degR_to_K         = 5.0/9.0          # K per degR
_ftps_to_mps       = 0.3048           # m/s per ft/s
_BTUlbmR_to_JkgK   = 4186.8           # J/(kg·K) per (BTU/lbm·°R)
_millipoise_to_Pas = 1e-4             # Pa·s per millipoise
_mcalcmKs_to_WmK   = 418.4            # W/(m·K) per mcal/(cm·K·s)

#%%==========================[USER INPUTS]=========================
ox               = 'N2O'
fuel             = 'Ethanol'
Material         = 'AlSi10Mg'  # 'Inconel', 'AlSi10Mg', 'ABD900'

Chamber_Pressure = 30 # bar
Feed_Pressure    = 40.0   # bar
Thrust_req       = 5000.0 # N
OF               = 3
CR               = 6    # contraction ratio Ac/At
Lstar_target_mm  = 750.0  # mm (chamber only)
tw               = 0.7e-3 # wall thickness [m]
channel_height   = 1.5e-3 # [m]
arc_angle        = 3.8    # degrees
n                = 50     # channels
fcp              = 0.275   # film-cooling percentage 

#Graphs 
Complete_Thermal_Analysis = True
Thermal_Diagnostic_3x3    = True
Film_Cooling_vs_No_FC     = True
SF_and_Temps_3D           = False #Looks Great, Takes like 5 Minutes to run, proceed with caution
Stress_Plot               = True
# Efficiencies
eta_cstar = 0.959
eta_Cf    = 0.923

#%%==========================[CEA / IDEAL PERFORMANCE]=========================
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

#%%==========================[GEOMETRY IN mm]=========================
def rao_nozzle(Rt, epsilon):
    """
    Generate a Rao bell nozzle contour and compute the divergent section volume.

    The profile is generated as:
    1. A short circular arc downstream of the throat (0.382*Rt radius).
    2. A parabolic bell section fitted between the arc end and the nozzle exit.

    Parameters:
    ----------
    Rt : float
        Throat radius [m]
    epsilon : float
        Nozzle expansion ratio (Ae / At)

    Returns:
    -------
    x_array : ndarray
        Axial coordinates of nozzle contour [m]
    y_array : ndarray
        Radius coordinates of nozzle contour [m]
    volume : float
        Internal volume of divergent section [m³]
    """
    import numpy as np
    import math

    # Approximate Rao bell length as 80% of equivalent 15° conical nozzle
    nozzle_length = 0.8 * ((math.sqrt(epsilon) - 1) * Rt / math.tan(math.radians(15)))

    # Wall angles (in radians) at:
    # - theta_n: start of parabolic bell (end of throat arc)
    # - theta_e: nozzle exit
    theta_n = math.radians(24)
    theta_e = math.radians(12)
    Rn = 0.382*Rt

    # -------------------
    # 1) Circular arc at throat
    # -------------------
    x_change = 0.382 * Rt * math.sin(theta_n)  # axial length of circular arc
    x_circ = np.linspace(0, x_change, 5)       # axial coordinate points along arc
    theta_circ = np.arcsin(x_circ / (0.382 * Rt))  # corresponding arc angles
    y_circ = Rt * (1.382 - 0.382 * np.cos(theta_circ))  # radial positions along arc

    # -------------------
    # 2) Parabolic section
    # -------------------
    # Start point (N) of parabola
    Nx = 0.382 * Rt * math.cos(theta_n - math.radians(90))
    Ny = 0.382 * Rt * math.sin(theta_n - math.radians(90)) + 0.382 * Rt + Rt

    # End point (E) of parabola (exit)
    Ex = nozzle_length
    Ey = np.sqrt(epsilon) * Rt

    # Slopes at start (m1) and end (m2)
    m1, m2 = math.tan(theta_n), math.tan(theta_e)

    # Intersection point (Q) of tangents from N and E
    C1, C2 = Ny - m1 * Nx, Ey - m2 * Ex
    Qx = (C2 - C1) / (m1 - m2)
    Qy = (m1 * C2 - m2 * C1) / (m1 - m2)

    # Quadratic Bezier curve parameter t
    t = np.linspace(0, 1, 96)
    x_para = (1 - t)**2 * Nx + 2 * (1 - t) * t * Qx + t**2 * Ex
    y_para = (1 - t)**2 * Ny + 2 * (1 - t) * t * Qy + t**2 * Ey

    # -------------------
    # 3) Combine arc + parabola
    # -------------------
    x_array = np.concatenate((x_circ, x_para[1:]))
    y_array = np.concatenate((y_circ, y_para[1:]))

    # -------------------
    # 4) Volume integration (solid of revolution)
    # -------------------
    volume = 0
    for i in range(len(x_array) - 1):
        volume += math.pi * y_array[i]**2 * (x_array[i+1] - x_array[i])

    return x_array, y_array, volume, Rn, theta_n, theta_e
def convergent_sizing(CR, Rt):
    """
    Generate a convergent section contour with:
    - Upstream chamber fillet R2 (fraction of maximum possible R2)
    - Straight conical section at chosen half-angle b
    - Upstream throat fillet R1

    Parameters:
    ----------
    CR : float
        Contraction ratio (Ac / At)
    Rt : float
        Throat radius [m]

    Returns:
    -------
    x_array : ndarray
        Axial coordinates of convergent contour [m]
    y_array : ndarray
        Radius coordinates of convergent contour [m]
    volume : float
        Internal volume of convergent section [m³]
    """
    import math
    import numpy as np

    # Upstream throat fillet (empirical ~1.5 * Rt)
    R1 = 1.5 * Rt

    # Convergent half-angle b (deg → rad)
    b = math.radians(30)

    # Chamber radius from contraction ratio
    Rc = np.sqrt(CR) * Rt

    # Maximum possible R2 before it touches R1 (two-arc blend case)
    R2max = (Rc - Rt) / (1 - math.cos(b)) - R1

    # Select fraction of maximum R2 (empirical choice: 0.35)
    R2 = R2max * 0.5

    # -------------------
    # 1) R2 arc from chamber wall to straight cone
    # -------------------
    theta2 = np.linspace(0, b, 40)
    x2_array = R2 * np.sin(theta2)
    y2_array = Rc - R2 + R2 * np.cos(theta2)

    # -------------------
    # 2) R1 arc from straight cone into throat
    # -------------------
    theta1 = np.linspace(b, 0, 40)
    x1_array = R1 * math.sin(b) - R1 * np.sin(theta1)
    y1_array = Rt + R1 - R1 * np.cos(theta1)

    # -------------------
    # 3) Straight cone segment between arcs
    # -------------------
    dely = y2_array[-1] - y1_array[0]       # radial drop between arcs
    delx = dely / math.tan(b)                # axial length of straight section
    xl_array = np.linspace(np.max(x2_array), np.max(x2_array) + delx, 24)
    yl_array = np.linspace(np.min(y2_array), np.min(y2_array) - dely, 24)

    # Shift R1 arc forward in x so it starts after straight section
    x1_array += np.max(xl_array)

    # -------------------
    # 4) Combine arcs + straight
    # -------------------
    x_array = np.concatenate((x2_array, xl_array[1:-1], x1_array))
    y_array = np.concatenate((y2_array, yl_array[1:-1], y1_array))

    # -------------------
    # 5) Volume integration (solid of revolution)
    # -------------------
    volume = 0
    for i in range(len(x_array) - 1):
        volume += math.pi * y_array[i]**2 * (x_array[i+1] - x_array[i])

    return x_array, y_array, volume, R1, R2, b
def cylinder_sizing(Lstar, other_volume, Rt, CR):
    """
    Size the cylindrical chamber section so total L* requirement is met.

    L* is the characteristic length: total chamber volume (including convergent)
    divided by throat area.

    Parameters:
    ----------
    Lstar : float
        Target characteristic length [m]
    other_volume : float
        Volume already present (convergent + divergent) [m³]
    Rt : float
        Throat radius [m]
    CR : float
        Contraction ratio (Ac / At)

    Returns:
    -------
    x_array : ndarray
        Axial coordinates of cylinder [m]
    y_array : ndarray
        Radius coordinates of cylinder [m] (constant)
    """
    import math
    import numpy as np

    # Required total chamber volume from L* target
    required_volume = Lstar * math.pi * Rt**2

    # Remaining volume to be filled by cylinder
    Volume = required_volume - other_volume

    # Length of cylinder = volume / chamber cross-section
    Length = Volume / (CR * math.pi * Rt**2)

    # Coordinates of straight cylinder
    x_array = np.linspace(0, Length, 100)
    y_array = np.ones_like(x_array) * Rt * math.sqrt(CR)

    return x_array, y_array, Length


Rt_mm = Rt_SI * 1e3
Re_mm = math.sqrt(Area_Ratio) * Rt_mm
Rc_mm = math.sqrt(CR) * Rt_mm

# Bell (Rao) nozzle
xn, yn, vn_mm3, Rn_mm, theta_n, theta_e = rao_nozzle(Rt_mm, Area_Ratio)

# Convergent (arc–cone–arc)
xc, yc, vc_mm3, R1_mm, R2_mm, b_rad = convergent_sizing(CR, Rt_mm)

# Cylinder sized by L*
xcyl, ycyl, Lcyl_mm = cylinder_sizing(Lstar_target_mm, vc_mm3, Rt_mm, CR)

# Stitch sections axially
xc   = xc + np.max(xcyl)
xn   = xn + np.max(xc)
x_nodes = np.concatenate((xcyl, xc[:-1], xn))
y_nodes = np.concatenate((ycyl, yc[:-1], yn))

x_seg   = (x_nodes[1:] - x_nodes[:-1])/2 + x_nodes[:-1]
y_seg   = (y_nodes[1:] - y_nodes[:-1])/2 + y_nodes[:-1]
# Throat index in forward arrays
throat_node_fwd = np.argmin(y_nodes)

#%%==========================[MATERIAL PROPERTIES]=========================
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

#%%==========================[REGEN PREP]=========================
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

#%%==========================[THERMAL ARRAYS]=========================
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

#%%==========================[MAIN BARTZ LOOP (reversed coolant direction)]=========================

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

def Nu_func(Re, f, Pr, c_height, c_width, Dh, L, mu_c, mu_tw, y1, y2, y3, x1, x2, x3):
    """
    Returns local Nusselt number for a regen channel with optional curvature enhancement.
    Inputs:
      Re, f, Pr        : Reynolds (local), Darcy friction factor (local), Prandtl (bulk)
      c_height, c_width: channel rectangle (a = larger side, b = smaller side)
      Dh, L            : hydraulic diameter, local distance from coolant inlet to this station
      mu_c, mu_tw      : dynamic viscosity at bulk and at coolant-side wall (for property correction)
      (x1,y1),(x2,y2),(x3,y3): three successive centerline points to estimate local radius of curvature
    """
    # ---------------- Radius of curvature (through 3 points) ----------------
    # Guard against collinearity / tiny triangle area
    denom = 2.0 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    s12 = np.hypot(x1-x2, y1-y2)
    s23 = np.hypot(x2-x3, y2-y3)
    s31 = np.hypot(x3-x1, y3-y1)
    if denom < 1e-12:
        Rc = np.inf          # effectively straight
    else:
        Rc = (s12 * s23 * s31) / denom

    # Dean number (curvature); safe if Rc is large
    De = Re * np.sqrt(Dh / max(2.0 * Rc, 1e-12))

    # ---------------- Helpers ----------------
    # Finite-length correction uses local x=L; cap near inlet
    L_eff = max(L, 3.0 * Dh)

    # Thermal entrance length (laminar estimate)
    xth = 0.05 * Re * Pr * Dh  # only meaningful if flow is laminar; harmless otherwise

    # Aspect ratio handling for fully developed laminar (Shah & London, H1 ~ uniform q'')
    a, b = (c_height, c_width) if c_height >= c_width else (c_width, c_height)
    ar_grid = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)
    Nu_fd_grid = np.array([3.608, 4.123, 5.331, 6.490], dtype=float)  # based on Dh, H1
    Nu_fd_rect = np.interp(a / max(b, 1e-12), ar_grid, Nu_fd_grid)

    # Property corrections 
    wall_corr_turb = (mu_c / max(mu_tw, 1e-12))**0.11   # stand-in for (Pr/Prw)^0.11 if Prw unavailable
    wall_corr_lam  = (mu_c / max(mu_tw, 1e-12))**0.14   # Sieder–Tate style for laminar

    # Curvature multipliers (defaults you can tune)
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
        # Turbulent side
        Nu_base = Nu_gnielinski(Re, Pr, f)
        len_corr = 1.0 + (Dh / L_eff)**(2.0/3.0)
        Nu_turb  = Nu_base * len_corr * wall_corr_turb
        Nu_turb  = Nu_turb * (1.0 + C_turb * (De**m_turb))

        # Laminar side (pick developing vs fully-developed by local entrance criterion)
        if L_eff > xth:
            Nu_lam = Nu_fd_rect * wall_corr_lam
        else:
            Nu_lam = Nu_hausen(Re, Pr, Dh, L_eff) * wall_corr_lam
        Nu_lam  = Nu_lam * (1.0 + C_lam * (De**m_lam))

        # Smooth blend (no bitwise ops!)
        gamma = (Re - 2300.0) / (4000.0 - 2300.0)
        gamma = min(max(gamma, 0.0), 1.0)
        Nu = (1.0 - gamma) * Nu_lam + gamma * Nu_turb

    else:
        # Re <= 2300: Laminar branch
        if L_eff > xth:
            # Fully developed laminar in rectangular duct
            Nu_lam = Nu_fd_rect * wall_corr_lam
        else:
            # Thermally developing laminar (Hausen)
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
    fuel_ref = Fluid(FluidsList.Ethanol).with_state(Input.pressure(Press_ref), Input.temperature(Tc_loc - 273.15))
    fuel_ref = fuel_phase(Tc_loc, Press_ref, fuel_ref)
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
        fuel_ref = Fluid(FluidsList.Ethanol).with_state(Input.pressure(Press_ref), Input.temperature(Tc_loc - 273.15))
        fuel_ref = fuel_phase(Tc_loc, Press_ref, fuel_ref)
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
    fuel_st = Fluid(FluidsList.Ethanol).with_state(Input.pressure(regen_press), Input.temperature(Tc_loc - 273.15))  
    fuel_st = fuel_phase(Tc_loc, regen_press, fuel_st)
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
    Nu   = Nu_func(Re_c, fD, Pr_c, channel_height, rev['w'][i], Dh_i, L, mu_f, mu_f, y1, y2, y3, x1, x2, x3)
    Nu_array[i] = Nu
    Re_array[i] = Re_c
    alpha_c = Nu * lam_c / Dh_i
    
   # --- Bartz alpha_T using MIDPOINT geometry ratios ---
    At_over_A = At / A_seg
    Dt_over_R = (2.0*Rt) / (rev['r'][i] *1e-3)
    
    # iterative wall solve (with cap + under-relax)
    Twg_0   = Te - 50.0
    tol     = 1e-4
    max_it  = 200
    relax   = 0.45
    
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



#%%==========================[FILM COOLING]=========================
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
for i in range(throat_idx):
    Tsat             = ethanol_Tsat_from_P(pressure_array[i])
    T_fcool_array[i] = T_fcoolant
    coolant_ref      = Fluid(FluidsList.Ethanol).with_state(Input.pressure(pressure_array[i]), Input.temperature(T_fcoolant - 273.15))
    cp_cool_liq      = coolant_ref.specific_heat  # J/(kg·K) from PyFluids
    h0               = alphaT_fwd[i]
    q_conv0          = h0 * (Te_fwd[i] - T_fcoolant)
    mdotvap          = q_conv0/max(ethanol_latent_heat_vap,1.0)

    wet_P            = 2 *np.pi * r_seg_mm[i] *1e-3
        
    DeltaT = max(Te_fwd[i] - T_fcoolant, 0.0)                 # Te_fwd is your Taw(core)
    H      = (cp_g_array[i] * Km_array[i] / ethanol_latent_heat_vap) * DeltaT
    h      = h0 * np.log1p(H) / max(H, 1e-12)                 # correction outside the log
    
    h_corr_array[i] = h
    
    Gamma_array[i] = Gamma
    
    # Use per-span heat: (W/m^2)*m -> W/m  (no perimeter here)
    dx_seg = dx_at(i)  # or your unified dx
    q_to_film_per_span = qg_fwd[i] * dx_seg
    
    # Liquid film temperature rise (per-span energy balance)
    dT_liq      = q_to_film_per_span / max(Gamma * cp_cool_liq, 1e-12)
    T_fcoolant += dT_liq
    
    # When at saturation: evaporate film (per-span mass removal)
    if T_fcoolant >= Tsat - 1e-9:
        T_fcoolant = Tsat
        # mdotvap is kg/(m^2·s); multiply by dx [m] -> kg/(m·s) per span
        dGamma = - mdotvap * dx_seg
        Gamma  = max(Gamma + dGamma, 1e-12)
        if Gamma < 1e-8:
            burnout_idx = i
            break

    
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
    ethanol_gas = Fluid(FluidsList.Ethanol).with_state(
        Input.pressure(pressure_array[i]),
        Input.temperature(Taw_fc_g[j] - 273.15)
    )
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
    fuel_ref = Fluid(FluidsList.Ethanol).with_state(Input.pressure(Press_ref), Input.temperature(Tc_loc - 273.15))
    fuel_ref = fuel_phase(Tc_loc, Press_ref, fuel_ref)
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
        fuel_ref = Fluid(FluidsList.Ethanol).with_state(Input.pressure(Press_ref), Input.temperature(Tc_loc - 273.15))
        fuel_ref = fuel_phase(Tc_loc, Press_ref, fuel_ref)
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
    Nu   = Nu_func(Re_c, fD, Pr_c, channel_height, rev['w'][i], Dh_i, L, mu_f, mu_f, y1, y2, y3, x1, x2, x3)
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


#%%--------------------------[RUN CODE UNTIL COOLANT TEMP CONVERGES]----------------------------------------
iter_count = 0
while abs(Tc_fc_rev[-1] - T_fcool_array[0])/T_fcool_array[0] >1e-4 and iter_count < 25:
    #-----------------------------------[LIQUID FILM]--------------------------------------------------
    #Initially, since film cooling is injected at x=0
    Gamma_init  = (fuel_mdot*fcp)/(2*np.pi*r_seg_mm[0]*1e-3) #Initial Mass flow of coolant per unit circumference
    T_fcoolant  = Tc_fc_rev[-1]
    dT_liq      = 0
    Gamma       = Gamma_init
    burnout_idx = None
    for i in range(throat_idx):
        Tsat             = ethanol_Tsat_from_P(pressure_array[i])
        T_fcool_array[i] = T_fcoolant
        coolant_ref      = Fluid(FluidsList.Ethanol).with_state(Input.pressure(pressure_array[i]), Input.temperature(T_fcoolant - 273.15))
        cp_cool_liq      = coolant_ref.specific_heat  # J/(kg·K) from PyFluids
        h0               = alphaT_fwd[i]
        q_conv0          = h0 * (Te_fwd[i] - T_fcoolant)
        mdotvap          = q_conv0/max(ethanol_latent_heat_vap,1.0)
    
        wet_P            = 2 *np.pi * r_seg_mm[i] *1e-3
            
        DeltaT = max(Te_fwd[i] - T_fcoolant, 0.0)                 # Te_fwd is your Taw(core)
        H      = (cp_g_array[i] * Km_array[i] / ethanol_latent_heat_vap) * DeltaT
        h      = h0 * np.log1p(H) / max(H, 1e-12)                 # correction outside the log
        
        h_corr_array[i] = h
        
        Gamma_array[i] = Gamma
        q_to_film   = qg_fwd[i] * wet_P * dx_at(i)  # W/m² * m * m = J/m (per circumference)
        dT_liq      = q_to_film / max(Gamma*cp_cool_liq, 1e-12)
        T_fcoolant += dT_liq
        if T_fcoolant >= Tsat - 1e-9:
            T_fcoolant = Tsat
            dGamma = - mdotvap * wet_P * dx_at(i)           # kg/s/m (per circumference)
            Gamma  = Gamma + dGamma
            if Gamma < 1e-8:
                burnout_idx = i
                break
        
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
        ethanol_gas = Fluid(FluidsList.Ethanol).with_state(
            Input.pressure(pressure_array[i]),
            Input.temperature(Taw_fc_g[j] - 273.15)
        )
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
        fuel_ref = Fluid(FluidsList.Ethanol).with_state(Input.pressure(Press_ref), Input.temperature(Tc_loc - 273.15))
        fuel_ref = fuel_phase(Tc_loc, Press_ref, fuel_ref)
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
            fuel_ref = Fluid(FluidsList.Ethanol).with_state(Input.pressure(Press_ref), Input.temperature(Tc_loc - 273.15))
            fuel_ref = fuel_phase(Tc_loc, Press_ref, fuel_ref)
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
        Nu   = Nu_func(Re_c, fD, Pr_c, channel_height, rev['w'][i], Dh_i, L, mu_f, mu_f, y1, y2, y3, x1, x2, x3)
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


# ======= Combined Temps (+Tsat) & Heat Flux with Cleaner Colours =======


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

if Complete_Thermal_Analysis == True:
    fig, ax1 = plt.subplots(figsize=(8.0, 4.8))
    ax2 = ax1.twinx()
    # --- Left axis (Temps + Tsat) ---
    l_wg,  = ax1.plot(x, Twg,  label=r'$T_{w,g}$',  color=RED,   linewidth=1.6, alpha=0.95)
    l_wc,  = ax1.plot(x, Twc,  label=r'$T_{w,c}$',  color=TEAL, linewidth=1.4, alpha=0.9,  linestyle='--')
    l_tc,  = ax1.plot(x, Tc_s, label=r'$T_c$',      color=BLUE,  linewidth=1.4, alpha=0.9,  linestyle='-.')
    l_sat, = ax1.plot(x, Tsat_curve, label=tsat_label, color=PINK, linewidth=1.4, alpha=0.85, linestyle=':')
    
    # --- Right axis (Heat flux) ---
    r_qg,  = ax2.plot(x, qg,   label=r'$q_g$',       color=ORANGE, linewidth=1.6, alpha=0.8)
    
    # --- Nozzle contour overlay (scaled into temp axis range) ---
    try:
        Tmin = np.nanmin([Twg.min(), Twc.min(), Tc_s.min(), Tsat_curve.min()])
        Tmax = np.nanmax([Twg.max(), Twc.max(), Tc_s.max(), Tsat_curve.max()])
        dr = (r_seg_mm - np.nanmin(r_seg_mm)) / max(1e-12, (np.nanmax(r_seg_mm) - np.nanmin(r_seg_mm)))
        band = 0.25*(Tmax - Tmin)
        baseline = Tmin - 0.15*(Tmax - Tmin)
        contour_scaled = baseline + band * dr
        l_cnt, = ax1.plot(x, contour_scaled, color=GREY, linewidth=1.0, alpha=0.5,
                          linestyle='-', label='Nozzle contour (scaled)')
    except Exception:
        l_cnt = None
    
    # Throat marker
    ax1.axvline(x=x_throat, color='k', linestyle=':', linewidth=1.0, label='Throat')
    
    # Labels, title, grid
    ax1.set_xlabel('Axial position [mm]')
    ax1.set_ylabel('Temperature [K]')
    ax2.set_ylabel(r'Heat flux [W/m$^2$]')
    ax1.set_title('Temperatures + Boiling Point (left) & Heat Flux (right)')
    ax1.grid(True, alpha=0.25)
    
    # Combined legend
    handles = [l_wg, l_wc, l_tc, l_sat, r_qg]
    if l_cnt is not None: handles.append(l_cnt)
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='upper left', frameon=True)
    
    plt.tight_layout()
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
# ============================ PLOTTING (FAST, NO-LaTeX) ===============================
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

