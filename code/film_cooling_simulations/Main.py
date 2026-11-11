# %% ------------------------- Imports and helpers -------------------------
import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage
import scipy.optimize
from scipy.interpolate import CubicSpline
from rocketcea.cea_obj import CEA_Obj
from pyfluids import Fluid, FluidsList, Input, Phases
from CoolProp.CoolProp import PropsSI

mpl.rcParams.update({"text.usetex": False, "text.latex.preamble": "", "font.family": "serif"})

# Ethanol critical point (approx.) and pressure floor used to keep property calls sane
_ETH_Tc = 514.0        # K
_ETH_Pc = 6.38e6       # Pa
P_FLOOR = 1e5          # Pa


def ethanol_Tsat_from_P(P_pa):
    """Antoine-equation saturation temperature of ethanol [K] (scalar or array input in Pa)."""
    P_mmhg = np.asarray(P_pa) / 133.322368
    A1, B1, C1 = 8.20417, 1642.89, 230.3
    A2, B2, C2 = 7.68117, 1332.04, 199.200
    logP = np.log10(P_mmhg)
    T1_C = B1 / (A1 - logP) - C1
    T2_C = B2 / (A2 - logP) - C2
    T_C = np.where(T1_C <= 78.3, T1_C, T2_C)
    return T_C + 273.15


def ethanol_hfg(P_pa):
    """Latent heat of vaporisation of ethanol [J/kg] at pressure P [Pa] (CoolProp)."""
    P = min(max(float(P_pa), 1e5), 0.98 * _ETH_Pc)
    return PropsSI('H', 'P', P, 'Q', 1, 'Ethanol') - PropsSI('H', 'P', P, 'Q', 0, 'Ethanol')


def haaland_fD(eps, Dh, Re):
    """Darcy friction factor, Haaland explicit approximation."""
    Re = max(Re, 1.0)
    return (-1.8 * math.log10((eps / (3.7 * Dh)) ** 1.11 + 6.9 / Re)) ** -2


# %% ------------------------- Unit conversions (RocketCEA -> SI) ------------------------
_degR_to_K         = 5.0 / 9.0        # K per degR
_ftps_to_mps       = 0.3048           # m/s per ft/s
_BTUlbmR_to_JkgK   = 4186.8           # J/(kg K) per (BTU/lbm degR)
_millipoise_to_Pas = 1e-4             # Pa s per millipoise

# %% ================================= User inputs ==================================
ox               = 'N2O'
fuel             = 'Ethanol'
Material         = 'AlSi10Mg'  # 'Inconel718', 'AlSi10Mg', 'ABD900'

Chamber_Pressure = 30     # bar
Feed_Pressure    = 40.0   # bar
Thrust_req       = 5000.0 # N

# Set ONE of these, leave the other as None.
# feed_OF = ox / total fuel (incl. film);  core_OF = ox / core fuel (excl. film)
core_OF          = None
feed_OF          = 3.0

CR               = 6      # contraction ratio Ac/At
Lstar_target_mm  = 750.0  # mm (chamber only)
tw               = 0.7e-3 # wall thickness [m]
channel_height   = 1.2e-3 # [m]
arc_angle        = 3.8    # degrees
n                = 50     # channels
fcp              = 0.1    # film-cooling fraction of TOTAL fuel flow

# Efficiencies
eta_cstar = 0.959
eta_Cf    = 0.923

# --- Model options ---
Roughness_Scale        = 10.0  # multiplier on the surface-roughness tables (original script used 10)
Build_Vertical         = True # True: build angle measured from the build plate = 90 deg - wall angle from axis
Use_Dean_Correction    = False # True: compute a real Dean number; False: De = 1 (original behaviour)
Film_Wall_At_Film_Temp = True  # True: liquid-film region uses film temperature as the driving temperature
Reevaluate_Sigma       = True  # True: re-evaluate the Bartz sigma term with the film-cooled wall temperature
Entrainment_Coeff      = 0.1963  # gas-film entrainment coefficient (SI units, mu in Pa s)
Film_Max_Iter          = 25
Film_Tol               = 1e-4
Film_Relax             = 1.0   # <1 under-relaxes the film inlet temperature update

# --- Graphs ---
Complete_Thermal_Analysis = True
Thermal_Diagnostic_3x3    = False
Film_Cooling_vs_No_FC     = True
SF_and_Temps_3D           = False  # Looks great, takes ~5 minutes to run, proceed with caution
Stress_Plot               = True
Nu_Plot                   = False

if core_OF is None and feed_OF is None:
    print("Define either Feed O/F or Core O/F.")
    sys.exit()
elif core_OF is not None and feed_OF is not None:
    print("Both Feed O/F and Core O/F defined, define only one.")
    sys.exit()
elif core_OF is None:
    core_OF = feed_OF / (1 - fcp)
else:
    feed_OF = core_OF * (1 - fcp)
OF = core_OF   # O/F seen by CEA (the core mixture)

# %% ========================= CEA / ideal performance =========================
ispObj = CEA_Obj(oxName=ox, fuelName=fuel)

Pc_SI   = Chamber_Pressure * 1e5       # Pa
Pc_psia = Chamber_Pressure * 14.5038   # psia
g0      = 9.80665

# Area ratio ~ Pe = 1 atm
Area_Ratio = ispObj.get_eps_at_PcOvPe(Pc=Pc_psia, MR=OF,
                                      PcOvPe=Chamber_Pressure / 1.013,
                                      frozen=0, frozenAtThroat=0)

# Ideal Isp & c*
ISP_vac_ideal = ispObj.get_Isp(Pc=Pc_psia, MR=OF, eps=Area_Ratio, frozen=0, frozenAtThroat=0)   # s
cstar_ideal   = ispObj.get_Cstar(Pc=Pc_psia, MR=OF) / 3.28                                      # m/s

# Ideal Cf at SL
Cf_list_SL  = ispObj.get_PambCf(Pamb=14.7, Pc=Pc_psia, MR=OF, eps=Area_Ratio)
Cf_SL_ideal = float(Cf_list_SL[0])

# Apply efficiencies
cstar_act = eta_cstar * cstar_ideal
Cf_SL_act = eta_Cf * Cf_SL_ideal

# Throat area from thrust target (SL)
At_SI = Thrust_req / (Cf_SL_act * Pc_SI)   # m^2
Rt_SI = math.sqrt(At_SI / math.pi)         # m

# Mass flow
mdot_act   = Pc_SI * At_SI / cstar_act     # nozzle (core) mass flow
ISP_SL_act = Cf_SL_act * cstar_act / g0    # core-flow Isp (film flow not counted)
F_SL       = Cf_SL_act * Pc_SI * At_SI     # = Thrust_req by construction

# %% ============================== Geometry (mm) ==============================
def rao_nozzle(Rt, epsilon):
    """Rao bell nozzle contour (80% bell), ~800 points."""
    nozzle_length = 0.8 * ((math.sqrt(epsilon) - 1) * Rt / math.tan(math.radians(15)))
    theta_n = math.radians(24)
    theta_e = math.radians(12)
    Rn = 0.382 * Rt

    # 1) Circular arc at throat
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

    t = np.linspace(0, 1, 751)
    x_para = (1 - t) ** 2 * Nx + 2 * (1 - t) * t * Qx + t ** 2 * Ex
    y_para = (1 - t) ** 2 * Ny + 2 * (1 - t) * t * Qy + t ** 2 * Ey

    x_array = np.concatenate((x_circ, x_para[1:]))
    y_array = np.concatenate((y_circ, y_para[1:]))

    volume = 0
    for i in range(len(x_array) - 1):
        volume += math.pi * y_array[i] ** 2 * (x_array[i + 1] - x_array[i])

    return x_array, y_array, volume, Rn, theta_n, theta_e


def convergent_sizing(CR, Rt):
    """Convergent section contour (arc - cone - arc), ~700 points."""
    R1 = 1.5 * Rt
    b = math.radians(30)
    Rc = np.sqrt(CR) * Rt
    R2max = (Rc - Rt) / (1 - math.cos(b)) - R1
    R2 = R2max * 0.5

    theta2 = np.linspace(0, b, 250)
    x2_array = R2 * np.sin(theta2)
    y2_array = Rc - R2 + R2 * np.cos(theta2)

    theta1 = np.linspace(b, 0, 250)
    x1_array = R1 * math.sin(b) - R1 * np.sin(theta1)
    y1_array = Rt + R1 - R1 * np.cos(theta1)

    dely = y2_array[-1] - y1_array[0]
    delx = dely / math.tan(b)
    xl_array = np.linspace(np.max(x2_array), np.max(x2_array) + delx, 200)
    yl_array = np.linspace(np.min(y2_array), np.min(y2_array) - dely, 200)

    x1_array += np.max(xl_array)

    x_array = np.concatenate((x2_array, xl_array[1:-1], x1_array))
    y_array = np.concatenate((y2_array, yl_array[1:-1], y1_array))

    volume = 0
    for i in range(len(x_array) - 1):
        volume += math.pi * y_array[i] ** 2 * (x_array[i + 1] - x_array[i])

    return x_array, y_array, volume, R1, R2, b


def cylinder_sizing(Lstar, other_volume, Rt, CR):
    """Cylindrical chamber sized by L*, 501 points."""
    required_volume = Lstar * math.pi * Rt ** 2
    Volume = required_volume - other_volume
    Length = Volume / (CR * math.pi * Rt ** 2)
    x_array = np.linspace(0, Length, 501)
    y_array = np.ones_like(x_array) * Rt * math.sqrt(CR)
    return x_array, y_array, Length


Rt_mm = Rt_SI * 1e3
Re_mm = math.sqrt(Area_Ratio) * Rt_mm
Rc_mm = math.sqrt(CR) * Rt_mm

xn, yn, vn_mm3, Rn_mm, theta_n, theta_e = rao_nozzle(Rt_mm, Area_Ratio)
xc, yc, vc_mm3, R1_mm, R2_mm, b_rad = convergent_sizing(CR, Rt_mm)
xcyl, ycyl, Lcyl_mm = cylinder_sizing(Lstar_target_mm, vc_mm3, Rt_mm, CR)

# Stitch sections axially
xc = xc + np.max(xcyl)
xn = xn + np.max(xc)

x_raw = np.concatenate((xcyl, xc, xn))
y_raw = np.concatenate((ycyl, yc, yn))

# Strictly increasing x for the spline (drop duplicate joints)
_, unique_indices = np.unique(x_raw, return_index=True)
x_clean = x_raw[unique_indices]
y_clean = y_raw[unique_indices]

cs = CubicSpline(x_clean, y_clean)
cs_d1 = cs.derivative(1)

# Resample at 1001 nodes -> 1000 uniform segments
x_nodes = np.linspace(x_clean.min(), x_clean.max(), 1001)
y_nodes = cs(x_nodes)
x_seg_mm = 0.5 * (x_nodes[:-1] + x_nodes[1:])      # mm
r_seg_mm = 0.5 * (y_nodes[:-1] + y_nodes[1:])      # mm
Nseg = len(x_seg_mm)

throat_node_fwd = int(np.argmin(y_nodes))
throat_idx_fwd  = int(np.argmin(r_seg_mm))
x_throat        = x_nodes[throat_node_fwd]

# Smoothed second derivative of the contour (used only for the optional Dean number)
ddy_raw    = cs.derivative(2)(x_nodes)
ddy_smooth = sp.ndimage.gaussian_filter1d(ddy_raw, sigma=30)
ddy_func   = CubicSpline(x_nodes, ddy_smooth)

# %% ========================== Material properties ==========================
if Material == 'Inconel718':
    Surf_roughness      = np.array([54, 38, 12, 6, 5]) * 1e-6 * Roughness_Scale
    manufacturing_angle = np.array([30, 45, 60, 75, 90]) * math.pi / 180
    v                   = 0.28
    conductivity        = 12
    cte                 = 16e-6
    modulus_temps       = np.array([21, 93, 204, 316, 427, 538, 649, 760, 871, 954]) + 273.15
    modulus             = np.array([208, 205, 202, 194, 186, 179, 172, 162, 127, 78]) * 1e9
    yield_temps         = np.array([93, 204, 316, 427, 538, 649, 760]) + 273.15
    yield_stress        = np.array([1172, 1124, 1096, 1076, 1069, 1027, 758]) * 1e6
    emissivity          = 0.3
elif Material == 'AlSi10Mg':
    Surf_roughness      = np.array([66, 56, 21, 12, 8]) * 1e-6 * Roughness_Scale
    manufacturing_angle = np.array([30, 45, 60, 75, 90]) * math.pi / 180
    modulus_temps       = np.array([25, 50, 100, 150, 200, 250, 300, 350, 400]) + 273.15
    modulus             = np.array([77.6, 75.5, 72.8, 63.2, 60, 55, 45, 37, 28]) * 1e9
    yield_temps         = np.array([298, 323, 373, 423, 473, 523, 573, 623, 673])
    yield_stress        = np.array([204, 198, 181, 182, 158, 132, 70, 30, 12]) * 1e6
    conductivity        = 140
    cte                 = 27e-6
    v                   = 0.33
    emissivity          = 0.650
elif Material == 'ABD900':
    Surf_roughness      = np.array([54, 38, 12, 6, 5]) * 1e-6 * 0.7 * Roughness_Scale
    manufacturing_angle = np.array([30, 45, 60, 75, 90]) * math.pi / 180
    modulus_temps       = np.array([25, 700, 800, 900]) + 273.15
    modulus             = np.array([192, 157, 131, 103]) * 1e9
    yield_temps         = np.array([25, 427, 538, 649, 732, 760, 788, 816, 843, 871, 927]) + 273.15
    yield_stress        = np.array([978, 931, 903, 903, 758, 680, 600, 503, 434, 352, 221]) * 1e6
    cte                 = np.array([11.4, 12.9, 13.7, 14.4, 15.5, 17.5, 19.2]) * 1e-6
    cte_temps           = np.array([50, 200, 400, 600, 800, 1000, 1200]) + 273.15
    conductivity        = np.array([11, 12.6, 15.7, 18.8, 23.2, 26.4, 30.1])
    conductivity_temps  = np.array([25, 200, 400, 600, 800, 1000, 1200]) + 273.15
    v                   = 0.28
    emissivity          = 0.3
else:
    raise ValueError("Unknown Material")


def lam_wall(T):
    """Wall thermal conductivity [W/m/K] (floored at 1)."""
    if isinstance(conductivity, np.ndarray):
        return max(float(np.interp(T, conductivity_temps, conductivity)), 1.0)
    return max(float(conductivity), 1.0)


# %% ============================ Regen preparation ============================
core_fuel_mdot = mdot_act / (OF + 1)
ox_mdot        = mdot_act * OF / (OF + 1)
fuel_mdot      = core_fuel_mdot / (1 - fcp)     # total fuel, incl. film
film_mdot      = fuel_mdot * fcp
feed_press     = Feed_Pressure * 1e5

# Channel width from printing arc
channel_width = (r_seg_mm + (tw * 1e3) + 0.5 * (channel_height * 1e3)) * math.sin(math.radians(arc_angle)) * 1e-3  # [m]
w  = channel_width
h  = channel_height
Dh = 2 * h * w / (h + w)

# Wall angle (radians) from the axis at each segment
angles_seg = np.arctan2(np.diff(y_nodes), np.diff(x_nodes))
angles_seg[0] = 0.0
angles_seg[-1] = 0.0
angles_abs = np.abs(angles_seg)
build_angle = (np.pi / 2 - angles_abs) if Build_Vertical else angles_abs
build_angle = np.clip(build_angle, manufacturing_angle.min(), manufacturing_angle.max())
roughness_node = np.interp(build_angle, manufacturing_angle, Surf_roughness)

# Reverse for coolant counterflow (coolant enters at the nozzle exit)
rev = {
    'x':       x_seg_mm[::-1],
    'x_nodes': x_nodes[::-1],
    'r':       r_seg_mm[::-1],
    'r_nodes': y_nodes[::-1],
    'Dh':      Dh[::-1],
    'w':       w[::-1],
    'eps':     roughness_node[::-1],
}
dr_array = np.abs(np.diff(rev['r_nodes'])) * 1e-3
dx_array = np.abs(np.diff(rev['x_nodes'])) * 1e-3
seg_len_rev  = np.sqrt(dr_array ** 2 + dx_array ** 2)   # [m]
L_cum_rev    = np.cumsum(seg_len_rev)                   # distance from coolant inlet [m]
R_rev_m      = rev['r'] * 1e-3
A_rev        = math.pi * R_rev_m ** 2

# %% =========================[CEA gas properties]=========================
T0_chamber = ispObj.get_Tcomb(Pc=Pc_psia, MR=OF)                                   # degR
cp_chamber, mu_chamber, _, Pr_chamber = ispObj.get_Chamber_Transport(Pc=Pc_psia, MR=OF, frozen=0)
cp_throat,  mu_throat,  _, Pr_throat  = ispObj.get_Throat_Transport(Pc=Pc_psia, MR=OF, frozen=0)
cp_exit,    mu_exit,    _, Pr_exit    = ispObj.get_Exit_Transport(Pc=Pc_psia, MR=OF, frozen=0)
Mg_chamber, gamma_chamber = ispObj.get_Chamber_MolWt_gamma(Pc=Pc_psia, MR=OF, eps=Area_Ratio)
_, gamma_throat = ispObj.get_Throat_MolWt_gamma(Pc=Pc_psia, MR=OF, eps=Area_Ratio)
_, gamma_exit   = ispObj.get_exit_MolWt_gamma(Pc=Pc_psia, MR=OF, eps=Area_Ratio)
c_star_chamber  = ispObj.get_Cstar(Pc=Pc_psia, MR=OF)                              # ft/s

_xk = (0, x_seg_mm[throat_idx_fwd], np.max(x_seg_mm))
cp_array     = np.interp(x_seg_mm, _xk, (cp_chamber, cp_throat, cp_exit)) * _BTUlbmR_to_JkgK
mu_array     = np.interp(x_seg_mm, _xk, (mu_chamber, mu_throat, mu_exit)) * _millipoise_to_Pas
Pr_array     = np.interp(x_seg_mm, _xk, (Pr_chamber, Pr_throat, Pr_exit))
gamma_interp = np.interp(x_seg_mm, _xk, (gamma_chamber, gamma_throat, gamma_exit))

cp_rev, mu_rev, Pr_rev, gamma_rev = cp_array[::-1], mu_array[::-1], Pr_array[::-1], gamma_interp[::-1]

T0    = T0_chamber * _degR_to_K
cstar = c_star_chamber * _ftps_to_mps
Rt = Rt_SI
Dt = 2.0 * Rt
At = At_SI

# Mach number along the contour (computed once; the cylinder shares a single CEA call)
area_ratio_fwd = (math.pi * (r_seg_mm * 1e-3) ** 2) / At
Mach_fwd = np.zeros(Nseg)
_mach_cache = {}
for j in range(Nseg):
    ar  = max(area_ratio_fwd[j], 1.0 + 1e-6)
    sub = j <= throat_idx_fwd
    key = (sub, round(float(ar), 6))
    if key not in _mach_cache:
        try:
            if sub:
                M_ = ispObj.get_Chamber_MachNumber(Pc=Pc_psia, MR=OF, fac_CR=ar)
            else:
                M_ = ispObj.get_MachNumber(Pc=Pc_psia, MR=OF, eps=ar, frozen=0, frozenAtThroat=0)
        except Exception:
            M_ = 1.0
        _mach_cache[key] = float(M_)
    Mach_fwd[j] = _mach_cache[key]
Mach_rev = Mach_fwd[::-1].copy()

# Static gas pressure along the contour [Pa]
pressure_array = Pc_SI / (1.0 + 0.5 * (gamma_interp - 1.0) * Mach_fwd ** 2) ** (gamma_interp / (gamma_interp - 1.0))

# Bartz coefficient prefactor per segment (everything except the sigma term)
Dt_over_R_rev = (2.0 * Rt) / R_rev_m
At_over_A_rev = At / A_rev
h_pref_rev = (0.026 * Dt ** -0.2 * mu_rev ** 0.2 * cp_rev * Pr_rev ** -0.6
              * (Pc_SI / cstar) ** 0.8 * Dt_over_R_rev ** 0.1 * At_over_A_rev ** 0.9)


def bartz_sigma(Twg, M, gam):
    base2 = np.maximum(1.0 + 0.5 * (gam - 1.0) * M * M, 1e-9)
    base1 = np.maximum(0.5 * (Twg / max(T0, 1e-6)) * base2 + 0.5, 1e-9)
    return base1 ** -0.68 * base2 ** -0.12


# %% ============================ Coolant helpers ============================
def fuel_phase(Tc_loc_K, regen_press, fuel_st, *, pressure_in_pa=True):
    """Robust ethanol state chooser for PyFluids (falls back on Tsat-based phase selection)."""
    P = float(regen_press if pressure_in_pa else regen_press * 1e5)  # Pa
    T_K = float(Tc_loc_K)
    T_C = T_K - 273.15
    if not (np.isfinite(P) and np.isfinite(T_K)) or P <= 0.0:
        raise ValueError(f"Bad inputs to fuel_phase: P={P} Pa, T={T_K} K")

    fs = Fluid(FluidsList.Ethanol)   # fresh instance so no phase carries over

    if (P > 0.99 * _ETH_Pc and T_K > 0.99 * _ETH_Tc):
        try:
            return fs.with_state(Input.pressure(P), Input.temperature(T_C))
        except ValueError:
            return fs.with_state(Input.pressure(P), Input.temperature((T_K + 0.2) - 273.15))

    try:
        return fs.with_state(Input.pressure(P), Input.temperature(T_C))
    except ValueError:
        try:
            Tsat_K = PropsSI('T', 'P', P, 'Q', 0, 'Ethanol')
        except Exception:
            return fs.with_state(Input.pressure(P), Input.temperature((T_K + 0.2) - 273.15))

        eps = 0.15  # K, stay off the coexistence curve
        if T_K < Tsat_K - eps:
            fs.specify_phase(Phases.Liquid)
            T_try = T_K + eps
        elif T_K > Tsat_K + eps:
            fs.specify_phase(Phases.Gas)
            T_try = T_K - eps
        else:
            fs.specify_phase(Phases.Liquid)
            T_try = Tsat_K - eps

        try:
            return fs.with_state(Input.pressure(P), Input.temperature(T_try - 273.15))
        except ValueError:
            return fs.with_state(Input.pressure(P), Input.temperature((T_try + 0.3) - 273.15))


def get_ethanol_state(T_K, P_pa, phase_hint=None, pressure_floor_pa=1e5, temp_floor_K=200.0, temp_cap_K=None):
    """Ethanol state with guards against unphysical marching values."""
    P_use = max(float(P_pa), pressure_floor_pa)
    T_use = max(float(T_K), temp_floor_K)
    if temp_cap_K is not None:
        T_use = min(T_use, float(temp_cap_K))

    fl = Fluid(FluidsList.Ethanol)
    if phase_hint == "gas":
        try:
            fl.specify_phase(Phases.Gas)
            return fl.with_state(Input.pressure(P_use), Input.temperature(T_use - 273.15))
        except ValueError:
            return fuel_phase(T_use, P_use, fl)
    if phase_hint == "liquid":
        try:
            fl.specify_phase(Phases.Liquid)
            return fl.with_state(Input.pressure(P_use), Input.temperature(T_use - 273.15))
        except ValueError:
            return fuel_phase(T_use, P_use, fl)
    return fuel_phase(T_use, P_use, fl)


def coolant_step(i, Tc_loc, P_in):
    """
    Hydraulics and properties for reversed segment i.
    Returns a dict with rho, mu, lam, cp, Pr, Re, fD, P_out and a flag for the 1 bar pressure floor.
    """
    Dh_i  = rev['Dh'][i]
    Aflow = channel_height * rev['w'][i]
    m_ch  = fuel_mdot / n
    seg_len = seg_len_rev[i]
    P_in  = max(float(P_in), P_FLOOR)
    Tc_loc = max(float(Tc_loc), 200.0)

    def dP_for(state):
        rho_, mu_ = state.density, state.dynamic_viscosity
        u_  = m_ch / (rho_ * Aflow)
        Re_ = rho_ * u_ * Dh_i / mu_
        fD_ = haaland_fD(rev['eps'][i], Dh_i, Re_)
        dyn = 0.5 * rho_ * u_ ** 2
        dP_ = fD_ * seg_len / Dh_i * dyn
        if i == 0 or i == (Nseg - 1):
            dP_ += 0.8 * dyn   # inlet / outlet loss
        return dP_

    st = get_ethanol_state(Tc_loc, P_in, phase_hint="liquid")
    dP = dP_for(st)
    P_raw = P_in - dP
    if dP / P_in > 5e-2:   # big drop in one segment: redo with properties at the mean pressure
        st_m = get_ethanol_state(Tc_loc, max(0.5 * (P_in + P_raw), P_FLOOR), phase_hint="liquid")
        dP = dP_for(st_m)
        P_raw = P_in - dP
    P_out = max(P_raw, P_FLOOR)

    st = get_ethanol_state(Tc_loc, P_out, phase_hint="liquid")
    rho, mu, lam, cp = st.density, st.dynamic_viscosity, st.conductivity, st.specific_heat
    u  = m_ch / (rho * Aflow)
    Re = rho * u * Dh_i / mu
    return {'rho': rho, 'mu': mu, 'lam': lam, 'cp': cp, 'Pr': cp * mu / lam, 'Re': Re,
            'fD': haaland_fD(rev['eps'][i], Dh_i, Re), 'P_out': P_out, 'floored': P_raw <= P_FLOOR}


# ---------------- Nusselt number ----------------
def _nu_gnielinski(Re, Pr, f):
    num = (f / 8.0) * (Re - 1000.0) * Pr
    den = 1.0 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0)
    return num / max(den, 1e-12)


def _nu_hausen(Re, Pr, Dh, L):
    Gz = Re * Pr * Dh / max(L, 1e-12)
    return 3.66 + 0.0668 * Gz / (1.0 + 0.04 * (Gz ** (2.0 / 3.0)))


def Nu_func(Re, f, Pr, c_height, c_width, Dh, L, mu_c, mu_tw, x_loc):
    """
    Local Nusselt number for a rectangular regen channel.
    x_loc is the axial position [mm] (used for the optional Dean-number curvature enhancement).
    """
    if Use_Dean_Correction:
        dy  = float(cs_d1(x_loc))
        ddy = float(ddy_func(x_loc))
        Rc_m = abs((1 + dy ** 2) ** 1.5 / max(abs(ddy), 1e-12)) * 1e-3   # contour radius of curvature [m]
        De = Re * math.sqrt(Dh / (2.0 * Rc_m))
    else:
        De = 1.0

    L_eff = max(L, 3.0 * Dh)
    xth = 0.05 * Re * Pr * Dh   # thermal entrance length (laminar)

    a, b = (c_height, c_width) if c_height >= c_width else (c_width, c_height)
    ar_grid = np.array([1.0, 2.0, 4.0, 8.0])
    Nu_fd_grid = np.array([3.608, 4.123, 5.331, 6.490])
    Nu_fd_rect = np.interp(a / max(b, 1e-12), ar_grid, Nu_fd_grid)

    wall_corr_turb = (mu_c / max(mu_tw, 1e-12)) ** 0.11
    wall_corr_lam  = (mu_c / max(mu_tw, 1e-12)) ** 0.14

    C_turb, m_turb = 0.06, 0.20
    C_lam,  m_lam  = 0.03, 0.50

    def turb():
        Nu_base = _nu_gnielinski(Re, Pr, f)
        len_corr = 1.0 + (Dh / L_eff) ** (2.0 / 3.0)
        return Nu_base * len_corr * wall_corr_turb * (1.0 + C_turb * (De ** m_turb))

    def lam():
        Nu_l = (Nu_fd_rect if L_eff > xth else _nu_hausen(Re, Pr, Dh, L_eff)) * wall_corr_lam
        return Nu_l * (1.0 + C_lam * (De ** m_lam))

    if Re >= 4000.0:
        Nu = turb()
    elif Re > 2300.0:
        gam = min(max((Re - 2300.0) / (4000.0 - 2300.0), 0.0), 1.0)
        Nu = (1.0 - gam) * lam() + gam * turb()
    else:
        Nu = lam()
    return float(Nu)


def flip(arr):
    return np.asarray(arr)[::-1].copy()


# %% ================= Pass 1: Bartz loop, no film (coolant marches exit -> injector) =================
qg_nf_rev     = np.zeros(Nseg)
alphaT_nf_rev = np.zeros(Nseg)
sigma0_rev    = np.zeros(Nseg)
Twg_nf_rev    = np.zeros(Nseg)
Twc_nf_rev    = np.zeros(Nseg)
Te_rev        = np.zeros(Nseg)
P_nf_rev      = np.zeros(Nseg)
tc_nf         = np.zeros(Nseg + 1)
tc_nf[0]      = 300.0   # K, coolant inlet

for i in range(Nseg):
    M, gam, Pr_g = Mach_rev[i], gamma_rev[i], Pr_rev[i]

    # Recovery (adiabatic wall) temperature
    Pr_use = float(Pr_g) if (np.isfinite(Pr_g) and Pr_g > 0.0) else 0.7
    r_rec  = Pr_use ** (1.0 / 3.0)
    den    = max(1.0 + 0.5 * (gam - 1.0) * M * M, 1e-8)
    Te     = T0 * (1.0 + r_rec * 0.5 * (gam - 1.0) * M * M) / den
    Te_rev[i] = Te

    P_in = feed_press if i == 0 else P_nf_rev[i - 1]
    cool = coolant_step(i, tc_nf[i], P_in)
    P_nf_rev[i] = cool['P_out']

    Nu = Nu_func(cool['Re'], cool['fD'], cool['Pr'], channel_height, rev['w'][i], rev['Dh'][i],
                 L_cum_rev[i], cool['mu'], cool['mu'], rev['x'][i])
    alpha_c = max(Nu * cool['lam'] / rev['Dh'][i], 50.0)
    Tc_loc  = max(tc_nf[i], 200.0)

    # Iterative (under-relaxed) wall solve
    Twg_0, relax, tol = Te - 50.0, 0.1, 1e-4
    for k in range(500):
        alpha_T = h_pref_rev[i] * bartz_sigma(Twg_0, M, gam)
        qg  = alpha_T * max(Te - Twg_0, 0.0)
        Twc = qg / alpha_c + Tc_loc
        Twg_new = Twc + qg * max(tw, 1e-4) / lam_wall(0.5 * (Twg_0 + Twc))
        Twg_1 = (1.0 - relax) * Twg_0 + relax * Twg_new
        converged = abs(Twg_1 - Twg_0) / max(abs(Twg_1), 1.0) < tol
        Twg_0 = Twg_1
        if converged:
            break

    sig0    = bartz_sigma(Twg_0, M, gam)
    alpha_T = h_pref_rev[i] * sig0
    qg  = alpha_T * max(Te - Twg_0, 0.0)
    Twc = qg / alpha_c + Tc_loc
    Twg_1 = Twc + qg * max(tw, 1e-4) / lam_wall(0.5 * (Twg_0 + Twc))

    qg_nf_rev[i], alphaT_nf_rev[i], sigma0_rev[i] = qg, alpha_T, sig0
    Twg_nf_rev[i], Twc_nf_rev[i] = Twg_1, Twc

    Ahot_seg = 2.0 * math.pi * R_rev_m[i] * seg_len_rev[i]
    tc_nf[i + 1] = tc_nf[i] + qg * Ahot_seg / (fuel_mdot * cool['cp'])

# Forward (gas-flow) order
qg_fwd       = flip(qg_nf_rev)
alphaT_fwd   = flip(alphaT_nf_rev)
Twg_fwd      = flip(Twg_nf_rev)
Twc_fwd      = flip(Twc_nf_rev)
Te_fwd       = flip(Te_rev)
Tc_fwd_nodes = flip(tc_nf)
Tc_seg       = 0.5 * (Tc_fwd_nodes[:-1] + Tc_fwd_nodes[1:])

# %% ============================== Film cooling ==============================
Mv = 46.0   # g/mol, ethanol
cp_g_film  = cp_chamber * _BTUlbmR_to_JkgK        # J/kg/K, chamber gas
mu_g_film  = mu_chamber * _millipoise_to_Pas      # Pa s (entrainment correlation needs SI)
Km_liq     = (Mg_chamber / Mv) ** 0.6 if Mg_chamber > Mv else (Mg_chamber / Mv) ** 0.35   # blowing correction
Km_gas     = (Mv / max(Mg_chamber, 1e-12)) ** 0.14
hfg_array  = np.array([ethanol_hfg(p) for p in pressure_array])    # J/kg, local pressure
throat_idx = throat_idx_fwd
dx_m       = np.diff(x_nodes) * 1e-3                               # segment length [m]
r_m_fwd    = r_seg_mm * 1e-3
Gamma_init = film_mdot / (2 * np.pi * r_m_fwd[0])                  # kg/(m s) at the injector


def solve_film(T_film_in):
    """
    Liquid film (sensible heating + evaporation) followed by the gaseous film (entrainment/mixing).
    Returns the blended film heat-transfer coefficient and driving temperature along the contour.
    """
    T_fcool = np.zeros(Nseg)
    h_corr  = np.zeros(Nseg)
    T_f = float(T_film_in)

    if T_f >= ethanol_Tsat_from_P(pressure_array[0]):
        burnout = 0                       # film flashes at the injector face
        T_fcool[:] = T_f
    else:
        Gamma = Gamma_init
        burnout = throat_idx
        for i in range(throat_idx):
            Tsat = float(ethanol_Tsat_from_P(pressure_array[i]))
            T_fcool[i] = T_f
            cp_l = get_ethanol_state(T_f, pressure_array[i], phase_hint="liquid").specific_heat
            h0   = alphaT_fwd[i]
            hfg  = max(hfg_array[i], 1.0)
            mdotvap = h0 * (Te_fwd[i] - T_f) / hfg          # kg/m^2/s
            DeltaT  = max(Te_fwd[i] - T_f, 0.0)
            H = (cp_g_film * Km_liq / hfg) * DeltaT
            h_corr[i] = h0 * np.log1p(H) / max(H, 1e-12)
            T_f += qg_fwd[i] * dx_m[i] / max(Gamma * cp_l, 1e-12)   # W/m per m of circumference
            if T_f >= Tsat - 1e-9:
                T_f = Tsat
                Gamma -= mdotvap * dx_m[i]                  # kg/(m s)
                if Gamma < 1e-8:
                    burnout = i
                    break

    # ---------------- Gas film: per-span mass flow m' [kg/(m s)] ----------------
    Lg  = Nseg - burnout
    mbl = np.zeros(Lg)
    P_burn = 2 * np.pi * max(r_m_fwd[burnout], 1e-9)
    mbl[0] = max(film_mdot / P_burn, 1e-8)       # all film mass is vapour from the burnout point on
    Taw_g  = np.zeros(Lg)
    Tsat_b = float(ethanol_Tsat_from_P(pressure_array[burnout]))
    Taw_g[0] = np.clip(Tsat_b, T_f, Te_fwd[burnout])
    mdot_total_core = ox_mdot + fuel_mdot * (1 - fcp)

    for i in range(burnout, Nseg - 1):
        j = i - burnout
        Pi    = 2 * np.pi * max(r_m_fwd[i], 1e-9)
        A_g   = np.pi * max(r_m_fwd[i], 1e-9) ** 2
        mdot_film      = mbl[j] * Pi
        entrained      = max(mdot_film - film_mdot, 0.0)               # core mass already pulled into the film
        G              = max(mdot_total_core - entrained, 1e-9) / A_g
        cp_c = get_ethanol_state(Taw_g[j], pressure_array[i], phase_hint="gas").specific_heat

        D  = 2.0 * max(r_m_fwd[i], 1e-9)
        dD = 2.0 * (r_m_fwd[i + 1] - r_m_fwd[i])
        dx = max(dx_m[i], 1e-12)

        m_i  = max(mbl[j], 1e-12)
        dmb1 = Entrainment_Coeff * G * (mu_g_film / m_i) ** 0.25 * dx            # predictor
        m_p  = max(mbl[j] + dmb1, 1e-12)
        dmb2 = Entrainment_Coeff * G * (mu_g_film / m_p) ** 0.25 * dx            # corrector
        dmb_e    = 0.5 * (dmb1 + dmb2)
        dmb_geom = -m_i * (dD / max(D, 1e-12))                                    # area-change term
        mbl[j + 1] = max(mbl[j] + dmb_e + dmb_geom, 1e-12)

        C_f = mbl[j] * Pi * max(cp_c, 1e-6)                    # film capacity rate
        C_e = max(dmb_e, 0.0) * Pi * Km_gas * max(cp_g_film, 1e-6)
        Taw_g[j + 1] = Taw_g[j] + (Te_fwd[i] - Taw_g[j]) * (C_e / max(C_f + C_e, 1e-9))

    # ---------------- Blend liquid -> gas film ----------------
    blend_width = 15
    half = blend_width // 2
    liq_T = T_fcool if Film_Wall_At_Film_Temp else Te_fwd
    h_out   = alphaT_fwd.copy()
    Taw_out = Te_fwd.copy()
    for i in range(Nseg):
        if burnout == 0:
            Taw_out[i] = Taw_g[i]
        elif i < burnout - half:
            h_out[i], Taw_out[i] = h_corr[i], liq_T[i]
        elif i > burnout + half:
            Taw_out[i] = Taw_g[min(i - burnout, Lg - 1)]
        else:
            w_b   = (i - (burnout - half)) / blend_width
            idx_g = min(max(0, i - burnout), Lg - 1)
            h_out[i]   = (1 - w_b) * h_corr[max(0, burnout - 1)] + w_b * alphaT_fwd[i]
            Taw_out[i] = (1 - w_b) * liq_T[max(0, burnout - 1)] + w_b * Taw_g[idx_g]

    return {'burnout': burnout, 'h': h_out, 'Taw': Taw_out, 'T_film_end': T_f}


def film_bartz(h_fc_fwd, Taw_fc_fwd):
    """Coolant march (exit -> injector) with film-cooled gas-side boundary conditions."""
    h_rev, Taw_rev = flip(h_fc_fwd), flip(Taw_fc_fwd)
    out = {k: np.zeros(Nseg) for k in ('Twg', 'Twc', 'qg', 'P', 'Re', 'Nu', 'h')}
    tc = np.zeros(Nseg + 1)
    tc[0] = 300.0
    floor_hits = 0

    for i in range(Nseg):
        P_in = feed_press if i == 0 else out['P'][i - 1]
        cool = coolant_step(i, tc[i], P_in)
        floor_hits += int(cool['floored'])
        out['P'][i] = cool['P_out']

        Nu = Nu_func(cool['Re'], cool['fD'], cool['Pr'], channel_height, rev['w'][i], rev['Dh'][i],
                     L_cum_rev[i], cool['mu'], cool['mu'], rev['x'][i])
        alpha_c = max(Nu * cool['lam'] / rev['Dh'][i], 50.0)
        Tc_loc  = max(tc[i], 200.0)
        Taw     = Taw_rev[i]
        M, gam  = Mach_rev[i], gamma_rev[i]

        Twg = Taw - 50.0
        for k in range(60):
            if Reevaluate_Sigma:
                h_loc = h_rev[i] * bartz_sigma(Twg, M, gam) / sigma0_rev[i]
            else:
                h_loc = h_rev[i]
            h_loc = max(h_loc, 1.0)
            Twc_est = Twg - 0.0   # initial guess for conductivity lookup
            lam_w = lam_wall(0.5 * (Twg + (Tc_loc + 0.5 * (Twg - Tc_loc))))
            qg  = max((Taw - Tc_loc) / (1.0 / h_loc + 1.0 / alpha_c + tw / lam_w), 0.0)
            Twc = qg / alpha_c + Tc_loc
            Twg_new = Taw - qg / h_loc
            if abs(Twg_new - Twg) < 0.05:
                Twg = Twg_new
                break
            Twg = 0.5 * (Twg + Twg_new)

        out['Twg'][i], out['Twc'][i], out['qg'][i] = Twg, Twc, qg
        out['Re'][i], out['Nu'][i], out['h'][i] = cool['Re'], Nu, h_loc
        tc[i + 1] = tc[i] + qg * 2.0 * math.pi * R_rev_m[i] * seg_len_rev[i] / (fuel_mdot * cool['cp'])

    out['tc'] = tc
    out['floor_hits'] = floor_hits
    return out


# %% -------------------- Iterate until the film coolant temperature converges --------------------
T_film_in = float(np.max(Tc_seg))   # initial guess: no-film regen outlet temperature
for iter_count in range(1, Film_Max_Iter + 1):
    film = solve_film(T_film_in)
    res  = film_bartz(film['h'], film['Taw'])
    T_out = res['tc'][-1]
    film_resid = abs(T_out - T_film_in) / T_film_in
    if film_resid < Film_Tol:
        break
    T_film_in = (1.0 - Film_Relax) * T_film_in + Film_Relax * T_out

burnout_idx = film['burnout']
film_phase_x = x_seg_mm[burnout_idx]

Twg_fc_array = flip(res['Twg'])
Twc_fc_array = flip(res['Twc'])
qg_fc_array  = flip(res['qg'])
Tc_fc_array  = flip(0.5 * (res['tc'][:-1] + res['tc'][1:]))   # bulk coolant temperature per segment
Pcool_fwd    = flip(res['P'])
h_used_fwd   = flip(res['h'])
Re_fwd       = flip(res['Re'])
Tc_out_film  = res['tc'][-1]

# %% ==========================[PLOTTING]=========================
x = x_seg_mm
Twg, Twc, qg, aT = Twg_fc_array, Twc_fc_array, qg_fc_array, h_used_fwd
Mach, Tc_s, Pcool = Mach_fwd, Tc_fc_array, Pcool_fwd

RED    = (1.0, 0.0, 0.0)
GREEN  = (0.0, 1.0, 0.0)
TEAL   = (0.0, 0.7, 0.7)
PINK   = (1.0, 0.0, 1.0)
BLUE   = (0.0, 0.0, 1.0)
GREY   = (0.4, 0.4, 0.4)
ORANGE = (1.0, 0.5, 0.0)

Tsat_curve = ethanol_Tsat_from_P(Pcool)
tsat_label = 'Ethanol $T_\\mathrm{sat}(P)$'

# Phase change: bulk boiling in the channels and film burnout
boiling_mask = Tc_s >= Tsat_curve
regen_boiling_idx = int(np.where(boiling_mask)[0][-1]) if boiling_mask.any() else None
wall_boil_mask = Twc >= Tsat_curve   # coolant-side wall above Tsat -> local (subcooled) boiling

if Complete_Thermal_Analysis:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(x, Twg,  label=r'$T_{w,g}$', color=RED,  linewidth=1.6)
    ax1.plot(x, Twc,  label=r'$T_{w,c}$', color=TEAL, linewidth=1.4, linestyle='--')
    ax1.plot(x, Tc_s, label=r'$T_c$ (Regen)', color=BLUE, linewidth=1.4, linestyle='-.')
    ax1.plot(x, Tsat_curve, label=tsat_label, color=PINK, linewidth=1.4, alpha=0.6, linestyle=':')
    ax2.plot(x, qg, label=r'$q_g$', color=ORANGE, linewidth=1.6, alpha=0.8)

    if regen_boiling_idx is not None:
        ax1.axvline(x=x[regen_boiling_idx], color='darkblue', linestyle='--', linewidth=2,
                    label='Regen Phase Change (Boiling)')
    ax1.axvline(x=film_phase_x, color='magenta', linestyle='-', linewidth=2, label='Film Phase Change (Burnout)')
    ax1.axvline(x=x_throat, color='k', linestyle=':', linewidth=1.2, label='Throat')

    ax1.set_xlabel('Axial position [mm]')
    ax1.set_ylabel('Temperature [K]')
    ax2.set_ylabel(r'Heat flux [W/m$^2$]')
    ax1.set_title('Custom Simulation Thermal Analysis')
    ax1.grid(True, alpha=0.2)

    handles, labels = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + h2, labels + l2, loc='upper center', bbox_to_anchor=(0.5, -0.12),
               ncol=3, frameon=True, fontsize='small')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.show()

colors = [(1.0, 0.0, 0.0), (0.0, 0.6, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.5, 0.0),
          (0.0, 0.7, 0.7), (0.6, 0.0, 0.6), (0.3, 0.3, 0.3), (0.6, 0.4, 0.0)]

if Thermal_Diagnostic_3x3:
    sigma_d = bartz_sigma(Twg, Mach, gamma_interp)
    At_over_A_09 = (At / (np.pi * r_m_fwd ** 2)) ** 0.9
    Dt_over_R_01 = ((2.0 * Rt_SI) / r_m_fwd) ** 0.1

    fig, axs = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
    (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs
    ax1.plot(x, qg, color=colors[0]);            ax1.set_title(r'$q_g$  [W/m$^2$]')
    ax2.plot(x, alphaT_fwd, color=colors[1]);    ax2.set_title(r'$\alpha_T$ (no film)  [W/m$^2$K]')
    ax3.plot(x, Mach, color=colors[2]);          ax3.set_title('Mach')
    ax4.plot(x, Te_fwd, color=colors[3]);        ax4.set_title(r'$T_{aw}$  [K]')
    ax5.plot(x, Twg, color=colors[4]);           ax5.set_title(r'$T_{w,g}$  [K]')
    ax6.plot(x, Te_fwd - Twg, color=colors[5]);  ax6.set_title(r'$\Delta T$  [K]')
    ax7.plot(x, sigma_d, color=colors[6]);       ax7.set_title(r'$\sigma$  [-]')
    ax8.plot(x, At_over_A_09, color=colors[7]);  ax8.set_title(r'$(A_t/A)^{0.9}$')
    ax9.plot(x, Dt_over_R_01, color=colors[8]);  ax9.set_title(r'$(D_t/R)^{0.1}$')
    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
        ax.axvline(x=x_throat, color='k', linestyle=':', linewidth=1.0)
    for ax in axs[-1, :]:
        ax.set_xlabel('Axial position [mm]')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if Film_Cooling_vs_No_FC:
    q_scale = 1e-6
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    axs[0, 0].plot(x, Twg_fwd, label='Twg (no film)', color=RED, linewidth=1.8)
    axs[0, 0].plot(x, Twg_fc_array, label='Twg (film)', color=BLUE, linewidth=1.8, linestyle='--')
    axs[0, 0].set_ylabel('Temperature [K]'); axs[0, 0].set_title('Gas-side wall temperature (Twg)')

    axs[0, 1].plot(x, Twc_fwd, label='Twc (no film)', color=GREY, linewidth=1.8)
    axs[0, 1].plot(x, Twc_fc_array, label='Twc (film)', color=TEAL, linewidth=1.8, linestyle='--')
    axs[0, 1].set_ylabel('Temperature [K]'); axs[0, 1].set_title('Coolant-side wall temperature (Twc)')

    axs[1, 0].plot(x, Tc_seg, label='Tc (no film)', color=GREEN, linewidth=1.8)
    axs[1, 0].plot(x, Tc_fc_array, label='Tc (film)', color=PINK, linewidth=1.8, linestyle='--')
    axs[1, 0].set_ylabel('Temperature [K]'); axs[1, 0].set_title('Coolant bulk temperature (Tc)')
    axs[1, 0].set_xlabel('Axial position x [mm]')

    axs[1, 1].plot(x, qg_fwd * q_scale, label='qg (no film)', color=ORANGE, linewidth=1.8)
    axs[1, 1].plot(x, qg_fc_array * q_scale, label='qg (film)', color=colors[6], linewidth=1.8, linestyle='--')
    axs[1, 1].set_ylabel('qg [MW/m^2]'); axs[1, 1].set_title('Hot-gas heat flux')
    axs[1, 1].set_xlabel('Axial position x [mm]')

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axvline(x=x_throat, color='k', linestyle=':', linewidth=1.0)
    plt.tight_layout()
    plt.show()

if Nu_Plot:
    plt.figure()
    plt.plot(x, flip(res['Nu']))
    plt.xlabel('Axial position [mm]'); plt.ylabel('Coolant-side Nu [-]')
    plt.grid(alpha=0.3)
    plt.show()

# %%==========================[STRESS CALCULATIONS]=====================================
ax_pos = x_seg_mm * 1e-3
radii  = r_seg_mm * 1e-3
pressures = pressure_array                  # static gas pressure from CEA Mach / gamma
channel_dP = Pcool_fwd - pressures          # coolant minus gas pressure across the wall

avg_innerwall_temp = (Twg_fc_array + Twc_fc_array) / 2
dT = Twg_fc_array - Twc_fc_array

E = np.interp(avg_innerwall_temp, modulus_temps, modulus) if isinstance(modulus, np.ndarray) else modulus
yield_strength = (np.interp(avg_innerwall_temp, yield_temps, yield_stress)
                  if isinstance(yield_stress, np.ndarray) else yield_stress)
coeff_thermal_expansion = (np.interp(avg_innerwall_temp, cte_temps, cte)
                           if isinstance(cte, np.ndarray) else cte)
thermal_conductivity = (np.interp(avg_innerwall_temp, conductivity_temps, conductivity)
                        if isinstance(conductivity, np.ndarray) else conductivity)

longitudinal_thermal_stress = E * coeff_thermal_expansion * dT / (2 * (1 - v))
tangential_thermal_stress   = (E * coeff_thermal_expansion * qg_fc_array * tw) / (2 * (1 - v) * thermal_conductivity)
tangential_pressure_stress  = channel_dP * (channel_width / tw) ** 2 * 0.5
crit_longitudinal_buckling_stress = E * tw / (np.sqrt(3 * (1 - v ** 2)) * radii)
tot_tangential_stress = tangential_pressure_stress + tangential_thermal_stress
von_mises_stress = np.sqrt(tot_tangential_stress ** 2 + longitudinal_thermal_stress ** 2
                           - tot_tangential_stress * longitudinal_thermal_stress)
yield_sf = yield_strength / von_mises_stress

# -------- 3D helpers --------
def setup_3d_subplot(fig, position, title, xlabel='X (mm)', ylabel='Y (mm)', zlabel='Axial Position (mm)'):
    ax = fig.add_subplot(position, projection='3d')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    ax.set_title(title)
    return ax


def set_equal_axes(ax, X, Y, Z):
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    eps = 1e-9 if max_range == 0 else 0
    ax.set_xlim(mid_x - max_range - eps, mid_x + max_range + eps)
    ax.set_ylim(mid_y - max_range - eps, mid_y + max_range + eps)
    ax.set_zlim(mid_z - max_range - eps, mid_z + max_range + eps)


def rotate_and_plot_surfaces_fast(ax, X, Y, Z, facecolors, num_channels_to_plot):
    phis = np.linspace(0, 2 * np.pi, num_channels_to_plot, endpoint=False)
    for ci, si in zip(np.cos(phis), np.sin(phis)):
        Xr = X * ci - Y * si
        Yr = X * si + Y * ci
        ax.plot_surface(Xr, Yr, Z, facecolors=facecolors, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False, alpha=1.0)


def annotate_extreme_point_fast(ax, X, Y, Z, target_array, label_coords=None, text_prefix="Value", is_max=True):
    if np.all(np.isnan(target_array)):
        return
    idx = np.unravel_index((np.nanargmax if is_max else np.nanargmin)(target_array), target_array.shape)
    if label_coords is None:
        r_plot = float(np.hypot(X, Y).max())
        label_coords = (1.15 * r_plot, -0.60 * r_plot, float(Z.min() + 0.35 * (Z.max() - Z.min())))
    lx, ly, lz = label_coords
    ax.text(lx, ly, lz, f"{text_prefix}: {float(target_array[idx]):.2f}", color='black',
            fontsize=12, fontweight='bold')


if SF_and_Temps_3D:
    thetas = np.linspace(0.0, np.deg2rad(arc_angle), 30)
    Nr, Nt = radii.size, thetas.size
    Theta, Radius = np.meshgrid(thetas, radii)
    X = Radius * np.cos(Theta)
    Y = Radius * np.sin(Theta)
    Z = np.tile(ax_pos[:, None], (1, Nt))

    DS_R = max(1, Nr // 220)
    DS_T = max(1, Nt // 48)
    Xs, Ys, Zs = X[::DS_R, ::DS_T], Y[::DS_R, ::DS_T], Z[::DS_R, ::DS_T]

    Yield_SF_s = np.tile(yield_sf[:, None], (1, Nt))[::DS_R, ::DS_T]
    Firewall_Temp_s = np.tile(Twg_fc_array[:, None], (1, Nt))[::DS_R, ::DS_T]

    cmap_sf = plt.get_cmap('jet_r')
    norm_sf = plt.Normalize(vmin=0, vmax=12)
    faces_sf = cmap_sf(norm_sf(np.clip(Yield_SF_s, 0, 12)))

    cmap_t = plt.get_cmap('hot_r')
    norm_t = plt.Normalize(vmin=float(np.nanmin(Firewall_Temp_s)), vmax=float(np.nanmax(Firewall_Temp_s)))
    faces_t = cmap_t(norm_t(Firewall_Temp_s))

    fig = plt.figure(figsize=(16, 8))

    ax1 = setup_3d_subplot(fig, 121, "Yield Safety Factor Contours")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_sf, norm=norm_sf), ax=ax1, shrink=0.6,
                 label='Yield Safety Factor (capped at 12)')
    set_equal_axes(ax1, Xs, Ys, Zs)
    ax1.invert_zaxis()
    rotate_and_plot_surfaces_fast(ax1, Xs, Ys, Zs, faces_sf, n)
    annotate_extreme_point_fast(ax1, Xs, Ys, Zs, Yield_SF_s, text_prefix="Minimum SF", is_max=False)
    ax1.view_init(elev=10, azim=210)

    ax2 = setup_3d_subplot(fig, 122, "Firewall Temperature Contours")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_t, norm=norm_t), ax=ax2, shrink=0.6, label='Firewall Temperature (K)')
    set_equal_axes(ax2, Xs, Ys, Zs)
    ax2.invert_zaxis()
    rotate_and_plot_surfaces_fast(ax2, Xs, Ys, Zs, faces_t, n)
    annotate_extreme_point_fast(ax2, Xs, Ys, Zs, Firewall_Temp_s, text_prefix="Max Temp. (K)", is_max=True)
    ax2.view_init(elev=10, azim=210)

    plt.tight_layout()
    plt.show()

if Stress_Plot:
    fig, axp = plt.subplots()
    axp.plot(ax_pos, tangential_pressure_stress / 1e6, label='Tangential pressure stress')
    axp.plot(ax_pos, tangential_thermal_stress / 1e6, label='Tangential thermal stress')
    axp.plot(ax_pos, von_mises_stress / 1e6, label='Von Mises', linestyle=':')
    axp.plot(ax_pos, yield_strength / 1e6, label='Yield strength', linestyle='--')
    axp.set_xlabel("Axial position [m]")
    axp.set_ylabel("Stress [MPa]")
    axp.set_title("Wall stresses")
    axp.legend(); axp.grid(alpha=0.3)
    plt.show()

# %% ========================== Export to RPA Format ==========================
output_filename = "Sim_Thermals.txt"
N = x.size

wc_export  = np.zeros(N)
rho_export = np.zeros(N)
Aflow_fwd  = channel_height * channel_width
mass_flow_rate_ch = fuel_mdot / n

for i in range(N):
    try:
        state = get_ethanol_state(Tc_s[i], Pcool[i], phase_hint="liquid")
        rho_export[i] = state.density
        wc_export[i] = mass_flow_rate_ch / (rho_export[i] * Aflow_fwd[i])
    except Exception:
        rho_export[i] = 0.0
        wc_export[i] = 0.0

with open(output_filename, 'w') as f:
    f.write("#" + "-" * 158 + "\n")
    f.write("# Results of thermal analysis\n")
    f.write("#" + "-" * 158 + "\n")
    f.write("#Location, \t Radius, mm\t Conv. heat\t Conv. heat\t Rad. heat \t Total heat\t     Twg, K\t     Twi, K\t     Twc, K\t      Tc, K\t    pc, MPa\t    wc, m/s\t rho, kg/m^3\t                       Comments\n")
    f.write("#        mm\t           \t  coeff., k\t  flux, kW/\t flux, kW/m\t  flux, kW/\t           \t           \t           \t           \t           \t           \t            \n")
    f.write("#          \t           \t    W/m^2-K\t        m^2\t         ^2\t        m^2\t           \t           \t           \t           \t           \t           \t            \n")
    f.write("#" + "-" * 158 + "\n")

    for i in range(N):
        loc = x[i]
        rad = r_seg_mm[i]
        h_conv = aT[i] / 1000.0
        q_conv = qg[i] / 1000.0
        q_rad = 0.0   # radiation not modelled
        q_tot = q_conv + q_rad
        t_wg = Twg[i]
        t_wi = Twg[i]   # no TBC defined
        t_wc = Twc[i]
        t_c = Tc_s[i]
        p_c = Pcool[i] / 1e6
        v_c = wc_export[i]
        rho_c = rho_export[i]

        comment = "regenerative cooling, opposite flow"
        if loc <= film_phase_x:
            comment += "; film coolant injection"

        row_str = (f"{loc:10.4f}\t{rad:11.4f}\t{h_conv:11.4f}\t{q_conv:11.4f}\t{q_rad:11.4f}\t{q_tot:11.4f}\t"
                   f"{t_wg:11.4f}\t{t_wi:11.4f}\t{t_wc:11.4f}\t{t_c:11.4f}\t{p_c:11.4f}\t{v_c:11.4f}\t"
                   f"{rho_c:11.4f}\t {comment}\n")
        f.write(row_str)

# %% ============================ Results summary ============================
def _section(title):
    print(f"\n{'=' * 8} {title} {'=' * (60 - len(title))}")


def _loc(arr, fn):
    """Index and axial position [mm] of the min/max of arr."""
    idx = int(fn(arr))
    return idx, x_seg_mm[idx]


_section("PERFORMANCE")
F_total_prop = ox_mdot + fuel_mdot
print(f"Propellants          : {ox} / {fuel}")
print(f"O/F                  : feed {feed_OF:.2f}  |  core {core_OF:.2f}")
print(f"Chamber pressure     : {Chamber_Pressure:.1f} bar  (feed {Feed_Pressure:.1f} bar)")
print(f"Thrust (SL, design)  : {F_SL:.1f} N")
print(f"Isp SL (core flow)   : {ISP_SL_act:.1f} s   |   effective incl. film: {F_SL / (g0 * F_total_prop):.1f} s")
print(f"Isp vac (ideal)      : {ISP_vac_ideal:.1f} s")
print(f"c* ideal / actual    : {cstar_ideal:.1f} / {cstar_act:.1f} m/s")
print(f"Cf SL ideal / actual : {Cf_SL_ideal:.3f} / {Cf_SL_act:.3f}")
print(f"Chamber temp (T0)    : {T0:.0f} K")
print(f"Mass flow            : {F_total_prop:.3f} kg/s total "
      f"({ox_mdot:.3f} ox, {fuel_mdot:.3f} fuel of which {film_mdot:.3f} film)")

_section("GEOMETRY")
print(f"Throat radius / dia  : {Rt_mm:.2f} mm / {2 * Rt_mm:.2f} mm")
print(f"Chamber radius       : {Rc_mm:.2f} mm  (CR = {CR:.1f})")
print(f"Exit radius          : {Re_mm:.2f} mm  (area ratio = {Area_Ratio:.2f})")
print(f"Cylinder length      : {Lcyl_mm:.1f} mm  (L* = {Lstar_target_mm:.0f} mm)")
print(f"Total length         : {x_nodes.max() - x_nodes.min():.1f} mm  (throat at x = {x_throat:.1f} mm)")
print(f"Material / wall      : {Material}, {tw * 1e3:.2f} mm wall")

_section("COOLING CHANNELS")
print(f"Channels             : {n} x {channel_height * 1e3:.2f} mm tall")
print(f"Channel width        : {np.min(channel_width) * 1e3:.2f} mm (min) to {np.max(channel_width) * 1e3:.2f} mm (max)")
print(f"Hydraulic diameter   : {np.min(Dh) * 1e3:.2f} mm (min) to {np.max(Dh) * 1e3:.2f} mm (max)")
print(f"Coolant Reynolds no. : {np.min(Re_fwd):.3g} to {np.max(Re_fwd):.3g}")
print(f"Wall roughness       : {np.min(roughness_node) * 1e6:.0f} to {np.max(roughness_node) * 1e6:.0f} um "
      f"(eps/Dh max {np.max(roughness_node / Dh):.2f})")

_section("COOLANT (ETHANOL)")
P_out_regen = Pcool_fwd[0]    # coolant pressure at the injector end
dP_regen = feed_press - P_out_regen
inj_margin = P_out_regen - Pc_SI
print(f"Inlet / outlet temp  : {tc_nf[0]:.1f} K / {Tc_out_film:.1f} K")
print(f"Max bulk temp        : {np.max(Tc_s):.1f} K")
print(f"Regen pressure drop  : {dP_regen / 1e5:.2f} bar ({100 * dP_regen / feed_press:.1f}% of feed)")
print(f"Injector dP available: {inj_margin / 1e5:.2f} bar ({100 * inj_margin / Pc_SI:.0f}% of Pc)")
if boiling_mask.any():
    print(f"Bulk boiling         : YES, x = {x[boiling_mask].min():.1f} to {x[boiling_mask].max():.1f} mm")
else:
    print("Bulk boiling         : none")
if wall_boil_mask.any():
    print(f"Wall boiling (Twc>Ts): YES, {int(wall_boil_mask.sum())} segments, "
          f"x = {x[wall_boil_mask].min():.1f} to {x[wall_boil_mask].max():.1f} mm")
else:
    print("Wall boiling (Twc>Ts): none")

_section("FILM COOLING")
print(f"Film fraction        : {fcp * 100:.1f}% of total fuel ({film_mdot:.3f} kg/s)")
if burnout_idx == 0:
    print("Film status          : flashes to vapour at injector face")
elif burnout_idx >= throat_idx:
    print(f"Film status          : liquid film survives to throat (x = {x_seg_mm[throat_idx]:.1f} mm)")
else:
    print(f"Film burnout         : x = {film_phase_x:.1f} mm "
          f"({film_phase_x / x_seg_mm[throat_idx] * 100:.0f}% of the way to the throat)")
print(f"Film inlet T         : {T_film_in:.1f} K")
print(f"Coolant T iteration  : {iter_count} iterations, residual {film_resid:.2e}")

_section("THERMAL")
i_twg, x_twg = _loc(Twg_fc_array, np.argmax)
i_qg, x_qg   = _loc(qg_fc_array, np.argmax)
i_twc, x_twc = _loc(Twc_fc_array, np.argmax)
print(f"Max gas-side wall T  : {Twg_fc_array[i_twg]:.0f} K at x = {x_twg:.1f} mm (no film: {np.max(Twg_fwd):.0f} K)")
print(f"Max coolant-side T   : {Twc_fc_array[i_twc]:.0f} K at x = {x_twc:.1f} mm")
print(f"Peak heat flux       : {qg_fc_array[i_qg] / 1e6:.2f} MW/m^2 at x = {x_qg:.1f} mm "
      f"(no film: {np.max(qg_fwd) / 1e6:.2f})")
Q_total = np.sum(qg_fc_array * 2 * np.pi * radii * np.diff(x_nodes) * 1e-3)
print(f"Total heat to coolant: {Q_total / 1e3:.1f} kW")

_section("STRUCTURAL")
i_sf, x_sf = _loc(yield_sf, np.nanargmin)
i_vm, x_vm = _loc(von_mises_stress, np.nanargmax)
print(f"Min yield SF         : {yield_sf[i_sf]:.2f} at x = {x_sf:.1f} mm (Twg there {Twg_fc_array[i_sf]:.0f} K)")
print(f"Max von Mises        : {von_mises_stress[i_vm] / 1e6:.1f} MPa at x = {x_vm:.1f} mm")
print(f"Max thermal / pressure tangential stress: "
      f"{np.nanmax(tangential_thermal_stress) / 1e6:.1f} / {np.nanmax(tangential_pressure_stress) / 1e6:.1f} MPa")

warnings_list = []
if np.nanmin(yield_sf) < 1.0:
    warnings_list.append(f"Yield SF below 1 ({np.nanmin(yield_sf):.2f}) at x = {x_sf:.1f} mm")
if boiling_mask.any():
    warnings_list.append("Coolant bulk temperature exceeds Tsat somewhere in the channels")
if wall_boil_mask.any():
    warnings_list.append(f"Coolant-side wall above Tsat in {int(wall_boil_mask.sum())} segments "
                         f"(local boiling, single-phase correlation not valid there)")
if res['floor_hits']:
    warnings_list.append(f"Coolant pressure hit the 1 bar floor in {res['floor_hits']} segments "
                         f"(feed pressure too low / dP too high)")
if inj_margin < 0.15 * Pc_SI:
    warnings_list.append(f"Injector dP available is only {100 * inj_margin / Pc_SI:.0f}% of Pc")
if film_resid > Film_Tol:
    warnings_list.append(f"Film coolant temperature loop did not converge (residual {film_resid:.2e})")
if np.max(roughness_node / Dh) > 0.05:
    warnings_list.append(f"Relative roughness up to {np.max(roughness_node / Dh):.2f}: "
                         f"outside the range where the Haaland correlation is reliable")

_section("WARNINGS")
if warnings_list:
    for w_msg in warnings_list:
        print(f"! {w_msg}")
else:
    print("None")
print("=" * 70)

print(f"Exported thermal results to {output_filename}")