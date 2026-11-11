import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from os import system
import scipy as sp
import os
channel_arc_angle = None
channel_width = None
system('cls')

## Data input

# material = 'Inconel718'
#material = '6082-T6'
material = 'AlSi10Mg'
# material = 'ABD900'

# channel_dp = 40 # pressure difference across the firewall (bar) - assume to be constant
channel_p = 40 # channel pressure (bar) - assume to be constant

t_w = 1e-3 * 1 # wall thickness (m)

channel_arc_angle = 3.8 # deg
# channel_width = 1e-3 * 0.4 # m

chamber_stagnation_temp = 2903
pc = 33.5
gamma = 1.1674

def machfunc(mach, r, throat_area, gamma):
    area_ratio = (np.pi*r**2) / throat_area
    if mach == 0:
        mach = 1e-7
    return (area_ratio - ((1.0/mach) * ((1 + 0.5*(gamma-1)*mach*mach) / ((gamma + 1)/2))**((gamma+1) / (2*(gamma-1)))))

## Equations
calc_gas_temp = np.vectorize(lambda Tc, gamma, mach: Tc / (1 + 0.5 * (gamma - 1) * mach**2))
calc_gas_pressure = np.vectorize(lambda gas_temp, chamber_temp, pc, gamma: pc * (gas_temp / chamber_temp)**(gamma / (gamma - 1)))
calc_channel_width_angle = lambda r, t_w, angle: 2 * (r + t_w) * np.sin(np.deg2rad(angle/2)) # m
calc_tangential_thermal_stress = np.vectorize(lambda q, E, cte, t_w, v, k: (E * cte * q * t_w) / (2 * (1 - v) * k))
calc_longitudinal_thermal_stress = np.vectorize(lambda firewall_t, coolant_wall_t, E, cte: E * cte * (firewall_t - coolant_wall_t))
temp_pressure_stress = lambda t_w, width, dp: ((width / t_w)**2 * dp * 0.5e5)
calc_tangential_pressure_stress = np.vectorize(temp_pressure_stress)
calc_crit_long_buckling_stress = np.vectorize(lambda r, E, t_w, v: E * t_w / (r * np.sqrt(3*(1 - v**2))))
calc_von_mises_stress = np.vectorize(lambda tang_t, tang_p, long_t: np.sqrt(0.5 * ((tang_t + tang_p - long_t)**2 + (long_t)**2 + (tang_t + tang_p)**2)))

## Read values from RPA
columns = ['axial_pos', 'radius', 'conv_hf_coeff', 'q_conv', 'q_rad', 'q_total', 'tbc_temp', 
           'firewall_temp', 'coolant_wall_temp', 'coolant_temp', 'channel_pressure', 
           'coolant_velocity', 'coolant_density']

base_path = os.path.dirname(__file__)
file_path = r"C:\Repos\iclr_simulations\code\film_cooling_simulations\Sim_Thermals.txt"
# file_path = os.path.join(base_path, 'Sim_Thermals.txt')
with open(file_path, 'r', encoding="utf8") as input_file:
    lines = input_file.readlines()

lines = lines[8:-1]
data = []

for line in lines:
    values = []
    for val in line.strip().split():
        if re.match(r'^[-+]?\d*\.?\d+([eE][-+]?\d+)?$', val):
            values.append(float(val))
    data.append(values)

df = pd.DataFrame(data, columns=columns)

## Read values into arrays
axial_pos = df["axial_pos"].to_numpy(dtype=np.float64) * 1e-3 # m
radius = df["radius"].to_numpy(dtype=np.float64) *1e-3 # m
conv_hf_coeff = df["conv_hf_coeff"].to_numpy(dtype=np.float64) # W/m^2K
q_conv = df["q_conv"].to_numpy(dtype=np.float64) * 1e3 # W/m^2
q_rad = df["q_rad"].to_numpy(dtype=np.float64) * 1e3 # W/m^2
q_total = df["q_total"].to_numpy(dtype=np.float64) * 1e3 # W/m^2
tbc_temp = df["tbc_temp"].to_numpy(dtype=np.float64) # K
firewall_temp = df["firewall_temp"].to_numpy(dtype=np.float64) # K
coolant_wall_temp = df["coolant_wall_temp"].to_numpy(dtype=np.float64) # K
coolant_temp = df["coolant_temp"].to_numpy(dtype=np.float64) # K
rpa_coolant_pressure = df["channel_pressure"].to_numpy(dtype=np.float64) * 10 # bar
coolant_velocity = df["coolant_velocity"].to_numpy(dtype=np.float64) # m/s
coolant_density = df["coolant_density"].to_numpy(dtype=np.float64) # kg/m^3

throat_index = np.argmin(radius)
axial_pos = (axial_pos - axial_pos[throat_index])
min_pos = axial_pos[0]
max_pos = axial_pos[-1]

throat_area = np.pi * radius[throat_index]**2 # m^2

gas_mach = np.zeros_like(radius)

for idx, r in enumerate(radius):
    if idx < throat_index:
        m = sp.optimize.root_scalar(machfunc, bracket=[0, 1], args=(r, throat_area, gamma), method='brentq').root
    if idx >= throat_index:
        m = sp.optimize.root_scalar(machfunc, bracket=[1, 10], args=(r, throat_area, gamma), method='brentq').root
    gas_mach[idx] = m

gas_temp = calc_gas_temp(chamber_stagnation_temp, gamma, gas_mach) # K
gas_pressure = calc_gas_pressure(gas_temp, chamber_stagnation_temp, pc, gamma) # Pa

channel_dp = channel_p - gas_pressure

def set_channel_width(radius, t_w, channel_arc_angle, channel_width):
    if channel_width is not None and channel_arc_angle is not None:
        raise ValueError("Only provide one of channel_width and channel_arc_angle.")
    
    if channel_width is None and channel_arc_angle is None:
        raise ValueError("Either channel_width or channel_arc_angle must be provided.")

    if channel_width is None:
        return calc_channel_width_angle(radius, t_w, channel_arc_angle)
    else:
        return np.full_like(radius, channel_width)

match material:
    case 'AlSi10Mg':
        modulus_temps = np.array([25, 50, 100, 150, 200, 250, 300, 350, 400]) + 273.15 # E Temps (K)
        modulus = np.array([77.6, 75.5, 72.8, 63.2, 60, 55, 45, 37, 28]) * 1e9 # E (Pa)
        yield_temps = np.array([298, 323, 373, 423, 473, 523, 573, 623, 673]) # Ys Temps (K)
        yield_stress = np.array([204, 198, 181, 182, 158, 132, 70, 30, 12]) * 1e6 # Ys (Pa)
        conductivity = 130 # thermal conductivity (W/mK)
        cte = 27e-6 # coefficient of thermal expansion (1/K)
        v = 0.33 # Poisson's ratio

    case '6082-T6':
        ys_0 = 260e6 # Yield stress at room temp (for under 6mm thickness)
        uts_0 = 310e6 # Ultimate tensile strength at room temp (for under 6mm thickness)
        v = 0.3 # Poisson's ratio
        conductivity = 0.07*(firewall_temp-273.15) + 190
        cte = 0.2e-7*(firewall_temp-273.15) + 22.5e-6
        modulus_temps = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 550]) + 273.15 # E Temps (K)
        modulus = np.array([70, 69.3, 67.9, 65.1, 60.2, 54.6, 47.6, 37.8, 28.0, 0]) * 1e9 # E
        yield_temps = np.array([20, 100, 150, 200, 250, 300, 350, 550]) + 273.15 # Ys Temps (K)
        yield_stress = np.array([1, 0.90, 0.79, 0.65, 0.38, 0.20, 0.11, 0]) * ys_0 # Ys

    case 'Inconel718':
        v = 0.28 # Poisson's ratio
        conductivity = 12 # thermal conductivity (W/mK)
        cte = 16e-6 # coefficient of thermal expansion (1/K)
        modulus_temps = np.array([21, 93, 204, 316, 427, 538, 649, 760, 871, 954]) + 273.15 # E Temps (K)
        modulus = np.array([208, 205, 202, 194, 186, 179, 172, 162, 127, 78]) * 1e9 # E (Pa)
        yield_temps = np.array([93, 204, 316, 427, 538, 649, 760]) + 273.15 # Ys Temps (K)
        yield_stress = np.array([1172, 1124, 1096, 1076, 1069, 1027, 758]) * 1e6 # Ys (Pa)

    case 'ABD900':
        modulus_temps = np.array([25, 700, 800, 900]) + 273.15 # E Temps (K)
        modulus = np.array([192, 157, 131, 103]) * 1e9 # E (Pa)
        yield_temps = np.array([25, 427, 538, 649, 732, 760, 788, 816, 843, 871, 927]) + 273.15 # Ys Temps (K)
        yield_stress = np.array([978, 931, 903, 903, 758, 680, 600, 503, 434, 352, 221]) * 1e6 # Ys (Pa)
        cte = np.array([11.4, 12.9, 13.7, 14.4, 15.5, 17.5, 19.2])*1e-6
        cte_temps = np.array([50, 200, 400, 600, 800, 1000, 1200]) +273.15 
        conductivity = np.array([11, 12.6, 15.7, 18.8, 23.2, 26.4, 30.1]) 
        conductivity_temps = np.array([25, 200, 400, 600, 800, 1000, 1200]) + 273.15
        v = 0.28

    case 'GRCop-42':
        modulus_temps = np.array([25, 50]) + 273.15 # E Temps (K)
        modulus = np.array([78.9, 78.9]) * 1e9 # E (Pa)
        yield_temps = np.array([300, 400, 500, 600, 700, 800, 900, 1000]) # Ys Temps (K)
        yield_stress = np.array([175, 170, 160, 150, 135, 120, 95, 70]) * 1e6 # Ys (Pa)
        cte = 20e-6
        conductivity = 250 
        v = 0.33

    case _:
        raise ValueError("Material not recognized. Please check the material name.")






if isinstance(modulus, np.ndarray):
    youngs_modulus = np.zeros_like(firewall_temp)
    valid_temps_modulus = (firewall_temp >= np.min(modulus_temps)) & (firewall_temp <= np.max(modulus_temps))
    youngs_modulus[valid_temps_modulus] = np.interp(firewall_temp[valid_temps_modulus], modulus_temps, modulus)
else:
    youngs_modulus = modulus


if isinstance(yield_stress, np.ndarray):
    yield_strength = np.zeros_like(firewall_temp)
    valid_temps_yield = (firewall_temp >= np.min(yield_temps)) & (firewall_temp <= np.max(yield_temps))
    yield_strength[valid_temps_yield] = np.interp(firewall_temp[valid_temps_yield], yield_temps, yield_stress)
else: 
    yield_strength = yield_stress

if isinstance(cte, np.ndarray):
    coeff_thermal_expansion = np.zeros_like(firewall_temp)
    valid_temps_cte = (firewall_temp >= np.min(cte_temps)) & (firewall_temp <= np.max(cte_temps))
    coeff_thermal_expansion[valid_temps_cte] = np.interp(firewall_temp[valid_temps_cte], cte_temps, cte)
else:
    coeff_thermal_expansion = cte

if isinstance(conductivity, np.ndarray):
    thermal_conductivity = np.zeros_like(firewall_temp)
    valid_temps_tc = (firewall_temp >= np.min(conductivity_temps)) & (firewall_temp <= np.max(conductivity_temps))
    thermal_conductivity[valid_temps_tc] = np.interp(firewall_temp[valid_temps_tc], conductivity_temps, conductivity)

else:
    thermal_conductivity = conductivity




channel_width = set_channel_width(radius, t_w, channel_arc_angle, channel_width)







## Calculations
tangential_thermal_stress = calc_tangential_thermal_stress(q_total, youngs_modulus, coeff_thermal_expansion, t_w, v, thermal_conductivity) # Pa
longitudinal_thermal_stress = calc_longitudinal_thermal_stress(firewall_temp, coolant_wall_temp, youngs_modulus, coeff_thermal_expansion) # Pa
dT = firewall_temp - coolant_wall_temp
tangential_pressure_stress = calc_tangential_pressure_stress(t_w, channel_width, channel_dp) # Pa
crit_long_buckling_stress = calc_crit_long_buckling_stress(radius, youngs_modulus, t_w, v) # Pa
von_mises_stress = calc_von_mises_stress(tangential_thermal_stress, tangential_pressure_stress, longitudinal_thermal_stress) # MPa

yield_sf = np.divide(yield_strength, von_mises_stress) # Safety Factor (Yield)
buckling_sf = np.divide(crit_long_buckling_stress, longitudinal_thermal_stress, out=np.full_like(longitudinal_thermal_stress, np.nan), where=longitudinal_thermal_stress!=0) # Safety Factor (Buckling)
strain = np.divide(von_mises_stress, youngs_modulus, out=np.full_like(youngs_modulus, np.nan), where=youngs_modulus!=0) * 100 # Strain (%)

min_sf_yield = np.min(yield_sf[~np.isnan(yield_sf)]) if np.any(~np.isnan(yield_sf)) else 0
min_sf_buckling = np.min(buckling_sf[~np.isnan(buckling_sf)]) if np.any(~np.isnan(buckling_sf)) else 0

idx_valid = np.where((q_conv > 0) & ~np.isnan(yield_sf) & ~np.isnan(buckling_sf))[0]
display_max_yield_sf = np.max(yield_sf[idx_valid]) if len(idx_valid) > 0 else np.max(yield_sf[~np.isnan(yield_sf)])
display_max_buckling_sf = np.max(buckling_sf[idx_valid]) if len(idx_valid) > 0 else np.max(buckling_sf[~np.isnan(buckling_sf)])

if min_sf_yield < min_sf_buckling:
    min_sf = min_sf_yield
    yield_first = True
else:
    min_sf = min_sf_buckling
    yield_first = False

## Calculate total heat flux (integrates using trapezoidal revolved area)
totHeatFluxInt = 0
for i in range(len(axial_pos)-1):
    dA = np.pi * (radius[i] + radius[i+1]) * np.sqrt((radius[i] - radius[i+1]) ** 2 + (axial_pos[i+1] - axial_pos[i]) ** 2) # mm^2
    totHeatFluxInt += q_total[i] * dA

print(f'Total Heat Flux: {totHeatFluxInt/1e3:.1f} kW')
print(f'Peak Heat Flux: {np.max(q_total)/1e3:.1f} kW/m^2')
print(f'Coolant temperature rise: {np.max(coolant_temp) - np.min(coolant_temp):.1f} deg C')
print(f'Min coolant density: {np.min(coolant_density):.1f} kg/m^3') # Useful to check if coolant is boiling in channels

# Plots
fig, ax = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
fig.suptitle(f"Engine Thermals - {material}")
ax1 = ax[0,0]
ax2 = ax1.twinx()

ax1.plot(axial_pos*1e3, yield_strength*1e-6, color="tab:green", label="Yield Stress")
ax1.plot(axial_pos*1e3, tangential_thermal_stress*1e-6, color="tab:pink", label="Tangential Thermal Stress")
ax1.plot(axial_pos*1e3, longitudinal_thermal_stress*1e-6, color="tab:purple", label="Longitudinal Thermal Stress")
ax1.plot(axial_pos*1e3, tangential_pressure_stress*1e-6, color="tab:orange", label="Tangential Pressure Stress")
ax1.plot(axial_pos*1e3, von_mises_stress*1e-6, color="tab:red", label="Von Mises Stress")
ax1.set_ylim(0, None)

ax2.plot(axial_pos*1e3, youngs_modulus*1e-9, color="tab:blue", label="Young's Modulus")
ax2.set_ylabel("Modulus (GPa)", color="tab:blue")
ax2.set_ylim(0, None)

ax1.set_ylabel("Stress (MPa)")
ax1.set_xlabel("Axial Distance From Throat (mm)")
ax1.set_xlim(min_pos*1e3, max_pos*1e3)
ax[0,0].grid()

legend_lines = ax1.lines + ax2.lines
legend_labels = [l.get_label() for l in legend_lines]
ax1.legend(legend_lines, legend_labels, loc='center left')

ax3 = ax[1,0]
ax4 = ax3.twinx()

ax3.plot(axial_pos*1e3, firewall_temp, label="Firewall Temp", color="tab:orange")
ax3.plot(axial_pos*1e3, coolant_wall_temp, label='Coolant Wall Temp', color="tab:blue")
ax3.plot(axial_pos*1e3, coolant_temp, label='Coolant Temp', color="tab:green")

ax4.plot(axial_pos*1e3, q_total*1e-6, color="tab:red", label="Heat Flux")
ax4.set_ylabel("Heat Flux (MW/m^2)", color="tab:red")
ax4.set_ylim(0, None)

ax3.set_ylabel("Temperature (K)")
ax3.set_xlabel("Axial Distance From Throat (mm)")
ax3.set_xlim(min_pos*1e3, max_pos*1e3)
ax3.grid()

legend_lines = ax3.lines + ax4.lines
legend_labels = [l.get_label() for l in legend_lines]
ax3.legend(legend_lines, legend_labels, loc='upper left')

axial_pos_mm = axial_pos * 1e3

ax5 = ax[0,1]
ax6 = ax5.twinx()
ax5.plot(axial_pos*1e3, yield_sf, color="tab:red", label="Safety Factor (Yield)")
ax6.plot(axial_pos*1e3, buckling_sf, color="tab:blue", label="Safety Factor (Buckling)")
ax6.set_ylabel("Safety Factor (Buckling)", color="tab:blue")
ax5.set_ylabel("Safety Factor (Yield)")
ax5.set_xlabel("Axial Distance From Throat (mm)")
margin = 1.1
ax5.set_ylim(0, display_max_yield_sf * margin)
ax6.set_ylim(0, display_max_buckling_sf * margin)
ax5.set_xlim(min_pos*1e3, max_pos*1e3)
ax5.grid()
if yield_first:
    ax5.set_title(f"Minimum Safety Factor: {min_sf:.3f} by Yield")
    ax5.axhline(y=min_sf, color='tab:red', linestyle='--', label="Minimum Safety Factor")
else:
    ax5.set_title(f"Minimum Safety Factor: {min_sf:.3f} by Buckling")
    ax6.axhline(y=min_sf, color='tab:blue', linestyle='--', label="Minimum Safety Factor")
legend_lines = ax5.lines + ax6.lines
legend_labels = [l.get_label() for l in legend_lines]
ax5.legend(legend_lines, legend_labels, loc='best')

ax7 = ax[1,1]
ax7.plot(axial_pos*1e3, strain, color="tab:blue")
ax7.set_ylabel("Strain (%)")
ax7.set_xlabel("Axial Distance From Throat (mm)")
ax7.set_xlim(min_pos*1e3, max_pos*1e3)
ax7.set_ylim(0, None)
ax7.grid()
ax7.legend(["Strain"])

# Add radius to each subplot
for i in range(2):
    for j in range(2):
        ax_twin = ax[i,j].twinx()
        ax_twin.plot(axial_pos*1e3, radius, color="tab:gray", alpha=1)
        ax_twin.set_ylim(0, np.max(radius)*4)
        ax_twin.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()