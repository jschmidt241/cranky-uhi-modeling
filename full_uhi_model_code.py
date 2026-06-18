"""
earth system modeling fall 2025 final project
models urban heat island effect with numerical methods

author: John Schmidt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Steady state calculations & visualization
ss_surf_temp = 10
ss_depth_temp = 20
ss_depth = 5
def ss_T_z(z):
    return ss_surf_temp + ((ss_depth_temp - ss_surf_temp) * (z / ss_depth))

fig, ax = plt.subplots()
ss_z = np.linspace(0.,5.,100,endpoint=True)
ss_temp = [ss_T_z(z) for z in ss_z]
ax.plot(ss_z, ss_temp, label='T(z)')
ax.set_title('Steady-State Diffusion for Depth d=5')
ax.set_xlabel('z (m)')
ax.set_ylabel('Temperature (◦C)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# Carslaw and Jaeger diffusion calculations
cj_T_avg = 15 #degrees C
cj_A_0 = 7 # degrees C
cj_omega = 7e-5
cj_eta = 3e-7
def cj_T_z(z,t):
    expterm = np.exp(-1 * z * np.sqrt(cj_omega / (2*cj_eta)))
    sinterm = np.sin((cj_omega * t) - (z * np.sqrt(cj_omega / (2*cj_eta))))
    return cj_T_avg + (cj_A_0 * expterm * sinterm)

fig, ax = plt.subplots()
ss_z = np.linspace(0.,5.,500,endpoint=True)
cj_time = 0
cj_temp = [cj_T_z(z,cj_time) for z in ss_z]
ax.plot(ss_z, cj_temp, label='T(z)')
ax.set_title(f'Carslaw and Jaeger Sinusoidal Diffusion at Time t={cj_time}s')
ax.set_xlabel('z (m)')
ax.set_ylabel('Temperature (◦C)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()


# Testing numerical model for diffusion with an analytical solution to sinusoidal forcing:
def sinusoid_surface_temp(t, T_avg, amp, omega):
    return T_avg + (amp * np.sin(omega * t))
fig, ax = plt.subplots()
sin_t = np.linspace(0.,86400.,1000,endpoint=True)
sin_temp = [sinusoid_surface_temp(t, 15, 7, 2 * np.pi / 86400) for t in sin_t]
ax.plot(sin_t, sin_temp, label='Temperature')
ax.set_title(f'')
ax.set_xlabel('Time t')
ax.set_ylabel('Temperature (◦C)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# thomas algorithm implementation from homework 6
def thomas(d, a, b, y):
    N = len(d)
    x = [0] * N
    a_p = []
    y_dp = []
    
    for i in range(N):
        if i == 0:
            a_p.append(a[i] / d[i])
            y_dp.append(y[i] / d[i])
        else:
            denom = d[i] - b[i] * a_p[i-1]
            a_p.append(a[i] / denom)
            y_dp.append((y[i] - b[i] * y_dp[i-1]) / denom)
    
    for i in range(N-1, -1, -1):
        if i == N-1:
            x[i] = y_dp[i]
        else:
            x[i] = y_dp[i] - a_p[i] * x[i+1]
    
    return x

# modified crank-nicolson, accepts dirichlet boundary on both sides instead of ground flux
def crank_nicolson_given_surface(D, dz, init_profile, depth_value, 
                                surface_temps, total_timesteps):
    N_total = len(init_profile) 
    N = N_total - 2 

    A_d = [1 + D] * N  
    A_a = [-D/2] * (N-1) + [0] 
    A_b = [0] + [-D/2] * (N-1) 
        
    T_history = [list(init_profile)]
    T_current = list(init_profile)
    
    for t in range(total_timesteps):
        T_surface = surface_temps[t]
        T_bottom = depth_value
        
        rhs = [0] * N
        for i in range(N):
            j = i + 1 
            rhs[i] = T_current[j] * (1 - D)
            if j > 1:  
                rhs[i] += (D/2) * T_current[j-1] 
            if j < N_total - 2:  
                rhs[i] += (D/2) * T_current[j+1] 
            if j == 1:  
                rhs[i] += (D/2) * T_current[0] 
                rhs[i] += (D/2) * T_surface    
            if j == N_total - 2:  
                rhs[i] += (D/2) * T_bottom 
                rhs[i] += (D/2) * T_bottom 
        T_interior = thomas(A_d, A_a, A_b, rhs)
        
        T_new = [T_surface] + T_interior + [T_bottom]
        
        T_current = T_new
        T_history.append(T_new)
    
    return T_history

# setup a simple example
dt = 1080  # seconds 
dz = 0.02  # m
eta = 3e-7  # m^2/s
D = (eta * dt) / (dz**2)

print(f"D = {D:.4f}")

total_time = 86400 * 3 
total_depth = 1.0 
N_total = int(total_depth / dz) + 1  
z = np.linspace(0, total_depth, N_total)

# Time array
time_array = np.arange(0, total_time, dt)
n_timesteps = len(time_array)

forced_surface = [sinusoid_surface_temp(t, 15, 7, 7e-5) for t in time_array]

# start at mean temperature everywhere
init_condition = [15.0] * N_total

T_history = crank_nicolson_given_surface(
    D, dz, init_condition, 15.0, forced_surface, n_timesteps
)

T_history = np.array(T_history)

# Compare with Carslaw & Jaeger at final time
def carslaw_jaeger(z, t, T_mean, A0, omega, eta):
    decay = np.exp(-z * np.sqrt(omega / (2 * eta)))
    phase = omega * t - z * np.sqrt(omega / (2 * eta))
    return T_mean + A0 * decay * np.sin(phase)

t_final = time_array[-1]
T_numerical = T_history[-1, :]
T_analytical = carslaw_jaeger(z, t_final, 15, 7, 7e-5, eta)

# Calculate RMSE
rmse = np.sqrt(np.mean((T_numerical - T_analytical)**2))

plt.figure(figsize=(10, 6))
plt.plot(z, T_analytical, 'k-', linewidth=2, label='Analytical (C&J)')
plt.plot(z, T_numerical, 'r--', linewidth=2, label=f'Numerical (RMSE={rmse:.4f}°C)')
plt.xlabel('Depth (m)')
plt.ylabel('Temperature (°C)')
plt.title(f'Temperature Profile at t={t_final/3600:.1f} hours')
plt.legend()
plt.grid(True)
plt.show()

print(f"RMSE: {rmse:.4f} °C")




# Full Crank-Nicolson implementation coupled with energy balance at the surface

#iteratively solve for surface temp
def solve_surface_temperature(params, SW_down, T_air, T_subsurface_top, T_surface_guess): 
    alpha = params['albedo']
    k = params['thermal_conductivity']
    dz = params['dz']
    sigma = params['stefan_boltzmann']
    rho = params['rho_air']
    cp = params['c_p']
    C_H = params['bulk_heat_transfer_coeff']
    u = params['wind_speed']
    f = params['evaporative_fraction']
    A = params['anthropogenic_heat']
        
    T_surface = T_surface_guess
    #only iterate 100 times max
    for _ in range(100):
        T_surface_old = T_surface
        T_surface_K = T_surface + 273.15
        T_air_K = T_air + 273.15
        
        # Calculate energy balance components
        R_net = (1 - alpha) * SW_down - sigma * (T_surface_K**4 - T_air_K**4)
        H = rho * cp * C_H * u * (T_surface - T_air)
        # Latent heat (only when R_net > 0)
        if f > 0 and R_net > 0:
            LE = f * R_net
        else:
            LE = 0.0
        numerator = R_net + A - LE + rho * cp * C_H * u * T_air \
                    + k * T_subsurface_top / dz
        denominator = k / dz + rho * cp * C_H * u
        
        T_surface_new = numerator / denominator
        T_surface = 0.5 * T_surface_new + (1 - 0.5) * T_surface_old
        
        if abs(T_surface - T_surface_old) < 0.1:
            T_surface_K = T_surface + 273.15
            R_net = (1 - alpha) * SW_down - sigma * (T_surface_K**4 - T_air_K**4)
            H = rho * cp * C_H * u * (T_surface - T_air)
            if f > 0 and R_net > 0:
                LE = f * R_net
            else:
                LE = 0.0
            G = R_net + A - H - LE
            
            components = {'R_net': R_net,'H': H,'LE': LE,'A': A,'G': G}
            return T_surface, G, components
    # if it didn't converge, will error out
    return (None,None,None)

# version of crank with neumann boundary at surface to accept ground flux
def crank_nicolson_coupled_energy_balance(params, SW_down_series, T_air_series, total_timesteps):
    eta = params['thermal_diffusivity']
    k = params['thermal_conductivity']
    dz = params['dz']
    dt = params['dt']
    depth = params['depth']
    T_bottom = params['T_depth']
    T_init = params['T_initial']
    
    #  setup grid
    z = np.arange(0, depth + dz, dz)
    n_depths = len(z)
    D = eta * dt / (dz**2)
    print(f"D = {D:.4f}")
    
    # setup temperature profile
    T_current = np.ones(n_depths) * T_init
    T_history = np.zeros((total_timesteps + 1, n_depths))
    T_history[0, :] = T_current
    
    G_series = np.zeros(total_timesteps)
    T_surface_series = np.zeros(total_timesteps)
    energy_components = []
    
    # build crank matrices
    N = n_depths - 1
    A_d = [1 + D/2] + [1 + D] * (N-1)
    A_a = [-D/2] * (N-1) + [0]
    A_b = [0] + [-D/2] * (N-1)
    
    T_surface_guess = T_init
    
    for n in range(total_timesteps):
        SW_down = SW_down_series[n]
        T_air = T_air_series[n]
        
        # solve for surface temperature
        T_surface, G, components = solve_surface_temperature(params, SW_down, T_air, T_current[1], T_surface_guess)
        
        # store results
        G_series[n] = G
        T_surface_series[n] = T_surface
        energy_components.append(components)

        # Build RHS
        rhs = [0] * N
        
        for i in range(N):
            j = i
            if j == 0:
                # Surface with Neumann BC, modified matrix
                rhs[i] = T_current[j] * (1 - D/2)
                rhs[i] += (D/2) * T_current[j+1]
                rhs[i] += (D * G * dz) / k
                
            elif j == N - 1:
                # Last interior point
                rhs[i] = T_current[j] * (1 - D)
                rhs[i] += (D/2) * T_current[j-1]
                rhs[i] += (D/2) * T_current[N]
                rhs[i] += (D/2) * T_bottom
                
            else:
                # interior
                rhs[i] = T_current[j] * (1 - D)
                rhs[i] += (D/2) * T_current[j-1]
                rhs[i] += (D/2) * T_current[j+1]
        
        # Solve tridiagonal system with thomas
        T_interior = thomas(A_d, A_a, A_b, rhs)
        
        # Update histories with new solution
        T_new = np.array(T_interior + [T_bottom])
        T_current = T_new
        T_history[n + 1, :] = T_current
        
        # guess for next iteration: use current T_surface
        T_surface_guess = T_surface
    
    diagnostics = {
        'G': G_series,
        'T_surface': T_surface_series,
        'energy_components': energy_components,
    }
    
    return T_history, z, diagnostics

# Physical constants
STEFAN_BOLTZMANN = 5.67e-8  # W/(m2·K4)
RHO_AIR = 1.2  # kg/m3
C_P = 1005  # J/(kg·K)

#define parameters for both urban and rural locations
urban_params = {
    'albedo': 0.2,
    'thermal_diffusivity': 0.5e-6,  # m2/s
    'thermal_conductivity': 1.2,  # W/(m·K)
    'bulk_heat_transfer_coeff': 0.004,
    'wind_speed': 2.5,  # m/s
    'evaporative_fraction': 0.0,
    'anthropogenic_heat': 0.0,  # W/m2
    'rho_air': RHO_AIR,
    'c_p': C_P,
    'stefan_boltzmann': STEFAN_BOLTZMANN,
    'depth': 0.5,  # m
    'dz': 0.01,  # m
    'dt': 1200,  # s (20 min)
    'T_depth': 20.0,  # degree C
    'T_initial': 20.0,  # degree C
}

rural_params = {
    'albedo': 0.25,
    'thermal_diffusivity': 0.25e-6,  # m2/s
    'thermal_conductivity': 0.8,  # W/(m·K)
    'bulk_heat_transfer_coeff': 0.002,
    'wind_speed': 2.5,  # m/s
    'evaporative_fraction': 0.6,
    'anthropogenic_heat': 0.0,  # W/m2
    'rho_air': RHO_AIR,
    'c_p': C_P,
    'stefan_boltzmann': STEFAN_BOLTZMANN,
    'depth': 0.5,  # m
    'dz': 0.01,  # m
    'dt': 1200,  # s (20 min)
    'T_depth': 20.0,  # degree C
    'T_initial': 20.0,  # degree C
}

# setup time series 
total_time = 86400 * 7  # 7 days
time_array = np.arange(0, total_time, urban_params['dt'])
n_timesteps = len(time_array)

# Solar radiation, sample data
omega = 2 * np.pi / 86400  # rad/s (24 hour period)
SW_max = 750  # W/m2 peak solar radiation
SW_down_series = np.array([
    max(0, SW_max * np.sin(omega * t)) for t in time_array
])

# Air temperature sample data
T_air_mean = 27.0  # °C
T_air_amplitude = 5.0  # °C
phase_lag = 3 * 3600  # 3 hour lag behind solar radiation
T_air_series = np.array([
    T_air_mean + T_air_amplitude * np.sin(omega * (t - phase_lag)) 
    for t in time_array
])

T_history_urban, z_urban, diag_urban = crank_nicolson_coupled_energy_balance(
    urban_params, SW_down_series, T_air_series, n_timesteps
)
T_history_rural, z_rural, diag_rural = crank_nicolson_coupled_energy_balance(
    rural_params, SW_down_series, T_air_series, n_timesteps
)

time_array_extended = np.concatenate([[0], time_array])

fig, ax = plt.subplots()
#2, 2, figsize=(15, 10)
# Surface temperatures
# ax = axes[0, 0]
ax.plot(time_array / 3600, diag_urban['T_surface'], 'r-', label='Urban', linewidth=2)
ax.plot(time_array / 3600, diag_rural['T_surface'], 'g-', label='Rural', linewidth=2)
ax.plot(time_array / 3600, T_air_series, 'k--', label='Air temp', alpha=0.5)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Surface Temperature Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([48, 120])  # Show days 2-5
plt.tight_layout()
plt.show()

# ground heat flux
fig, ax = plt.subplots()
ax.plot(time_array / 3600, diag_urban['G'], 'r-', label='Urban', linewidth=2)
ax.plot(time_array / 3600, diag_rural['G'], 'g-', label='Rural', linewidth=2)
ax.axhline(0, color='k', linestyle=':', alpha=0.5)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Ground Heat Flux (W/m²)')
ax.set_title('Ground Heat Flux Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([48, 120])
plt.tight_layout()
plt.show()

# uhi magnitude (urban - rural surface temp)
fig, ax = plt.subplots()
UHI = diag_urban['T_surface'] - diag_rural['T_surface']
ax.plot(time_array / 3600, UHI, 'purple', linewidth=2)
ax.axhline(0, color='k', linestyle=':', alpha=0.5)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('UHI Magnitude (°C)')
ax.set_title('Urban Heat Island Effect (Urban - Rural)')
ax.grid(True, alpha=0.3)
ax.set_xlim([48, 120])
plt.tight_layout()
plt.show()

# tmep profiles at specific time 
fig, ax = plt.subplots()
idx_3pm_day3 = int((72 + 15) * 3600 / urban_params['dt']) 
ax.plot(z_urban, T_history_urban[idx_3pm_day3, :], 'r-', linewidth=2, label='Urban')
ax.plot(z_rural, T_history_rural[idx_3pm_day3, :], 'g-', linewidth=2, label='Rural')
ax.set_xlabel('Depth (m)')
ax.set_ylabel('Temperature (°C)')
ax.set_title(f'Temperature Profile at t={time_array[idx_3pm_day3-1]/3600:.1f} hrs')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#colormap for rural
# transpose to make visualizing easier
X = np.array(T_history_rural).T 
time = np.arange(X.shape[1])
depth_m = np.arange(X.shape[0]) / 100

plt.figure(figsize=(12, 6))
plt.pcolormesh(time, depth_m, X, shading='auto', cmap='RdBu_r')
plt.colorbar(label='temperature')
plt.clim(15,30)
plt.xlabel('Time (hours)')
plt.ylabel('Depth (m)')
plt.gca().invert_yaxis()
plt.title('Rural depth temperature colormap ')
plt.show()

#urban colormap
X = np.array(T_history_urban).T 
time = np.arange(X.shape[1])
depth_m = np.arange(X.shape[0]) / 100

plt.figure(figsize=(12, 6))
plt.pcolormesh(time, depth_m, X, shading='auto', cmap='RdBu_r')
plt.colorbar(label='temperature')
plt.clim(15,30)
plt.xlabel('Time (hours)')
plt.ylabel('Depth (m)')
plt.gca().invert_yaxis()
plt.title('Urban depth temperature colormap ')
plt.show()