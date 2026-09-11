import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

class TransmissionLine:
    def __init__(self, resistance, inductance, conductance, capacitance, frequency):
        self.resistance = resistance
        self.inductance = inductance
        self.conductance = conductance
        self.capacitance = capacitance
        self.frequency = frequency
        self.angular_freq = 2 * np.pi * frequency
        self._compute_params()

    def _compute_params(self):
        z_series = self.resistance + 1j * self.angular_freq * self.inductance
        y_shunt = self.conductance + 1j * self.angular_freq * self.capacitance
        
        self.char_impedance = np.sqrt(z_series / y_shunt)
        self.prop_const = np.sqrt(z_series * y_shunt)
        self.attenuation_const = self.prop_const.real
        self.phase_const = self.prop_const.imag
        self.wave_len = 2 * np.pi / self.phase_const if self.phase_const != 0 else np.inf
        self.phase_velocity = self.angular_freq / self.phase_const if self.phase_const != 0 else np.inf
        
        if self.resistance == 0 and self.conductance == 0:
            self.classification = "Lossless"
        elif self.inductance != 0 and self.capacitance != 0 and np.isclose(self.resistance / self.inductance, self.conductance / self.capacitance):
            self.classification = "Distortionless"
        else:
            self.classification = "Lossy"

    def plot_alpha_beta_ratio(self, freq_array):
        ang_freq_arr = 2 * np.pi * freq_array
        gamma_arr = np.sqrt((self.resistance + 1j * ang_freq_arr * self.inductance) * (self.conductance + 1j * ang_freq_arr * self.capacitance))
        ratio_arr = gamma_arr.real / gamma_arr.imag
            
        plt.figure(figsize=(8, 5))
        plt.plot(freq_array / 1e9, ratio_arr, 'b-')
        plt.grid(True)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel(r'$\alpha / \beta$')
        plt.title(r'Ratio of $\alpha$ to $\beta$ vs Frequency')
        plt.tight_layout()
        plt.savefig('alpha_beta_ratio.png')
        plt.close()

    def plot_waves_3d(self, num_periods=3, num_wavelengths=4, load_refl_coeff=1.0):
        period_duration = 1.0 / self.frequency
        
        time_array = np.linspace(0, num_periods * period_duration, 50)
        space_array = np.linspace(0, num_wavelengths * self.wave_len, 50)
        
        space_grid, time_grid = np.meshgrid(space_array, time_array)
        
        fwd_wave = np.exp(1j * (self.angular_freq * time_grid - self.phase_const * space_grid))
        
        bwd_wave = np.exp(1j * (self.angular_freq * time_grid + self.phase_const * space_grid))
        
        total_wave = fwd_wave + load_refl_coeff * bwd_wave
        
        fig = plt.figure(figsize=(18, 5))
        
        ax_fwd = fig.add_subplot(131, projection='3d')
        ax_fwd.plot_surface(space_grid, time_grid, fwd_wave.real, cmap='viridis')
        ax_fwd.set_xlabel('Space (x)')
        ax_fwd.set_ylabel('Time (t)')
        ax_fwd.set_zlabel('Amplitude')
        ax_fwd.set_title('Forward Wave')
        
        ax_bwd = fig.add_subplot(132, projection='3d')
        ax_bwd.plot_surface(space_grid, time_grid, bwd_wave.real, cmap='viridis')
        ax_bwd.set_xlabel('Space (x)')
        ax_bwd.set_ylabel('Time (t)')
        ax_bwd.set_zlabel('Amplitude')
        ax_bwd.set_title('Backward Wave')
        
        ax_tot = fig.add_subplot(133, projection='3d')
        ax_tot.plot_surface(space_grid, time_grid, total_wave.real, cmap='viridis')
        ax_tot.set_xlabel('Space (x)')
        ax_tot.set_ylabel('Time (t)')
        ax_tot.set_zlabel('Amplitude')
        ax_tot.set_title(rf'Interfered Wave ($\Gamma = {load_refl_coeff}$)')
        
        plt.tight_layout()
        plt.savefig('wave_visualization.png')
        plt.close()

    def impedance_transformation(self, load_impedance, line_length):
        refl_coeff_load = (load_impedance - self.char_impedance) / (load_impedance + self.char_impedance)
        refl_coeff_in = refl_coeff_load * np.exp(-2j * self.phase_const * line_length)
        return self.char_impedance * (1 + refl_coeff_in) / (1 - refl_coeff_in)

    def admittance_transformation(self, load_admittance, line_length):
        load_imp = 1.0 / load_admittance if load_admittance != 0 else np.inf
        input_imp = self.impedance_transformation(load_imp, line_length)
        return 1.0 / input_imp

    def analyze_circuit(self, line_length, load_impedance, generator_voltage, generator_impedance):
        refl_coeff_load = (load_impedance - self.char_impedance) / (load_impedance + self.char_impedance)
        mag_gamma = np.abs(refl_coeff_load)
        vswr = (1 + mag_gamma) / (1 - mag_gamma) if mag_gamma != 1 else np.inf
        return_loss = -20 * np.log10(mag_gamma) if mag_gamma != 0 else np.inf
        
        input_impedance = self.impedance_transformation(load_impedance, line_length)
        
        v_in = generator_voltage * input_impedance / (input_impedance + generator_impedance)
        
        v0_plus = v_in / (np.exp(self.prop_const * line_length) + refl_coeff_load * np.exp(-self.prop_const * line_length))
        v0_minus = v0_plus * refl_coeff_load
        
        theta_gamma = np.angle(refl_coeff_load)
        n = np.arange(-10, 10)
        
        d_max = (theta_gamma + 2 * n * np.pi) / (2 * self.phase_const)
        d_max = d_max[(d_max >= 0) & (d_max <= line_length)]
        
        d_min = (theta_gamma + (2 * n - 1) * np.pi) / (2 * self.phase_const)
        d_min = d_min[(d_min >= 0) & (d_min <= line_length)]
        
        return {
            "v_p": self.phase_velocity,
            "classification": self.classification,
            "Gamma_L": refl_coeff_load,
            "VSWR": vswr,
            "ReturnLoss": return_loss,
            "Z_in": input_impedance,
            "V0_plus": np.abs(v0_plus),
            "V0_minus": np.abs(v0_minus),
            "d_max": d_max.tolist(),
            "d_min": d_min.tolist()
        }

if __name__ == "__main__":
    res = 0.5
    ind = 0.25e-6
    cond = 0.0
    cap = 100e-12
    freq_val = 1e9

    tl_obj = TransmissionLine(res, ind, cond, cap, freq_val)
    print(f"Z0 = {tl_obj.char_impedance:.2f}")
    print(f"gamma = {tl_obj.prop_const:.4f}")
    print(f"Phase Velocity = {tl_obj.phase_velocity:.2e} m/s")
    print(f"Classification = {tl_obj.classification}")
    
    tl_obj.plot_alpha_beta_ratio(np.linspace(1e9, 5e9, 41))
    tl_obj.plot_waves_3d()

    load_z = 100 + 50j
    input_z = tl_obj.impedance_transformation(load_z, 1.0)
    print(f"Input impedance Z_in at l=1.0m: {input_z:.2f}")
    
    circuit_params = tl_obj.analyze_circuit(1.0, load_z, 10.0, 50.0)
    print(f"VSWR = {circuit_params['VSWR']:.4f}")
    print(f"Return Loss = {circuit_params['ReturnLoss']:.2f} dB")
