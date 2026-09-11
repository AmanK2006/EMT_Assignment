import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

class TransmissionLine:
    def __init__(self, R, L, G, C, freq):
        self.R = R
        self.L = L
        self.G = G
        self.C = C
        self.freq = freq
        self.omega = 2 * np.pi * freq
        self.calculate_secondary_parameters()

    def calculate_secondary_parameters(self):
        """Calculates characteristic impedance Z0 and propagation constant gamma."""
        series_Z = self.R + 1j * self.omega * self.L
        shunt_Y = self.G + 1j * self.omega * self.C
        
        self.Z0 = np.sqrt(series_Z / shunt_Y)
        self.gamma = np.sqrt(series_Z * shunt_Y)
        self.alpha = self.gamma.real
        self.beta = self.gamma.imag
        self.wavelength = 2 * np.pi / self.beta

    def plot_alpha_beta_ratio(self, freq_range):
        """Plots the ratio of alpha/beta over a frequency range."""
        ratios = []
        for f in freq_range:
            w = 2 * np.pi * f
            gamma = np.sqrt((self.R + 1j*w*self.L) * (self.G + 1j*w*self.C))
            ratios.append(gamma.real / gamma.imag)
            
        plt.figure(figsize=(8, 5))
        plt.plot(freq_range / 1e9, ratios, 'b-')
        plt.grid(True)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel(r'$\alpha / \beta$')
        plt.title(r'Ratio of $\alpha$ to $\beta$ vs Frequency')
        plt.tight_layout()
        plt.savefig('alpha_beta_ratio.png')
        plt.close()

    def plot_waves_3d(self, T_periods=3, X_wavelengths=4, Gamma_L=1.0):
        """Visualizes forward, backward, and standing waves in 3D."""
        T_period = 1.0 / self.freq
        
        t_vals = np.linspace(0, T_periods * T_period, 50)
        x_vals = np.linspace(0, X_wavelengths * self.wavelength, 50)
        
        X, T = np.meshgrid(x_vals, t_vals)
        
        # Forward wave: exp(j(wt - kx))
        forward = np.exp(1j * (self.omega * T - self.beta * X))
        
        # Backward wave: exp(j(wt + kx))
        backward = np.exp(1j * (self.omega * T + self.beta * X))
        
        # Interfered wave
        interfered = forward + Gamma_L * backward
        
        fig = plt.figure(figsize=(18, 5))
        
        # Forward Wave
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(X, T, forward.real, cmap='viridis')
        ax1.set_xlabel('Space (x)')
        ax1.set_ylabel('Time (t)')
        ax1.set_zlabel('Amplitude')
        ax1.set_title('Forward Wave')
        
        # Backward Wave
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(X, T, backward.real, cmap='viridis')
        ax2.set_xlabel('Space (x)')
        ax2.set_ylabel('Time (t)')
        ax2.set_zlabel('Amplitude')
        ax2.set_title('Backward Wave')
        
        # Standing Wave
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot_surface(X, T, interfered.real, cmap='viridis')
        ax3.set_xlabel('Space (x)')
        ax3.set_ylabel('Time (t)')
        ax3.set_zlabel('Amplitude')
        ax3.set_title(rf'Interfered Wave ($\Gamma = {Gamma_L}$)')
        
        plt.tight_layout()
        plt.savefig('wave_visualization.png')
        plt.close()

    def impedance_transformation(self, Z_L, length):
        """Transforms load impedance over length l."""
        # Using cos and sin to avoid tan asymptotes
        num = Z_L * np.cos(self.beta * length) + 1j * self.Z0 * np.sin(self.beta * length)
        den = self.Z0 * np.cos(self.beta * length) + 1j * Z_L * np.sin(self.beta * length)
        return self.Z0 * (num / den)

    def admittance_transformation(self, Y_L, length):
        """Transforms load admittance over length l."""
        Z_L = 1.0 / Y_L if Y_L != 0 else np.inf
        Z_in = self.impedance_transformation(Z_L, length)
        return 1.0 / Z_in

# Example usage to verify
if __name__ == "__main__":
    R = 0.5
    L = 0.25e-6
    G = 0.0
    C = 100e-12
    freq = 1e9

    tl = TransmissionLine(R, L, G, C, freq)
    print(f"Z0 = {tl.Z0:.2f}")
    print(f"gamma = {tl.gamma:.4f}")
    
    tl.plot_alpha_beta_ratio(np.linspace(1e9, 5e9, 41))
    tl.plot_waves_3d()

    Z_L = 75 + 100j
    Z_in = tl.impedance_transformation(Z_L, 0.1)
    print(f"Input impedance Z_in at l=0.1m: {Z_in:.2f}")

