import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tx_line_analytics import TransmissionLine

def generate_dataset(num_samples=5000, seed=42):
    np.random.seed(seed)
    
    R_vals = np.random.uniform(0.1, 1.0, num_samples)
    L_vals = np.random.uniform(0.1e-6, 0.5e-6, num_samples)
    G_vals = np.random.uniform(0.0, 0.01, num_samples)
    C_vals = np.random.uniform(50e-12, 150e-12, num_samples)
    f_vals = np.random.uniform(100e6, 5e9, num_samples)
    
    X = []
    y = []
    
    for i in range(num_samples):
        tl = TransmissionLine(R_vals[i], L_vals[i], G_vals[i], C_vals[i], f_vals[i])
        omega = 2 * np.pi * f_vals[i]
        
        # Feature Engineering: add omega, wL, wC
        features = [R_vals[i], L_vals[i], G_vals[i], C_vals[i], f_vals[i], 
                    omega, omega*L_vals[i], omega*C_vals[i]]
        
        # Targets: Z0_mag, beta
        targets = [np.abs(tl.Z0), tl.beta]
        
        X.append(features)
        y.append(targets)
        
    return np.array(X), np.array(y)

def train_and_evaluate():
    print("Generating dataset...")
    X, y = generate_dataset(5000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    print(f"R^2 Scores for [Z0_mag, beta]: {r2}")
    print(f"MAE for [Z0_mag, beta]: {mae}")
    print(f"Average Accuracy (R^2): {np.mean(r2)*100:.2f}%")
    
    # 1. Scatter Plot: True vs Predicted Z0 magnitude
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.4, color='navy', s=15)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
    plt.xlabel(r'True $|Z_0|$ ($\Omega$)')
    plt.ylabel(r'Predicted $|Z_0|$ ($\Omega$)')
    plt.title(r'True vs Predicted $|Z_0|$ (Test Set)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('ml_prediction_Z0.png', dpi=300)
    plt.close()

    # 2. Feature Importance Plot
    feature_names = ['R', 'L', 'G', 'C', 'f', r'$\omega$', r'$\omega L$', r'$\omega C$']
    importances = model.feature_importances_
    plt.figure(figsize=(7, 4))
    bars = plt.bar(feature_names, importances, color='teal')
    plt.ylabel('Relative Importance')
    plt.title('Random Forest Feature Importance Analysis')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()

    # 3. Accuracy vs Frequency Analysis (In-Distribution vs Frequency Bins)
    freq_test = X_test[:, 4]
    rel_error_Z0 = np.abs(y_test[:, 0] - y_pred[:, 0]) / y_test[:, 0] * 100
    
    plt.figure(figsize=(7, 4))
    plt.scatter(freq_test / 1e9, rel_error_Z0, alpha=0.5, color='crimson', s=12)
    plt.axhline(np.mean(rel_error_Z0), color='black', linestyle='--', label=f'Mean Rel Error ({np.mean(rel_error_Z0):.2f}%)')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'Relative Percentage Error in $|Z_0|$ (%)')
    plt.title('Prediction Error Distribution across Frequency Spectrum')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('accuracy_vs_frequency.png', dpi=300)
    plt.close()

    # 4. Out-of-Distribution Extrapolation Test (0.01 GHz to 15 GHz)
    ext_freqs = np.linspace(10e6, 15e9, 200) # Trained on 100MHz - 5GHz
    ext_R, ext_L, ext_G, ext_C = 0.5, 0.25e-6, 0.005, 100e-12
    
    ext_X = []
    ext_true_Z0 = []
    for f in ext_freqs:
        tl = TransmissionLine(ext_R, ext_L, ext_G, ext_C, f)
        w = 2 * np.pi * f
        ext_X.append([ext_R, ext_L, ext_G, ext_C, f, w, w*ext_L, w*ext_C])
        ext_true_Z0.append(np.abs(tl.Z0))
        
    ext_X = np.array(ext_X)
    ext_pred = model.predict(ext_X)
    ext_pred_Z0 = ext_pred[:, 0]
    
    plt.figure(figsize=(8, 4.5))
    plt.plot(ext_freqs / 1e9, ext_true_Z0, 'b-', label=r'Analytical Exact $|Z_0|$', lw=2)
    plt.plot(ext_freqs / 1e9, ext_pred_Z0, 'r--', label=r'ML Model Predicted $|Z_0|$', lw=2)
    plt.axvspan(0.1, 5.0, color='green', alpha=0.15, label='Training Region (In-Distribution)')
    plt.axvspan(0.01, 0.1, color='red', alpha=0.1, label='Extrapolation Region')
    plt.axvspan(5.0, 15.0, color='red', alpha=0.1)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'$|Z_0|$ ($\Omega$)')
    plt.title('Model In-Distribution vs Out-of-Distribution Extrapolation Limits')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('extrapolation_error.png', dpi=300)
    plt.close()

    return model, r2, mae

if __name__ == "__main__":
    train_and_evaluate()
