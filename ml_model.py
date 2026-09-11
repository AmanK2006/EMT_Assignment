import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from transmission_line_analytics import TransmissionLine

def create_data_samples(sample_count=5000, rnd_seed=42):
    np.random.seed(rnd_seed)
    
    res_arr = np.random.uniform(0.1, 1.0, sample_count)
    ind_arr = np.random.uniform(0.1e-6, 0.5e-6, sample_count)
    cond_arr = np.random.uniform(0.0, 0.01, sample_count)
    cap_arr = np.random.uniform(50e-12, 150e-12, sample_count)
    freq_arr = np.random.uniform(100e6, 5e9, sample_count)
    
    feature_matrix = []
    target_matrix = []
    
    for idx in range(sample_count):
        tl_inst = TransmissionLine(res_arr[idx], ind_arr[idx], cond_arr[idx], cap_arr[idx], freq_arr[idx])
        ang_f = 2 * np.pi * freq_arr[idx]
        
        feats = [res_arr[idx], ind_arr[idx], cond_arr[idx], cap_arr[idx], freq_arr[idx], 
                    ang_f, ang_f*ind_arr[idx], ang_f*cap_arr[idx]]
        
        targs = [np.abs(tl_inst.char_impedance), tl_inst.phase_const]
        
        feature_matrix.append(feats)
        target_matrix.append(targs)
        
    return np.array(feature_matrix), np.array(target_matrix)

def build_and_test_model():
    print("Generating dataset...")
    feats_all, targs_all = create_data_samples(5000)
    
    feats_tr, feats_te, targs_tr, targs_te = train_test_split(feats_all, targs_all, test_size=0.2, random_state=42)
    
    print("Training Extra Trees Regressor...")
    reg_model = ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42)
    reg_model.fit(feats_tr, targs_tr)
    
    targs_pred = reg_model.predict(feats_te)
    
    r2_vals = r2_score(targs_te, targs_pred, multioutput='raw_values')
    mae_vals = mean_absolute_error(targs_te, targs_pred, multioutput='raw_values')
    print(f"R^2 Scores for [Z0_mag, beta]: {r2_vals}")
    print(f"MAE for [Z0_mag, beta]: {mae_vals}")
    print(f"Average Accuracy (R^2): {np.mean(r2_vals)*100:.2f}%")
    
    plt.figure(figsize=(6, 5))
    plt.scatter(targs_te[:, 0], targs_pred[:, 0], alpha=0.4, color='navy', s=15)
    plt.plot([targs_te[:, 0].min(), targs_te[:, 0].max()], [targs_te[:, 0].min(), targs_te[:, 0].max()], 'r--', lw=2)
    plt.xlabel(r'True $|Z_0|$ ($\Omega$)')
    plt.ylabel(r'Predicted $|Z_0|$ ($\Omega$)')
    plt.title(r'True vs Predicted $|Z_0|$ (Test Set)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('ml_prediction_Z0.png', dpi=300)
    plt.close()

    feat_labels = ['R', 'L', 'G', 'C', 'f', r'$\omega$', r'$\omega L$', r'$\omega C$']
    feat_imps = reg_model.feature_importances_
    plt.figure(figsize=(7, 4))
    bar_plot = plt.bar(feat_labels, feat_imps, color='teal')
    plt.ylabel('Relative Importance')
    plt.title('Extra Trees Feature Importance Analysis')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    for b in bar_plot:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()

    freq_te = feats_te[:, 4]
    err_rel_z0 = np.abs(targs_te[:, 0] - targs_pred[:, 0]) / targs_te[:, 0] * 100
    
    plt.figure(figsize=(7, 4))
    plt.scatter(freq_te / 1e9, err_rel_z0, alpha=0.5, color='crimson', s=12)
    mean_err = np.mean(err_rel_z0)
    plt.axhline(mean_err, color='black', linestyle='--', label=f'Mean Rel Error ({mean_err:.2f}%)')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'Relative Percentage Error in $|Z_0|$ (%)')
    plt.title('Prediction Error Distribution across Frequency Spectrum')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('accuracy_vs_frequency.png', dpi=300)
    plt.close()

    freqs_ext = np.linspace(10e6, 15e9, 200)
    r_ext, l_ext, g_ext, c_ext = 0.5, 0.25e-6, 0.005, 100e-12
    
    feats_ext = []
    targs_ext_true = []
    for f_val in freqs_ext:
        tl_ext = TransmissionLine(r_ext, l_ext, g_ext, c_ext, f_val)
        w_ext = 2 * np.pi * f_val
        feats_ext.append([r_ext, l_ext, g_ext, c_ext, f_val, w_ext, w_ext*l_ext, w_ext*c_ext])
        targs_ext_true.append(np.abs(tl_ext.char_impedance))
        
    feats_ext = np.array(feats_ext)
    targs_ext_pred_all = reg_model.predict(feats_ext)
    targs_ext_pred_z0 = targs_ext_pred_all[:, 0]
    
    plt.figure(figsize=(8, 4.5))
    plt.plot(freqs_ext / 1e9, targs_ext_true, 'b-', label=r'Analytical Exact $|Z_0|$', lw=2)
    plt.plot(freqs_ext / 1e9, targs_ext_pred_z0, 'r--', label=r'ML Model Predicted $|Z_0|$', lw=2)
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

    return reg_model, r2_vals, mae_vals

if __name__ == "__main__":
    build_and_test_model()
