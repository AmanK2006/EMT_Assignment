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
    
    len_arr = np.random.uniform(0.1, 10.0, sample_count)
    load_r_arr = np.random.uniform(10.0, 200.0, sample_count)
    load_x_arr = np.random.uniform(-100.0, 100.0, sample_count)
    gen_v_arr = np.random.uniform(1.0, 20.0, sample_count)
    gen_z_arr = np.random.uniform(10.0, 100.0, sample_count)
    
    feature_matrix = []
    target_matrix = []
    
    for idx in range(sample_count):
        tl_inst = TransmissionLine(res_arr[idx], ind_arr[idx], cond_arr[idx], cap_arr[idx], freq_arr[idx])
        ang_f = 2 * np.pi * freq_arr[idx]
        
        feats = [res_arr[idx], ind_arr[idx], cond_arr[idx], cap_arr[idx], freq_arr[idx], 
                    ang_f, len_arr[idx], load_r_arr[idx], load_x_arr[idx], gen_v_arr[idx], gen_z_arr[idx]]
        
        load_z = load_r_arr[idx] + 1j * load_x_arr[idx]
        circ = tl_inst.analyze_circuit(len_arr[idx], load_z, gen_v_arr[idx], gen_z_arr[idx])
        
        # Ensure return loss is finite for ML model
        ret_loss = circ["ReturnLoss"] if circ["ReturnLoss"] != np.inf else 100.0
        vswr = circ["VSWR"] if circ["VSWR"] != np.inf else 100.0
        
        targs = [
            np.abs(tl_inst.char_impedance), 
            tl_inst.phase_const,
            tl_inst.phase_velocity,
            np.abs(circ["Gamma_L"]),
            vswr,
            ret_loss,
            np.abs(circ["Z_in"]),
            circ["V0_plus"],
            circ["V0_minus"]
        ]
        
        feature_matrix.append(feats)
        target_matrix.append(targs)
        
    return np.array(feature_matrix), np.array(target_matrix)

def build_and_test_model():
    print("Generating dataset...")
    feats_all, targs_all = create_data_samples(5000)
    
    feats_tr, feats_te, targs_tr, targs_te = train_test_split(feats_all, targs_all, test_size=0.2, random_state=42)
    
    print("Training Extra Trees Regressor...")
    reg_model = ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42)
    reg_model.fit(feats_tr, targs_tr)
    
    targs_pred = reg_model.predict(feats_te)
    
    r2_vals = r2_score(targs_te, targs_pred, multioutput='raw_values')
    mae_vals = mean_absolute_error(targs_te, targs_pred, multioutput='raw_values')
    
    target_names = ["|Z0|", "beta", "v_p", "|Gamma_L|", "VSWR", "ReturnLoss", "|Z_in|", "V0+", "V0-"]
    
    print("\nModel Evaluation Results:")
    for i, name in enumerate(target_names):
        print(f"{name} -> R^2: {r2_vals[i]:.4f}, MAE: {mae_vals[i]:.4f}")
        
    print(f"\nAverage Accuracy (R^2): {np.mean(r2_vals)*100:.2f}%")
    
    return reg_model, r2_vals, mae_vals

if __name__ == "__main__":
    build_and_test_model()
