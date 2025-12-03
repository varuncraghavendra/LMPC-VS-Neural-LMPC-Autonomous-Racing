# ----------------------------------------------------------------------------------------------------------------------
# THREE-WAY ML COMPARISON 
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('fnc/simulator')
sys.path.append('fnc/controller')
sys.path.append('fnc')

import matplotlib.pyplot as plt
from plot import plotTrajectory, plotClosedLoopLMPC
from initControllerParameters import initMPCParams, initLMPCParams
from PredictiveControllers import MPC, LMPC
from PredictiveModel import PredictiveModel
from Utilities import Regression, PID
from SysModel import Simulator
from Track import Map
from NeuralNetworkDynamics import NNDynamicsPredictor
from KFoldDynamics import KFoldDynamicsPredictor
from GaussianProcessDynamics import GPDynamicsPredictor
import numpy as np
import os

def main():
    print("\n" + "="*100)
    print(" "*20 + "FAIR THREE-WAY ML COMPARISON (Unbiased Evaluation)")
    print("="*100 + "\n")
    
    N, n, d = 14, 6, 2
    x0 = np.array([0.5, 0, 0, 0, 0, 0])
    xS = [x0, x0]
    dt, vt = 0.1, 0.8
    map = Map(0.4)

    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps, _, QterminalSlack, lmpcParameters = initLMPCParams(map, N)
    
    simulator = Simulator(map)
    LMPCsimulator = Simulator(map, multiLap=False, flagLMPC=True)
    
    os.makedirs("models", exist_ok=True)
    
    # Initialize models
    print("="*100)
    print("INITIALIZING IMPROVED ML MODELS")
    print("="*100 + "\n")
    
    nn_predictor = NNDynamicsPredictor(state_dim=n, input_dim=d, hidden_dim=256, learning_rate=1e-3, device='cpu')
    kfold_predictor = KFoldDynamicsPredictor(state_dim=n, input_dim=d, n_folds=5)  # ← FIXED
    gp_predictor = GPDynamicsPredictor(state_dim=n, input_dim=d, noise_level=0.1)
    
    print("✓ Neural Network: 256 units, dropout, early stopping")
    print("✓ K-Fold: 5 folds with Random Forest")
    print("✓ Gaussian Process: Matern kernel, regularized\n")

    # Collect data
    print("="*100)
    print("COLLECTING BASELINE DATA")
    print("="*100 + "\n")
    
    PIDController = PID(vt)
    xPID, uPID, xPID_glob, _ = simulator.sim(xS, PIDController)
    print(f"✓ PID: {xPID.shape[0]} samples")
    
    A_baseline, B_baseline, _ = Regression(xPID, uPID, 1e-7)
    mpcParam.A, mpcParam.B = A_baseline, B_baseline
    mpc = MPC(mpcParam)
    xMPC, uMPC, xMPC_glob, _ = simulator.sim(xS, mpc)
    print(f"✓ MPC: {xMPC.shape[0]} samples")
    
    predictiveModel = PredictiveModel(n, d, map, 1)
    predictiveModel.addTrajectory(xPID, uPID)
    ltvmpcParam.timeVarying = True
    mpc = MPC(ltvmpcParam, predictiveModel)
    xTVMPC, uTVMPC, xTVMPC_glob, _ = simulator.sim(xS, mpc)
    print(f"✓ TV-MPC: {xTVMPC.shape[0]} samples\n")
    
    # Combine data
    all_states = np.vstack([xPID[:-1], xMPC[:-1], xTVMPC[:-1]])
    all_actions = np.vstack([uPID, uMPC, uTVMPC])
    all_next_states = np.vstack([xPID[1:], xMPC[1:], xTVMPC[1:]])
    
    print(f"Total samples: {len(all_states)}")
    
    # Proper 80/20 train/test split
    n_samples = len(all_states)
    indices = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}\n")
    
    # Add training data only
    for i in train_idx:
        nn_predictor.all_states.append(all_states[i])
        nn_predictor.all_actions.append(all_actions[i])
        nn_predictor.all_next_states.append(all_next_states[i])
        
        kfold_predictor.all_states.append(all_states[i])
        kfold_predictor.all_actions.append(all_actions[i])
        kfold_predictor.all_next_states.append(all_next_states[i])
        
        gp_predictor.all_states.append(all_states[i])
        gp_predictor.all_actions.append(all_actions[i])
        gp_predictor.all_next_states.append(all_next_states[i])

    # Train
    print("="*100)
    print("TRAINING ALL THREE ML MODELS")
    print("="*100 + "\n")
    
    print("1 Neural Network...")
    print("-" * 80)
    nn_predictor.train(epochs=200, batch_size=128, verbose=True)
    nn_predictor.save_model("models/nn_improved.pth")
    
    print("\n 2 K-Fold...")
    print("-" * 80)
    kfold_predictor.train(verbose=True)
    kfold_predictor.save_model("models/kfold_improved.pkl")
    
    print("3 Gaussian Process...")
    print("-" * 80)
    gp_predictor.train(verbose=True, use_subset=True, max_samples=1500)
    gp_predictor.save_model("models/gp_improved.pkl")

    # Test on held-out data
    print("\n" + "="*100)
    print("TESTING ON HELD-OUT DATA")
    print("="*100 + "\n")
    
    errors_baseline, errors_nn, errors_kfold, errors_gp = [], [], [], []
    
    for idx in test_idx:
        state, action, true_next = all_states[idx], all_actions[idx], all_next_states[idx]
        
        pred_baseline = A_baseline @ state + B_baseline @ action
        errors_baseline.append(np.abs(pred_baseline - true_next))
        
        pred_nn = nn_predictor.predict(state, action)
        errors_nn.append(np.abs(pred_nn - true_next))
        
        pred_kfold = kfold_predictor.predict(state, action, return_std=False)
        errors_kfold.append(np.abs(pred_kfold - true_next))
        
        pred_gp, _ = gp_predictor.predict(state, action, return_std=True)
        errors_gp.append(np.abs(pred_gp - true_next))
    
    mae_baseline = np.mean(errors_baseline, axis=0)
    mae_nn = np.mean(errors_nn, axis=0)
    mae_kfold = np.mean(errors_kfold, axis=0)
    mae_gp = np.mean(errors_gp, axis=0)
    
    state_names = ['vx', 'vy', 'wz', 'epsi', 's', 'ey']
    
    print("MAE on HELD-OUT TEST DATA:")
    print("┌─────────┬────────────┬────────────┬────────────┬────────────┐")
    print("│  State  │  Baseline  │     NN     │   K-Fold   │     GP     │")
    print("├─────────┼────────────┼────────────┼────────────┼────────────┤")
    
    for i, name in enumerate(state_names):
        print(f"│ {name:7s} │ {mae_baseline[i]:10.6f} │ {mae_nn[i]:10.6f} │ {mae_kfold[i]:10.6f} │ {mae_gp[i]:10.6f} │")
    
    print("└─────────┴────────────┴────────────┴────────────┴────────────┘\n")
    
    # Overall
    overall_baseline = np.mean(mae_baseline)
    overall_nn = np.mean(mae_nn)
    overall_kfold = np.mean(mae_kfold)
    overall_gp = np.mean(mae_gp)
    
    imp_nn = ((overall_baseline - overall_nn) / overall_baseline) * 100
    imp_kfold = ((overall_baseline - overall_kfold) / overall_baseline) * 100
    imp_gp = ((overall_baseline - overall_gp) / overall_baseline) * 100
    
    print("Overall (Unbiased):")
    print(f"  Baseline:    {overall_baseline:.6f}")
    print(f"  NN:          {overall_nn:.6f} ({imp_nn:+.1f}%)")
    print(f"  K-Fold:      {overall_kfold:.6f} ({imp_kfold:+.1f}%)")
    print(f"  GP:          {overall_gp:.6f} ({imp_gp:+.1f}%)\n")
    
    methods = {'NN': overall_nn, 'K-Fold': overall_kfold, 'GP': overall_gp}
    winner = min(methods, key=methods.get)
    print(f" WINNER: {winner}\n")
    
    if imp_gp > 60:
        print("⚠ GP >60% may indicate overfitting on deterministic data\n")
    
    # Plots
    print("="*100)
    print("GENERATING PLOTS")
    print("="*100 + "\n")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1
    x_pos = np.arange(len(state_names))
    width = 0.2
    
    ax1.bar(x_pos - 1.5*width, mae_baseline, width, label='Baseline', color='gray', alpha=0.7)
    ax1.bar(x_pos - 0.5*width, mae_nn, width, label='NN', color='darkgreen', alpha=0.8)
    ax1.bar(x_pos + 0.5*width, mae_kfold, width, label='K-Fold', color='steelblue', alpha=0.8)
    ax1.bar(x_pos + 1.5*width, mae_gp, width, label='GP', color='darkorange', alpha=0.8)
    
    ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax1.set_title('Per-State Accuracy (Held-Out Data)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(state_names)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Plot 2
    methods_names = ['Baseline', 'NN', 'K-Fold', 'GP']
    overall_errors = [overall_baseline, overall_nn, overall_kfold, overall_gp]
    colors = ['gray', 'darkgreen', 'steelblue', 'darkorange']
    
    bars = ax2.bar(methods_names, overall_errors, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Overall MAE', fontsize=12, fontweight='bold')
    ax2.set_title('Overall Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, error in zip(bars, overall_errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.5f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3
    improvements = [0, imp_nn, imp_kfold, imp_gp]
    
    bars = ax3.bar(methods_names, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Relative Improvement', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        offset = 2 if height > 0 else -5
        ax3.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{imp:+.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Plot 4
    imp_per_state = {
        'NN': [(mae_baseline[i] - mae_nn[i]) / mae_baseline[i] * 100 for i in range(6)],
        'K-Fold': [(mae_baseline[i] - mae_kfold[i]) / mae_baseline[i] * 100 for i in range(6)],
        'GP': [(mae_baseline[i] - mae_gp[i]) / mae_baseline[i] * 100 for i in range(6)]
    }
    
    x = np.arange(6)
    width = 0.25
    
    ax4.bar(x - width, imp_per_state['NN'], width, label='NN', color='darkgreen', alpha=0.8)
    ax4.bar(x, imp_per_state['K-Fold'], width, label='K-Fold', color='steelblue', alpha=0.8)
    ax4.bar(x + width, imp_per_state['GP'], width, label='GP', color='darkorange', alpha=0.8)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Per-State Improvement', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(state_names)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('Fair_Comparison.png', dpi=150)
    print("✓ Saved: Fair_Comparison.png\n")
    
    # Results
    print("="*100)
    print("FINAL RESULTS")
    print("="*100 + "\n")
    
    print("┌──────────────┬─────────────┬──────────────┐")
    print("│    Method    │ Overall MAE │ Improvement  │")
    print("├──────────────┼─────────────┼──────────────┤")
    print(f"│ Baseline     │ {overall_baseline:.6f}   │     0.0%     │")
    print(f"│ Neural Net   │ {overall_nn:.6f}   │   {imp_nn:+6.1f}%    │")
    print(f"│ K-Fold       │ {overall_kfold:.6f}   │   {imp_kfold:+6.1f}%    │")
    print(f"│ Gaussian Proc│ {overall_gp:.6f}   │   {imp_gp:+6.1f}%    │")
    print("└──────────────┴─────────────┴──────────────┘\n")
    
    print(f" Best: {winner}\n")
    print("="*100 + "\n")
    
    plt.show()

if __name__== "__main__":
    main()
