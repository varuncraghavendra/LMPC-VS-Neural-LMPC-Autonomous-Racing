# ML Model Comparison for Autonomous Vehicle Dynamics

The Learning Model Predictive Control (LMPC) is a data-driven control framework developed at UCB in the MPC lab. In this example, we implemented the LMPC for the autonomous racing problem. The controller drives several laps on race track and it learns from experience how to drive faster.


<p align="center">
<img src="https://github.com/urosolia/RacingLMPC/blob/master/src/ClosedLoop_multiLap.gif" width="500" />
</p>

##Abstract

Autonomous racing presents an extreme and highly dynamic control environment in which a vehicle
must not only operate at the limits of tire friction and handling capability but must also do so repeatedly
and reliably across multiple laps. This creates a unique opportunity for learning-based controllers to
leverage past experiences in order to progressively refine performance. In particular, Learning Model
Predictive Control (LMPC) has emerged as a promising iterative framework in which past closed-loop
trajectories contribute to a safe set and cost-to-go approximation, enabling increasingly optimized future
behavior. The success of LMPC critically depends on accurate vehicle dynamics prediction, which can
be achieved through data-driven machine learning models trained on racing trajectory data.
This report presents a comprehensive benchmarking study of three machine learning regression mod-
els K-Fold Random Forests, Gaussian Process Regression, and Neural Networks for predicting vehicle
dynamics in autonomous racing contexts. The primary objective is to evaluate which approach provides
the most reliable predictions of critical vehicle state variables including longitudinal velocity, lateral ve-
locity, yaw rate, heading error, track progress, and lateral deviation. All models are trained and evaluated
on a 2,997-sample dataset generated from a three-phase data collection protocol combining PID control,
Model Predictive Control, and Time-Varying MPC trajectories. To assess generalization robustness, ex-
periments are conducted across multiple train-test splits (70:30, 75:25, 80:20) with statistical validation
using five random seeds per configuration.
The experimental results establish a clear performance hierarchy. Gaussian Processes achieve the
highest predictive accuracy across all metrics, delivering up to 54.4% improvement over baseline linear
regression with particularly strong performance on longitudinal velocity, yaw rate, and lateral position
prediction. K-Fold Random Forests provide an optimal balance between accuracy and computational
efficiency, achieving approximately 30% improvement with excellent robustness across data partitions.
Neural Networks underperform due to limited dataset size, indicating that deep learning approaches re-
quire substantially larger training corpora for competitive performance in vehicle dynamics prediction.
Importantly, per-state analysis reveals that different models excel at different state variables, suggesting
that hybrid or state-specific model selection strategies could yield even greater predictive fidelity. These
findings provide actionable insights for integrating data-driven dynamics models into Learning Model
Predictive Control frameworks, where Gaussian Processes offer the highest accuracy when computa-
tional resources permit, while K-Fold Random Forests provide the best option for real-time closed-loop
control applications.
