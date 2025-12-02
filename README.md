# ML Model Comparison for Autonomous Vehicle Dynamics

The Learning Model Predictive Control (LMPC) is a data-driven control framework developed at UCB in the MPC lab. In this example, we implemented the LMPC for the autonomous racing problem. The controller drives several laps on race track and it learns from experience how to drive faster.


<p align="center">
<img src="https://github.com/urosolia/RacingLMPC/blob/master/src/ClosedLoop_multiLap.gif" width="500" />
</p>

### Abstract 

Autonomous racing presents an extreme and highly dynamic control environment in which a vehicle
must not only operate at the limits of conditions like tire friction and external force handling capabilities,
but must also do so repeatedly and reliably across multiple laps. This creates a unique opportunity for
learning-based controllers to leverage past experiences in order to progressively refine performance. In
particular, Learning Model Predictive Control (LMPC) has emerged as a promising iterative framework
in which past closed-loop trajectories contribute to a safe set and cost-to-go approximation, enabling
increasingly optimized future behavior. The success of LMPC critically depends on accurate vehicle
dynamics prediction, which can be achieved through data-driven machine learning models trained on
racing trajectory data.
This project presents a comprehensive benchmarking study of MPC with three machine learning
models; K-Fold Random Forests, Gaussian Process Regression, and Neural Networks for predicting
vehicle dynamics in autonomous racing contexts. The primary objective is to evaluate which approach
provides the most reliable predictions of critical vehicle state variables (longitudinal velocity, lateral
velocity, yaw rate, heading error, track progress, and lateral deviation). All models are trained and
evaluated on a 2,997-sample dataset generated from a three-phase data collection protocol combining
PID control, Model Predictive Control, and Time-Varying MPC trajectories. To assess generalization
robustness, experiments are conducted across multiple train-test splits with statistical validation.
