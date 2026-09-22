# Methodology Summary

## Study target

The project estimates cardiorespiratory fitness through VO₂max values indirectly obtained from the Queen's College Step Test (QCST). The reference equations are:

```text
Men:   VO₂max = 111.33 − 0.42 × post-test heart rate
Women: VO₂max = 65.81 − 0.1847 × post-test heart rate
```

VO₂max is expressed in mL·kg⁻¹·min⁻¹.

## Data acquisition

The study included 60 participants. Inertial and orientation measurements were collected at 50 Hz from wearable units located on the chest, hands, and knees. Heart rate and SpO₂ were acquired during the same session. Participants completed rest periods and semi-structured daily activities including folding clothes, sweeping, treadmill walking, moving a box, and stationary cycling.

The protocol was approved by the Institutional Review Board of the University of Puerto Rico at Mayagüez under CPHSI/IRB-UPRM No. 2024070003. Participants provided written informed consent and completed the Physical Activity Readiness Questionnaire before participation.

## Preprocessing

The application restricts both sensor streams to their overlapping interval. IMU signals are processed using high-pass filtering, low-pass filtering, and per-column z-score normalization. The synchronized signals are then divided into overlapping windows.

## Stage 1: continuous MET estimation

Stage 1 converts inertial measurements into a continuous representation of physical intensity without using activity identity as a predictor. Five-second IMU windows are summarized using structured intensity descriptors and provided to a multilayer perceptron regressor.

The saved checkpoint contains:

- Network weights.
- Feature-scaling mean and scale.
- Input feature count.
- Optimized model parameters.

Raw MET predictions are post-processed using median and moving-average smoothing. Derived information includes MET change, absolute MET change, accumulated stability duration, and intensity zone.

## Stage 2: VO₂max estimation

Stage 2 combines movement, estimated MET intensity, heart rate, SpO₂, cross-modal relationships, and participant characteristics. Each observation window is represented by 116 features, including:

- Movement time-, frequency-, and dynamics-based descriptors.
- Coordination among body locations.
- Heart-rate magnitude, variability, trend, and temporal behavior.
- SpO₂ magnitude and temporal behavior.
- Heart-rate response relative to estimated MET intensity.
- Movement and heart-rate coupling.
- MET summary and stability characteristics.
- Gender, age, height, weight, body fat, BMI, resting heart rate, and resting SpO₂.

An XGBoost regressor predicts VO₂max for each valid Stage 2 window. Since VO₂max is a participant-level property, window-level predictions are aggregated using their arithmetic mean.

## Validation

### Leave-One-Subject-Out

In each LOSO fold, all windows from one participant are excluded from training and used only for testing. The model produces one participant-level prediction from the held-out participant's valid windows.

Reported metrics include:

- Average participant-level RMSE.
- Global RMSE across participant-level predictions.
- R².
- Pearson correlation and p-value.
- Linear-fit slope and intercept.

### Leave-One-Activity-Out

LOAO evaluates generalization to complete activities excluded from model training. Activity identity is used only to define the experimental split; it is not included as a Stage 2 predictor. The analysis considers rest, folding clothes, sweeping, walking, box carrying, and cycling.

## Temporal-context evaluation

Stage 2 observation durations of 2, 5, 10, 30, 60, and 90 seconds are evaluated with 50% overlap. Each duration is evaluated with and without the MET-derived stability criterion. The final thesis configuration uses 60-second windows without stability-based exclusion.

## Scope of the repository

The repository contains the inference pipeline, portable training/evaluation scripts, trained final models, experimental model configurations, and documentation. Raw participant data and identifiable metadata are intentionally excluded.
