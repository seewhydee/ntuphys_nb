L  = [0.10, 0.16, 0.22, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.64];
dL = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
T  = [0.71, 0.76, 0.91, 1.00, 1.20, 1.14, 1.44, 1.40, 1.53, 1.58];
dT = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05];

Tsq = T.*T;
Tsq_error = 2*T.*dT;

%% Fit L and Tsq with weighted linear least squares
wt = 1 ./ (Tsq_error.^2);   % Vector of weights
[est, dev, stats] = glmfit(L, Tsq, 'normal', 'weights', wt);

%% Estimators and their errors:
a = est(2);                 % Estimate of a
b = est(1);                 % Estimate of b
da = stats.se(2);           % Standard error of a
db = stats.se(1);           % Standard error of b

%% Compute the estimate of g
g  = 4*pi*pi/a;
dg = (g/a)*da;

%% Plot the data points and fitted curve
errorbar(L, Tsq, Tsq_error, Tsq_error, dL, dL, "o");
xlabel("Pendulum length L (m)");
ylabel("Squared period T^2 (s^2)");
%% State the fitted value of g in the figure
title(sprintf("g = %.1f +/- %.1f m/s^2", g, dg));
%% Plot the fitted curve
L2 = linspace(0, 0.7, 100);
hold on; plot(L2, a*L2+b); hold off;