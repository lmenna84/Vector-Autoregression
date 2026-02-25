%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR bayes_maxshareSVAR
%% Author: Lorenzo Menna
%% This script demonstrates all features of the bayes_maxshareSVAR function
%% (Bayesian Max share identification with Minnesota prior)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) system for Bayesian max share identification

rng(12345);  % Set seed for reproducibility

% Parameters
K = 3;        % Number of variables
T = 200;      % Sample size
nlags = 2;    % Number of lags

% True VAR coefficient matrices (stable system)
A1 = [0.5  0.1  0.0;
      0.1  0.6  0.1;
      0.0  0.1  0.4];

A2 = [0.2  0.0  0.1;
      0.0  0.1  0.0;
      0.1  0.0  0.2];

nu = [0.5; 1.0; 0.2];  % Constants

% True structural impact matrix
B_true = [1.0  0.0  0.0;
          0.5  1.2  0.0;
          0.3  0.4  0.8];

% Generate structural shocks (orthogonal)
epsilon = randn(K, T);

% Generate VAR data
data = zeros(T, K);
data(1:nlags, :) = randn(nlags, K);  % Initial values

for t = (nlags+1):T
    structural_shock = B_true * epsilon(:, t);
    data(t, :) = nu' + data(t-1, :) * A1' + data(t-2, :) * A2' + structural_shock';
end

disp('========================================');
disp('Data simulation complete');
disp(['Sample size: ' num2str(T)]);
disp(['Number of variables: ' num2str(K)]);
disp('CRITICAL: Variable ordering matters for max share!');
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic Bayesian max share SVAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Bayesian max share SVAR - identifying TFP shock');
disp('First variable = Output (whose variance we maximize)');
disp('========================================');

% IMPORTANT: Put the variable of interest FIRST!
var_names = {'Output', 'Hours', 'Inflation'};
shock_names = {'TFP', 'LaborSupply', 'MonetaryPolicy'};

H = 20;        % Maximize variance at 20 quarters (medium-run)
accum = 1;     % Cumulative variance (default)

EQ1 = bayes_maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                         40, 2000, true);
close all;

disp('Posterior mean of impact matrix B:');
disp(EQ1.B);
fprintf('\n');

disp('Variance decomposition of Output at selected horizons (posterior median):');
disp('Horizon    TFP        LaborSupply   MonetaryPolicy');
horizons = [1, 4, 20, 40];
for h = horizons
    fprintf('%3d        %.2f%%      %.2f%%          %.2f%%\n', h, ...
            EQ1.vardecshock_TFP(1,h), ...
            EQ1.vardecshock_LaborSupply(1,h), ...
            EQ1.vardecshock_MonetaryPolicy(1,h));
end
fprintf('\n');
fprintf('Note: TFP shock explains maximum variance at H=%d (cumulative)\n', H);
fprintf('TFP contribution at H=%d: %.2f%%\n', H, EQ1.vardecshock_TFP(1,H));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Effect of horizon H on identification
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Comparing different horizons H');
disp('Short-run (H=1), Medium-run (H=20), Long-run (H=40)');
disp('Using fewer Gibbs iterations for speed');
disp('========================================');

EQ_H1 = bayes_maxshareSVAR(data, 1, nlags, 1, var_names, shock_names, ...
                           40, 1000, true);
EQ_H20 = bayes_maxshareSVAR(data, 20, nlags, 1, var_names, shock_names, ...
                            40, 1000, true);
EQ_H40 = bayes_maxshareSVAR(data, 40, nlags, 1, var_names, shock_names, ...
                            40, 1000, true);
close all;

disp('TFP shock variance contribution to Output (posterior median):');
disp('Horizon    H=1       H=20      H=40');
for h = [1, 4, 20, 40]
    fprintf('%3d        %.2f%%    %.2f%%    %.2f%%\n', h, ...
            EQ_H1.vardecshock_TFP(1,h), ...
            EQ_H20.vardecshock_TFP(1,h), ...
            EQ_H40.vardecshock_TFP(1,h));
end
fprintf('\n');
disp('Note: Different H identifies different shocks!');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Cumulative (accum=1) vs Point (accum=0) variance
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Cumulative vs Point-in-time variance maximization');
disp('========================================');

H = 20;
EQ_cumul = bayes_maxshareSVAR(data, H, nlags, 1, var_names, shock_names, ...
                              40, 1000, true);  % accum=1
EQ_point = bayes_maxshareSVAR(data, H, nlags, 0, var_names, shock_names, ...
                              40, 1000, true);  % accum=0
close all;

disp('TFP shock variance contribution to Output (posterior median):');
disp('Horizon    Cumulative   Point-in-time');
for h = [1, 10, 20, 30, 40]
    fprintf('%3d        %.2f%%        %.2f%%\n', h, ...
            EQ_cumul.vardecshock_TFP(1,h), ...
            EQ_point.vardecshock_TFP(1,h));
end
fprintf('\n');
fprintf('At H=%d:\n', H);
fprintf('  Cumulative (accum=1): TFP explains %.2f%% of cumulative variance\n', ...
        EQ_cumul.vardecshock_TFP(1,H));
fprintf('  Point (accum=0):      TFP explains %.2f%% of cumulative variance (0 to %d)\n', ...
        EQ_point.vardecshock_TFP(1,H), H);
fprintf('\n');
disp('IMPORTANT: Reported variance decomposition is ALWAYS cumulative!');
disp('With accum=0, each draw explains 100% at H, but median may differ.');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: Variable ordering matters!
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: Demonstrating that variable ordering matters');
disp('========================================');

% Original ordering: Output first
data_order1 = data;
var_names_order1 = {'Output', 'Hours', 'Inflation'};
shock_names_order1 = {'TFP', 'Shock2', 'Shock3'};

% Alternative ordering: Hours first
data_order2 = data(:, [2, 1, 3]);  % Reorder columns
var_names_order2 = {'Hours', 'Output', 'Inflation'};
shock_names_order2 = {'LaborSupply', 'Shock2', 'Shock3'};

H = 20;
EQ_order1 = bayes_maxshareSVAR(data_order1, H, nlags, 1, var_names_order1, ...
                               shock_names_order1, 40, 1000, true);
EQ_order2 = bayes_maxshareSVAR(data_order2, H, nlags, 1, var_names_order2, ...
                               shock_names_order2, 40, 1000, true);
close all;

disp('Order 1: Output first → TFP shock identified');
fprintf('  TFP explains %.2f%% of Output variance at H=%d\n', ...
        EQ_order1.vardecshock_TFP(1,H), H);

disp('Order 2: Hours first → Labor Supply shock identified');
fprintf('  LaborSupply explains %.2f%% of Hours variance at H=%d\n', ...
        EQ_order2.vardecshock_LaborSupply(1,H), H);

fprintf('\n');
disp('CRITICAL: Different orderings identify different shocks!');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Custom shock sizes
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: Bayesian max share SVAR with custom shock sizes');
disp('========================================');

shock_size = 2 * diag(EQ1.B);  % 2 std deviations
EQ3 = bayes_maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                         40, 1000, true, [], [], [], shock_size);
close all;

disp('Comparison of IRF magnitudes at impact (horizon 2):');
disp('           1-SD shock    2-SD shock');
fprintf('Output:    %.4f        %.4f\n', ...
        EQ1.Output_TFP(2,2), EQ3.Output_TFP(2,2));
fprintf('Expected ratio: ~2.0, Actual ratio: %.4f\n', ...
        EQ3.Output_TFP(2,2) / EQ1.Output_TFP(2,2));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: Different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: Bayesian max share with different confidence intervals');
disp('========================================');

EQ4a = bayes_maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                          40, 1000, true, [], [], [], [], 5);    % 90% CI
EQ4b = bayes_maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                          40, 1000, true, [], [], [], [], 16);   % 68% CI (default)
close all;

disp('Output response to TFP shock at horizon 10 (posterior percentiles):');
fprintf('90%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ4a.Output_TFP(3,10), EQ4a.Output_TFP(1,10), ...
        EQ4a.Output_TFP(1,10) - EQ4a.Output_TFP(3,10));
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ4b.Output_TFP(3,10), EQ4b.Output_TFP(1,10), ...
        EQ4b.Output_TFP(1,10) - EQ4b.Output_TFP(3,10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: With seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: Bayesian max share with quarterly seasonal dummies');
disp('========================================');

EQ5 = bayes_maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                         40, 1000, true, [], 'quarter');
close all;

disp('Seasonal patterns controlled for in estimation');
disp('TFP shock maximizes deseasonalized Output variance');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: With exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: Bayesian max share with exogenous oil price shock');
disp('========================================');

% Create oil price shock dummy
exog = zeros(T, 1);
exog(80:85) = 1;  % Oil shock period

% Add oil shock effect to data
data_oil = data;
oil_effect = B_true * [0; 0.5; 1];  % Affects hours and inflation
for t = 80:85
    data_oil(t, :) = data_oil(t, :) + oil_effect';
end

EQ6 = bayes_maxshareSVAR(data_oil, H, nlags, accum, var_names, shock_names, ...
                         40, 1000, true, [], [], exog);
close all;

disp('Oil price shock controlled for via exogenous dummy');
disp('TFP shock maximizes Output variance net of oil effects');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: With restricted reduced-form VAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Bayesian max share with restricted reduced-form VAR');
disp('========================================');

lr = [1 1 0;   % Output affected by output, hours only
      1 1 0;   % Hours affected by output, hours only
      1 1 1];  % Inflation affected by all variables

EQ7 = bayes_maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                         40, 1000, true, lr);
close all;

disp('Reduced-form restrictions on lags imposed via lr matrix');
disp('Max share identification applied to restricted Bayesian VAR');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: Custom Bayesian hyperparameters
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: Custom Minnesota prior hyperparameters');
disp('========================================');

% Default: parbayes = [1, 0.2, 1, 2, 100, 100]
% Tighter prior: reduce variance on own lags
parbayes_tight = [1, 0.1, 1, 2, 100, 100];  % Tighter on own lags

% Looser prior: increase variance
parbayes_loose = [1, 0.5, 1, 2, 100, 100];  % Looser on own lags

EQ_default = bayes_maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                                40, 1000, true, [], [], [], [], 16, []);
EQ_tight = bayes_maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                              40, 1000, true, [], [], [], [], 16, parbayes_tight);
EQ_loose = bayes_maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                              40, 1000, true, [], [], [], [], 16, parbayes_loose);
close all;

disp('Output response to TFP shock at horizon 10:');
fprintf('Default prior: %.4f  [%.4f, %.4f]\n', ...
        EQ_default.Output_TFP(2,10), ...
        EQ_default.Output_TFP(3,10), EQ_default.Output_TFP(1,10));
fprintf('Tight prior:   %.4f  [%.4f, %.4f]\n', ...
        EQ_tight.Output_TFP(2,10), ...
        EQ_tight.Output_TFP(3,10), EQ_tight.Output_TFP(1,10));
fprintf('Loose prior:   %.4f  [%.4f, %.4f]\n', ...
        EQ_loose.Output_TFP(2,10), ...
        EQ_loose.Output_TFP(3,10), EQ_loose.Output_TFP(1,10));
fprintf('\n');
disp('Note: Tighter prior shrinks estimates toward prior mean');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 11: Posterior distribution analysis
%% ========================================================================
disp('========================================');
disp('EXAMPLE 11: Analyzing posterior distributions');
disp('Using EQ.irf_sim to examine posterior uncertainty');
disp('========================================');

% Extract TFP shock impact on Output at horizon 10
irf_draws = squeeze(EQ1.irf_sim(1, 10, 1, :));  % Variable 1, horizon 10, shock 1

% Summary statistics
disp('TFP shock → Output at horizon 10:');
fprintf('  Posterior mean:   %.4f\n', mean(irf_draws));
fprintf('  Posterior median: %.4f\n', median(irf_draws));
fprintf('  Posterior std:    %.4f\n', std(irf_draws));
fprintf('  16th percentile:  %.4f\n', prctile(irf_draws, 16));
fprintf('  84th percentile:  %.4f\n', prctile(irf_draws, 84));
fprintf('\n');

% Plot posterior distribution
figure('Position', [100 100 800 400]);
subplot(1, 2, 1);
histogram(irf_draws, 50, 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7]);
hold on;
xline(median(irf_draws), 'r-', 'LineWidth', 2, 'Label', 'Median');
xline(prctile(irf_draws, 16), 'b--', 'LineWidth', 1.5, 'Label', '16th pct');
xline(prctile(irf_draws, 84), 'b--', 'LineWidth', 1.5, 'Label', '84th pct');
title('Posterior Distribution: TFP → Output (h=10)');
xlabel('IRF value');
ylabel('Density');
grid on;

% Trace plot
subplot(1, 2, 2);
plot(irf_draws, 'k-', 'LineWidth', 0.5);
hold on;
yline(median(irf_draws), 'r-', 'LineWidth', 2);
title('Posterior Draws: TFP → Output (h=10)');
xlabel('Gibbs iteration');
ylabel('IRF value');
grid on;

fprintf('Posterior distribution plot created (Figure 1)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 12: Comparison with frequentist max share
%% ========================================================================
disp('========================================');
disp('EXAMPLE 12: Bayesian vs Frequentist max share SVAR');
disp('========================================');

% Frequentist
EQ_freq = maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                       40, 1000, true);
close all;

% Bayesian
EQ_bayes = bayes_maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                              40, 2000, true);
close all;

disp('Output response to TFP shock at horizon 10:');
fprintf('Frequentist: %.4f  [%.4f, %.4f]\n', ...
        EQ_freq.Output_TFP(2,10), ...
        EQ_freq.Output_TFP(3,10), EQ_freq.Output_TFP(1,10));
fprintf('Bayesian:    %.4f  [%.4f, %.4f]\n', ...
        EQ_bayes.Output_TFP(2,10), ...
        EQ_bayes.Output_TFP(3,10), EQ_bayes.Output_TFP(1,10));
fprintf('\n');

% Plot comparison
figure('Position', [100 100 1000 400]);
subplot(1, 2, 1);
x_fill = [1:40, fliplr(1:40)];
y_fill_freq = [EQ_freq.Output_TFP(1,:), fliplr(EQ_freq.Output_TFP(3,:))];
y_fill_bayes = [EQ_bayes.Output_TFP(1,:), fliplr(EQ_bayes.Output_TFP(3,:))];
fill(x_fill, y_fill_freq, [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
hold on;
fill(x_fill, y_fill_bayes, [0.5 0.5 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(1:40, EQ_freq.Output_TFP(2,:), 'k-', 'LineWidth', 2, 'DisplayName', 'Frequentist');
plot(1:40, EQ_bayes.Output_TFP(2,:), 'b--', 'LineWidth', 2, 'DisplayName', 'Bayesian');
plot(1:40, zeros(1,40), 'k-', 'LineWidth', 0.5);
title('TFP Shock → Output');
xlabel('Horizon');
ylabel('Response');
legend('Location', 'best');
grid on;

subplot(1, 2, 2);
plot(1:40, EQ_freq.vardecshock_TFP(1,:), 'k-', 'LineWidth', 2, 'DisplayName', 'Frequentist');
hold on;
plot(1:40, EQ_bayes.vardecshock_TFP(1,:), 'b--', 'LineWidth', 2, 'DisplayName', 'Bayesian');
xline(H, 'r--', 'LineWidth', 1.5);
title('TFP Contribution to Output Variance');
xlabel('Horizon');
ylabel('Percentage');
legend('Location', 'best');
ylim([0 105]);
grid on;

sgtitle('Frequentist vs Bayesian Max Share');
fprintf('Comparison plot created (Figure 2)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 13: Burn-in and verbose options
%% ========================================================================
disp('========================================');
disp('EXAMPLE 13: Using burn-in and verbose parameters');
disp('========================================');

disp('Running with verbose=true (will print iterations)...');
EQ8 = bayes_maxshareSVAR(data, H, nlags, accum, var_names, shock_names, ...
                         40, 500, true, [], [], [], [], 16, [], 200, true);
close all;

disp(' ');
disp('Burn-in: 200 iterations discarded');
disp('Gibbs: 500 iterations kept for inference');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 14: Only first shock is identified
%% ========================================================================
disp('========================================');
disp('EXAMPLE 14: Understanding partial identification in Bayesian setting');
disp('Only the first shock (TFP) is identified; others are not unique');
disp('========================================');

disp('Key point: Max share identifies ONLY the first shock!');
disp('The remaining K-1 shocks are orthogonal to the first shock,');
disp('but their ordering is arbitrary (rotational indeterminacy).');
fprintf('\n');

% Show that we typically only analyze the first shock
disp('Typical usage: Focus on first shock only');
disp('IRF fields available:');
fprintf('  EQ.Output_TFP           ← Analyze this (identified)\n');
fprintf('  EQ.Output_LaborSupply   ← Usually ignore (not identified)\n');
fprintf('  EQ.Output_MonetaryPolicy← Usually ignore (not identified)\n');
fprintf('\n');

% Show IRF of first shock only with posterior bands
figure('Position', [100 100 800 600]);
for i = 1:K
    subplot(K, 1, i);
    
    % Extract IRF data for first shock only
    eval(['irf_data = EQ1.' var_names{i} '_' shock_names{1} ';']);
    
    % Plot confidence bands
    x_fill = [1:40, fliplr(1:40)];
    y_fill = [irf_data(1,:), fliplr(irf_data(3,:))];
    fill(x_fill, y_fill, [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    hold on;
    
    % Plot posterior median
    plot(1:40, irf_data(2,:), 'k-', 'LineWidth', 2);
    plot(1:40, zeros(1,40), 'k--', 'LineWidth', 0.5);
    
    title([var_names{i} ' response to ' shock_names{1} ' shock (Bayesian)']);
    xlabel('Horizon');
    ylabel('Response');
    grid on;
end
sgtitle('Bayesian Max Share SVAR: Only First Shock (TFP) is Analyzed');
fprintf('First shock IRF visualization created (Figure 3)\n');
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key Bayesian Max Share SVAR features demonstrated:');
disp('- Variable ordering matters (first variable is special)');
disp('- Only first shock is identified (partial identification)');
disp('- Horizon H determines which shock is identified');
disp('- Cumulative (accum=1) vs point (accum=0) variance maximization');
disp('- Posterior median and credible intervals');
disp('- Custom shock sizes and confidence levels');
disp('- Seasonal dummies and exogenous variables');
disp('- Restricted reduced-form VAR');
disp('- Minnesota prior hyperparameters (parbayes)');
disp('- Posterior distribution analysis using EQ.irf_sim');
disp('- Comparison with frequentist max share');
disp('- Burn-in and verbose options');
disp('- Understanding partial identification');