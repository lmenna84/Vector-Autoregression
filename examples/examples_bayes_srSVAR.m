%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR bayes_srSVAR
%% Author: Lorenzo Menna
%% This script demonstrates all features of the bayes_srSVAR function
%% (Bayesian Structural VAR with short-run restrictions via Cholesky)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) system with known parameters

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

% True structural impact matrix (lower triangular for Cholesky)
B_true = [1.0  0.0  0.0;
          0.3  0.9  0.0;
          0.1  0.2  0.8];

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
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic Bayesian SVAR with default Minnesota prior
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Basic Bayesian SVAR(2) with Minnesota prior');
disp('Running Gibbs sampler with 2000 draws + 500 burn-in...');
disp('========================================');

% Define variable names
var_names = {'Output', 'Inflation', 'InterestRate'};

tic;
EQ1 = bayes_srSVAR(data, 2, var_names, 40, 2000, true);
elapsed = toc;
close all

fprintf('Gibbs sampler completed in %.2f seconds\n', elapsed);
fprintf('\n');

disp('Posterior mean of Cholesky matrix:');
disp(EQ1.P);
disp('True structural impact matrix:');
disp(B_true);
fprintf('\n');

disp('Posterior mean of constants:');
disp(EQ1.constant_mean);
disp('True constants:');
disp(nu);
fprintf('\n');

% Display IRF for Output responding to Output shock
disp('IRF: Output response to Output shock (first 10 periods):');
disp('Period    84% CI    Median    16% CI');
for i = 1:10
    fprintf('%3d    %7.4f  %7.4f  %7.4f\n', i, ...
            EQ1.Output_Output(1,i), EQ1.Output_Output(2,i), EQ1.Output_Output(3,i));
end
fprintf('\n');

% Variance decomposition at horizon 10
disp('Variance decomposition of Output at horizon 10 (posterior mean):');
fprintf('Output shock:        %.2f%%\n', EQ1.vardecshock_Output(1, 10));
fprintf('Inflation shock:     %.2f%%\n', EQ1.vardecshock_Inflation(1, 10));
fprintf('InterestRate shock:  %.2f%%\n', EQ1.vardecshock_InterestRate(1, 10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Custom Minnesota prior hyperparameters
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Bayesian SVAR with custom prior');
disp('Tighter prior: smaller variance on first own lag');
disp('========================================');

% Custom hyperparameters: [prior mean, prior var, cross-lag ratio, decay, const var, exog var]
% Default:  [1, 0.2, 1, 2, 100, 100]
% Tighter:  [1, 0.1, 1, 2, 100, 100] - half the variance on first own lag
parbayes_tight = [1, 0.1, 1, 2, 100, 100];

EQ2 = bayes_srSVAR(data, 2, var_names, 40, 2000, true, [], [], 16, parbayes_tight);
close all

disp('Comparison of posterior means (first own lag coefficients):');
disp('         Default    Tighter    True');
for i = 1:K
    fprintf('Var %d:   %.4f     %.4f    %.4f\n', i, ...
            EQ1.coefficient_mean(i, i), EQ2.coefficient_mean(i, i), [A1(1,1); A1(2,2); A1(3,3)]);
end
fprintf('Note: Tighter prior pulls estimates closer to prior mean (1.0)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Bayesian SVAR with 95%% credible intervals');
disp('Default is 68%% (16%% bands), now using 95%% (2.5%% bands)');
disp('========================================');

EQ3 = bayes_srSVAR(data, 2, var_names, 40, 2000, true, [], [], 2.5);
close all

disp('Comparison of credible intervals (Output response to Output shock at horizon 5):');
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ1.Output_Output(3,5), EQ1.Output_Output(1,5), ...
        EQ1.Output_Output(1,5) - EQ1.Output_Output(3,5));
fprintf('95%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ3.Output_Output(3,5), EQ3.Output_Output(1,5), ...
        EQ3.Output_Output(1,5) - EQ3.Output_Output(3,5));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: Custom burn-in and verbose mode
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: Custom burn-in with verbose output');
disp('Using 1000 burn-in iterations (default is 500)');
disp('Showing first 10 and last 10 iterations...');
disp('========================================');

% We'll use fewer total draws for demonstration
EQ4 = bayes_srSVAR(data, 2, var_names, 40, 100, true, [], [], 16, [], 1000, true);
close all

fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Restricted Bayesian SVAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: Bayesian SVAR with restrictions');
disp('Restriction: Interest rate does not affect output or inflation');
disp('========================================');

% Restriction matrix
lr = [1 1 0;   % Output affected by output, inflation only
      1 1 0;   % Inflation affected by output, inflation only
      1 1 1];  % Interest rate affected by all variables

EQ5 = bayes_srSVAR(data, 2, var_names, 40, 2000, true, lr);
close all

disp('Posterior mean coefficients on interest rate lags (should be near zero):');
disp('Equation    Lag 1      Lag 2');
fprintf('Output:    %.6f   %.6f\n', EQ5.coefficient_mean(1, 3), EQ5.coefficient_mean(1, 6));
fprintf('Inflation: %.6f   %.6f\n', EQ5.coefficient_mean(2, 3), EQ5.coefficient_mean(2, 6));
fprintf('(Very small but not exactly zero due to soft restriction)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: Bayesian SVAR with exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: Bayesian SVAR with exogenous crisis dummy');
disp('Crisis period: observations 80-90');
disp('========================================');

% Create crisis dummy
exog = zeros(T, 1);
exog(80:90) = 1;

% Add crisis effect to data
data_crisis = data;
crisis_effect = B_true * [-2; -1.5; -1];
for t = 80:90
    data_crisis(t, :) = data_crisis(t, :) + crisis_effect';
end

EQ6 = bayes_srSVAR(data_crisis, 2, var_names, 40, 2000, true, [], exog);
close all

disp('Posterior mean of crisis dummy coefficients:');
disp('           Coefficient');
for i = 1:K
    fprintf('%-15s %.4f\n', var_names{i}, EQ6.exo_mean(i));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: Analyzing posterior distributions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: Posterior distribution analysis');
disp('========================================');

% Analyze posterior of first own-lag coefficient for Output equation
output_own_lag = squeeze(EQ1.coeff_sim(1, 1, :));  % Assuming const, adjust index if needed

% If there's a constant, the first own lag is in position 2
if isfield(EQ1, 'constant_mean')
    output_own_lag = squeeze(EQ1.coeff_sim(1, 2, :));
end

disp('Posterior statistics for Output first own-lag coefficient:');
fprintf('Mean:         %.4f\n', mean(output_own_lag));
fprintf('Median:       %.4f\n', median(output_own_lag));
fprintf('Std Dev:      %.4f\n', std(output_own_lag));
fprintf('2.5%% quant:   %.4f\n', quantile(output_own_lag, 0.025));
fprintf('97.5%% quant:  %.4f\n', quantile(output_own_lag, 0.975));
fprintf('True value:   %.4f\n', A1(1,1));
fprintf('\n');

% Plot posterior distribution
figure('Position', [100 100 800 400]);
subplot(1, 2, 1);
histogram(output_own_lag, 50, 'Normalization', 'pdf', 'FaceColor', [0.3 0.3 0.8]);
hold on;
xline(A1(1,1), 'r--', 'LineWidth', 2, 'Label', 'True value');
xline(mean(output_own_lag), 'k-', 'LineWidth', 2, 'Label', 'Posterior mean');
xlabel('Coefficient value');
ylabel('Density');
title('Posterior: Output first own-lag coefficient');
legend('Location', 'best');
grid on;

% Plot trace of Gibbs draws
subplot(1, 2, 2);
plot(1:2000, output_own_lag, 'k-', 'LineWidth', 0.5);
hold on;
yline(A1(1,1), 'r--', 'LineWidth', 2);
xlabel('Gibbs iteration');
ylabel('Coefficient value');
title('Trace plot (convergence check)');
legend('Draws', 'True value', 'Location', 'best');
grid on;

sgtitle('Posterior Analysis');
fprintf('Posterior distribution plot created (Figure 1)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: Posterior IRF distributions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: Analyzing IRF posterior distributions');
disp('========================================');

% Extract IRF draws for Output response to Output shock at horizon 5
irf_output_h5 = squeeze(EQ1.irf_sim(1, 5, 1, :));  % Variable 1, horizon 5, shock 1, all draws

disp('Posterior statistics for IRF (Output response to Output shock at h=5):');
fprintf('Mean:         %.4f\n', mean(irf_output_h5));
fprintf('Median:       %.4f\n', median(irf_output_h5));
fprintf('Std Dev:      %.4f\n', std(irf_output_h5));
fprintf('16%% quant:    %.4f\n', quantile(irf_output_h5, 0.16));
fprintf('84%% quant:    %.4f\n', quantile(irf_output_h5, 0.84));
fprintf('\n');

% Plot IRF posterior distribution
figure('Position', [100 100 1200 400]);

% Histogram of IRF at single horizon
subplot(1, 3, 1);
histogram(irf_output_h5, 50, 'Normalization', 'pdf', 'FaceColor', [0.8 0.3 0.3]);
hold on;
xline(median(irf_output_h5), 'k-', 'LineWidth', 2, 'Label', 'Median');
xlabel('IRF value');
ylabel('Density');
title('Posterior: IRF at horizon 5');
legend('Location', 'best');
grid on;

% Fan chart: multiple IRF draws
subplot(1, 3, 2);
hold on;
% Plot 100 random draws in light gray
n_draws_plot = 100;
random_idx = randperm(2000, n_draws_plot);
for i = 1:n_draws_plot
    plot(1:40, squeeze(EQ1.irf_sim(1, :, 1, random_idx(i))), ...
         'Color', [0.7 0.7 0.7 0.3], 'LineWidth', 0.5);
end
% Plot median on top
plot(1:40, EQ1.Output_Output(2,:), 'k-', 'LineWidth', 2);
% Plot credible bands
plot(1:40, EQ1.Output_Output(1,:), 'r--', 'LineWidth', 1.5);
plot(1:40, EQ1.Output_Output(3,:), 'r--', 'LineWidth', 1.5);
xlabel('Horizon');
ylabel('Response');
title('IRF Fan Chart (100 posterior draws)');
legend('Posterior draws', '', 'Median', '68% CI', '', 'Location', 'best');
grid on;

% Highest Posterior Density region
subplot(1, 3, 3);
hold on;
% Calculate percentiles for shaded region
irf_p05 = squeeze(quantile(EQ1.irf_sim(1, :, 1, :), 0.05, 4));
irf_p16 = squeeze(quantile(EQ1.irf_sim(1, :, 1, :), 0.16, 4));
irf_p50 = squeeze(quantile(EQ1.irf_sim(1, :, 1, :), 0.50, 4));
irf_p84 = squeeze(quantile(EQ1.irf_sim(1, :, 1, :), 0.84, 4));
irf_p95 = squeeze(quantile(EQ1.irf_sim(1, :, 1, :), 0.95, 4));

% Shaded regions
x_fill = [1:40, fliplr(1:40)];
fill(x_fill, [irf_p05, fliplr(irf_p95)], [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
fill(x_fill, [irf_p16, fliplr(irf_p84)], [0.6 0.6 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
plot(1:40, irf_p50, 'k-', 'LineWidth', 2);
xlabel('Horizon');
ylabel('Response');
title('IRF with Credible Intervals');
legend('90% CI', '68% CI', 'Median', 'Location', 'best');
grid on;

sgtitle('IRF Posterior Distributions');
fprintf('IRF posterior analysis plot created (Figure 2)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: Comparison with frequentist SVAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Bayesian vs Frequentist comparison');
disp('========================================');

% Estimate frequentist SVAR for comparison
EQ_freq = srSVAR(data, 2, var_names, 40, 1000, true);
close all

disp('Comparison of point estimates at horizon 5:');
disp('Output response to Output shock');
fprintf('Bayesian (median):    %.4f  [%.4f, %.4f]\n', ...
        EQ1.Output_Output(2, 5), EQ1.Output_Output(3, 5), EQ1.Output_Output(1, 5));
fprintf('Frequentist (point):  %.4f  [%.4f, %.4f]\n', ...
        EQ_freq.Output_Output(2, 5), EQ_freq.Output_Output(3, 5), EQ_freq.Output_Output(1, 5));
fprintf('\n');

disp('Note: Bayesian credible intervals (68%%) vs Frequentist confidence intervals (90%%)');
disp('Bayesian intervals are typically narrower due to prior information');
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key Bayesian features demonstrated:');
disp('- Minnesota prior specification');
disp('- Gibbs sampler with custom burn-in');
disp('- Posterior distributions of all parameters');
disp('- Credible intervals vs confidence intervals');
disp('- Soft restrictions on VAR coefficients');
disp('- Posterior predictive analysis');