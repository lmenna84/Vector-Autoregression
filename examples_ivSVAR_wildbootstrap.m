%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR ivSVAR_wildbootstrap
%% Author: Lorenzo Menna
%% This script demonstrates all features of the ivSVAR_wildbootstrap function
%% (Proxy-SVAR with external instruments, wild bootstrap confidence intervals)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a proxy-SVAR system with:
% - OilPrice (instrumented, K1=1): identified via external instrument
% - Output and Inflation (non-instrumented, K2=2): affected by oil shock
% - External instrument: oil supply news proxy (correlated with oil shock)

rng(12345);  % Set seed for reproducibility

% Parameters
K1 = 1;       % Number of instrumented variables (oil price)
K2 = 2;       % Number of non-instrumented variables (output, inflation)
K  = K1 + K2; % Total variables = 3
T  = 200;     % Sample size
nlags = 2;    % Number of lags

% True VAR coefficient matrices (stable system)
A1 = [0.5  0.0  0.0;   % OilPrice: mostly exogenous
      0.3  0.5  0.1;   % Output: affected by oil and own lags
      0.1  0.2  0.4];  % Inflation: affected by all variables

A2 = [0.2  0.0  0.0;
      0.1  0.2  0.0;
      0.0  0.1  0.2];

nu = [0.3; 0.8; 0.5];  % Constants

% True structural impact matrix (first column = identified oil supply shock)
B_true = [1.0  0.0  0.0;
          0.6  1.0  0.0;
          0.3  0.2  0.9];

% Generate structural shocks (orthogonal)
epsilon = randn(K, T);

% Generate VAR data
data = zeros(T, K);
data(1:nlags,:) = randn(nlags, K);  % Initial values

for t = (nlags+1):T
    structural_shock = B_true * epsilon(:,t);
    data(t,:) = nu' + data(t-1,:)*A1' + data(t-2,:)*A2' + structural_shock';
end

% Construct external instrument (proxy for oil supply shock)
% iv is correlated with epsilon(1,:) but orthogonal to epsilon(2,:) and epsilon(3,:)
iv_relevance = 0.8;
iv = iv_relevance * epsilon(1,:)' + sqrt(1 - iv_relevance^2) * randn(T, 1);

% Split into instrumented and non-instrumented data
data_inst  = data(:, 1:K1);       % OilPrice
data_other = data(:, K1+1:K);     % Output, Inflation

disp('========================================');
disp('Data simulation complete');
disp(['Instrumented variables (K1=' num2str(K1) '): OilPrice']);
disp(['Non-instrumented variables (K2=' num2str(K2) '): Output, Inflation']);
disp(['Sample size: ' num2str(T) ' observations, ' num2str(nlags) ' lags']);
disp('Instrument: oil supply news proxy (relevance = 0.8)');
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic proxy-SVAR with external instrument
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Basic proxy-SVAR (Mertens & Ravn 2013)');
disp('Single external instrument identifies one structural shock');
disp('Wild bootstrap uses Rademacher distribution (+1/-1 with equal probability)');
disp('========================================');

var_names   = {'OilPrice', 'Output', 'Inflation'};
shock_names = {'OilShock'};

EQ1 = ivSVAR_wildbootstrap(data_inst, data_other, iv, nlags, var_names, shock_names, 40, 1000, true);
close all;

disp('Partially-identified impact matrix P (first column identified):');
disp(EQ1.P(:,1));
fprintf('\n');
disp('True first column of B_true (oil shock impact):');
disp(B_true(:,1));
fprintf('\n');
disp('IRF: Output response to Oil Shock (first 10 periods):');
disp('Period    95% CI     Point     5% CI');
for i = 1:10
    fprintf('%3d    %8.4f  %8.4f  %8.4f\n', i, ...
            EQ1.Output_OilShock(1,i), EQ1.Output_OilShock(2,i), EQ1.Output_OilShock(3,i));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Instrument strength diagnostics
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Instrument strength diagnostics');
disp('First-stage robust F-statistic (Stock & Watson 2018)');
disp('========================================');

fprintf('Robust F-statistic for OilShock instrument: %.4f\n', EQ1.robustF);
fprintf('\n');
disp('Rule of thumb: F > 10 indicates a strong instrument');
if EQ1.robustF > 10
    disp('  => Instrument is STRONG (F > 10)');
else
    disp('  => Instrument may be WEAK (F <= 10)');
end
fprintf('\n');
disp('Weak instruments lead to biased and imprecise IV estimates.');
disp('Always report the robust F-statistic when using proxy-SVARs.');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Absolute vs relative IRF normalization
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Absolute vs relative IRF normalization');
disp('rel=0: unit structural shock (default, Mertens & Ravn normalization)');
disp('rel=1: unit shock to the instrumented variable itself');
disp('========================================');

rng(12345);
EQ_abs = ivSVAR_wildbootstrap(data_inst, data_other, iv, nlags, var_names, ...
                               shock_names, 40, 1000, true);
rng(12345);
EQ_rel = ivSVAR_wildbootstrap(data_inst, data_other, iv, nlags, var_names, ...
                               shock_names, 40, 1000, true, [], [], [], [], 5, 1);
close all;

disp('OilPrice impact response (horizon 2):');
fprintf('  Absolute (rel=0): %.4f\n', EQ_abs.OilPrice_OilShock(2,2));
fprintf('  Relative (rel=1): %.4f  (normalized so oil price moves by 1 unit)\n', ...
        EQ_rel.OilPrice_OilShock(2,2));
fprintf('\n');
disp('Output impact response (horizon 2):');
fprintf('  Absolute (rel=0): %.4f\n', EQ_abs.Output_OilShock(2,2));
fprintf('  Relative (rel=1): %.4f\n', EQ_rel.Output_OilShock(2,2));
fprintf('\n');
disp('Use rel=1 when non-invertibility is a concern or when you prefer');
disp('to normalize the shock so the instrumented variable moves by 1 unit at impact.');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: Custom shock size
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: Custom shock size');
disp('Scaling the oil shock to 10 units (e.g., $10 oil price increase)');
disp('========================================');

rng(12345);
EQ_unit = ivSVAR_wildbootstrap(data_inst, data_other, iv, nlags, var_names, ...
                                shock_names, 40, 1000, true);
rng(12345);
EQ_10   = ivSVAR_wildbootstrap(data_inst, data_other, iv, nlags, var_names, ...
                                shock_names, 40, 1000, true, [], [], [], 10);
close all;

disp('Output response to Oil Shock at horizon 10:');
fprintf('  Unit shock (size=1):    %.4f\n', EQ_unit.Output_OilShock(2,10));
fprintf('  Large shock (size=10):  %.4f\n', EQ_10.Output_OilShock(2,10));
fprintf('  Ratio: %.4f (expected ~10.0)\n', ...
        EQ_10.Output_OilShock(2,10) / EQ_unit.Output_OilShock(2,10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: Wild bootstrap with different confidence levels');
disp('========================================');

EQ_90 = ivSVAR_wildbootstrap(data_inst, data_other, iv, nlags, var_names, ...
                              shock_names, 40, 1000, true, [], [], [], [], 5);    % 90% CI
EQ_68 = ivSVAR_wildbootstrap(data_inst, data_other, iv, nlags, var_names, ...
                              shock_names, 40, 1000, true, [], [], [], [], 16);   % 68% CI
close all;

disp('Output response to Oil Shock at horizon 10:');
fprintf('90%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ_90.Output_OilShock(3,10), EQ_90.Output_OilShock(1,10), ...
        EQ_90.Output_OilShock(1,10) - EQ_90.Output_OilShock(3,10));
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ_68.Output_OilShock(3,10), EQ_68.Output_OilShock(1,10), ...
        EQ_68.Output_OilShock(1,10) - EQ_68.Output_OilShock(3,10));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: With seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: Proxy-SVAR with quarterly seasonal dummies');
disp('========================================');

EQ_seas = ivSVAR_wildbootstrap(data_inst, data_other, iv, nlags, var_names, ...
                                shock_names, 40, 1000, true, [], 'quarter');
close all;

disp('Quarterly seasonal dummies included in reduced-form VAR');
disp('Wild bootstrap applied: residuals multiplied by +1/-1 (Rademacher weights)');
disp('Seasonal patterns controlled for before wild bootstrap resampling');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: With exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: Proxy-SVAR with exogenous crisis dummy');
disp('Financial crisis period: observations 100-110');
disp('========================================');

% Create financial crisis dummy
exog = zeros(T, 1);
exog(100:110) = 1;

% Add crisis effect to data
data_crisis_inst  = data_inst;
data_crisis_other = data_other;
crisis_effect = B_true * [-1; -2; -1.5];
for t = 100:110
    data_crisis_inst(t,:)  = data_crisis_inst(t,:)  + crisis_effect(1:K1)';
    data_crisis_other(t,:) = data_crisis_other(t,:) + crisis_effect(K1+1:K)';
end

EQ_exog = ivSVAR_wildbootstrap(data_crisis_inst, data_crisis_other, iv, nlags, ...
                                var_names, shock_names, 40, 1000, true, [], [], exog);
close all;

disp('Crisis dummy controlled for via exogenous variable');
disp('IRFs reflect structural oil shock dynamics outside the crisis period');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: With restricted reduced-form VAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: Proxy-SVAR with reduced-form restrictions');
disp('Restriction: Output and Inflation do not Granger-cause OilPrice');
disp('Consistent with treating oil price as predetermined (small open economy)');
disp('========================================');

% lr(i,j)=0: variable j does not affect variable i in the reduced form
lr = [1 0 0;   % OilPrice: not affected by output or inflation
      1 1 1;   % Output: affected by all variables
      1 1 1];  % Inflation: affected by all variables

EQ_lr = ivSVAR_wildbootstrap(data_inst, data_other, iv, nlags, var_names, ...
                              shock_names, 40, 1000, true, lr);
close all;

disp('Granger non-causality restrictions imposed on reduced-form VAR');
disp('Structural identification via IV on restricted residuals');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: Identified structural shocks
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Identified structural shocks');
disp('========================================');

T_eff = T - nlags;
disp('EQ.struc contains the K1 identified structural shocks');
fprintf('Size: %dx%d (%d identified shock x %d effective observations)\n', ...
        size(EQ1.struc,1), size(EQ1.struc,2), K1, T_eff);
fprintf('\n');

disp('Oil shock summary statistics:');
fprintf('  Mean:    %.4f  (expected ~0)\n', mean(EQ1.struc(1,:)));
fprintf('  Std Dev: %.4f\n', std(EQ1.struc(1,:)));
fprintf('  Min:     %.4f\n', min(EQ1.struc(1,:)));
fprintf('  Max:     %.4f\n', max(EQ1.struc(1,:)));
fprintf('\n');

% Plot identified structural shocks
figure('Position', [100 100 1000 300]);
plot(1:T_eff, EQ1.struc(1,:), 'k-', 'LineWidth', 1);
hold on;
plot(1:T_eff, zeros(1,T_eff), 'r--', 'LineWidth', 0.5);
title('Identified Oil Supply Structural Shock');
xlabel('Time'); ylabel('Shock'); grid on;
fprintf('Structural shocks plot created (Figure 1)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: Partial variance decomposition
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: Partial variance decomposition');
disp('========================================');

disp('Note: Variance decomposition is PARTIAL.');
disp('Only one shock is identified (K1=1), so shares do not sum to 100%.');
disp('Shares reflect the fraction of variance explained by the oil shock alone.');
fprintf('\n');

disp('Variance of each variable explained by Oil Shock:');
disp('Horizon    OilPrice    Output    Inflation');
for h = [1, 4, 10, 20, 40]
    fprintf('%3d        %.2f%%      %.2f%%    %.2f%%\n', h, ...
            EQ1.vardecshock_OilShock(1,h), ...
            EQ1.vardecshock_OilShock(2,h), ...
            EQ1.vardecshock_OilShock(3,h));
end
fprintf('\n');

% Plot partial variance decomposition
figure('Position', [100 100 1200 400]);

subplot(1,3,1);
plot(1:40, EQ1.vardecshock_OilShock(1,:), 'k-', 'LineWidth', 2);
title('OilPrice: share from Oil Shock');
xlabel('Horizon'); ylabel('Percentage'); ylim([0 100]); grid on;

subplot(1,3,2);
plot(1:40, EQ1.vardecshock_OilShock(2,:), 'k-', 'LineWidth', 2);
title('Output: share from Oil Shock');
xlabel('Horizon'); ylabel('Percentage'); ylim([0 100]); grid on;

subplot(1,3,3);
plot(1:40, EQ1.vardecshock_OilShock(3,:), 'k-', 'LineWidth', 2);
title('Inflation: share from Oil Shock');
xlabel('Horizon'); ylabel('Percentage'); ylim([0 100]); grid on;

sgtitle('Partial Variance Decomposition (Oil Shock)');
fprintf('Variance decomposition plot created (Figure 2)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 11: IRF visualization with shaded confidence bands
%% ========================================================================
disp('========================================');
disp('EXAMPLE 11: IRF visualization with shaded confidence bands');
disp('========================================');

figure('Position', [100 100 1200 400]);
responses  = {EQ1.OilPrice_OilShock, EQ1.Output_OilShock, EQ1.Inflation_OilShock};
resp_names = {'Oil Price', 'Output', 'Inflation'};

for i = 1:K
    subplot(1, K, i);
    x_fill = [1:40, fliplr(1:40)];
    y_fill = [responses{i}(1,:), fliplr(responses{i}(3,:))];
    fill(x_fill, y_fill, [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    hold on;
    plot(1:40, responses{i}(2,:), 'k-', 'LineWidth', 2);
    plot(1:40, zeros(1,40), 'r--', 'LineWidth', 0.5);
    title([resp_names{i} ' \leftarrow Oil Shock']);
    xlabel('Horizon'); ylabel('Response'); grid on;
end
sgtitle('IRFs to Oil Supply Shock (90% Wild Bootstrap CI)');
fprintf('IRF visualization plot created (Figure 3)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 12: Verbose option
%% ========================================================================
disp('========================================');
disp('EXAMPLE 12: Verbose option for bootstrap monitoring');
disp('========================================');

disp('Running with verbose=true (will print bootstrap iteration numbers)...');
EQ_verb = ivSVAR_wildbootstrap(data_inst, data_other, iv, nlags, var_names, ...
                                shock_names, 40, 100, true, [], [], [], [], 5, 0, true);
close all;

disp(' ');
disp('With verbose=false (default), no iteration numbers are printed.');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 13: When to use wild bootstrap
%% ========================================================================
disp('========================================');
disp('EXAMPLE 13: When to use wild bootstrap for proxy-SVARs');
disp('========================================');

disp('Wild bootstrap is appropriate when:');
disp('  1. Residuals exhibit heteroskedasticity over time');
disp('     - The Rademacher distribution (+1/-1) scales each residual independently');
disp('     - Conditional variance patterns in the data are preserved');
disp('     - Robust to ARCH effects and regime changes in volatility');
disp('  ');
disp('  2. Sample size is short (T < 100)');
disp('     - Wild bootstrap does not require long blocks');
disp('     - No block length selection needed (unlike block bootstrap)');
disp('  ');
disp('  3. Shocks exhibit time-varying volatility');
disp('     - Common in financial data and post-crisis macroeconomic data');
disp('     - Wild bootstrap resamples the sign, not the magnitude, of residuals');
fprintf('\n');

disp('Comparison with block bootstrap:');
disp('  Wild:   Robust to heteroskedasticity, no block length choice');
disp('  Block:  Preserves autocorrelation, requires block length selection');
fprintf('\n');

disp('Wild bootstrap reference: Gonçalves & Kilian (2004, Journal of Econometrics)');
disp('Proxy-SVAR reference:     Mertens & Ravn (2013, AER)');
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key proxy-SVAR (wild bootstrap) features demonstrated:');
disp('- Basic proxy-SVAR with external instrument');
disp('- Instrument strength via robust F-statistic');
disp('- Absolute vs relative IRF normalization (rel parameter)');
disp('- Custom shock sizes');
disp('- Different confidence levels (90% and 68% CI)');
disp('- Quarterly seasonal dummies');
disp('- Exogenous event dummies');
disp('- Restricted reduced-form VAR');
disp('- Identified structural shocks');
disp('- Partial variance decomposition');
disp('- IRF visualization with shaded confidence bands');
disp('- Verbose bootstrap monitoring');
disp('- When to use wild bootstrap');
disp(' ');
disp('Key concepts:');
disp('- Proxy-SVAR identifies K1 shocks using K1 external instruments');
disp('- Partial identification: only K1 shocks out of K are identified');
disp('- Wild bootstrap (Rademacher) is robust to heteroskedasticity');
disp('- Robust F-statistic diagnoses instrument weakness (threshold: F > 10)');
disp('- rel=0: absolute IRFs (unit structural shock, default)');
disp('- rel=1: relative IRFs (instrumented variable moves by 1 unit at impact)');
