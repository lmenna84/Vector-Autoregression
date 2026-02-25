%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR gmmSRSVAR
%% Author: Lorenzo Menna
%% This script demonstrates all features of the gmmSRSVAR function
%% (GMM-estimated Structural VAR with short-run restrictions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) system with known structural shocks

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

% True structural impact matrix (non-recursive pattern for demonstration)
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
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic GMM SVAR with just-identified restrictions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: GMM SVAR with recursive (just-identified) restrictions');
disp('========================================');

var_names = {'Output', 'Inflation', 'InterestRate'};
shock_names = {'Supply', 'Demand', 'MonetaryPolicy'};

% Recursive identification matrix (lower triangular)
idmat_rec = [1  0  0;
             1  1  0;
             1  1  1];

EQ1 = gmmSRSVAR(data, 2, idmat_rec, var_names, shock_names, 40, 1000, true);
close all;

disp('Estimated impact matrix P (GMM):');
disp(EQ1.P);
disp('True impact matrix B:');
disp(B_true);
fprintf('\n');

% Display IRF
disp('IRF: Output response to Supply shock (first 10 periods):');
disp('Period    95% CI    Point    5% CI');
for i = 1:10
    fprintf('%3d    %7.4f  %7.4f  %7.4f\n', i, ...
            EQ1.Output_Supply(1,i), EQ1.Output_Supply(2,i), EQ1.Output_Supply(3,i));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Overidentified GMM SVAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: GMM SVAR with overidentifying restrictions');
disp('More than K(K-1)/2 = 3 zero restrictions');
disp('========================================');

% Overidentified: 4 zero restrictions (one more than minimum)
% Monetary policy shock affects only interest rate on impact
idmat_over = [1  1  0;
              1  1  0;
              0  0  1];

EQ2 = gmmSRSVAR(data, 2, idmat_over, var_names, shock_names, 40, 1000, true);
close all;

disp('Estimated impact matrix P (overidentified):');
disp(EQ2.P);
fprintf('Number of zero restrictions: %d\n', sum(idmat_over(:)==0));
fprintf('Minimum required for identification: %d\n', K*(K-1)/2);
fprintf('Overidentifying restrictions: %d\n', sum(idmat_over(:)==0) - K*(K-1)/2);
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Custom shock sizes
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: GMM SVAR with 2-standard-deviation shocks');
disp('========================================');

shock_size = 2 * diag(EQ1.P);  % 2 std deviations
EQ3 = gmmSRSVAR(data, 2, idmat_rec, var_names, shock_names, 40, 1000, ...
                true, [], [], [], shock_size);
close all;

disp('Comparison of IRF magnitudes at horizon 1:');
disp('           1-SD shock    2-SD shock');
fprintf('Output:    %.4f        %.4f\n', ...
        EQ1.Output_Supply(2,1), EQ3.Output_Supply(2,1));
fprintf('Expected ratio: ~2.0, Actual ratio: %.4f\n', ...
        EQ3.Output_Supply(2,1) / EQ1.Output_Supply(2,1));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: Different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: GMM SVAR with different confidence intervals');
disp('Comparing 90% (default) vs 68% confidence bands');
disp('========================================');

EQ4a = gmmSRSVAR(data, 2, idmat_rec, var_names, shock_names, 40, 1000, ...
                 true, [], [], [], [], 5);    % 90% CI (5% bands)
EQ4b = gmmSRSVAR(data, 2, idmat_rec, var_names, shock_names, 40, 1000, ...
                 true, [], [], [], [], 16);   % 68% CI (16% bands)
close all;

disp('Output response to Supply shock at horizon 5:');
fprintf('90%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ4a.Output_Supply(3,5), EQ4a.Output_Supply(1,5), ...
        EQ4a.Output_Supply(1,5) - EQ4a.Output_Supply(3,5));
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ4b.Output_Supply(3,5), EQ4b.Output_Supply(1,5), ...
        EQ4b.Output_Supply(1,5) - EQ4b.Output_Supply(3,5));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: With seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: GMM SVAR with quarterly seasonal dummies');
disp('========================================');

EQ5 = gmmSRSVAR(data, 2, idmat_rec, var_names, shock_names, 40, 1000, ...
                true, [], 'quarter');
close all;

disp('Note: Seasonal dummies are accounted for in estimation');
disp('IRFs are computed controlling for seasonal patterns');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: With exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: GMM SVAR with exogenous crisis dummy');
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

EQ6 = gmmSRSVAR(data_crisis, 2, idmat_rec, var_names, shock_names, 40, ...
                1000, true, [], [], exog);
close all;

disp('Exogenous variable coefficients estimated');
disp('Crisis dummy controlled for in GMM estimation');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: With restricted reduced-form VAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: GMM SVAR with restricted reduced-form VAR');
disp('Restriction: InterestRate does not affect Output/Inflation in lags');
disp('========================================');

lr = [1 1 0;   % Output affected by output, inflation only
      1 1 0;   % Inflation affected by output, inflation only
      1 1 1];  % Interest rate affected by all variables

EQ7 = gmmSRSVAR(data, 2, idmat_rec, var_names, shock_names, 40, 1000, ...
                true, lr);
close all;

disp('Reduced-form restrictions imposed via lr matrix');
disp('Structural identification via GMM on restricted VAR');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: Analyzing structural shocks
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: Structural shocks from GMM estimation');
disp('========================================');

% Structural shocks should be orthogonal
shocks = EQ1.struc;
corr_matrix = corr(shocks');

disp('Correlation matrix of structural shocks:');
disp('(Should be approximately identity matrix)');
disp(corr_matrix);

% Summary statistics
disp('Structural shock statistics:');
disp('           Mean       Std Dev');
for i = 1:K
    fprintf('%-15s %.4f     %.4f\n', shock_names{i}, ...
            mean(shocks(i,:)), std(shocks(i,:)));
end

% Plot structural shocks
figure('Position', [100 100 1000 600]);
for i = 1:K
    subplot(K, 1, i);
    plot(shocks(i,:), 'k-', 'LineWidth', 1);
    title([shock_names{i} ' shock']);
    ylabel('Magnitude');
    grid on;
end
xlabel('Time');
sgtitle('Structural Shocks (GMM Estimation)');
fprintf('Structural shocks plot created (Figure 1)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: Variance decomposition analysis
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Variance decomposition from GMM SVAR');
disp('========================================');

% Extract variance decomposition at specific horizons
horizons = [1, 4, 12, 40];
disp('Variance decomposition of Output:');
disp('Horizon    Supply     Demand     MonPolicy');
for h = horizons
    fprintf('%3d        %.2f%%      %.2f%%      %.2f%%\n', h, ...
            EQ1.vardecshock_Supply(1,h), ...
            EQ1.vardecshock_Demand(1,h), ...
            EQ1.vardecshock_MonetaryPolicy(1,h));
end
fprintf('\n');

% Plot variance decomposition
figure('Position', [100 100 1200 400]);
subplot(1, 3, 1);
plot(1:40, EQ1.vardecshock_Supply(1,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_Demand(1,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_MonetaryPolicy(1,:), 'k:', 'LineWidth', 2);
title('Output Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Supply', 'Demand', 'Mon. Policy', 'Location', 'best');
grid on;

subplot(1, 3, 2);
plot(1:40, EQ1.vardecshock_Supply(2,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_Demand(2,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_MonetaryPolicy(2,:), 'k:', 'LineWidth', 2);
title('Inflation Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Supply', 'Demand', 'Mon. Policy', 'Location', 'best');
grid on;

subplot(1, 3, 3);
plot(1:40, EQ1.vardecshock_Supply(3,:), 'k-', 'LineWidth', 2);
hold on;
plot(1:40, EQ1.vardecshock_Demand(3,:), 'k--', 'LineWidth', 2);
plot(1:40, EQ1.vardecshock_MonetaryPolicy(3,:), 'k:', 'LineWidth', 2);
title('Interest Rate Variance Decomposition');
xlabel('Horizon');
ylabel('Percentage');
legend('Supply', 'Demand', 'Mon. Policy', 'Location', 'best');
grid on;

sgtitle('Variance Decomposition (GMM SVAR)');
fprintf('Variance decomposition plot created (Figure 2)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: GMM vs Cholesky comparison
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: Comparing GMM with Cholesky identification');
disp('For recursive restrictions, GMM should match Cholesky');
disp('========================================');

% Estimate with Cholesky
EQ_chol = srSVAR(data, 2, var_names, 40, 1000, true);
close all;

disp('Impact matrices comparison:');
disp('GMM P:');
disp(EQ1.P);
disp('Cholesky P:');
disp(EQ_chol.P);

disp('Difference (should be small):');
disp(abs(EQ1.P - EQ_chol.P));
fprintf('Max absolute difference: %.6f\n', max(abs(EQ1.P(:) - EQ_chol.P(:))));
fprintf('\n');

% Compare IRFs
figure('Position', [100 100 1200 400]);
for i = 1:K
    subplot(1, 3, i);
    plot(1:40, EQ1.Output_Supply(2,:), 'k-', 'LineWidth', 2);
    hold on;
    plot(1:40, EQ_chol.Output_Output(2,:), 'r--', 'LineWidth', 2);
    title([var_names{i} ' response to first shock']);
    xlabel('Horizon');
    ylabel('Response');
    legend('GMM', 'Cholesky', 'Location', 'best');
    grid on;
end
sgtitle('GMM vs Cholesky: IRF Comparison');
fprintf('GMM vs Cholesky comparison plot created (Figure 3)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 11: Testing overidentifying restrictions
%% ========================================================================
disp('========================================');
disp('EXAMPLE 11: Overidentification test via GMM objective value');
disp('========================================');

% Just-identified model (exactly K(K-1)/2 = 3 zero restrictions)
idmat_just = [1  0  0;
              1  1  0;
              1  1  1];

% Overidentified model (4 zero restrictions = one extra)
% Monetary policy shock affects only interest rate on impact
idmat_over = [1  0  0;
              1  1  0;
              0  0  1];

EQ_just = gmmSRSVAR(data, 2, idmat_just, var_names, shock_names, 40, ...
                    1000, true);
EQ_over = gmmSRSVAR(data, 2, idmat_over, var_names, shock_names, 40, ...
                    1000, true);
close all;

disp('Just-identified model:');
fprintf('  Zero restrictions: %d (minimum required)\n', sum(idmat_just(:)==0));
disp('  Impact matrix P:');
disp(EQ_just.P);
fprintf('\n');

disp('Overidentified model:');
fprintf('  Zero restrictions: %d (one extra restriction)\n', sum(idmat_over(:)==0));
disp('  Impact matrix P:');
disp(EQ_over.P);
fprintf('\n');

disp('Note: GMM objective value can be used for overidentification test');
disp('(J-statistic has chi-squared distribution under null of valid restrictions)');
fprintf('Degrees of freedom for J-test: %d\n', sum(idmat_over(:)==0) - K*(K-1)/2);
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 12: Visualization of confidence bands
%% ========================================================================
disp('========================================');
disp('EXAMPLE 12: Detailed IRF visualization with confidence bands');
disp('========================================');

% Plot IRFs with shaded confidence regions
figure('Position', [100 100 1200 800]);
shock_idx = 1;  % Supply shock
for var_idx = 1:K
    subplot(K, 1, var_idx);
    
    % Extract IRF data
    eval(['irf_data = EQ1.' var_names{var_idx} '_' shock_names{shock_idx} ';']);
    
    % Plot confidence bands as shaded area
    x_fill = [1:40, fliplr(1:40)];
    y_fill = [irf_data(1,:), fliplr(irf_data(3,:))];
    fill(x_fill, y_fill, [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    hold on;
    
    % Plot point estimate
    plot(1:40, irf_data(2,:), 'k-', 'LineWidth', 2);
    
    % Add zero line
    plot(1:40, zeros(1,40), 'k--', 'LineWidth', 0.5);
    
    title([var_names{var_idx} ' response to ' shock_names{shock_idx} ' shock']);
    xlabel('Horizon');
    ylabel('Response');
    grid on;
end
sgtitle('IRF with 90% Confidence Bands (GMM SVAR)');
fprintf('Detailed IRF visualization plot created (Figure 4)\n');
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key GMM SVAR features demonstrated:');
disp('- GMM estimation with short-run zero restrictions');
disp('- Overidentification (more restrictions than minimum)');
disp('- Custom shock sizes and confidence levels');
disp('- Seasonal dummies and exogenous variables');
disp('- Restricted reduced-form VAR');
disp('- Structural shock analysis and orthogonality');
disp('- Variance decomposition');
disp('- Comparison with Cholesky identification');
disp('- Overidentification testing framework');