%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR lrSVAR
%% Author: Lorenzo Menna
%% This script demonstrates all features of the lrSVAR function
%% (Structural VAR with long-run restrictions à la Blanchard-Quah)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) system with long-run restrictions
% Following Blanchard-Quah: some shocks have no long-run effects on some variables

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

% True structural impact matrix satisfying long-run restrictions
% B matrix such that (I - A1 - A2) * B is lower triangular
% This ensures shock 2 has no long-run effect on variable 1,
% and shock 3 has no long-run effect on variables 1 and 2
B_true = [1.0  0.0  0.0;
          0.4  0.9  0.0;
          0.3  0.2  0.8];

% Verify long-run multiplier is lower triangular
B_longrun = (eye(K) - A1 - A2);
LR_multiplier = inv(B_longrun) * B_true;
disp('True long-run multiplier matrix (should be lower triangular):');
disp(LR_multiplier);

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
disp('True impact matrix B:');
disp(B_true);
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic long-run SVAR (Blanchard-Quah)
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Long-run SVAR (Blanchard-Quah identification)');
disp('========================================');

% Define variable names and shock names
var_names = {'Output', 'Inflation', 'Unemployment'};
shock_names = {'Supply', 'Demand', 'Labor'};

EQ1 = lrSVAR(data, 2, var_names, shock_names, 40, 1000, true);
close all;

disp('Estimated impact matrix P:');
disp(EQ1.P);
disp('True impact matrix B:');
disp(B_true);
fprintf('\n');

% Verify long-run restrictions
Q_check = reducedformVAR(data, 2, true);
A_sum = Q_check.coefficients(:, 1:K) + Q_check.coefficients(:, K+1:2*K);
B_check = (eye(K) - A_sum);
LR_estimated = inv(B_check) * EQ1.P;

disp('Estimated long-run multiplier matrix:');
disp(LR_estimated);
disp('Upper triangular elements (should be near zero):');
fprintf('LR(1,2) = %.6f  (Demand shock long-run effect on Output)\n', LR_estimated(1,2));
fprintf('LR(1,3) = %.6f  (Labor shock long-run effect on Output)\n', LR_estimated(1,3));
fprintf('LR(2,3) = %.6f  (Labor shock long-run effect on Inflation)\n', LR_estimated(2,3));
fprintf('\n');

% Display IRF
disp('IRF: Output response to Supply shock (first 10 periods):');
disp('Period    95% CI    IRF    5% CI');
for i = 1:10
    fprintf('%3d    %7.4f  %7.4f  %7.4f\n', i, ...
            EQ1.Output_Supply(1,i), EQ1.Output_Supply(2,i), EQ1.Output_Supply(3,i));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Long-run effects interpretation
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: Interpreting long-run effects');
disp('========================================');

% Compute cumulative IRFs (approximate long-run effects)
cumsum_irf_output = zeros(K, 3);
for shock = 1:K
    if shock == 1
        cumsum_irf_output(shock, :) = [sum(EQ1.Output_Supply(2,:)), ...
                                        sum(EQ1.Inflation_Supply(2,:)), ...
                                        sum(EQ1.Unemployment_Supply(2,:))];
    elseif shock == 2
        cumsum_irf_output(shock, :) = [sum(EQ1.Output_Demand(2,:)), ...
                                        sum(EQ1.Inflation_Demand(2,:)), ...
                                        sum(EQ1.Unemployment_Demand(2,:))];
    else
        cumsum_irf_output(shock, :) = [sum(EQ1.Output_Labor(2,:)), ...
                                        sum(EQ1.Inflation_Labor(2,:)), ...
                                        sum(EQ1.Unemployment_Labor(2,:))];
    end
end

disp('Cumulative IRFs (approximation to long-run effects):');
disp('                  Output    Inflation    Unemployment');
fprintf('Supply shock:     %.4f    %.4f       %.4f\n', cumsum_irf_output(1,:));
fprintf('Demand shock:     %.4f    %.4f       %.4f\n', cumsum_irf_output(2,:));
fprintf('Labor shock:      %.4f    %.4f       %.4f\n', cumsum_irf_output(3,:));
fprintf('\n');
fprintf('Note: Demand shock should have near-zero long-run effect on Output\n');
fprintf('      Labor shock should have near-zero long-run effect on Output and Inflation\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Comparison with short-run (Cholesky) identification
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Long-run vs Short-run identification');
disp('========================================');

EQ_lr = lrSVAR(data, 2, var_names, shock_names, 40, 1000, true);
close all;
EQ_sr = srSVAR(data, 2, var_names, 40, 1000, true);
close all;

disp('Long-run identification impact matrix:');
disp(EQ_lr.P);
fprintf('\n');

disp('Short-run (Cholesky) identification impact matrix:');
disp(EQ_sr.P);
fprintf('\n');

% Compare IRFs
disp('Comparison of IRFs (Output response to first shock at horizon 10):');
fprintf('Long-run ID:  %.4f [%.4f, %.4f]\n', ...
        EQ_lr.Output_Supply(2,10), EQ_lr.Output_Supply(3,10), EQ_lr.Output_Supply(1,10));
fprintf('Short-run ID: %.4f [%.4f, %.4f]\n', ...
        EQ_sr.Output_Output(2,10), EQ_sr.Output_Output(3,10), EQ_sr.Output_Output(1,10));
fprintf('\n');
disp('Note: Different identification schemes yield different structural interpretations');
disp('      Long-run uses shock names (Supply, Demand, Labor)');
disp('      Short-run uses variable ordering for shock interpretation');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: Custom shock sizes
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: Long-run SVAR with custom shock sizes');
disp('2 standard deviation shocks');
disp('========================================');

shock_size = 2 * diag(EQ1.P);

EQ4 = lrSVAR(data, 2, var_names, shock_names, 40, 1000, true, [], [], [], shock_size);
close all;

disp('Comparison of IRFs (Output response to Supply shock at horizon 5):');
fprintf('1 std shock:  %.4f [%.4f, %.4f]\n', ...
        EQ1.Output_Supply(2,5), EQ1.Output_Supply(3,5), EQ1.Output_Supply(1,5));
fprintf('2 std shock:  %.4f [%.4f, %.4f]\n', ...
        EQ4.Output_Supply(2,5), EQ4.Output_Supply(3,5), EQ4.Output_Supply(1,5));
fprintf('Ratio:        %.4f (should be ~2.0)\n', ...
        EQ4.Output_Supply(2,5) / EQ1.Output_Supply(2,5));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Different confidence levels
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: Long-run SVAR with different confidence levels');
disp('========================================');

EQ5a = lrSVAR(data, 2, var_names, shock_names, 40, 1000, true, [], [], [], [], 5);   % 90% CI
close all;
EQ5b = lrSVAR(data, 2, var_names, shock_names, 40, 1000, true, [], [], [], [], 16);  % 68% CI
close all;

disp('Comparison of confidence intervals (Output response to Supply shock at horizon 5):');
fprintf('90%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ5a.Output_Supply(3,5), EQ5a.Output_Supply(1,5), ...
        EQ5a.Output_Supply(1,5) - EQ5a.Output_Supply(3,5));
fprintf('68%% CI: [%.4f, %.4f]  Width: %.4f\n', ...
        EQ5b.Output_Supply(3,5), EQ5b.Output_Supply(1,5), ...
        EQ5b.Output_Supply(1,5) - EQ5b.Output_Supply(3,5));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: With seasonal dummies
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: Long-run SVAR with quarterly seasonal dummies');
disp('========================================');

% Simulate quarterly data with seasonal pattern
T_q = 120;
data_seasonal = zeros(T_q, K);
data_seasonal(1:nlags, :) = randn(nlags, K);

seasonal_pattern = [0.5; -0.3; 0.2; -0.4];

for t = (nlags+1):T_q
    quarter = mod(t-1, 4) + 1;
    seasonal_effect = seasonal_pattern(quarter) * [1; 0.5; 0.3]';
    structural_shock = B_true * randn(K, 1);
    
    data_seasonal(t, :) = nu' + data_seasonal(t-1, :) * A1' + ...
                          data_seasonal(t-2, :) * A2' + seasonal_effect + structural_shock';
end

EQ6 = lrSVAR(data_seasonal, 2, var_names, shock_names, 40, 1000, true, [], 'quarter');
close all;

disp('Long-run SVAR with quarterly dummies estimated successfully');
disp('IRF: Output response to Supply shock (periods 1-5):');
for i = 1:5
    fprintf('Period %d:  %.4f [%.4f, %.4f]\n', i, ...
            EQ6.Output_Supply(2,i), EQ6.Output_Supply(3,i), EQ6.Output_Supply(1,i));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: With exogenous variables
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: Long-run SVAR with exogenous crisis dummy');
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

EQ7 = lrSVAR(data_crisis, 2, var_names, shock_names, 40, 1000, true, [], [], exog);
close all;

disp('Long-run SVAR with crisis dummy estimated');
disp('Structural shocks during crisis period (observations 80-85):');
disp('Obs    Supply    Demand    Labor');
for i = 80:85
    fprintf('%3d   %7.3f   %7.3f   %7.3f\n', i, ...
            EQ7.struc(1,i-nlags), EQ7.struc(2,i-nlags), EQ7.struc(3,i-nlags));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: Restricted reduced-form VAR
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: Long-run SVAR with restricted reduced-form VAR');
disp('Unemployment does not affect output or inflation in lags');
disp('========================================');

lr = [1 1 0;   % Output affected by output, inflation only
      1 1 0;   % Inflation affected by output, inflation only
      1 1 1];  % Unemployment affected by all variables

EQ8 = lrSVAR(data, 2, var_names, shock_names, 40, 1000, true, lr);
close all;

disp('Long-run identification + reduced-form restrictions');
disp('Impact matrix P:');
disp(EQ8.P);
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: Structural shocks analysis
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Analyzing structural shocks');
disp('========================================');

% Plot structural shocks
figure('Position', [100 100 1200 800]);

for i = 1:K
    subplot(K, 1, i);
    plot(1:T-nlags, EQ1.struc(i,:), 'k-', 'LineWidth', 1);
    yline(0, 'r--', 'LineWidth', 0.5);
    xlabel('Time');
    ylabel('Shock');
    title([shock_names{i} ' structural shock']);
    grid on;
end

sgtitle('Estimated Structural Shocks (Long-run Identification)');
fprintf('Structural shocks plotted (Figure 1)\n');
fprintf('\n');

% Summary statistics
disp('Summary statistics of structural shocks:');
disp('Shock           Mean      Std Dev    Min       Max');
for i = 1:K
    fprintf('%-15s %.4f    %.4f     %.4f    %.4f\n', ...
            shock_names{i}, mean(EQ1.struc(i,:)), std(EQ1.struc(i,:)), ...
            min(EQ1.struc(i,:)), max(EQ1.struc(i,:)));
end
fprintf('\n');

% Correlation of structural shocks
disp('Correlation matrix of structural shocks:');
disp('         Supply    Demand    Labor');
corr_matrix = corr(EQ1.struc');
for i = 1:K
    fprintf('%-8s', shock_names{i});
    for j = 1:K
        fprintf(' %7.4f', corr_matrix(i,j));
    end
    fprintf('\n');
end
fprintf('Note: Shocks should be approximately uncorrelated\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: Variance decomposition
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: Forecast error variance decomposition');
disp('========================================');

horizons = [1, 4, 10, 20, 40];

disp('Variance decomposition of Output:');
disp('Horizon  Supply    Demand    Labor');
for h = horizons
    fprintf('%3d      %6.2f%%   %6.2f%%   %6.2f%%\n', h, ...
            EQ1.vardecshock_Supply(1, h), ...
            EQ1.vardecshock_Demand(1, h), ...
            EQ1.vardecshock_Labor(1, h));
end
fprintf('\n');

disp('Note: In long-run, only Supply shock should explain Output variance');
fprintf('\n');

% Plot variance decomposition
figure('Position', [100 100 1200 400]);
for i = 1:K
    subplot(1, K, i);
    
    if i == 1
        vardec = [EQ1.vardecshock_Supply(1,:); ...
                  EQ1.vardecshock_Demand(1,:); ...
                  EQ1.vardecshock_Labor(1,:)];
    elseif i == 2
        vardec = [EQ1.vardecshock_Supply(2,:); ...
                  EQ1.vardecshock_Demand(2,:); ...
                  EQ1.vardecshock_Labor(2,:)];
    else
        vardec = [EQ1.vardecshock_Supply(3,:); ...
                  EQ1.vardecshock_Demand(3,:); ...
                  EQ1.vardecshock_Labor(3,:)];
    end
    
    hold on;
    plot(1:40, vardec(1,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Supply');
    plot(1:40, vardec(2,:), 'r-', 'LineWidth', 2, 'DisplayName', 'Demand');
    plot(1:40, vardec(3,:), 'g-', 'LineWidth', 2, 'DisplayName', 'Labor');
    hold off;
    
    xlabel('Horizon');
    ylabel('Percentage');
    title([var_names{i} ' variance decomposition']);
    legend('Location', 'best');
    grid on;
    ylim([0 100]);
end

sgtitle('Forecast Error Variance Decomposition (Long-run Identification)');
fprintf('Variance decomposition plot created (Figure 2)\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 11: Visualizing long-run vs short-run effects
%% ========================================================================
disp('========================================');
disp('EXAMPLE 11: Short-run vs long-run effects comparison');
disp('========================================');

% Plot IRFs showing economic interpretation
figure('Position', [100 100 1200 800]);

for shock = 1:K
    for var = 1:K
        subplot(K, K, (var-1)*K + shock);
        
        % Get appropriate IRF
        if shock == 1 && var == 1
            irf_data = EQ1.Output_Supply;
        elseif shock == 1 && var == 2
            irf_data = EQ1.Inflation_Supply;
        elseif shock == 1 && var == 3
            irf_data = EQ1.Unemployment_Supply;
        elseif shock == 2 && var == 1
            irf_data = EQ1.Output_Demand;
        elseif shock == 2 && var == 2
            irf_data = EQ1.Inflation_Demand;
        elseif shock == 2 && var == 3
            irf_data = EQ1.Unemployment_Demand;
        elseif shock == 3 && var == 1
            irf_data = EQ1.Output_Labor;
        elseif shock == 3 && var == 2
            irf_data = EQ1.Inflation_Labor;
        else
            irf_data = EQ1.Unemployment_Labor;
        end
        
        hold on;
        plot(1:40, irf_data(1,:), 'k--', 'LineWidth', 1);
        plot(1:40, irf_data(3,:), 'k--', 'LineWidth', 1);
        plot(1:40, irf_data(2,:), 'b-', 'LineWidth', 2);
        yline(0, 'k:', 'LineWidth', 0.5);
        
        % Highlight long-run restriction zones
        if (shock > var)
            % This combination should have zero long-run effect
            text(30, max(irf_data(2,:))*0.8, 'LR=0', 'FontSize', 8, 'Color', 'r');
        end
        
        if var == 1
            title([shock_names{shock} ' shock']);
        end
        if shock == 1
            ylabel(var_names{var});
        end
        grid on;
        hold off;
    end
end

sgtitle('Impulse Response Functions with Long-run Restrictions');
fprintf('IRF comparison plot created (Figure 3)\n');
fprintf('\n\n');

%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key features demonstrated:');
disp('- Long-run restrictions (Blanchard-Quah identification)');
disp('- Economic shock names separate from variable names');
disp('- Verification of long-run multiplier structure');
disp('- Comparison with short-run (Cholesky) identification');
disp('- Custom shock sizes and confidence levels');
disp('- Seasonal dummies and exogenous variables');
disp('- Restricted reduced-form VARs');
disp('- Structural shocks recovery and analysis');
disp('- Variance decomposition interpretation');