%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLES FOR granger
%% Author: Lorenzo Menna
%% This script demonstrates all features of the granger function
%% (Pairwise Granger causality tests on vector autoregressions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% ========================================================================
%% DATA SIMULATION
%% ========================================================================
% We simulate a 3-variable VAR(2) with a known one-directional causal chain:
%   Output -> Inflation -> InterestRate
% with no reverse causality by construction:
%   InterestRate does NOT Granger-cause Output or Inflation
%   Inflation    does NOT Granger-cause Output

rng(42);

K = 3;
T = 200;
nlags = 2;

% Coefficient matrices encoding the causal chain
A1 = [0.5  0.0  0.0;   % Output:       own lag only
      0.3  0.4  0.0;   % Inflation:    caused by Output (0.3), own lag
      0.0  0.3  0.4];  % InterestRate: caused by Inflation (0.3), own lag

A2 = [0.1  0.0  0.0;
      0.1  0.1  0.0;
      0.0  0.1  0.1];

nu = [0.5; 1.0; 0.5];  % Constants

B_true = [1.0  0.0  0.0;
          0.2  0.8  0.0;
          0.1  0.1  0.7];

epsilon = randn(K, T);
data = zeros(T, K);
data(1:nlags, :) = randn(nlags, K);

for t = (nlags+1):T
    structural_shock = B_true * epsilon(:, t);
    data(t, :) = nu' + data(t-1, :) * A1' + data(t-2, :) * A2' + structural_shock';
end

var_names = {'Output', 'Inflation', 'InterestRate'};

disp('========================================');
disp('Data simulation complete');
fprintf('Sample size: %d, Variables: %s\n', T, strjoin(var_names, ', '));
disp('True causal chain: Output -> Inflation -> InterestRate');
disp('No reverse causality by construction');
disp('========================================');
fprintf('\n');

%% ========================================================================
%% EXAMPLE 1: Basic Granger causality test with verbose output
%% ========================================================================
disp('========================================');
disp('EXAMPLE 1: Basic Granger causality test');
disp('VAR(2) with constant, full verbose output');
disp('========================================');

Q1 = granger(data, nlags, var_names, true);

fprintf('F-statistic matrix (row i = caused by col j):\n');
disp(Q1.lambdaF);
fprintf('P-value matrix:\n');
disp(Q1.Pf);
fprintf('Results match the true causal structure:\n');
fprintf('  Output->Inflation   significant (p=%.4f), as expected.\n', Q1.Pf(2,1));
fprintf('  Inflation->IntRate  significant (p=%.4f), as expected.\n', Q1.Pf(3,2));
fprintf('  IntRate->Output     NOT significant (p=%.4f), as expected.\n', Q1.Pf(1,3));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 2: Effect of including a constant
%% ========================================================================
disp('========================================');
disp('EXAMPLE 2: With constant vs without constant');
disp('========================================');

Q2_nc = granger(data, nlags, var_names, false, false);
Q2_c  = granger(data, nlags, var_names, true,  false);

fprintf('%-30s  const=false    const=true\n', 'Test (F-stat / p-val)');
fprintf('%s\n', repmat('-', 1, 60));
pairs = {2,1,'Output -> Inflation';
         3,2,'Inflation -> IntRate';
         1,2,'Inflation -> Output';
         1,3,'IntRate -> Output'};
for pp = 1:size(pairs,1)
    ii = pairs{pp,1};
    jj = pairs{pp,2};
    label = pairs{pp,3};
    fprintf('%-30s  F=%.3f p=%.3f   F=%.3f p=%.3f\n', label, ...
        Q2_nc.lambdaF(ii,jj), Q2_nc.Pf(ii,jj), ...
        Q2_c.lambdaF(ii,jj),  Q2_c.Pf(ii,jj));
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 3: Silent mode — programmatic access to results
%% ========================================================================
disp('========================================');
disp('EXAMPLE 3: Silent mode (verbose=false)');
disp('Extracting and interpreting specific pairs');
disp('========================================');

Q3 = granger(data, nlags, var_names, true, false);

all_pairs = {2,1,'Output -> Inflation';
             3,2,'Inflation -> IntRate';
             1,2,'Inflation -> Output';
             2,3,'IntRate -> Inflation';
             3,1,'Output -> IntRate';
             1,3,'IntRate -> Output'};

fprintf('%-28s   F-stat   p-value   Result\n', 'Null: X does not cause Y');
fprintf('%s\n', repmat('-', 1, 65));
for pp = 1:size(all_pairs,1)
    ii = all_pairs{pp,1};
    jj = all_pairs{pp,2};
    label = all_pairs{pp,3};
    fval = Q3.lambdaF(ii,jj);
    pval = Q3.Pf(ii,jj);
    if     pval < 0.01; verdict = 'Reject H0 ***';
    elseif pval < 0.05; verdict = 'Reject H0 **';
    elseif pval < 0.10; verdict = 'Reject H0 *';
    else;               verdict = 'Fail to reject';
    end
    fprintf('%-28s   %6.3f   %6.4f    %s\n', label, fval, pval, verdict);
end
fprintf('*** p<0.01  ** p<0.05  * p<0.10\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 4: Sensitivity to lag length
%% ========================================================================
disp('========================================');
disp('EXAMPLE 4: Lag length sensitivity');
disp('True model has 2 lags; testing nlags = 1 to 4');
disp('========================================');

fprintf('nlags  Out->Inf          Inf->IntRate      IntRate->Out\n');
fprintf('       F-stat  p-val    F-stat  p-val    F-stat  p-val\n');
fprintf('%s\n', repmat('-', 1, 65));
for lag = 1:4
    Qlag = granger(data, lag, var_names, true, false);
    fprintf('  %d    %6.3f  %.4f   %6.3f  %.4f   %6.3f  %.4f\n', lag, ...
        Qlag.lambdaF(2,1), Qlag.Pf(2,1), ...
        Qlag.lambdaF(3,2), Qlag.Pf(3,2), ...
        Qlag.lambdaF(1,3), Qlag.Pf(1,3));
end
fprintf('Note: signal is strongest when lag length matches the true model.\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 5: Chi-squared vs F-statistic
%% ========================================================================
disp('========================================');
disp('EXAMPLE 5: Chi-squared vs F-statistic');
disp('The Wald Chi2 = nlags * F by construction');
disp('========================================');

Q5 = granger(data, nlags, var_names, true, false);

fprintf('Output -> Inflation:\n');
fprintf('  F-stat   = %8.4f  (p = %.4f)\n', Q5.lambdaF(2,1),   Q5.Pf(2,1));
fprintf('  Chi2     = %8.4f  (p = %.4f)\n', Q5.lambdaCHI(2,1), Q5.Pchi(2,1));
fprintf('  Chi2 / F = %8.4f  (should equal nlags = %d)\n\n', ...
    Q5.lambdaCHI(2,1) / Q5.lambdaF(2,1), nlags);

fprintf('Inflation -> InterestRate:\n');
fprintf('  F-stat   = %8.4f  (p = %.4f)\n', Q5.lambdaF(3,2),   Q5.Pf(3,2));
fprintf('  Chi2     = %8.4f  (p = %.4f)\n', Q5.lambdaCHI(3,2), Q5.Pchi(3,2));
fprintf('  Chi2 / F = %8.4f  (should equal nlags = %d)\n', ...
    Q5.lambdaCHI(3,2) / Q5.lambdaF(3,2), nlags);
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 6: Full p-value table
%% ========================================================================
disp('========================================');
disp('EXAMPLE 6: Full p-value table (F-test)');
disp('Row i = caused variable, Col j = causing variable');
disp('========================================');

Q6 = granger(data, nlags, var_names, true, false);

colwidth = 18;
fprintf('%-16s', '');
for j = 1:K
    fprintf(['%-' num2str(colwidth) 's'], var_names{j});
end
fprintf('\n%s\n', repmat('-', 1, 16 + colwidth*K));
for i = 1:K
    fprintf('%-16s', var_names{i});
    for j = 1:K
        if i == j
            fprintf(['%-' num2str(colwidth) 's'], '---');
        else
            stars = '';
            if     Q6.Pf(i,j) < 0.01; stars = '***';
            elseif Q6.Pf(i,j) < 0.05; stars = '**';
            elseif Q6.Pf(i,j) < 0.10; stars = '*';
            end
            cell_str = [sprintf('%.4f', Q6.Pf(i,j)) stars];
            fprintf(['%-' num2str(colwidth) 's'], cell_str);
        end
    end
    fprintf('\n');
end
fprintf('*** p<0.01  ** p<0.05  * p<0.10\n');
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 7: Bidirectional Granger causality (feedback system)
%% ========================================================================
disp('========================================');
disp('EXAMPLE 7: Bidirectional causality');
disp('Feedback system: SeriesA <-> SeriesB');
disp('========================================');

rng(99);
A1_bi = [0.5  0.3;
         0.3  0.5];
A2_bi = [0.1  0.1;
         0.1  0.1];
T_bi = 200;
data_bi = zeros(T_bi, 2);
data_bi(1:2, :) = randn(2, 2);
for t = 3:T_bi
    data_bi(t,:) = data_bi(t-1,:)*A1_bi' + data_bi(t-2,:)*A2_bi' + randn(1,2);
end
var_names_bi = {'SeriesA', 'SeriesB'};

Q7 = granger(data_bi, 2, var_names_bi, false);

fprintf('Expected: significant in both directions (A<->B).\n');
fprintf('SeriesA -> SeriesB:  F=%.4f, p=%.4f\n', Q7.lambdaF(2,1), Q7.Pf(2,1));
fprintf('SeriesB -> SeriesA:  F=%.4f, p=%.4f\n', Q7.lambdaF(1,2), Q7.Pf(1,2));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 8: No Granger causality — independent AR processes
%% ========================================================================
disp('========================================');
disp('EXAMPLE 8: No causality (independent AR(2) processes)');
disp('All cross-variable p-values should be large on average');
disp('========================================');

rng(77);
T_ind = 300;
K_ind = 3;
data_ind = zeros(T_ind, K_ind);
data_ind(1:2, :) = randn(2, K_ind);
for t = 3:T_ind
    data_ind(t,:) = 0.5*data_ind(t-1,:) + 0.1*data_ind(t-2,:) + randn(1,K_ind);
end
var_names_ind = {'AR1', 'AR2', 'AR3'};

Q8 = granger(data_ind, 2, var_names_ind, false, false);

fprintf('Cross-variable p-values (all should be large; ~10%% false positives expected at 10%% level):\n');
for i = 1:K_ind
    for j = 1:K_ind
        if i ~= j
            fprintf('  %s -> %s:  p = %.4f\n', var_names_ind{j}, var_names_ind{i}, Q8.Pf(i,j));
        end
    end
end
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 9: Using default variable names (no names provided)
%% ========================================================================
disp('========================================');
disp('EXAMPLE 9: Default variable names');
disp('Running with only required arguments');
disp('========================================');

% With only 2 arguments: var_names auto-generated as var1, var2, var3; const=false; verbose=true
Q9 = granger(data, nlags);

fprintf('Results also accessible numerically — e.g., Q9.lambdaF(2,1) = %.4f\n', Q9.lambdaF(2,1));
fprintf('\n\n');

%% ========================================================================
%% EXAMPLE 10: 4-variable system
%% ========================================================================
disp('========================================');
disp('EXAMPLE 10: 4-variable Granger causality');
disp('Output -> Inflation -> IntRate; Output/IntRate -> ExchangeRate');
disp('========================================');

rng(321);
K4 = 4;
T4 = 250;
% Causal structure encoded in A1:
%   Output      -> Inflation (0.3), ExchangeRate (0.1)
%   Inflation   -> InterestRate (0.3)
%   InterestRate-> ExchangeRate (0.2)
A1_4 = [0.5  0.0  0.0  0.0;   % Output
         0.3  0.4  0.0  0.0;   % Inflation
         0.0  0.3  0.4  0.0;   % InterestRate
         0.1  0.0  0.2  0.4];  % ExchangeRate
A2_4 = 0.1 * eye(K4);
nu4  = [0.5; 1.0; 0.5; 0.5];

data4 = zeros(T4, K4);
data4(1:2, :) = randn(2, K4);
for t = 3:T4
    data4(t,:) = nu4' + data4(t-1,:)*A1_4' + data4(t-2,:)*A2_4' + randn(1,K4);
end
var_names4 = {'Output', 'Inflation', 'IntRate', 'ExchangeRate'};

Q10 = granger(data4, 2, var_names4, true);

fprintf('P-value matrix:\n');
disp(Q10.Pf);
fprintf('\n\n');

%% ========================================================================
%% SUMMARY
%% ========================================================================
disp('========================================');
disp('All examples completed successfully!');
disp('========================================');
disp('Key granger features demonstrated:');
disp('  - Full verbose output with significance stars');
disp('  - With and without constant (const=true/false)');
disp('  - Silent mode + programmatic extraction of results');
disp('  - Lag length sensitivity');
disp('  - Chi-squared vs F-statistic (Chi2 = nlags * F)');
disp('  - Full KxK p-value table');
disp('  - Bidirectional (feedback) causality');
disp('  - No-causality null not rejected');
disp('  - Default variable names (no var_names provided)');
disp('  - 4-variable system');
disp(' ');
disp('Key concepts:');
disp('  - Q.lambdaF(i,j): F-stat for "j Granger-causes i"');
disp('  - Q.Pf(i,j):      p-value for the same test');
disp('  - Q.lambdaCHI / Q.Pchi: Wald Chi2 versions');
disp('  - Chi2 = nlags * F exactly');
disp('  - Stars: *** p<0.01, ** p<0.05, * p<0.10');
