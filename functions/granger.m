function Q=granger(data,nlags,var_names,const,verbose)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose:
% Granger causality test on a vector autoregression (no maximum number of lags)
% -----------------------------------
% Inputs:
% data = NxK matrix (N number of observations, K number of variables)
% nlags = desired number of lags
% Optional:
% var_names = 1xK cell vector with names of the variables
% const = true if the VAR is estimated with a constant and = false if without.
%         Default is false.
% verbose = true to display results, false to suppress. Default is true.
% -----------------------------------
% Returns:
% Q = structure
% Q.lambdaF = KxK matrix of F statistics. Element (i,j) tests whether 
%             variable j Granger-causes variable i. Stata-compatible.
% Q.Pf = KxK matrix of p-values corresponding to Q.lambdaF
% Q.lambdaCHI = KxK matrix of Wald Chi-squared statistics
% Q.Pchi = KxK matrix of p-values corresponding to Q.lambdaCHI
% ------------------------------------
% The solution method follows Lütkepohl (2005)

if nargin<5
    verbose=true;
end
if nargin<4
    verbose=true;
    const=false;
end
if nargin<3
    verbose=true;
    const=false;
    for xx=1:size(data,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end

if isempty(var_names)==1
    for xx=1:size(data,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end

T=size(data,1)-nlags;
K=size(data,2);
Qfull=reducedformVAR(data,nlags,const);

% Construct regressor matrix Z
Zprep=flipud(data(1:size(data,1)-1,:))';
Zprep=reshape(Zprep,(size(data,1)-1)*K,1);
Zprep1=ones(1,T);
Zprep2=zeros(0,0);
count=0;
for xx=1:T
    Zprep2=[Zprep2 Zprep(1+count:nlags*K+count,1)];
    count=count+K;
end
Zprep2=fliplr(Zprep2);
Z=[Zprep1;Zprep2];
clear Zprep Zprep1 Zprep2

if const==0
    Z=Z(2:end,:);
end

% Stack coefficients into vector
if const==0
    bet=reshape(Qfull.coefficients,K^2*nlags,1);
    C=zeros(nlags,K^2*nlags,K^2);
else 
    bet=reshape([Qfull.constants Qfull.coefficients],K^2*nlags+K,1);
    C=zeros(nlags,K^2*nlags+K,K^2+K);
end

% Construct restriction matrices
for xx=1:K^2
    for yy=1:nlags
        if const==0
            C(yy,xx+K^2*(yy-1),xx)=1;
        else 
            C(yy,xx+K^2*(yy-1)+K,xx)=1;
        end
    end
end

% Compute test statistics
count=0;
for xx=1:K
    for yy=1:K
        count=count+1;
        % Wald Chi-squared statistic
        lambdaCHI(yy,xx)=bet'*C(:,:,count)'*inv(C(:,:,count)*kron(inv(Z*Z'),Qfull.sigma)*C(:,:,count)')*C(:,:,count)*bet;
        % Wald F statistic
        lambdaF(yy,xx)=bet'*C(:,:,count)'*inv(C(:,:,count)*kron(inv(Z*Z'),Qfull.sigma)*C(:,:,count)')*C(:,:,count)*bet/nlags;
    end
end

% Compute p-values
if const==0
    df2 = T - K*nlags;
else
    df2 = T - K*nlags - 1;
end

for xx=1:K
    for yy=1:K
        Pchi(xx,yy)=(1-chi2cdf(lambdaCHI(xx,yy),nlags));
        Pf(xx,yy)=(1-fcdf(lambdaF(xx,yy),nlags,df2));
    end
end

% Display results if verbose
if verbose==true
    disp('========================================');
    disp('Granger Causality Test Results');
    disp('========================================');
    for xx=1:K
        for yy=1:K
            if xx~=yy  % Don't display variable causing itself
                fprintf('%s does not Granger-cause %s: F = %.4f, p = %.4f', ...
                        var_names{yy}, var_names{xx}, lambdaF(xx,yy), Pf(xx,yy));
                if Pf(xx,yy) < 0.01
                    fprintf(' ***\n');
                elseif Pf(xx,yy) < 0.05
                    fprintf(' **\n');
                elseif Pf(xx,yy) < 0.10
                    fprintf(' *\n');
                else
                    fprintf('\n');
                end
            end
        end
    end
    disp('*** p<0.01, ** p<0.05, * p<0.10');
    disp('========================================');
end

% Store results
Q.lambdaCHI=lambdaCHI;
Q.lambdaF=lambdaF;
Q.Pchi=Pchi;
Q.Pf=Pf;

end