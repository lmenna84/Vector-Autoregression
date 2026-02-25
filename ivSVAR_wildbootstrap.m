function EQ=ivSVAR_wildbootstrap(data_inst,other_data,iv,nlags,var_names,shock_names,irf_lenght,bootstrap_num,const,lr,dum,exog,shock,confianza,rel,verbose)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose: 
% Impulse responses and Variance decomposition of a Structural Vector Autoregression
% with external instruments (proxy-SVAR) using wild bootstrap for confidence intervals.
% -----------------------------------
% Inputs:
% data_inst = NxK1 matrix (N number of observations, K1 number of instrumented variables). 
% other_data = NxK2 matrix (N number of observations, K2 number of non-instrumented variables). 
% iv = NxK1 matrix of instrumental variables (external instruments).
% nlags = desired number of lags (no maximum limit enforced)
% Optional:
% var_names = 1x(K1+K2) cell vector with names of the variables
% shock_names = 1xK1 cell vector with names of the identified shocks
% irf_lenght = desired length of the IRFs. Default is 40.
% bootstrap_num = number of wild bootstrap repetitions for the computation of
%                 the confidence intervals of the IRFs. Default is 1000.
% const = true if the VAR is estimated with a constant and false if without.
%         Default is false.
% lr = (K1+K2)x(K1+K2) matrix of zeros and ones specifying restrictions on 
%      the reduced-form VAR. Element (i,j) = 0 restricts variable j from 
%      affecting variable i. Default is no restrictions.
% dum = seasonal dummies. Specify 'month' for 11 monthly dummies or 
%       'quarter' for 3 quarterly dummies. Default is no dummies.
% exog = NxE matrix of exogenous variables entering without lags 
%        (non-dynamic regressors, e.g., event dummies). Default is no exogenous variables.
% shock = K1x1 vector containing the size of the shock for each identified shock. 
%         Default is unit shocks (appropriate for proxy-SVARs).
% confianza = level in % of the lowest confidence band. Default is 5%.
% rel = 1 for relative IRFs (unit shock to variable), 0 for absolute IRFs 
%       (unit structural shock). Default is 0 (absolute). Use rel=1 when 
%       non-invertibility is a concern.
% verbose = true to print bootstrap iteration numbers, false to suppress. Default is false.
% -----------------------------------
% Returns:
% EQ = structure with the following fields:
% EQ.vari_shockj = K1 collections of 3xirf_lenght matrices containing in the first row the
%                  95% confidence band, in the second row the IRF, and in the third row the
%                  5% confidence band. i is the variable, j is the identified shock.
% EQ.P = (K1+K2)xK1 partially-identified impact matrix. Only first K1 shocks identified.
% EQ.vardecshock_shockj = K1 matrices of size (K1+K2)xirf_lenght, where each row shows
%                         the percentage of variance of that variable explained by shock j.
% EQ.stdirf = (K1+K2)xirf_lenghtxK1 array containing standard deviations of IRFs
% EQ.struc = K1x(N-nlags) matrix of identified structural shocks
% EQ.irf_sim = (K1+K2)xirf_lenghtxbootstrap_numxK1 array of all IRF draws from bootstrap
% EQ.robustF = K1x1 vector of robust F-statistics for first-stage instrument relevance
% ------------------------------------
% The solution method follows Mertens & Ravn (2013) and Stock & Watson (2012/2018)
% Wild bootstrap uses Rademacher distribution, robust to heteroskedasticity

if nargin<16
    verbose=false;
end
if nargin<15
    verbose=false;
    rel=0;
end
if nargin<14
    verbose=false;
    rel=0;
    confianza=5;
end
if nargin<13
    verbose=false;
    rel=0;
    confianza=5;
    shock=[];
end
if nargin<12
    verbose=false;
    rel=0;
    confianza=5;
    shock=[];
    exog=[];
end
if nargin<11
    verbose=false;
    rel=0;
    confianza=5;
    shock=[];
    exog=[];
    dum=[];
end
if nargin<10
    verbose=false;
    rel=0;
    confianza=5;
    shock=[];
    exog=[];
    dum=[];
    lr=[];
end
if nargin<9
    verbose=false;
    rel=0;
    confianza=5;
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
end
if nargin<8
    verbose=false;
    rel=0;
    confianza=5;
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    bootstrap_num=1000;
end
if nargin<7
    verbose=false;
    rel=0;
    confianza=5;
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    bootstrap_num=1000;
    irf_lenght=40;
end
if nargin<6
    verbose=false;
    rel=0;
    confianza=5;
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    bootstrap_num=1000;
    irf_lenght=40;
    for xx=1:size(data_inst,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end
if nargin<5
    verbose=false;
    rel=0;
    confianza=5;
    shock=[];
    exog=[];
    dum=[];
    const=false;
    lr=[];
    bootstrap_num=1000;
    irf_lenght=40;
    for xx=1:size(data_inst,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
    for xx=1:size(data_inst,2)+size(other_data,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end

if isempty(verbose)==1
    verbose=false;
end
if isempty(rel)==1
    rel=0;
end
if isempty(confianza)==1
    confianza=5;
end
if isempty(shock)==1
    shock=[];
end
if isempty(bootstrap_num)==1
    bootstrap_num=1000;
end
if isempty(irf_lenght)==1
    irf_lenght=40;
end
if isempty(shock_names)==1
    for xx=1:size(data_inst,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end
if isempty(var_names)==1
    for xx=1:size(data_inst,2)+size(other_data,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end
if isempty(dum)==1
    numdum=0;
elseif strcmpi('month',dum)==1
    numdum=11;
elseif strcmpi('quarter',dum)==1
    numdum=3;
end

K1=size(data_inst,2);

if size(iv,2)~=K1
    display('Mistake: number of instruments different from number of instrumented variables');
    return
end

K2=size(other_data,2);
data=[data_inst,other_data];
K=size(data,2);
Q=reducedformVAR(data,nlags,const,lr,[],dum,exog);
varcovar=Q.sigma;
resid=Q.resid;

% Obtain estimates of structural parameters using external instruments
resid1=resid(:,1:K1);
resid2=resid(:,K1+1:K);
iv=iv(nlags+1:size(iv,1),:);

% First stage: regress instrumented residuals on instruments
for xx=1:K1
    X_fs = iv;
    y_fs = resid1(:,xx);
    beta_fs = X_fs\y_fs;
    first_stage(xx,:) = beta_fs';
    
    % Compute robust standard errors
    resid_fs = y_fs - X_fs*beta_fs;
    meat = X_fs' * diag(resid_fs.^2) * X_fs;
    bread = inv(X_fs' * X_fs);
    vcov_robust = bread * meat * bread;
    SE_robust(xx,:) = sqrt(diag(vcov_robust))';
    
    pred_firststage(:,xx) = X_fs * beta_fs;
    EQ.robustF(xx,:) = (first_stage(xx,:) / SE_robust(xx,:))^2;
end

clear SE_robust

% Second stage: effects on non-instrumented variables
for xx=1:K2
    X_ss = pred_firststage;
    y_ss = resid2(:,xx);
    beta_ss = X_ss\y_ss;
    X(xx,:) = beta_ss';
end

X=X(:,1);

% Compute absolute IRFs following Mertens & Ravn (2013)
varcovar11=varcovar(1:K1,1:K1);
varcovar12=varcovar(1:K1,K1+1:K);
varcovar21=varcovar(K1+1:K,1:K1);
varcovar22=varcovar(K1+1:K,K1+1:K);
Z=X*varcovar11*X'-(varcovar21*X'+X*varcovar21')+varcovar22;
bet12_bet12t=(varcovar21-X*varcovar11)'*inv(Z)*(varcovar21-X*varcovar11);
bet22_bet22t=varcovar22+X*(bet12_bet12t-varcovar11)*X';
bet11_bet11t=varcovar11-bet12_bet12t;
bet12_inv_bet22=(bet12_bet12t*X'+(varcovar21-X*varcovar11)')*inv(bet22_bet22t);

S1_S1t=(eye(K1)-bet12_inv_bet22*X)*bet11_bet11t*(eye(K1)-bet12_inv_bet22*X)';
S1=chol(S1_S1t)';
bet11=inv(eye(K1)-bet12_inv_bet22*X)*S1;
bet21=X*inv(eye(K1)-bet12_inv_bet22*X)*S1;
bet1=[bet11' bet21']';

P=[bet1 zeros(K,K-K1)];
EQ.P=P;

if isempty(shock)==1
    shock=ones(K1,1);  % Unit shocks for proxy-SVARs
end

% Compute IRFs to the K1 identified structural shocks
V=zeros(K*nlags,1+numdum);
demm=isfield(Q,'constants');
if demm==1
    V(1:K,1)=Q.constants;
end
Vexo=zeros(K*nlags,size(exog,2));
if isempty(exog)==0
    Vexo(1:K,:)=Q.exo;
end
demmdum=isfield(Q,'dummies');
if demmdum==1
    V(1:K,2:numdum+1)=Q.dummies;
end
A=zeros(K*nlags,K*nlags);
A(1:K,:)=Q.coefficients;
J_mat=zeros(K,K*nlags);
J_mat(:,1:K)=eye(K);
if nlags>1
    A(K+1:K*nlags,1:K*nlags-K)=eye(K*nlags-K);
end
Irf=zeros(K*nlags,irf_lenght,K1);

for xx=1:K1
    w=zeros(K,1);
    if rel==1
        w(xx,1)=shock(xx)/P(xx,xx);  % Relative: unit shock to variable
    elseif rel==0
        w(xx,1)=shock(xx);            % Absolute: unit structural shock
    end
    W=zeros(K*nlags,irf_lenght);
    W(1:K,1)=P*w;
    for yy=1:irf_lenght
    Irf(:,yy+1,xx)=A*Irf(:,yy,xx)+W(:,yy);
    end
end
for xx=1:K1
    for yy=1:irf_lenght
        irf(:,yy,xx)=J_mat*Irf(:,yy,xx);
    end
end

% Wild bootstrap for confidence intervals 
u=zeros(K,size(resid,1),bootstrap_num);
iv_sim=zeros(size(iv,1),K1,bootstrap_num);

% Rademacher distribution (+1 or -1 with equal probability)
for xx=1:bootstrap_num
    if verbose==true
        xx
    end
    rs_unif=rand(1,size(data,1)-nlags);
    for hh=1:size(rs_unif,2)
        if rs_unif(1,hh)>0.5
        rs(1,hh)=1;
        else rs(1,hh)=-1;
        end
    end
    for yy=1:size(data,1)-nlags
    u(:,yy,xx)=resid(yy,:)'.*rs(1,yy);
    iv_sim(yy,:,xx)=iv(yy,:).*rs(1,yy);
    end
end
    
U=zeros(K*nlags,size(data,1)-nlags,bootstrap_num);
U(1:K,:,:)=u;
Y=zeros(K*nlags,size(data,1)-nlags+1,bootstrap_num);
stucaz=data(1:nlags,:)';
stucaz=flip(stucaz,2);
for xx=1:bootstrap_num
    Y(:,1,xx)=reshape(stucaz,K*nlags,1);
end

for xx=1:bootstrap_num
    conta=0;
    for yy=1:size(data,1)-nlags
        conta=conta+1;
        if isempty(dum)==0
                vec_dummies=zeros(size(V,2)-1,1);
                if mod(conta,numdum)~=0
                    prot=mod(conta,numdum);
                else prot=numdum;
                end
            vec_dummies(prot,1)=1;
            vec_dummies=[1;vec_dummies];
            else vec_dummies=1;
        end
        if isempty(exog)==0
            Y(:,yy+1,xx)=V*vec_dummies+Vexo*exog(yy,:)'+A*Y(:,yy,xx)+U(:,yy,xx);
        else Y(:,yy+1,xx)=V*vec_dummies+A*Y(:,yy,xx)+U(:,yy,xx);
        end
    end
end

y=zeros(K,size(data,1),bootstrap_num);
for xx=1:bootstrap_num
    y(:,1:nlags,xx)=data(1:nlags,:)';
end
clear stucaz
Y=Y(:,2:size(Y,2),:);
for xx=1:bootstrap_num
    for yy=1:size(data,1)-nlags
        y(:,yy+nlags,xx)=J_mat*Y(:,yy,xx);
    end
end
for xx=1:bootstrap_num
    temp(:,:,xx)=y(:,:,xx)';
end
y=temp;
clear temp

for jj=1:bootstrap_num
    Q_temp=reducedformVAR(y(:,:,jj),nlags,const,lr,[],dum,exog);
    varcovar_temp=Q_temp.sigma;
    resid_temp=Q_temp.resid;
    resid1_temp=resid_temp(:,1:K1);
    resid2_temp=resid_temp(:,K1+1:K);
    
    for xx=1:K1
        X_fs_temp = iv_sim(:,:,jj);
        y_fs_temp = resid1_temp(:,xx);
        beta_fs_temp = X_fs_temp\y_fs_temp;
        first_stage_temp(xx,:) = beta_fs_temp';
        pred_firststage_temp(:,xx) = X_fs_temp * beta_fs_temp;
    end
    
    clear X_temp
    for xx=1:K2
        X_ss_temp = pred_firststage_temp;
        y_ss_temp = resid2_temp(:,xx);
        beta_ss_temp = X_ss_temp\y_ss_temp;
        X_temp(xx,:) = beta_ss_temp';
    end
    
    X_temp=X_temp(:,1);
    varcovar11_temp=varcovar_temp(1:K1,1:K1);
    varcovar12_temp=varcovar_temp(1:K1,K1+1:K);
    varcovar21_temp=varcovar_temp(K1+1:K,1:K1);
    varcovar22_temp=varcovar_temp(K1+1:K,K1+1:K);
    Z_temp=X_temp*varcovar11_temp*X_temp'-(varcovar21_temp*X_temp'+X_temp*varcovar21_temp')+varcovar22_temp;
    bet12_bet12t_temp=(varcovar21_temp-X_temp*varcovar11_temp)'*inv(Z_temp)*(varcovar21_temp-X_temp*varcovar11_temp);
    bet22_bet22t_temp=varcovar22_temp+X_temp*(bet12_bet12t_temp-varcovar11_temp)*X_temp';
    bet11_bet11t_temp=varcovar11_temp-bet12_bet12t_temp;
    bet12_inv_bet22_temp=(bet12_bet12t_temp*X_temp'+(varcovar21_temp-X_temp*varcovar11_temp)')*inv(bet22_bet22t_temp);
    S1_S1t_temp=(eye(K1)-bet12_inv_bet22_temp*X_temp)*bet11_bet11t_temp*(eye(K1)-bet12_inv_bet22_temp*X_temp)';
    S1_temp=chol(S1_S1t_temp)';
    bet11_temp=inv(eye(K1)-bet12_inv_bet22_temp*X_temp)*S1_temp;
    bet21_temp=X_temp*inv(eye(K1)-bet12_inv_bet22_temp*X_temp)*S1_temp;
    bet1_temp=[bet11_temp' bet21_temp']';
    P_temp=[bet1_temp zeros(K,K-K1)];
    A_temp=zeros(K*nlags,K*nlags);
    A_temp(1:K,:)=Q_temp.coefficients;
    if nlags>1
        A_temp(K+1:K*nlags,1:K*nlags-K)=eye(K*nlags-K);
    end
    Irf_temp=zeros(K*nlags,irf_lenght,K1);
    for xx=1:K1
        w_temp=zeros(K,1);
        if rel==1
            w_temp(xx,1)=shock(xx)/P_temp(xx,xx);
        elseif rel==0
            w_temp(xx,1)=shock(xx);
        end
        W_temp=zeros(K*nlags,irf_lenght);
        W_temp(1:K,1)=P_temp*w_temp;
        for yy=1:irf_lenght
        Irf_temp(:,yy+1,xx)=A_temp*Irf_temp(:,yy,xx)+W_temp(:,yy);
        end
    end
    for xx=1:K1
        for yy=1:irf_lenght
            irf_sim(:,yy,jj,xx)=J_mat*Irf_temp(:,yy,xx);
        end
    end
end

EQ.irf_sim=irf_sim;
perc_irf_up=zeros(K,irf_lenght,K1);
perc_irf_down=zeros(K,irf_lenght,K1);
banda1=confianza;
banda2=100-confianza;
for xx=1:K1
    for yy=1:irf_lenght
        for hh=1:K
            temp=sort(irf_sim(hh,yy,:,xx));
            temp1=[temp(floor(bootstrap_num*banda2/100));temp(ceil(bootstrap_num*banda1/100))];
            perc_irf_up(hh,yy,xx)=temp1(1);
            perc_irf_down(hh,yy,xx)=temp1(2);
            irf_std(hh,yy,xx)=std(irf_sim(hh,yy,:,xx));
        end
    end
end

for xx=1:K1
    for yy=1:K
        eval([var_names{yy} '_' shock_names{xx} '=[perc_irf_up(yy,:,xx);irf(yy,:,xx);perc_irf_down(yy,:,xx)];']);
    end
end

for xx=1:K1
    count=0;
    figure(xx);
    for yy=1:K
        count=count+1;
        subplot(K,1,count)
        eval(['plot(1:irf_lenght,' var_names{yy} '_' shock_names{xx} '(1,:),'':k'',1:irf_lenght,' var_names{yy} '_' shock_names{xx} '(2,:),''k'',1:irf_lenght,' var_names{yy} '_' shock_names{xx} '(3,:),'':k'');']);
        eval(['title(''' var_names{yy} ' ' shock_names{xx} ''');']);
    end
end

count=0;
for xx=1:K1
    for yy=1:K
        count=count+1;
        eval(['EQ.' var_names{yy} '_' shock_names{xx} '=' var_names{yy} '_' shock_names{xx} ';']);
    end
end

% Variance decomposition
C=zeros(K*nlags,K);
C(1:K,1:K)=P;
C1=zeros(K*nlags,K*nlags);
C1(1:K,1:K)=varcovar;
Totvar=zeros(K*nlags,K*nlags,irf_lenght);
for xx=1:irf_lenght
    for yy=1:xx
        Totvar(:,:,xx)=A^(yy-1)*C1*(A^(yy-1))'+Totvar(:,:,xx);
    end
end
totvar=zeros(K,K,irf_lenght);
for xx=1:irf_lenght
    totvar(:,:,xx)=J_mat*Totvar(:,:,xx)*J_mat';
end
Var=zeros(K*nlags,K*nlags,irf_lenght,K1);
for xx=1:K1
    s=zeros(K,K);
    s(xx,xx)=1;
    for yy=1:irf_lenght
        for hh=1:yy
            Var(:,:,yy,xx)=A^(hh-1)*C*s*C'*(A^(hh-1))'+Var(:,:,yy,xx);
        end
    end
end
var=zeros(K,K,irf_lenght,K1);
for xx=1:K1
    for yy=1:irf_lenght
    var(:,:,yy,xx)=J_mat*Var(:,:,yy,xx)*J_mat';
    end
end
            
var_dec=zeros(K,K,irf_lenght,K1);
for xx=1:K1
    var_dec(:,:,:,xx)=var(:,:,:,xx)./totvar(:,:,:)*100;
end

for xx=1:K1
    for yy=1:irf_lenght
    eval(['vardecshock_' shock_names{xx} '(:,yy)=diag(var_dec(:,:,yy,xx));']);
    end
end

for xx=1:K1
    eval(['EQ.vardecshock_' shock_names{xx} '=vardecshock_' shock_names{xx} ';']);
end

EQ.stdirf=irf_std;

% Recover structural shocks (Lunsford, 2015)
phi=((1/(size(data,1)-K*nlags-1)*iv'*resid)*inv(varcovar)*(1/(size(data,1)-K*nlags-1)*resid'*iv))^(1/2);
pai=varcovar^(-1)*(size(data,1)-K*nlags-1)^(-1)*resid'*iv;
EQ.struc=(resid*pai*phi^(-1))';

end