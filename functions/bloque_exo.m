function EQ=bloque_exo(data_bloqueexo,data_bloqueendo,nlags,var_names,shock_names,irf_lenght,monte_carlo,const,lrexo,lrendo,dum,exog,shock,confianza,verbose)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose: 
% Impulse responses and Variance decomposition of a Structural Vector Autoregression
% with short run (Cholesky) restrictions and a block-exogenous structure where
% the exogenous block has a possibly larger sample than the endogenous block.
% -------------------------------------------------------------------------
% Inputs:
% data_bloqueexo = NexoxK1 matrix (Nexo number of observations, K1 number of variables).
%                  Variables in the exogenous block (ordered first, affect endogenous block
%                  but are not affected by it).
% data_bloqueendo = NendoxK2 matrix (Nendo number of observations, K2 number of variables).
%                   Variables in the endogenous block. It is assumed that Nendo < Nexo and 
%                   the endogenous block corresponds to the last Nendo dates of the exogenous block.
% nlags = desired number of lags (no maximum limit enforced)
% Optional:
% var_names = 1x(K1+K2) cell vector with names of all variables (exogenous block first, 
%             then endogenous block)
% shock_names = 1x(K1+K2) cell vector with names of the structural shocks
% irf_lenght = desired length of the IRFs. Default is 40.
% monte_carlo = number of Monte Carlo repetitions for the computation of
%               the confidence intervals of the IRFs. Default is 1000.
% const = true if the VAR is estimated with a constant and false if without.
%         Default is false.
% lrexo = K1xK1 matrix of zeros and ones, zeros for variables that do not have
%         effects on other variables, one otherwise, in the exogenous block
% lrendo = K2xK2 matrix of zeros and ones, zeros for variables that do not have
%          effects on other variables, one otherwise, in the endogenous block
% dum = seasonal dummies. Specify 'month' for 11 monthly dummies or 
%       'quarter' for 3 quarterly dummies. Default is no dummies.
% exog = NxE matrix, where E is the number of exogenous variables that
%        enter without lags (non-dynamic regressors, e.g., event dummies).
% shock = (K1+K2)x1 vector containing the size of the shock for each variable. 
%         Default is one standard deviation shocks.
% confianza = level in % of the lowest confidence band. Default is 5%.
% verbose = true to print Monte Carlo iteration numbers, false to suppress. Default is false.
% -----------------------------------
% Returns:
% EQ = structure with the following fields:
% EQ.vari_shockj = (K1+K2)*(K1+K2) 3xirf_lenght matrices containing in the first row the
%                  95% confidence line, in the second row the IRF, and in the third row the
%                  5% confidence line. i is the variable, j is the shock.
% EQ.P = (K1+K2)x(K1+K2) block-triangular Cholesky matrix of the variance covariance matrix.
%        Upper-left K1xK1 block is the exogenous block identification.
%        Lower-right K2xK2 block is the endogenous block identification conditional on exogenous.
% EQ.vardecshock_shockj = (K1+K2) matrices of size (K1+K2)xirf_lenght containing in each row j the
%                         percentage of variance of variable j due to shock i. Each column is 
%                         the number of periods ahead.
% EQ.stdirf = (K1+K2)xirf_lenghtx(K1+K2) array containing standard deviations of IRFs from Monte Carlo
% EQ.struc = (K1+K2)xT2 matrix of structural shocks for the overlapping sample period
% EQ.constants = (K1+K2)x1 vector of constants (if const=true)
% EQ.coefficients = (K1+K2)x((K1+K2)*nlags) matrix of VAR coefficients
% EQ.exo = (K1+K2)xE matrix of coefficients on exogenous variables (if exog provided)
% EQ.dummies = (K1+K2)xnumdum matrix of seasonal dummy coefficients (if dum provided)
% ------------------------------------
% The solution method follows Lütkepohl (2005)

if nargin<15
    verbose=false;
end
if nargin<14
    verbose=false;
    confianza=5;
end
if nargin<13
    verbose=false;
    shock=[];
    confianza=5;
end
if nargin<12
    verbose=false;
    shock=[];
    confianza=5;
    exog=[];
end
if nargin<11
    verbose=false;
    shock=[];
    confianza=5;
    dum=[];
    exog=[];
end
if nargin<10
    verbose=false;
    shock=[];
    confianza=5;
    dum=[];
    exog=[];
    lrendo=[];
end
if nargin<9
    verbose=false;
    shock=[];
    confianza=5;
    dum=[];
    exog=[];
    lrexo=[];
    lrendo=[];
end
if nargin<8
    verbose=false;
    shock=[];
    confianza=5; 
    exog=[];
    dum=[];
    lrexo=[];
    lrendo=[];
    const=false;
end
if nargin<7
    verbose=false;
    shock=[];
    confianza=5;   
    exog=[];
    dum=[];
    lrexo=[];
    lrendo=[];
    const=false;
    monte_carlo=1000;
end
if nargin<6
    verbose=false;
    shock=[];
    confianza=5;    
    exog=[];
    dum=[];
    lrexo=[];
    lrendo=[];
    const=false;
    monte_carlo=1000;
    irf_lenght=40;
end
if nargin<5
    verbose=false;
    shock=[];
    confianza=5;   
    exog=[];
    dum=[];
    lrexo=[];
    lrendo=[];
    const=false;
    monte_carlo=1000;
    irf_lenght=40;
    for xx=1:size(data_bloqueexo,2)+size(data_bloqueendo,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end
if nargin<4
    verbose=false;
    shock=[];
    confianza=5;   
    exog=[];
    dum=[];
    lrexo=[];
    lrendo=[];
    const=false;
    monte_carlo=1000;
    irf_lenght=40;
    for xx=1:size(data_bloqueexo,2)+size(data_bloqueendo,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
    for xx=1:size(data_bloqueexo,2)+size(data_bloqueendo,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end

if isempty(verbose)==1
    verbose=false;
end
if isempty(monte_carlo)==1
    monte_carlo=1000;
end
if isempty(irf_lenght)==1
    irf_lenght=40;
end
if isempty(confianza)==1
    confianza=5;
end
if isempty(shock_names)==1
    for xx=1:size(data_bloqueexo,2)+size(data_bloqueendo,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end
if isempty(var_names)==1
    for xx=1:size(data_bloqueexo,2)+size(data_bloqueendo,2)
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

K1=size(data_bloqueexo,2);
K2=size(data_bloqueendo,2);
T1=size(data_bloqueexo,1);
T2=size(data_bloqueendo,1);
K=K1+K2;

% Estimate exogenous block and recover shocks
exogeno_rf=reducedformVAR([data_bloqueexo],nlags,const,lrexo,[],dum,exog);
ccc=1/size(exogeno_rf.resid,1)*exogeno_rf.resid'*exogeno_rf.resid;
c=chol(ccc)';
choques=(inv(c)*exogeno_rf.resid');
choques=choques';
P=c;

% Estimate reduced form of full model
if isempty(lrexo)==1 & isempty(lrendo)==1
    lr=[];
elseif isempty(lrexo)==0 & isempty(lrendo)==1
    lr=[lrexo ones(K1,K2);ones(K2,K1) ones(K2,K2)];
elseif isempty(lrexo)==1 & isempty(lrendo)==0
    lr=[ones(K1,K1) ones(K1,K2);ones(K2,K1) lrendo];
elseif isempty(lrexo)==0 & isempty(lrendo)==0
    lr=[lrexo ones(K1,K2);ones(K2,K1) lrendo];
end
if isempty(exog)==1
    Q=reducedformVAR([data_bloqueexo(end-T2+1:end,:) data_bloqueendo],nlags,const,lr,[],dum);
else 
    Q=reducedformVAR([data_bloqueexo(end-T2+1:end,:) data_bloqueendo],nlags,const,lr,[],dum,exog(end-T2+1:end,:));
end

% Use OLS to recover Cholesky coefficients of the endogenous block on the
% structural shocks to variables in the exogenous block
for xx=K1+1:K2+K1
    X = choques(T1-T2+1:end,:);
    y = Q.resid(:,xx);
    beta = X\y;
    e(xx,:) = beta';
    resid(:,xx) = y - X*beta;
end
e(1:K1,:)=[];
resid(:,1:K1)=[];

% Now recover Cholesky coefficients between variables of the endogenous block
ccc=1/size(resid,1)*resid'*resid;
c=chol(ccc)';

% Complete Cholesky matrix (block-triangular structure)
P=[P zeros(K1,K2);
    e c];
EQ.P=P;

residuali=[exogeno_rf.resid(T1-T2+1:end,:)'; Q.resid(:,K1+1:K)'];
EQ.struc=inv(P)*residuali; 

if isempty(shock)==1
    shock=diag(P);
end

V=zeros(K*nlags,1+numdum);
demm=isfield(exogeno_rf,'constants');
if demm==1
    V(1:K,1)=[exogeno_rf.constants; Q.constants(K1+1:K2+K1)];
    EQ.constants=[exogeno_rf.constants; Q.constants(K1+1:K2+K1)];
end

Vexo=zeros(K*nlags,size(exog,2));
demmexo=isfield(exogeno_rf,'exo');
if demmexo==1
    Vexo(1:K,:)=[exogeno_rf.exo; Q.exo(K1+1:K2+K1)];
    EQ.exo=[exogeno_rf.exo; Q.exo(K1+1:K2+K1)];
end

demmdum=isfield(exogeno_rf,'dummies');
if demmdum==1
    V(1:K,2:numdum+1)=[exogeno_rf.dummies; Q.dummies(K1+1:K, :)];
    EQ.dummies=[exogeno_rf.dummies; Q.dummies(K1+1:K, :)];
end

A=zeros(K*nlags,K*nlags);
for xx=1:nlags
    A(1:K1,1+(xx-1)*K:K1+(xx-1)*K)=exogeno_rf.coefficients(:,1+(xx-1)*K1:K1+(xx-1)*K1);
end
A(K1+1:K,:)=[Q.coefficients(K1+1:K2+K1,:)];
EQ.coefficients=A(1:K,:);

J_mat=zeros(K,K*nlags);
J_mat(:,1:K)=eye(K);
if nlags>1
    A(K+1:K*nlags,1:K*nlags-K)=eye(K*nlags-K);
end

Irf=zeros(K*nlags,irf_lenght,K);

for xx=1:K
    w=zeros(K,1);
    w(xx,1)=shock(xx)/P(xx,xx);
    W=zeros(K*nlags,irf_lenght);
    W(1:K,1)=P*w;
    for yy=1:irf_lenght
    Irf(:,yy+1,xx)=A*Irf(:,yy,xx)+W(:,yy);
    end
end
for xx=1:K
    for yy=1:irf_lenght
        irf(:,yy,xx)=J_mat*Irf(:,yy,xx);
    end
end

% Monte Carlo simulation for confidence bands
u=randn(K,size(data_bloqueexo,1)+100,monte_carlo);
for xx=1:monte_carlo
    u(:,:,xx)=P*u(:,:,xx);
end
U=zeros(K*nlags,size(data_bloqueexo,1)+100,monte_carlo);
U(1:K,:,:)=u;
Y=zeros(K*nlags,size(data_bloqueexo,1)+100,monte_carlo);
if isempty(exog)==0
    exo_u=[zeros(size(exog,2),100) exog'];
end

for xx=1:monte_carlo
    if verbose==true
        xx
    end
    conta=0;
    for yy=1:size(data_bloqueexo,1)+99
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
            Y(:,yy+1,xx)=V*vec_dummies+Vexo*exo_u(:,yy)+A*Y(:,yy,xx)+U(:,yy+1,xx);
        else Y(:,yy+1,xx)=V*vec_dummies+A*Y(:,yy,xx)+U(:,yy+1,xx);
        end
    end
end

Y=Y(:,101:size(Y,2),:);
y=zeros(K,size(data_bloqueexo,1),monte_carlo);
for xx=1:monte_carlo
    for yy=1:size(data_bloqueexo,1)
        y(:,yy,xx)=J_mat*Y(:,yy,xx);
    end
end
for xx=1:monte_carlo
    temp(:,:,xx)=y(:,:,xx)';
end
y=temp;
clear temp

for jj=1:monte_carlo
    exogeno_rf_temp=reducedformVAR(y(:,1:K1,jj),nlags,const,lrexo,[],dum,exog);
    ccc_temp=1/size(exogeno_rf_temp.resid,1)*exogeno_rf_temp.resid'*exogeno_rf_temp.resid;
    c_temp=chol(ccc_temp)';
    choques_temp=(inv(c_temp)*exogeno_rf_temp.resid');
    choques_temp=choques_temp';
    P_temp=c_temp;
    if isempty(exog)==1
        Q_temp=reducedformVAR([y(end-T2+1:end,1:K1,jj) y(end-T2+1:end,K1+1:K,jj)],nlags,const,lr,[],dum);
    else 
        Q_temp=reducedformVAR([y(end-T2+1:end,1:K1,jj) y(end-T2+1:end,K1+1:K,jj)],nlags,const,lr,[],dum,exog(end-T2+1:end,:));
    end
    for xx=K1+1:K2+K1
        X_temp = choques_temp(T1-T2+1:end,:);
        y_temp = Q_temp.resid(:,xx);
        beta_temp = X_temp\y_temp;
        e_temp(xx,:) = beta_temp';
        resid_temp(:,xx) = y_temp - X_temp*beta_temp;
    end
    e_temp(1:K1,:)=[];
    resid_temp(:,1:K1)=[];
    ccc_temp=1/size(resid_temp,1)*resid_temp'*resid_temp;
    c_temp=chol(ccc_temp)';
    P_temp=[P_temp zeros(K1,K2);
        e_temp c_temp];
    A_temp=zeros(K*nlags,K*nlags);
    for xx=1:nlags
        A_temp(1:K1,1+(xx-1)*K:K1+(xx-1)*K)=exogeno_rf_temp.coefficients(:,1+(xx-1)*K1:K1+(xx-1)*K1);
    end
    A_temp(K1+1:K,:)=[Q_temp.coefficients(K1+1:K2+K1,:)];
    J_temp=zeros(K,K*nlags);
    J_temp(:,1:K)=eye(K);
    if nlags>1
        A_temp(K+1:K*nlags,1:K*nlags-K)=eye(K*nlags-K);
    end
    Irf_temp=zeros(K*nlags,irf_lenght,K);
    for xx=1:K
        w_temp=zeros(K,1);
        w_temp(xx,1)=shock(xx)/P_temp(xx,xx);
        W_temp=zeros(K*nlags,irf_lenght);
        W_temp(1:K,1)=P_temp*w_temp;
        for yy=1:irf_lenght
        Irf_temp(:,yy+1,xx)=A_temp*Irf_temp(:,yy,xx)+W_temp(:,yy);
        end
    end
    for xx=1:K
        for yy=1:irf_lenght
            irf_sim(:,yy,jj,xx)=J_temp*Irf_temp(:,yy,xx);
        end
    end
end

perc_irf_up=zeros(K,irf_lenght,K);
perc_irf_down=zeros(K,irf_lenght,K);
banda1=confianza;
banda2=100-confianza;
for xx=1:K
    for yy=1:irf_lenght
        for hh=1:K
            temp=sort(irf_sim(hh,yy,:,xx));
            temp1=[temp(floor(monte_carlo*banda2/100));temp(ceil(monte_carlo*banda1/100))];
            perc_irf_up(hh,yy,xx)=temp1(1);
            perc_irf_down(hh,yy,xx)=temp1(2);
            irf_std(hh,yy,xx)=std(irf_sim(hh,yy,:,xx));
        end            
    end
end

for xx=1:K
    for yy=1:K
        eval([var_names{yy} '_' shock_names{xx} '=[perc_irf_up(yy,:,xx);irf(yy,:,xx);perc_irf_down(yy,:,xx)];']);
    end
end

for xx=1:K
    count=0;
    figure(xx);
    for yy=1:K
        count=count+1;
        subplot(K,1,count)
        eval(['plot(1:irf_lenght,' var_names{yy} '_' shock_names{xx} '(1,:),'':k'',1:irf_lenght,' var_names{yy} '_' shock_names{xx} '(2,:),''k'',1:irf_lenght,' var_names{yy} '_' shock_names{xx} '(3,:),'':k'');']);
        eval(['title(''' var_names{yy} ' ' shock_names{xx} ''');']);
    end
end

C=zeros(K*nlags,K);
C(1:K,1:K)=P;
Totvar=zeros(K*nlags,K*nlags,irf_lenght);
for xx=1:irf_lenght
    for yy=1:xx
        Totvar(:,:,xx)=A^(yy-1)*C*C'*(A^(yy-1))'+Totvar(:,:,xx);
    end
end
totvar=zeros(K,K,irf_lenght);
for xx=1:irf_lenght
    totvar(:,:,xx)=J_mat*Totvar(:,:,xx)*J_mat';
end
Var=zeros(K*nlags,K*nlags,irf_lenght,K);
for xx=1:K
    s=zeros(K,K);
    s(xx,xx)=1;
    for yy=1:irf_lenght
        for hh=1:yy
            Var(:,:,yy,xx)=A^(hh-1)*C*s*C'*(A^(hh-1))'+Var(:,:,yy,xx);
        end
    end
end
var=zeros(K,K,irf_lenght,K);
for xx=1:K
    for yy=1:irf_lenght
    var(:,:,yy,xx)=J_mat*Var(:,:,yy,xx)*J_mat';
    end
end
            
var_dec=zeros(K,K,irf_lenght,K);
for xx=1:K
    var_dec(:,:,:,xx)=var(:,:,:,xx)./totvar(:,:,:)*100;
end

for xx=1:K
    for yy=1:irf_lenght
    eval(['vardecshock_' shock_names{xx} '(:,yy)=diag(var_dec(:,:,yy,xx));']);
    end
end

count=0;
for xx=1:K
    for yy=1:K
        count=count+1;
        eval(['EQ.' var_names{yy} '_' shock_names{xx} '=' var_names{yy} '_' shock_names{xx} ';']);
    end
end

for xx=1:K
    eval(['EQ.vardecshock_' shock_names{xx} '=vardecshock_' shock_names{xx} ';']);
end

EQ.stdirf=irf_std;

end