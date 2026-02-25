function EQ=srSVAR(data,nlags,var_names,irf_lenght,monte_carlo,const,lr,dum,exog,shock,confianza)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose: 
% Impulse responses and Variance decomposition of a Structural Vector Autoregression
% with short run restrictions (Cholesky decomposition), with optional restrictions on
% variables' interactions
% -----------------------------------
% Inputs:
% data = NxK matrix (N number of observations, K number of variables)
% nlags = desired number of lags (no maximum limit enforced)
% Optional:
% var_names = 1xK cell vector with names of the variables
% irf_lenght = desired length of the IRFs
% monte_carlo = number of Monte Carlo repetitions for the computation of
% the confidence intervals of the IRFs. Default is 1000.
% const = true if the VAR is estimated with a constant and false if without.
% Default is false.
% dum = seasonal dummies as in reducedformVAR
% exog = NxE matrix, where E is the number of exogenous variables that
% enter without lags (non-dynamic regressors, e.g., event dummies). 
% These variables are taken as deterministic in the construction of the
% confidence bands.
% lr = KxK matrix of zeros and ones; zeros indicate that the column 
% variable does not affect the row variable, one indicates effect is allowed.
% Default is no restrictions.
% shock = Kx1 vector containing the size of the shock for each variable. 
% Default is one standard deviation shocks.
% confianza = level in % of the lowest confidence band. Default is 5%.
% -----------------------------------
% Returns:
% EQ = structure with the following fields:
% EQ.vari_varj = K*K collections of 3xirf_lenght matrices containing in the first row the
% 95% confidence band, in the second row the IRF, and in the third row the
% 5% confidence band. i is the variable, j is the shock.
% EQ.P = KxK Cholesky matrix of the variance-covariance matrix of the
% residuals
% EQ.vardecshock_vari = K matrices of size Kxirf_lenght, where each row j contains the
% percentage of variance of variable j explained by shock i at each horizon
% EQ.stdirf = Kxirf_lenghtxK array containing the standard deviations of
% the IRF coefficients 
% EQ.struc = KxN matrix containing estimated structural shocks
% ------------------------------------
% The solution method follows Lütkepohl (2005)

if nargin<11
    confianza=5;
end
if nargin<10
    shock=[];
    confianza=5;
end
if nargin<9
    shock=[];
    confianza=5;
    exog=[];
end
if nargin<8
    shock=[];
    confianza=5;
    dum=[];
    exog=[];
end
if nargin<7
    shock=[];
    confianza=5; 
    exog=[];
    dum=[];
    lr=[];
end
if nargin<6
    shock=[];
    confianza=5; 
    exog=[];
    dum=[];
    const=false;
    lr=[];
end
if nargin<5
    shock=[];
    confianza=5;   
    exog=[];
    dum=[];
    const=false;
    lr=[];
    monte_carlo=1000;
end
if nargin<4
    shock=[];
    confianza=5;    
    exog=[];
    dum=[];
    const=false;
    lr=[];
    monte_carlo=1000;
    irf_lenght=40;
end
if nargin<3
    shock=[];
    confianza=5;   
    exog=[];
    dum=[];
    const=false;
    lr=[];
    monte_carlo=1000;
    irf_lenght=40;
    for xx=1:size(data,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
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
if isempty(var_names)==1
    for xx=1:size(data,2)
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

K=size(data,2);
Q=reducedformVAR(data,nlags,const,lr,[],dum,exog);
varcovar=Q.sigma;
P=chol(varcovar)';
EQ.P=P;
if isempty(shock)==1
    shock=diag(P);
end

V=zeros(K*nlags,1+numdum);
demm=isfield(Q,'constants');
if demm==1
    V(1:K,1)=Q.constants;
end
Vexo=zeros(K*nlags,size(exog,2));
if isempty(exog)==0
    Vexo(1:K,:)=Q.exo;
end
if isempty(dum)==0
    V(1:K,2:numdum+1)=Q.dummies;
end
A=zeros(K*nlags,K*nlags);
A(1:K,:)=Q.coefficients;
J=zeros(K,K*nlags);
J(:,1:K)=eye(K);
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
        irf(:,yy,xx)=J*Irf(:,yy,xx);
    end
end

u=randn(K,size(data,1)+100,monte_carlo);
for xx=1:monte_carlo
    u(:,:,xx)=P*u(:,:,xx);
end
U=zeros(K*nlags,size(data,1)+100,monte_carlo);
U(1:K,:,:)=u;
Y=zeros(K*nlags,size(data,1)+100,monte_carlo);
if isempty(exog)==0
    exo_u=[zeros(size(exog,2),100) exog'];
end
for xx=1:monte_carlo
    conta=0;
    for yy=1:size(data,1)+99
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
y=zeros(K,size(data,1),monte_carlo);
for xx=1:monte_carlo
    for yy=1:size(data,1)
        y(:,yy,xx)=J*Y(:,yy,xx);
    end
end
for xx=1:monte_carlo
    temp(:,:,xx)=y(:,:,xx)';
end
y=temp;
clear temp

for jj=1:monte_carlo
    Q_temp=reducedformVAR(y(:,:,jj),nlags,const,lr,[],dum,exog);
    varcovar_temp=Q_temp.sigma;
    P_temp=chol(varcovar_temp)';
    A_temp=zeros(K*nlags,K*nlags);
    A_temp(1:K,:)=Q_temp.coefficients;
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
            irf_sim(:,yy,jj,xx)=J*Irf_temp(:,yy,xx);
        end
    end
end

EQ.irf_sim=irf_sim;
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
        eval([var_names{yy} '_' var_names{xx} '=[perc_irf_up(yy,:,xx);irf(yy,:,xx);perc_irf_down(yy,:,xx)];']);
    end
end

% Plot IRFs
for xx=1:K
    count=0;
    figure(xx);
    for yy=1:K
        count=count+1;
        subplot(K,1,count)
        eval(['plot(1:irf_lenght,' var_names{yy} '_' var_names{xx} '(1,:),'':k'',1:irf_lenght,' var_names{yy} '_' var_names{xx} '(2,:),''k'',1:irf_lenght,' var_names{yy} '_' var_names{xx} '(3,:),'':k'');']);
        eval(['title(''' var_names{yy} ' ' var_names{xx} ''');']);
    end
end

count=0;
for xx=1:K
    for yy=1:K
        count=count+1;
        eval(['EQ.' var_names{yy} '_' var_names{xx} '=' var_names{yy} '_' var_names{xx} ';']);
    end
end

C=zeros(K*nlags,K);
C(1:K,1:K)=P;
Totvar=zeros(K*nlags,K*nlags,irf_lenght);
count=0;
for xx=1:irf_lenght
    count=count+1;
    for yy=1:count
        Totvar(:,:,xx)=A^(yy-1)*C*C'*(A^(yy-1))'+Totvar(:,:,xx);
    end
end
totvar=zeros(K,K,irf_lenght);
for xx=1:irf_lenght
    totvar(:,:,xx)=J*Totvar(:,:,xx)*J';
end
Var=zeros(K*nlags,K*nlags,irf_lenght,K);
for xx=1:K
    s=zeros(K,K);
    s(xx,xx)=1;
    count=0;
    for yy=1:irf_lenght
        count=count+1;
        for hh=1:count
            Var(:,:,yy,xx)=A^(hh-1)*C*s*C'*(A^(hh-1))'+Var(:,:,yy,xx);
        end
    end
end
var=zeros(K,K,irf_lenght,K);
for xx=1:K
    for yy=1:irf_lenght
    var(:,:,yy,xx)=J*Var(:,:,yy,xx)*J';
    end
end
            
var_dec=zeros(K,K,irf_lenght,K);
for xx=1:K
    var_dec(:,:,:,xx)=var(:,:,:,xx)./totvar(:,:,:)*100;
end

for xx=1:K
    for yy=1:irf_lenght
    eval(['vardecshock_' var_names{xx} '(:,yy)=diag(var_dec(:,:,yy,xx));']);
    end
end

for xx=1:K
    eval(['EQ.vardecshock_' var_names{xx} '=vardecshock_' var_names{xx} ';']);
end

EQ.stdirf=irf_std;
EQ.struc=(inv(P)*Q.resid');

end