function EQ=nolucas(data,constants,coefficients,idmatrix,strucshocks,shock,withvar,withconst,dum,dummies,exog,coef_exog,forecast)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose: 
% Compute counterfactual contribution of shocks in SVAR models, possibly constraining
% transmission through some variables. NOT robust to Lucas critique when channels are
% shut down (agents' behavior would change if you actually blocked a transmission channel).
% -----------------------------------
% Inputs:
% data = NxK matrix (N number of observations, K number of variables)
% constants = Kx1 vector containing the constants of the reduced form VAR
% coefficients = KxK*nlags matrix containing the autoregressive coefficients of the reduced form VAR
% idmatrix = KxK matrix containing the identification matrix of the SVAR (impact matrix P)
% strucshocks = Kx(N-nlags) matrix containing the structural shocks 
% shock = scalar containing the index of the shock whose contribution the user wants to obtain
% withvar = 1xK vector containing ones in positions of variables whose transmission to RETAIN
%           and zeros in positions of variables whose transmission to SHUT OFF
% withconst = 1 if excluded variables should affect the deterministic part, 
%             0 if excluded variables should also be blocked in deterministic dynamics
% Optional:
% dum = 'month' for 11 monthly dummies, 'quarter' for 3 quarterly dummies, [] for none
% dummies = Kxnumdum matrix of seasonal dummy coefficients from VAR estimation
% exog = NxE matrix of exogenous variables
% coef_exog = KxE matrix of coefficients on exogenous variables
% forecast = number of periods ahead for forecasting. Default is 0.
% -----------------------------------
% Returns:
% EQ = structure
% EQ.contribution = KxN matrix containing the counterfactual dynamic of variables 
%                   with only the shock of interest and restricted transmission
% EQ.onlyshock = KxN matrix with pure shock contribution (no deterministic part)
% EQ.deterministic = KxN matrix with only the deterministic part
% EQ.forecast_contribution = Kxforecast matrix (if forecast>0)
% EQ.forecast_onlyshock = Kxforecast matrix (if forecast>0)
% EQ.forecast_deterministic = Kxforecast matrix (if forecast>0)
% ------------------------------------

if nargin<13
    forecast=0;
end
if nargin<12
    coef_exog=[];
end
if nargin<11
    coef_exog=[];
    exog=[];
end
if nargin<10
    dummies=[];
end
if nargin<9
    dummies=[];
    dum=[];
end

K=size(data,2);
nlags=size(coefficients,2)/K;
J=zeros(K,K*nlags);
J(:,1:K)=eye(K);

% Setup seasonal dummies
if isempty(dum)==1
    numdum=0;
elseif strcmpi('month',dum)==1
    numdum=11;
elseif strcmpi('quarter',dum)==1
    numdum=3;
else
    numdum=0;
end

stucaz=data(1:nlags,:)';
stucaz=flip(stucaz,2);

% Contribution of shock with restricted transmission
P=idmatrix;
Y=zeros(K*nlags,size(data,1)-nlags+1);
struc=strucshocks(shock,:);
u(:,:)=zeros(K,size(struc,2));
u(:,:)=P(:,shock)*struc;
U=zeros(K*nlags,size(data,1)-nlags);
U(1:K,:)=u;
A=zeros(K*nlags,K*nlags);
A(1:K,:)=coefficients;
if nlags>1
    A(K+1:K*nlags,1:K*nlags-K)=eye(K*nlags-K);
end

% Create restricted A matrix (block transmission through excluded variables)
Aless=A;
for zz=1:K
    if withvar(zz)==0
        for yy=1:nlags
            Aless(1:K,zz+K*(yy-1))=zeros(K,1);
        end
    end
end

% Simulate shock contribution with restricted dynamics
for yy=1:size(data,1)-nlags
    Y(:,yy+1)=Aless*Y(:,yy)+U(:,yy);
end
y=zeros(K,size(data,1));
Y=Y(:,2:size(Y,2));
for yy=1:size(data,1)-nlags
    y(:,yy+nlags)=J*Y(:,yy);
end

% Forecast shock contribution if requested
if forecast~=0
    Yfor(:,1)=Y(:,end);
    for yy=1:forecast
        Yfor(:,yy+1)=Aless*Yfor(:,yy);
    end
    yfor=zeros(K,forecast);
    Yfor=Yfor(:,2:size(Yfor,2));
    for yy=1:forecast
        yfor(:,yy)=J*Yfor(:,yy);
    end
end

% Deterministic component (constants + dummies + exogenous + dynamics)
V=zeros(K*nlags,1+numdum);
V(1:K,1)=constants;
if isempty(dum)==0 && ~isempty(dummies)
    V(1:K,2:numdum+1)=dummies;
end

Vexo=zeros(K*nlags,size(exog,2));
if isempty(exog)==0
    Vexo(1:K,:)=coef_exog;
end

Y=zeros(K*nlags,size(data,1)-nlags+1);
Y(:,1)=reshape(stucaz,K*nlags,1);

conta=0;
for yy=1:size(data,1)-nlags
    conta=conta+1;
    
    % Create time-varying dummy vector
    if isempty(dum)==0
        vec_dummies=zeros(size(V,2)-1,1);
        if mod(conta,numdum)~=0
            prot=mod(conta,numdum);
        else 
            prot=numdum;
        end
        vec_dummies(prot,1)=1;
        vec_dummies=[1;vec_dummies];
    else 
        vec_dummies=1;
    end
    
    if withconst==1
        % Full dynamics (excluded variables affect deterministic part)
        if isempty(exog)==1
            Y(:,yy+1)=V*vec_dummies+A*Y(:,yy);
        else 
            Y(:,yy+1)=V*vec_dummies+Vexo*exog(nlags+yy,:)'+A*Y(:,yy);
        end
    elseif withconst==0
        % Restricted dynamics (excluded variables also blocked in deterministic)
        if isempty(exog)==1
            Y(:,yy+1)=V*vec_dummies+Aless*Y(:,yy);
        else 
            Y(:,yy+1)=V*vec_dummies+Vexo*exog(nlags+yy,:)'+Aless*Y(:,yy);
        end
    end
end

y_hat=zeros(K,size(data,1));
y_hat(:,1:nlags)=data(1:nlags,:)';
Y=Y(:,2:size(Y,2));
for yy=1:size(data,1)-nlags
    y_hat(:,yy+nlags)=J*Y(:,yy);
end

% Forecast deterministic component if requested
if forecast~=0
    clear Yfor
    Yfor(:,1)=Y(:,end);
    conta_for=conta; % Continue dummy sequence
    
    for yy=1:forecast
        conta_for=conta_for+1;
        
        % Create time-varying dummy vector for forecast
        if isempty(dum)==0
            vec_dummies_for=zeros(size(V,2)-1,1);
            if mod(conta_for,numdum)~=0
                prot=mod(conta_for,numdum);
            else 
                prot=numdum;
            end
            vec_dummies_for(prot,1)=1;
            vec_dummies_for=[1;vec_dummies_for];
        else 
            vec_dummies_for=1;
        end
        
        if withconst==1
            if isempty(exog)==1
                Yfor(:,yy+1)=V*vec_dummies_for+A*Yfor(:,yy);
            else 
                Yfor(:,yy+1)=V*vec_dummies_for+Vexo*exog(end,:)'+A*Yfor(:,yy);
            end
        elseif withconst==0
            if isempty(exog)==1
                Yfor(:,yy+1)=V*vec_dummies_for+Aless*Yfor(:,yy);
            else 
                Yfor(:,yy+1)=V*vec_dummies_for+Vexo*exog(end,:)'+Aless*Yfor(:,yy);
            end
        end
    end
    Yfor=Yfor(:,2:size(Yfor,2));
    for yy=1:forecast
        y_forhat(:,yy)=J*Yfor(:,yy);
    end
end

% Final decomposition: contribution = onlyshock + deterministic
EQ.contribution=y+y_hat;
EQ.onlyshock=y;
EQ.deterministic=y_hat;

if forecast~=0
    EQ.forecast_contribution=yfor+y_forhat;
    EQ.forecast_onlyshock=yfor;
    EQ.forecast_deterministic=y_forhat;
end

end