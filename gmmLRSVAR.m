function EQ=gmmLRSVAR(data,nlags,idmat,var_names,shock_names,irf_lenght,monte_carlo,const,lr,dum,exog,shock,confianza)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose: 
% Impulse responses and Variance decomposition of a Structural Vector Autoregression
% with long run (possibly) overidentified zero restrictions, with optional restrictions on
% variables' interactions. Uses GMM estimation.
% -----------------------------------
% Inputs:
% data = NxK matrix (N number of observations, K number of variables)
% nlags = desired number of lags (no maximum limit enforced)
% idmat: KxK matrix of zeros and ones, zeros for structural shock that do not
% affect the corresponding variable in the long run and ones for those that do.
% These are LONG-RUN restrictions on the long-run multiplier matrix Theta(1).
% Optional:
% var_names = 1xK cell vector with names of the variables
% shock_names = 1xK cell vector with names of the shocks
% irf_lenght = desired length of the IRFs
% monte_carlo = number of Monte Carlo repetitions for the computation of
% the confidence intervals of the IRFs. Default is 1000.
% const = true if the VAR is estimated with a constant and false if without.
% Default is false.
% lr = KxK matrix of zeros and ones, zeros for variables that do not have
% effects on other variables, one otherwise. Default is no restrictions.
% dum = seasonal dummies as in reducedformVAR
% exog = NxE matrix, where E is the number of exogenous variables that
% enter without lags (non-dynamic regressors, e.g., event dummies). 
% These variables are taken as deterministic in the construction of the
% confidence bands.
% shock = Kx1 vector containing the size of the shock for each variable. 
% Default is one standard deviation shocks.
% confianza = level in % of the lowest confidence band. Default is 5%.
% -----------------------------------
% Returns:
% EQ = structure with the following fields:
% EQ.vari_shockj = K*K 3xirf_lenght matrices containing in the first row the
% 95% confidence line, in the second row the irf, and in the third row the
% 5% confidence line. i is the variable, j is the shock.
% EQ.P = KxK GMM-estimated impact matrix satisfying long-run restrictions
% EQ.vardecshock_shockj = K Kxirf_lenght matrices containing in each row j the
% percentage of variance of variable j due to shock i. Each column number
% is the number of periods ahead.
% EQ.stdirf = Kxirf_lenghtxK array containing the standard deviations of
% the IRF coefficients
% EQ.struc = Kx(N-nlags) matrix containing estimated structural shocks
% ------------------------------------
% The GMM estimation method follows Lütkepohl (2005).
% This function uses fminsearchcon for constrained optimization,
% authored by John D'Errico and available on MATLAB File Exchange:
% https://www.mathworks.com/matlabcentral/fileexchange/8277

if nargin<13
    confianza=5;
end
if nargin<12
    shock=[];
    confianza=5;
end
if nargin<11
    shock=[];
    confianza=5;
    exog=[];
end
if nargin<10
    shock=[];
    confianza=5;
    exog=[];
    dum=[];
end
if nargin<9
    shock=[];
    confianza=5;
    exog=[];
    dum=[];
    lr=[];
end
if nargin<8
    shock=[];
    confianza=5;
    exog=[];
    dum=[];
    const=false;
    lr=[];
end
if nargin<7
    shock=[];
    confianza=5;
    exog=[];
    dum=[];
    const=false;
    lr=[];
    monte_carlo=1000;
end
if nargin<6
    shock=[];
    confianza=5;
    exog=[];
    dum=[];
    const=false;
    lr=[];
    monte_carlo=1000;
    irf_lenght=40;
end
if nargin<5
    shock=[];
    confianza=5;
    exog=[];
    dum=[];
    const=false;
    lr=[];
    monte_carlo=1000;
    irf_lenght=40;
    for xx=1:size(data,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
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
    for xx=1:size(data,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
    for xx=1:size(data,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end

if isempty(confianza)==1
    confianza=5;
end
if isempty(monte_carlo)==1
    monte_carlo=1000;
end
if isempty(irf_lenght)==1
    irf_lenght=40;
end
if isempty(shock_names)==1
    for xx=1:size(data,2)
        eval(['shock_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
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

somma=zeros(K,K);
count=0;
for ii=1:nlags
    somma=Q.coefficients(1:K,1+count:K+count)+somma;
    count=count+K;
end
QQ=inv(eye(K)-somma)*varcovar*(inv(eye(K)-somma))';
vechQQ=vectorize(QQ,1);
% build the weighting matrix
temp=zeros(size(Q.resid,2),size(Q.resid,2));
temp=vectorize(temp,1);
% average vectorized squared residuals
for xx=1:size(Q.resid,1)
    temp=temp+vectorize(Q.resid(xx,:)'*Q.resid(xx,:),1);
end
uu1mean=1/size(Q.resid,1)*temp;

temp=zeros(size(uu1mean,1),size(uu1mean,1));
for xx=1:size(Q.resid,1)
    temp=temp+(vectorize(Q.resid(xx,:)'*Q.resid(xx,:),1)-uu1mean)*(vectorize(Q.resid(xx,:)'*Q.resid(xx,:),1)-uu1mean)';
end
SIGM=1/size(Q.resid,1)*temp;
% objective function
inizio=chol(QQ,'upper');
J=@(B) size(Q.resid,1)*(vechQQ-vectorize(B*B',1)).'*inv(SIGM)*(vechQQ-vectorize(B*B',1));
for xx=1:size(idmat,1)
    for yy=1:size(idmat,2)
        if idmat(xx,yy)==1
            LB(xx,yy)=-inf;
            UB(xx,yy)=inf;
        elseif idmat(xx,yy)==0
            LB(xx,yy)=0;
            UB(xx,yy)=0;
        end
    end
end            
options = optimset('Display','iter','MaxIter',15000,'MaxFunEvals',20000,'TolFun',1e-8);
[B,fval]=fminsearchcon(J,inizio,LB,UB,[],[],[],options);
P=(eye(K)-somma)*B;
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

u=randn(K,size(data,1)+100,monte_carlo);
for xx=1:monte_carlo
    u(:,:,xx)=chol(varcovar)'*u(:,:,xx);
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
        y(:,yy,xx)=J_mat*Y(:,yy,xx);
    end
end
for xx=1:monte_carlo
    temp1(:,:,xx)=y(:,:,xx)';
end
y=temp1;
clear temp1

for jj=1:monte_carlo
    Q_temp=reducedformVAR(y(:,:,jj),nlags,const,lr,[],dum,exog);
    varcovar_temp=Q_temp.sigma;
    somma_temp=zeros(K,K);
    count=0;
    for ii=1:nlags
        somma_temp=Q_temp.coefficients(1:K,1+count:K+count)+somma_temp;
        count=count+K;
    end
    QQ_temp=inv(eye(K)-somma_temp)*varcovar_temp*(inv(eye(K)-somma_temp))';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    vechQQ_temp=vectorize(QQ_temp,1);
    % build the weighting matrix
    temp=zeros(size(Q_temp.resid,2),size(Q_temp.resid,2));
    temp=vectorize(temp,1);
    % average vectorized squared residuals
    for xx=1:size(Q_temp.resid,1)
        temp=temp+vectorize(Q_temp.resid(xx,:)'*Q_temp.resid(xx,:),1);
    end
    uu1mean_temp=1/size(Q_temp.resid,1)*temp;

    temp=zeros(size(uu1mean_temp,1),size(uu1mean_temp,1));
    for xx=1:size(Q_temp.resid,1)
        temp=temp+(vectorize(Q_temp.resid(xx,:)'*Q_temp.resid(xx,:),1)-uu1mean_temp)*(vectorize(Q_temp.resid(xx,:)'*Q_temp.resid(xx,:),1)-uu1mean_temp)';
    end
    SIGM_temp=1/size(Q_temp.resid,1)*temp;
    % objective function
    inizio_temp=chol(QQ_temp,'upper');
    J_temp=@(B_temp) size(Q_temp.resid,1)*(vechQQ_temp-vectorize(B_temp*B_temp',1)).'*inv(SIGM_temp)*(vechQQ_temp-vectorize(B_temp*B_temp',1));
    for xx=1:size(idmat,1)
        for yy=1:size(idmat,2)
            if idmat(xx,yy)==1
                LB_temp(xx,yy)=-inf;
                UB_temp(xx,yy)=inf;
            elseif idmat(xx,yy)==0
                LB_temp(xx,yy)=0;
                UB_temp(xx,yy)=0;
            end
        end
    end            
    [B_temp,fval]=fminsearchcon(J_temp,inizio_temp,LB_temp,UB_temp,[],[],[],options);
    P_temp=(eye(K)-somma_temp)*B_temp;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
            irf_sim(:,yy,jj,xx)=J_mat*Irf_temp(:,yy,xx);
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
EQ.struc=(inv(P)*Q.resid');

end

%% ========================================================================
%% SUBFUNCTIONS
%% ========================================================================

%% Subfunction: vectorize
function y=vectorize(x,h,u)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose:
% Vectorize or half-vectorize symmetric matrix
% -----------------------------------
% Inputs:
% x = NxK matrix to be vectorized
% Optional:
% h = scalar that takes value 1 if half-vectorization and 0 if
% vectorization. Default is 0.
% u = scalar that takes value 1 if upper-half vectorization and 0
% otherwise. Default is 1.
% -----------------------------------
% Returns:
% y = (NNx1) vector in case of vectorization or half-vectorized vector
% ------------------------------------
if nargin<3
    u=1;
end
if nargin<2
    u=1;
    h=0;
end
N=size(x,1);
K=size(x,2);
if N~=K && h==1
    display('The matrix has to be symmetric for half-vectorization')
return
end
if h==0
    y=reshape(x,N*K,1);
elseif h==1 && u==1
    y=zeros(0,0);
for xx=1:N
        y=[y;x(1:xx,xx)];
end
elseif h==1 && u==0
    y=zeros(0,0);
for xx=1:N
        y=[y;x(1:end-xx+1,xx)];
end
end
end

%% Subfunction: fminsearchcon
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: John D'Errico (woodchips@rochester.rr.com) %%
%% Source: MATLAB File Exchange %%
%% https://www.mathworks.com/matlabcentral/fileexchange/8277 %%
%% Release: 1.0, Release date: 12/16/06 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extension of fminsearch with bound and general inequality constraints
% Used here for GMM estimation with zero restrictions on long-run multiplier

function [x,fval,exitflag,output]=fminsearchcon(fun,x0,LB,UB,A,b,nonlcon,options,varargin)
% FMINSEARCHCON: Extension of FMINSEARCHBND with general inequality constraints
% [Full documentation same as in gmmSRSVAR - omitted for brevity]

% size checks
xsize = size(x0);
x0 = x0(:);
n=length(x0);

if (nargin<3) || isempty(LB)
  LB = repmat(-inf,n,1);
else
  LB = LB(:);
end
if (nargin<4) || isempty(UB)
  UB = repmat(inf,n,1);
else
  UB = UB(:);
end
if (n~=length(LB)) || (n~=length(UB))
  error 'x0 is incompatible in size with either LB or UB.'
end

% defaults for A,b
if (nargin<5) || isempty(A)
  A = [];
end
if (nargin<6) || isempty(b)
  b = [];
end
nA = [];
nb = [];
if (isempty(A)&&~isempty(b)) || (isempty(b)&&~isempty(A)) 
  error 'Sizes of A and b are incompatible'
elseif ~isempty(A)
  nA = size(A);
  b = b(:);
  nb = size(b,1);
  if nA(1)~=nb  
    error 'Sizes of A and b are incompatible'
  end
  if nA(2)~=n
    error 'A is incompatible in size with x0'
  end
end

% defaults for nonlcon
if (nargin<7) || isempty(nonlcon)
  nonlcon = [];
end

% test for feasibility of the initial value
% against any general inequality constraints
if ~isempty(A)
  if any(A*x0>b)
    error 'Infeasible starting values (linear inequalities failed).'
  end
end
if ~isempty(nonlcon)
  if any(feval(nonlcon,(reshape(x0,xsize)),varargin{:})>0)
    error 'Infeasible starting values (nonlinear inequalities failed).'
  end
end

% set default options if necessary
if (nargin<8) || isempty(options)
  options = optimset('fminsearch');
end

% stuff into a struct to pass around
params.args = varargin;
params.LB = LB;
params.UB = UB;
params.fun = fun;
params.n = n;
params.xsize = xsize;
params.OutputFcn = [];
params.A = A;
params.b = b;
params.nonlcon = nonlcon;

% 0 --> unconstrained variable
% 1 --> lower bound only
% 2 --> upper bound only
% 3 --> dual finite bounds
% 4 --> fixed variable
params.BoundClass = zeros(n,1);
for i=1:n
  k = isfinite(LB(i)) + 2*isfinite(UB(i));
  params.BoundClass(i) = k;
  if (k==3) && (LB(i)==UB(i))
    params.BoundClass(i) = 4;
  end
end

% transform starting values into their unconstrained
% surrogates. Check for infeasible starting guesses.
x0u = x0;
k=1;
for i = 1:n
  switch params.BoundClass(i)
    case 1
      % lower bound only
      if x0(i)<=LB(i)
        % infeasible starting value. Use bound.
        x0u(k) = 0;
      else
        x0u(k) = sqrt(x0(i) - LB(i));
      end
      
      % increment k
      k=k+1;
    case 2
      % upper bound only
      if x0(i)>=UB(i)
        % infeasible starting value. use bound.
        x0u(k) = 0;
      else
        x0u(k) = sqrt(UB(i) - x0(i));
      end
      
      % increment k
      k=k+1;
    case 3
      % lower and upper bounds
      if x0(i)<=LB(i)
        % infeasible starting value
        x0u(k) = -pi/2;
      elseif x0(i)>=UB(i)
        % infeasible starting value
        x0u(k) = pi/2;
      else
        x0u(k) = 2*(x0(i) - LB(i))/(UB(i)-LB(i)) - 1;
        % shift by 2*pi to avoid problems at zero in fminsearch
        % otherwise, the initial simplex is vanishingly small
        x0u(k) = 2*pi+asin(max(-1,min(1,x0u(k))));
      end
      
      % increment k
      k=k+1;
    case 0
      % unconstrained variable. x0u(i) is set.
      x0u(k) = x0(i);
      
      % increment k
      k=k+1;
    case 4
      % fixed variable. drop it before fminsearch sees it.
      % k is not incremented for this variable.
  end
  
end
% if any of the unknowns were fixed, then we need to shorten
% x0u now.
if k<=n
  x0u(k:n) = [];
end

% were all the variables fixed?
if isempty(x0u)
  % All variables were fixed. quit immediately, setting the
  % appropriate parameters, then return.
  
  % undo the variable transformations into the original space
  x = xtransform(x0u,params);
  
  % final reshape
  x = reshape(x,xsize);
  
  % stuff fval with the final value
  fval = feval(params.fun,x,params.args{:});
  
  % fminsearchbnd was not called
  exitflag = 0;
  
  output.iterations = 0;
  output.funcCount = 1;
  output.algorithm = 'fminsearch';
  output.message = 'All variables were held fixed by the applied bounds';
  
  % return with no call at all to fminsearch
  return
end

% Check for an outputfcn. If there is any, then substitute my
% own wrapper function.
if ~isempty(options.OutputFcn)
  params.OutputFcn = options.OutputFcn;
  options.OutputFcn = @outfun_wrapper;
end

% now we can call fminsearch, but with our own
% intra-objective function.
[xu,fval,exitflag,output] = fminsearch(@intrafun,x0u,options,params);

% undo the variable transformations into the original space
x = xtransform(xu,params);

% final reshape
x = reshape(x,xsize);

% Use a nested function as the OutputFcn wrapper
  function stop = outfun_wrapper(x,varargin);
    % we need to transform x first
    xtrans = xtransform(x,params);
    
    % then call the user supplied OutputFcn
    stop = params.OutputFcn(xtrans,varargin{1:(end-1)});
    
  end

end % mainline end

% ======================================
% ========= begin subfunctions =========
% ======================================
function fval = intrafun(x,params)
% transform variables, test constraints, then call original function

% transform
xtrans = xtransform(x,params);

% test constraints before the function call
% First, do the linear inequality constraints, if any
if ~isempty(params.A)
  % Required: A*xtrans <= b
  if any(params.A*xtrans(:) > params.b)
    % linear inequality constraints failed. Just return inf.
    fval = inf;
    return
  end
end

% resize xtrans to be the correct size for the nonlcon
% and objective function calls
xtrans = reshape(xtrans,params.xsize);

% Next, do the nonlinear inequality constraints
if ~isempty(params.nonlcon)
  % Required: nonlcon(xtrans) <= 0
  cons = feval(params.nonlcon,xtrans,params.args{:});
  if any(cons(:) > 0)
    % nonlinear inequality constraints failed. Just return inf.
    fval = inf;
    return
  end
end

% we survived the general inequality constraints. Only now
% do we evaluate the objective function.
% append any additional parameters to the argument list
fval = feval(params.fun,xtrans,params.args{:});

end % sub function intrafun end

% ======================================
function xtrans = xtransform(x,params)
% converts unconstrained variables into their original domains

xtrans = zeros(1,params.n);
% k allows some variables to be fixed, thus dropped from the
% optimization.
k=1;
for i = 1:params.n
  switch params.BoundClass(i)
    case 1
      % lower bound only
      xtrans(i) = params.LB(i) + x(k).^2;
      
      k=k+1;
    case 2
      % upper bound only
      xtrans(i) = params.UB(i) - x(k).^2;
      
      k=k+1;
    case 3
      % lower and upper bounds
      xtrans(i) = (sin(x(k))+1)/2;
      xtrans(i) = xtrans(i)*(params.UB(i) - params.LB(i)) + params.LB(i);
      % just in case of any floating point problems
      xtrans(i) = max(params.LB(i),min(params.UB(i),xtrans(i)));
      
      k=k+1;
    case 4
      % fixed variable, bounds are equal, set it at either bound
      xtrans(i) = params.LB(i);
    case 0
      % unconstrained variable.
      xtrans(i) = x(k);
      
      k=k+1;
  end
end

end % sub function xtransform end