function EQ=givSVAR(data_inst,other_data,iv,nlags,nleads,nlagsclean,var_names,irf_lenght,bootstrap_num,const,lr,dum,exog,rel)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: Lorenzo Menna %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose: 
% Impulse responses and Variance decomposition of a non-invertible Structural Vector Autoregression
% with external instruments as in Forni, Gambetti and Ricco. Compute
% relative irf if the shock is also non-recoverable. Compute absolute irf
% if the shock is recoverable
% -----------------------------------
% Inputs:
% data_inst = Nx1 matrix indicator instrumented variable. 
% other_data = NxK2 matrix (N number of observations, K2 number of non-instrumented variables). 
% iv = Nx1 matrix of instrumental variables.
% nlags = desired number of lags (max. 12)
% nleads = desired number of leads in the regression of proxy vs VAR
% residuals (should capture the lags of the effect of non fundamental
% shocks
% nlagsclean = number of lags in the clening equation. Cannot be larger
% than nlags
% Optional:
% var_names = 1xK cell vector with names of the variables
% irf_lenght = desired lenght of the IRFs.
% bootstrap_num = number of bootstrap repetitions for the computation of
% the confidence intervals of the IRFs
% const = true if the VAR is estimated with a constant and = false if without.
% Default is without.
% lr = KxK matrix of zeros and ones, zeros for variables that do not have
% effects on other variables, one otherwise
% exog = NxE matrix, where E is the number of exogenous variables that
% enter without lags. These variables are not dynamic. E.g. dummies. They
% are assumed to be zero in the construction of the forecast.
% rel = 1 if relative irfs and 0 if absolute. Default is absolute
% -----------------------------------
% Returns:
% EQ = structure
% EQ.vari = K 3xirf_lenght matrices containing in the first row the
% 95% confidence line, in the second row the irf, and in the third row the
% 5% confidence line. i is the the variable, j is the shock.
% EQ.struct = K1xN matrix containing estimated structural shocks
% ------------------------------------
% The solution method follows Luthkepol

if nargin<14
    rel=0;
end
if nargin<13
    rel=0;
    exog=[];
end
if nargin<12
    rel=0;
    exog=[];
    dum=[];
end
if nargin<11
    rel=0;
    exog=[];
    dum=[];
    lr=[];
end
if nargin<10
    rel=0;
    exog=[];
    dum=[];
    const=false;
    lr=[];
end
if nargin<9
    rel=0;
    exog=[];
    dum=[];
    const=false;
    lr=[];
    bootstrap_num=1000;
end
if nargin<8
    rel=0;
    exog=[];
    dum=[];
    const=false;
    lr=[];
    bootstrap_num=1000;
    irf_lenght=40;
end
if nargin<7
    rel=0;
    exog=[];
    dum=[];
    const=false;
    lr=[];
    bootstrap_num=1000;
    irf_lenght=40;
    for xx=1:size(data_inst,2)+size(other_data,2)
        eval(['var_names{' int2str(xx) '}=''var' int2str(xx) ''';']);
    end
end

if isempty(rel)==1
    rel=0;
end
if isempty(bootstrap_num)==1
    bootstrap_num=1000;
end
if isempty(irf_lenght)==1
    irf_lenght=40;
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

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Beginning

K1=1;
K2=size(other_data,2);
data=[data_inst,other_data];
K=size(data,2);
T=size(data,1)-nlags;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clean instrument
if nlagsclean==0
    qq.u=iv(nlags+1:end);
else qq=distlagOLS([iv data],nlagsclean,const,[zeros(K+1,1) ones(K+1,nlagsclean)]);
    qq.u=qq.u(1+(nlags-nlagsclean):end);
end
iv=qq.u;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reduced Form VAR
Q=reducedformVAR(data,nlags,const,lr,[],dum,exog);
varcovar=Q.sigma;
resid=Q.resid;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Projection on proxy variable
for xx=1:K
    qq=distlagOLS([resid(:,xx) iv],nleads,0,[zeros(1,1+nleads);ones(1,1+nleads)]);
    X(xx,:)=qq.coefficients(2,:);
    if xx==1
        EQ.F=qq.F;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Impulse response function

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
Irf=zeros(K*nlags,irf_lenght);
W=zeros(K*nlags,irf_lenght);
W(1:K,1:nleads+1)=X;
for yy=1:irf_lenght
    Irf(:,yy+1)=A*Irf(:,yy)+W(:,yy);
end
for yy=1:irf_lenght
    irf(:,yy)=J*Irf(:,yy);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute absolute impulse response or relative irf and bounds of the absolute impulse response if rel=1

if rel==1
    % Relative irf
    relirf=irf./irf(1,2);
    % If relative irf, compute bounds
    % Upper bound
    t = 0:(nleads);
    max_weight=0;
    max_f=0;
    for f=1:T
        thetaf=2*pi*f/T;
        exp_vector = exp(1j * thetaf * t);
        Xfreq=X*exp_vector';
        weightfreq=real(Xfreq'*inv(varcovar)*Xfreq); 
        if weightfreq > max_weight
            max_weight=weightfreq;
            max_f=f;
        end
    end
    up_irf=irf./max_weight;
    % Lower bound
    low_irf=irf.*std(iv);
else somma=0;
    for xx=1:nleads+1
        somma=somma+X(:,xx)'*inv(varcovar)*X(:,xx);
    end
    weight=sqrt(somma);
    absirf=irf./weight;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute shock series: only when rel=0
% Project instrument on forward residuals
waldres=[Q.resid(1:end-nleads,:)];
for xx=1:nleads
    waldres=[waldres Q.resid(1+xx:end-nleads+xx,:)];
end
olsleads=OLSmontecarlo(iv(1:end-nleads),waldres,0,10,10);
if rel==0
    num_shock=olsleads.e'*[ waldres'];
    e=reshape(olsleads.e,K,nleads+1);
    somma=0;
    for xx=1:nleads+1
        somma=somma+e(:,xx)'*varcovar*e(:,xx);
    end
    weight=sqrt(somma);
    EQ.struct=num_shock./weight;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Inference bootstrap

% Compute confidence intervals using block bootstrap as in Lundsford
u=zeros(K,size(resid,1),bootstrap_num);
iv_sim=zeros(size(iv,1),K1,bootstrap_num);

l=(size(data,1)-nlags)^(1/4);
l=ceil(l);
NN=(size(data,1)-nlags)/l;
NN=ceil(NN);
for xx=1:size(data,1)-nlags-l+1
    UU(:,:,xx)=resid(xx:xx+l-1,:)';
    MM(:,:,xx)=iv(xx:xx+l-1,:)';
end

for xx=1:bootstrap_num
    rs=randi(size(data,1)-nlags-l+1,[NN,1]);
    u_1=zeros(K,l*NN);
    iv_sim_1=zeros(K1,l*NN);
    for yy=1:size(rs,1)
        u_1(:,(yy-1)*l+1:yy*l)=UU(:,:,rs(yy));
        iv_sim_1(:,(yy-1)*l+1:yy*l)=MM(:,:,rs(yy));
    end
    u_1=u_1(:,1:size(u_1,2)-(NN*l-(size(data,1)-nlags)));
    iv_sim_1=iv_sim_1(:,1:size(iv_sim_1,2)-(NN*l-(size(data,1)-nlags)));
    mean_u1=mean(u_1,2);
    mean_ivsim1=mean(iv_sim_1,2);
    for yy=1:size(u_1,2)
        u(:,yy,xx)=u_1(:,yy)-mean_u1;
        iv_sim(yy,:,xx)=(iv_sim_1(:,yy)-mean_ivsim1)';
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
        y(:,yy+nlags,xx)=J*Y(:,yy,xx);
    end
end
for xx=1:bootstrap_num
    temp(:,:,xx)=y(:,:,xx)';
end
y=temp;
clear temp


% Compute confidence intervals using bootstrap as in Forni et al
% intg=randi(T-nleads,[1,T-nleads,bootstrap_num]);
% u=zeros(K,T,bootstrap_num);
% nu=zeros(K1,T-nleads,bootstrap_num);
% Y=zeros(K*nlags,T+1,bootstrap_num);
% ivsim=zeros(K1,T,bootstrap_num);
% % Set initial values
% stucaz=data(1:nlags,:)';
% stucaz=flip(stucaz,2);
% for xx=1:bootstrap_num
%     Y(:,1,xx)=reshape(stucaz,K*nlags,1);
% end
% % % Set final values
% for xx=1:bootstrap_num
%     u(:,T-nleads+1:T,xx)=resid(T-nleads+1:T,:)';
% end
% for xx=1:bootstrap_num
%     ivsim(:,T-nleads+1:T,xx)=iv(T-nleads+1:T,:)';
% end
% % % set intermediate values
% for xx=1:bootstrap_num
%     for yy=1:T-nleads
%         u(:,yy,xx)=resid(intg(1,yy,xx),:)';
%         nu(1,yy,xx)=olsleads.resid(intg(1,yy,xx),1)';
%     end
% end
% U=zeros(K*nlags,T,bootstrap_num);
% U(1:K,:,:)=u;
% for xx=1:bootstrap_num
%     conta=0;
%     for yy=1:T
%         conta=conta+1;
%         if isempty(dum)==0
%                 vec_dummies=zeros(size(V,2)-1,1);
%                 if mod(conta,numdum)~=0
%                     prot=mod(conta,numdum);
%                 else prot=numdum;
%                 end
%             vec_dummies(prot,1)=1;
%             vec_dummies=[1;vec_dummies];
%             else vec_dummies=1;
%         end
%         if isempty(exog)==0
%             Y(:,yy+1,xx)=V*vec_dummies+Vexo*exog(yy,:)'+A*Y(:,yy,xx)+U(:,yy,xx);
%         else Y(:,yy+1,xx)=V*vec_dummies+A*Y(:,yy,xx)+U(:,yy,xx);
%         end
%     end
% end
% for xx=1:bootstrap_num
%     for yy=1:T-nleads
%         waldressim=[u(:,yy,xx)];
%         for hh=1:nleads
%             waldressim=[waldressim; u(:,yy+hh,xx)];
%         end
%         ivsim(1,yy,xx)=olsleads.e'*waldressim+nu(1,yy,xx);
%     end
% end
% % return to non-companion form
% y=zeros(K,T+nlags,bootstrap_num);
% for xx=1:bootstrap_num
%     y(:,1:nlags,xx)=data(1:nlags,:)';
% end
% clear stucaz
% Y=Y(:,2:size(Y,2),:);
% for xx=1:bootstrap_num
%     for yy=1:T
%         y(:,yy+nlags,xx)=J*Y(:,yy,xx);
%     end
% end
% for xx=1:bootstrap_num
%     temp1(:,:,xx)=y(:,:,xx)';
%     temp2(:,:,xx)=ivsim(:,:,xx)';
% end
% y=temp1;
% iv_sim=temp2;
% clear temp1 temp2
% estimate
for jj=1:bootstrap_num
    jj
    Q_temp=reducedformVAR(y(:,:,jj),nlags,const,lr,[],dum,exog);
    varcovar_temp=Q_temp.sigma;
    resid_temp=Q_temp.resid;
    for xx=1:K
        qq=distlagOLS([resid_temp(:,xx) iv_sim(:,:,jj)],nleads,0,[zeros(1,1+nleads);ones(1,1+nleads)]);
        X_temp(xx,:)=qq.coefficients(2,:);
    end
    A_temp=zeros(K*nlags,K*nlags);
    A_temp(1:K,:)=Q_temp.coefficients;
    if nlags>1
        A_temp(K+1:K*nlags,1:K*nlags-K)=eye(K*nlags-K);
    end
    Irf_temp=zeros(K*nlags,irf_lenght,K1);
    W=zeros(K*nlags,irf_lenght);
    W(1:K,1:nleads+1)=X_temp;
    for yy=1:irf_lenght
        Irf_temp(:,yy+1)=A_temp*Irf_temp(:,yy)+W(:,yy);
    end
    for yy=1:irf_lenght
        irf_temp(:,yy)=J*Irf_temp(:,yy);
    end
    if rel==1
        % Relative irf
        relirf_sim(:,:,jj)=irf_temp./irf_temp(1,2);
        % If relative irf, compute bounds
        % Upper bound
        t = 0:(nleads);
        max_weight=0;
        max_f=0;
        for f=1:T
            thetaf=2*pi*f/T;
            exp_vector = exp(1j * thetaf * t);
            Xfreq=X_temp*exp_vector';
            weightfreq=real(Xfreq'*inv(varcovar_temp)*Xfreq); 
            if weightfreq > max_weight
                max_weight=weightfreq;
                max_f=f;
            end
        end
        up_irf_sim(:,:,jj)=irf_temp./max_weight;
        % Lower bound
        low_irf_sim(:,:,jj)=irf_temp.*std(iv);
    else somma=0;
        for xx=1:nleads+1
            somma=somma+X_temp(:,xx)'*inv(varcovar_temp)*X_temp(:,xx);
        end
        weight=sqrt(somma);
        absirf_sim(:,:,jj)=irf_temp./weight;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prepare output

if rel==1
    EQ.relirf_sim=relirf_sim;
    EQ.up_irf_sim=up_irf_sim;
    EQ.low_irf_sim=low_irf_sim;
elseif rel==0 
    EQ.absirf_sim=absirf_sim;
end

perc_irf_up=zeros(K,irf_lenght);
perc_irf_down=zeros(K,irf_lenght);
if rel==1
    for yy=1:irf_lenght
        for hh=1:K
            temp=sort(relirf_sim(hh,yy,:));
            temp1=[temp(floor(bootstrap_num*95/100));temp(floor(bootstrap_num*5/100))];
            perc_relirf_up(hh,yy)=temp1(1);
            perc_relirf_down(hh,yy)=temp1(2);
            temp=sort(up_irf_sim(hh,yy,:));
            temp1=[temp(floor(bootstrap_num*95/100));temp(floor(bootstrap_num*5/100))];
            perc_upirf_up(hh,yy)=temp1(1);
            perc_upirf_down(hh,yy)=temp1(2);
            temp=sort(low_irf_sim(hh,yy,:));
            temp1=[temp(floor(bootstrap_num*95/100));temp(floor(bootstrap_num*5/100))];
            perc_lowirf_up(hh,yy)=temp1(1);
            perc_lowirf_down(hh,yy)=temp1(2);
        end
    end
elseif rel==0
    for yy=1:irf_lenght
        for hh=1:K
            temp=sort(absirf_sim(hh,yy,:));
            temp1=[temp(floor(bootstrap_num*95/100));temp(floor(bootstrap_num*5/100))];
            perc_absirf_up(hh,yy)=temp1(1);
            perc_absirf_down(hh,yy)=temp1(2);
        end
    end
end

if rel==1
    for yy=1:K
        eval(['EQ.rel_' var_names{yy} '=[perc_relirf_up(yy,:);relirf(yy,:);perc_relirf_down(yy,:)];']);
    end
    for yy=1:K
        eval(['EQ.upbound_' var_names{yy} '=[perc_upirf_up(yy,:);up_irf(yy,:);perc_upirf_down(yy,:)];']);
    end
    for yy=1:K
        eval(['EQ.lowbound_' var_names{yy} '=[perc_lowirf_up(yy,:);low_irf(yy,:);perc_lowirf_down(yy,:)];']);
    end
elseif rel==0
    for yy=1:K
        eval(['EQ.abs_' var_names{yy} '=[perc_absirf_up(yy,:);absirf(yy,:);perc_absirf_down(yy,:)];']);
    end
end

if rel==1
    count=0;
    figure(1);
    for yy=1:K
        count=count+1;
        subplot(K,1,count)
        eval(['plot(1:irf_lenght,EQ.rel_' var_names{yy} '(1,:),'':k'',1:irf_lenght,EQ.rel_' var_names{yy} '(2,:),''k'',1:irf_lenght,EQ.rel_' var_names{yy} '(3,:),'':k'');']);
        eval(['title(''REL' var_names{yy} ''');']);
    end
    count=0;
    figure(2);
    for yy=1:K
        count=count+1;
        subplot(K,1,count)
        eval(['plot(1:irf_lenght,EQ.upbound_' var_names{yy} '(1,:),'':k'',1:irf_lenght,EQ.upbound_' var_names{yy} '(2,:),''k'',1:irf_lenght,EQ.upbound_' var_names{yy} '(3,:),'':k'');']);
        eval(['title(''UPBOUND' var_names{yy} ''');']);
    end
    count=0;
    figure(3);
    for yy=1:K
        count=count+1;
        subplot(K,1,count)
        eval(['plot(1:irf_lenght,EQ.lowbound_' var_names{yy} '(1,:),'':k'',1:irf_lenght,EQ.lowbound_' var_names{yy} '(2,:),''k'',1:irf_lenght,EQ.lowbound_' var_names{yy} '(3,:),'':k'');']);
        eval(['title(''LOWBOUND' var_names{yy} ''');']);
    end
elseif rel==0
    count=0;
    figure(1);
    for yy=1:K
        count=count+1;
        subplot(K,1,count)
        eval(['plot(1:irf_lenght,EQ.abs_' var_names{yy} '(1,:),'':k'',1:irf_lenght,EQ.abs_' var_names{yy} '(2,:),''k'',1:irf_lenght,EQ.abs_' var_names{yy} '(3,:),'':k'');']);
        eval(['title(''ABS' var_names{yy} ''');']);
    end
end


end