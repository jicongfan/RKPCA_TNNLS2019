function [X,E,K,obj]=RKPCA_PLMAdSS(X,ker,lambda,options) 
% This is the codes for the RKPCA algorithm solved via proximal linearized minimization with adaptive step size
% more details are in the paper:
% "Exactly Robust Kernel Principal Component Analysis" J Fan, TWS Chow.
% IEEE transactions on neural networks and learning systems. 2019.
% input----
%   X: dxn data matrix, d is the number of features, n is the number of
%       samples.
%   ker: ker.type , should be 'rbf; ker.par, parameter of 'rbf' kernel; 
%       if ker.par=0, it will be estimated based on ker.c if exists.
%   lambda: parameter for the ||E||_1, control the sparsity of E.
%   options:
%       p: the parameter of Schatten-p norm, the default value is 1;
%       r_svd: 0 for full-svd (default); 1 for random svd;
%       gamma: hyperparameter for estimating the parameter r of random-svd;
%       maxIter: maximum of iterations
%       e: tolerance
%       c: hyper-parameter for updating the stepsize, c=0.1 as default
[d,n]=size(X);
E=zeros(d,n);
lambda=lambda/(sum(abs(X(:))))*n;
disp(['lambda=' num2str(lambda)])
%
if strcmp(ker.type,'rbf') && ker.par==0
    XX=sum(X.*X,1);
    D=repmat(XX,n,1) + repmat(XX',1,n) - 2*X'*X;
    if isfield(ker,'c')
        ker.par=(ker.c*mean(real(D(:).^0.5)))^2;
    else
        ker.par=(1*mean(real(D(:).^0.5)))^2;
    end
end
disp(['kernel type: ' ker.type '  kernel parameter(s):' num2str(ker.par)]) 
%
if isfield(options,'p')
    p=options.p;
else
    p=1;
end
if isfield(options,'r_svd')
    r_svd=options.r_svd;
else
    r_svd=0;
end
if n>2000
    r_svd=1;
end
r=n;
if r_svd~=0
    if isfield(options,'gamma')
        gamma=options.gamma;
    else
        gamma=1e-5;
    end
    r=min(round(n/2),d*5);
    disp(['Perform randomized SVD with initial rank=' num2str(r)])
end
%    
if isfield(options,'maxIter')
    maxIter=options.maxIter;
else
    maxIter=500;
end
%
if isfield(options,'e')
    e=options.e;
else
    e=1e-4;
end
%
if isfield(options,'c')
    c=options.c;
else
    c=0.1;
end
%
normF_X=norm(X,'fro');
I=eye(n);
%
iter=0;
%
while iter<maxIter
    iter=iter+1;
    K=ker_x(X-E,ker);
    if r_svd==0
        Kp2=real(K^(p/2));
        gLgK=real(p/2*Kp2*(K+eye(n)*1e-5)^(-1));
    else
        [U,S,V]=rsvd(K,r);
        S=diag(S);
        Kp2=U*diag(S.^(p/2))*V';
        gLgK=real(p/2*(U*diag((S+1*gamma).^(p/2-1))*V'));
        r=length(find(S>gamma));
    end
%
    f=trace(Kp2);
    obj(iter)=f+lambda*sum(abs(E(:)));
    [gE,L]=gLgX_m(gLgK,K,X,E,ker);
    tau=c*L;
    temp=E-gE/tau;
    E_new=max(0,temp-lambda/tau)+min(0,temp+lambda/tau);
%     E_new=solve_l1l2(temp,lambda/tau);
    if iter>1
        if obj(iter)>obj(iter-1)
            c=min(5,c*1.2);
        end
    end
%%
    stopC=norm(E_new-E,'fro')/normF_X;
    %
    isstopC=stopC<e;
    if mod(iter,100)==0||isstopC||iter==1
        disp(['iteration=' num2str(iter) '/' num2str(maxIter)  '  obj=' num2str(obj(iter)) '  stopC=' num2str(stopC)...
            '  tau=' num2str(tau)  '  c=' num2str(c) '  r=' num2str(r)])
    end
    if isstopC && iter>50
        disp('converged')
        break;
    end
    E=E_new;  
end
X=X-E;
end
%%
function K=ker_x(X,ker)
[d,n]=size(X);
XX=sum(X.*X,1);
D=repmat(XX,n,1) + repmat(XX',1,n) - 2*X'*X;
K=exp(-D/2/ker.par); 
end
%%
function [g,L]=gLgX_m(gLgK,K,X,E,ker)
[d,n]=size(X);
T=gLgK.*K;
B=repmat(sum(T),d,1);
g=-2/ker.par*((X-E)*T-(X-E).*B);
L=normest(2/ker.par*(T-eye(n)*mean(B(:))));
end