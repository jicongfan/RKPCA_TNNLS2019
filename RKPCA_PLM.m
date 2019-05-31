function [X,E,K,obj]=RKPCA_PLM(X,ker,lambda,p,maxIter) 
% This is the codes for the RKPCA algorithm solved via proximal linearized minimization with adaptive step size
% more details are in the paper:
% "Exactly Robust Kernel Principal Component Analysis" J Fan, TWS Chow.
% IEEE transactions on neural networks and learning systems. 2019.
[d,n]=size(X);
%
if ker.par==0
    XX=sum(X.*X,1);
    D=repmat(XX,n,1) + repmat(XX',1,n) - 2*X'*X;
    ker.par=(1.5*mean(real(D(:).^0.5)))^2;
end 
disp(['RBF kernel \sigma^2=' num2str(ker.par)])  
%
E=zeros(d,n);
%
if ~exist('maxIter')
    maxIter=1000;
end
lambda=lambda/(d*mean(abs(X(:))));
disp(['lambda=' num2str(lambda)])
e=1e-4;
%
normF_X=norm(X,'fro');
%
iter=0;
%
I=eye(n);
c=0.1;
%
while iter<maxIter
    iter=iter+1;
    K=ker_x(X-E,ker);
    Kp2=real(K^(p/2));
    f=trace(Kp2);
    obj(iter)=f+lambda*sum(abs(E(:)));
    gLgK=real(p/2*Kp2*(K+eye(n)*1e-5)^(-1));
    [gE,L]=gLgX_m(gLgK,K,X,E,ker);
    tau=c*L;
    temp=E-gE/tau;
    E_new=max(0,temp-lambda/tau)+min(0,temp+lambda/tau);
    rz=length(find(E_new(:)~=0))/d/n;
    if iter>1
        if obj(iter)>obj(iter-1)
            c=c*1.2;
        end
    end
    %%
    stopC=norm(E_new-E,'fro')/normF_X;
    isstopC=stopC<e;
    if mod(iter,100)==0||isstopC||iter==1
        disp(['iteration=' num2str(iter) '/' num2str(maxIter)  '  obj=' num2str(obj(iter)) ' rzeros=' num2str(rz) ])
        disp(['stopC=' num2str(stopC)  '  tau=' num2str(tau) '......' 'c=' num2str(c)])
    end
    if isstopC
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
L=normest(2/ker.par*(T-mean(B(:))));
end