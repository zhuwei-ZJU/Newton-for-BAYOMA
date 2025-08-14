%==========================================
% Input
%==========================================
% f: [m,1]
% z: [m,1]
% PHI: [n,m]
% S: [m,m]
% Se: scalar
% ff: Frequency, [nf,1]
% F: FFT of measured data, [nf,n]
% xx: vectoried modal parameters [ntheta,1]
%==========================================
% Output
%==========================================
% L: Negative log-likelihood function
% g: Gradient
% H: Hessian matrix
%==========================================
% Syntax
%==========================================
% L = calNLLF(ff,F,f,z,PHI,S,Se)
% L = calNLLF(ff,F,xx)
% g,H are optional output
%==========================================
function [L,g,H] = calNLLF(ff,F,varargin)
if nargin == 3
    xx = varargin{1};
    n = size(F,2);
    m = (-(n+2) + sqrt((n+2)^2+4*(length(xx)-1)))/2;
    [f,z,PHI,S,Se] = mpvec2mat(xx,n,m);
else
    [f,z,PHI,S,Se] = varargin{:};
end
F3(:,1,:) = F.';   % [n,1,nf]
n = size(PHI,1);   % n: number of dofs
beta(:,1,:) = f./ff.';   % [m,1,nf]
h = 1./(1 - beta.^2 - 1i*2*beta.*z);   % [m,1,nf]
H = h.*S.*pagectranspose(h);   % [m,m,nf]
H = (H+pagectranspose(H))/2;
P = pageinv(H) + (PHI.'*PHI)/Se;
P = (P+pagectranspose(P))/2;
L1 = sum(n*log(pi) + n*log(Se) + log(pagedet(H)) + log(pagedet(P)));
L2 = sum(pagemtimes(F3,'ctranspose',F3,'none')/Se,3);
F3PHI = pagemtimes(F3,'ctranspose',PHI,'none');
L3 = sum(pagemtimes(pagemrdivide(F3PHI,P),'none',F3PHI,'ctranspose'),3)/Se^2;
% L3 = sum(pagemtimes2(F3PHI,pageinv(P),F3PHI,'ctranspose'),3)/Se^2;
L = real(L1+L2-L3);

% iSe = diag(1./diag(Se));
% P = pageinv(H) + PHI.'*iSe*PHI;   % [m,m,nf]
% L = sum(n*log(pi) + n*log(Se) + log(pagedet(H)) + log(pagedet(P))) +...
%     sum(1/Se*pagemtimes(F3,'ctranspose',F3,'none') -...
%     1/Se^2*pagemtimes2(F3,'ctranspose',PHI,'none',pageinv(P),'none',PHI,'transpose',F3,'none'),3);
% L = real(L);
if nargout == 2   % gradient required
    g = NLLFHess_scalarSe(f,z,PHI,S,Se,ff,F);
if nargout == 3   % Hessian required
    [g,H] = NLLFHess_scalarSe(f,z,PHI,S,Se,ff,F);
end
end
end