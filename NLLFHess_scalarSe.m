%% Calculate the gradient and hessian matrix of the NLLF (negative log-liklihood function)
% An indirect way by Louis Identity
%========================================================================
% 220904-Firstly written by Wei at ZJUI
% 221111-Consistent with the page-wise operation
% 230523-More condensed form for Ak terms
%========================================================================
% Input
%========================================================================
% f: MPV of natural frequency [m,1]
% z: MPV of damping ratio [m,1]
% PHI: MPV of mode shape [n,m]
% S: MPV of modal force PSD [m,m]
% Se: MPV of prediction error scalar
% ff: Frequency, [nf,1]
% F: FFT of measured data, [nf,n]
%========================================================================
% Output
%========================================================================
% NLLFgrad: Gradient of the NLLF
% NLLFHess: Hessian matrix of the NLLF
%========================================================================
function [grad,Hess] = NLLFHess_scalarSe(f,z,PHI,S,Se,ff,F)
[n,m] = size(PHI);   % [number of dof, number of modes]
nf = length(ff);   % number of fft points
Rm = full(vec2diag(m));  Dh = dup_mat(m,'hh');   Ds = dup_mat(m,'sk');
[Mu,Sigma,Mu2,Nu2] = EYitak12(f,z,PHI,S,Se,ff,F);
[A,HQ] = calAHQ(f,z,PHI,S,Se,ff,F,Mu,Mu2);
M = [Nu2;Mu;conj(Mu);ones(1,1,nf)];
D = blkdiag(eye(2*m+m*n),[Rm.',Dh,1i*Ds],1);
%========================================================================
% Fisher's identity: dQ/dθ' = dL/dθ'
%========================================================================
G = pagemtimes(A,M);
grad = -real(sum(pagemtimes(D,'transpose',G,'none'),3).'); 
if nargout > 1
    [Mu_tilde2,Nu3,Nu_tilde3,Nu4] = EYitak34(Mu,Sigma,Mu2);
    %========================================================================
    % The expected product of 2 first derivatives
    %========================================================================
    MM = [Nu4,                    Nu_tilde3,          Nu3,                Nu2
        pagetranspose(Nu_tilde3), Mu_tilde2,          Mu2,                Mu
        pagetranspose(Nu3),       pagetranspose(Mu2), conj(Mu_tilde2),    conj(Mu)
        pagetranspose(Nu2),       pagetranspose(Mu),  pagectranspose(Mu), ones(1,1,nf)];
    E1 = pagemtimes2(A,MM,A,'transpose');
    %========================================================================
    % The expectation of the second derivatives
    %========================================================================
    E2 = HQ;
    %========================================================================
    % The product of the first derivative of Q function
    %========================================================================
    E3 = pagemtimes(G,'none',G,'transpose');
    %========================================================================
    % Louis identity
    %========================================================================
    H = sum(-E1 -E2 + E3,3);
    %========================================================
    % Consider the Hermitian constraints
    %========================================================
    Hess = real(D.'*H*D);
    Hess = (Hess+Hess.')/2;
end
end