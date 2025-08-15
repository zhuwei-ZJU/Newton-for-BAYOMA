%% A new BAYOMA algorithm based on EM and Newton's method to fast compute the MPV and PCM
% Single setup and scalar Se
%========================================================================
% 230304-Firstly written by Wei at ZJUI
% 230523-More condensed matrix form
% 230531-Better initial guess for phi
%========================================================================
% Input
%========================================================================
% in: Struct contains:
% # Mandatory fields: tdata, fs, f1f2, f0
% # Optional fields:
%   ## tol_cvg: tolerance of convergence, 1e-3 by default
%   ## maxEMIter: maximum iteration of EM, 100 by default
%   ## maxNewtonIter: maximum iterations of Newton's method, 10 by default
%   ## iband: band to calculate
%========================================================================
% Output
%========================================================================
% out: Struct contains:
% {f,z,phi,S,Se}: optimum of modal parameters
% cofv: struct contains the posterior c.o.v. of modal parameters
% XX: modal parameters under iteration {nb,1}cell: [nIter,ntheta]
% loglik: log-likelihood values {nb,1}cell: [nIter,1]
% grad: gradients of the NLLF {nb,1}cell: [ntheta,nIter]
% normgrad: Frobenius of grad {nb,1}cell: [nIter,1]
% CM: covariance matrix under Newton iteration {nb,nIterNewton}cell
% nEMIter: Number of EM iterations
% nNewtonIter: Number of Newton's method iterations
% time_EM: consuming time of EM iterations
% tim_Newton: consuming time of Newton's method iterations
% time_req: total required time
%========================================================================
function out = Newton4bayoma(in)
% Fast Bayesian FFT modal identification using robust Newton method
%--------------------------------------------------------------------------
% out = Newton4bayoma(in)
% Perform Bayesian modal identification in frequency domain.
%
% If you use this program, please cite:
%   1. Zhu, Wei; Li, Binbin; Billie F, Spencer; From EM to Newton: 
%       fast and reliable computation for Bayesian FFT modal identification.
%       Reliability Engineering and System Safety.
%   2. Li, Binbin; Au, Siu-Kui. An expectation-maximization algorithm for 
%       Bayesian operational modal analysis with multiple (possibly close)
%       modes. Mechanical Systems and Signal Processing, 2019,132:490-511.
%   3. Zhu, Wei and Li, Binbin, EM-Aided Fast Posterior Covariance Computation
%       in Bayesian FFT Method. Mechanical Systems and Signal Processing, 2024, 211:111211.
% IN contains the following mandatory fields:
%   tdata = (nt,n) ambient data; nt = no. of time points; n = no. of dofs
%   fs = scalar, sampling rate (Hz)
%   f1f2 = (nb,2), f1f2(:,1) and f1f2(:,2) gives the lower and upper bound 
%          of frequency bands used for modal identification
%   f0 = {nb,1}, cell array of initial guesses of natural frequencies
%
% The following optional fields can be supplied in IN:
%  tol_cvg: tolerance of convergence, 1e-3 by default
%  maxEMIter: maximum iteration of EM, 100 by default
%  maxNewtonIter: maximum iterations of Newton's method, 10 by default
%  iband: band to calculate
%  The measurementn error of data is assumed to have constant PSD in the
%  selected band.
%
% OUT is a structure with the following fields:
% 1. MPV of modal parameters:
%  f = (1,m) natural frequencies (Hz)
%  z = (1,m) damping ratios
%  phi = (n,m) mode shape, each column normalized to unity
%  Se = scalar, prediction error variance
%  S = (m,m) PSD matrix of modal force
%
% Note: If tdata is given in g and fs in Hz, then Se and S are two-sided
% with a unit of g^2/Hz
% 
% 2. Posterior uncertainty
%  coefv = cell containing blocks of c.o.v. for 
%    f(:),z(:),S(:),Se,phi(:,1),phi(:,2),...,phi(:,m),
%  Note that off-diagonal terms of c.o.v. of S is the c.o.v. of coherence

% copyright statement
disp('============================================');
disp('BAYOMA_Newton - BAYesian Operational Modal Analysis Using Newton method');
disp('-- Wei Zhu (weizhu321@hotmail.com) - Aug 15, 2025');
disp('-- Binbin Li (bbl@zju.edu.cn) - Aug 15, 2025');
disp('============================================');
%%
%===============================
% Defaults setting
%===============================
if ~isfield(in,'tol_cvg') || isempty(in.tol_cvg)
    in.tol_cvg = 1e-3;
end
if ~isfield(in,'minEMIter') || isempty(in.minEMIter)
    in.minEMIter = 10;
end
if ~isfield(in,'maxEMIter') || isempty(in.maxEMIter)
    in.maxEMIter = 100;
end
if ~isfield(in,'maxfzIter') || isempty(in.maxfzIter)
    in.maxfzIter = 20;
end
if ~isfield(in,'steplentol') || isempty(in.steplentol)
    in.steplentol = 1/2^20;
end
if ~isfield(in,'minNewtonIter') || isempty(in.minNewtonIter)
    in.minNewtonIter = 1;
end
if ~isfield(in,'maxNewtonIter') || isempty(in.maxNewtonIter)
    in.maxNewtonIter = 100;
end
%========================================================
% Check input (Refer to the 'commonproc' function)
%========================================================
% check mandatory fields from in
if ~isfield(in,'tdata')
    error('tdata is a required field!');
end
if ~isfield(in,'fs')
    error('fs is a required field!');
end
if ~isfield(in,'f0')
    error('f0 is a required field!');
end
if ~isfield(in,'f1f2')
    error('f1f2 is a required field!');
end
% check legitimacy
% fs
if length(in.fs(:))>1
    warning('Only fs(1) will be used.');
end
in.fs = in.fs(1);
% f1f2
if ~isnumeric(in.f1f2)
    error('f1f2 should be a numerical array!');
end
if size(in.f1f2,2)~=2
    error('f1f2 should have 2 columns!')
end
% f0
if ~iscell(in.f0)
    error('f0 should be a cell!');
end
nb = size(in.f1f2,1);
if ~isequal(size(in.f0),[nb 1])
    error('The 1st dim. of f0 and f1f2 should be equal!');
end
% iband
if isfield(in,'iband')
    if ~isempty(in.iband)
        in.iband = round(in.iband); % force to integer
        if any(in.iband(:)<1 | in.iband(:)>size(in.f1f2,1))
            tmpstr = 'Each entry in iband(:) must be an integer ';
            tmpstr = [tmpstr,'between 1 and 1st dim. of f1f2.'];
            error(tmpstr);
        end
        in.f0 = in.f0(in.iband);
        in.f1f2 = in.f1f2(in.iband,:);
        nb = length(in.iband);   % no. of freq. bands
    end
end
%========================================================
% Calculate MPV and PCM band-by-band
%========================================================
f1f2 = in.f1f2;   f0 = in.f0;
outib = cell(nb,1);
for ib = 1:nb
    in.f1f2 = f1f2(ib,:);
    in.f0 = f0{ib};
    outib{ib} = Newton4bayoma_oneband(in);
end
%========================================================
% Set output
%========================================================
out = outib{1};
Scov = outib{1}.coefv;
covfields = fieldnames(Scov);
oo = rmfield(outib{1},'coefv');
outfields = fieldnames(oo);
for ib = 2:nb
    for jj = 1:numel(covfields)
        jjfield = covfields{jj};
        out.coefv.(jjfield) = cat(2,out.coefv.(jjfield),outib{ib}.coefv.(jjfield));
    end
    for ii = 1:numel(outfields)
        iifield = outfields{ii};
        out.(iifield) = cat(2,out.(iifield),outib{ib}.(iifield));
    end
end
end