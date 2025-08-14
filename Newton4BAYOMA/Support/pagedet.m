function d = pagedet(A)
% Page-wise matrix determinant
% Syntax:
% -----------------------------------------------------------------------------------
% d = pagedet(A)
% -----------------------------------------------------------------------------------
%
% Inputs:
% -----------------------------------------------------------------------------------
% A : allow arbitrary number of input matrices
% -----------------------------------------------------------------------------------
%
% Outputs:
% -----------------------------------------------------------------------------------
% d : Page-wise determinant of A
% -----------------------------------------------------------------------------------

IA = size(A);
Ndim1 = IA(1);
Ndim2 = IA(2);
if Ndim1~=Ndim2
    error('Input has to be square!')
else
    Ndim = Ndim1;
end
Nele = Ndim^2;
siz = IA(3:end);
Nmat = prod(siz);
d = zeros([1,1,siz]);
for imat = 1:Nmat
    ind = (Nele*(imat-1)+1):(Nele*imat);
    A_tep = zeros(Ndim,Ndim);
    A_tep(:) = A(ind);
    d(imat) = det(A_tep);
end
return