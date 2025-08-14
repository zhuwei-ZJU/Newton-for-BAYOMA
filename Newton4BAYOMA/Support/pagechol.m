function [L,ierr] = pagechol(A)
% Page-wise matrix cholesky decomposiiton
% Syntax:
% -----------------------------------------------------------------------------------
% [L,ierr] = pagechol(A)
% -----------------------------------------------------------------------------------
%
% Inputs:
% -----------------------------------------------------------------------------------
% A : allow arbitrary number of input matrices
% -----------------------------------------------------------------------------------
%
% Outputs:
% -----------------------------------------------------------------------------------
% d : Page-wise cholesky decomposiiton of A
% -----------------------------------------------------------------------------------

sze_A = size(A);
N = size(A,1);
L = zeros(sze_A);
ierr = 0;
for k = 1:N
    % exit if A is not positive definite
%     if (A(k,k,:) <= 0), ierr = k; 
%         fprintf('Error in Choleski decomposition - Matrix must be positive definite\n');
%         return
%     end
    % Compute main diagonal elt. and then scale the k-th column
    A(k,k,:) = sqrt(A(k,k,:));
    L(k,k,:) = A(k,k,:);
    A(k+1:N,k,:) = A(k+1:N,k,:)./A(k,k,:);
    L(k+1:N,k,:) = A(k+1:N,k,:);
    % Update lower triangle of the trailing (n-k) by (n-k) block
    for j = k+1:N
        A(j:N,j,:) = A(j:N,j,:) - A(j:N,k,:).*A(j,k,:);
        L(j:N,j,:) = A(j:N,j,:);
    end
end
return