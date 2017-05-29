function [M] = normalisem (M, C)
C = inv(sqrtm(C));
M = C * M * C;
