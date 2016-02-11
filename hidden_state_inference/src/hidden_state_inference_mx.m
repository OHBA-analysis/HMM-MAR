function [gamma, Xi] = hidden_state_inference_mx(B, Pi_0, P, order)
%HIDDEN_STATE_INFERENCE using foward-backward propagation
% [GAMMA,XI,SCALE] = HIDDEN_STATE_INFERENCE_MX(B, PI_0, P, order)
%    returns the updated object SELF, together with probabilities 
%   of each state given all previous and future data points, XI, and the
%   scalings on alpha, SCALE. 
%
%
%
%   The function uses the marginal likelihood of the data given the priors 
%   for each state, B,  the initial probabilities, PI_0, the transition matrix P.
%   There is also an order term, relevant for MAR models. If you're not
%   using a MAR model, set order = 0. 
%
%   You MUST have four inputs and three outputs. If you don't want to make
%   / assign variables, use ~ as a dummy variable 
%   e.g. [g,x,~] = hidden_state_inference_mx(B,PI,P,ORDER)
%
% To use this file, you may need to recompile, changing the top of the
% makefile to reflect your matlab installation

%	Copyright 2015 OHBA
%	This program is free software: you can redistribute it and/or modify
%	it under the terms of the GNU General Public License as published by
%	the Free Software Foundation, either version 3 of the License, or
%	(at your option) any later version.
%	
%	This program is distributed in the hope that it will be useful,
%	but WITHOUT ANY WARRANTY; without even the implied warranty of
%	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%	GNU General Public License for more details.
%	
%	You should have received a copy of the GNU General Public License
%	along with this program.  If not, see <http://www.gnu.org/licenses/>.


%	$LastChangedBy: GilesColclough $
%	$Revision: 763 $
%	$LastChangedDate: 2015-10-21 11:52:19 +0100 (Wed, 21 Oct 2015) $
%	Contact: giles.colclough@magd.ox.ac.uk
%	Originally written on: MACI64 by Giles Colclough, 31-Mar-2015 20:10:16


error([mfilename ':NoMexFunction'], ...
      '%s: mex function must be compiled for this to work. \n', ...
      mfilename);
end