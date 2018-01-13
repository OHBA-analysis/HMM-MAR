function significant = spectsignificance(tests,alpha)
%
% From a estiamtion of p-values through the specttest function,
% It returns 0 or 1 for each power of coherence value, if the p-value is
% below the level of significance 'alpha'. 
%
% INPUTS: 
%
% tests                 The output of specttest, i.e. a cell 
%                       where each element corresponds to one subject
% alpha                 Significance level
%
% OUTPUT:
% 
% significant           Struct with 4 fields: lower, higher, lower_corr, higher_corr. 
%                       ('_corr' reflects to multiple comparisons correction)
%                       Each field has a field 'states', just like each
%                       element of tests, with fields state(k).psd and 
%                       state(k).coh containing 1 if the value is
%                       significant and 0 otherwise
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

if nargin < 2, alpha = 0.01; end

significant = struct();
significant.higher = struct();
significant.higher.state = struct();
significant.higher_corr = struct();
significant.higher_corr.state = struct();
significant.lower = struct();
significant.lower.state = struct();
significant.lower_corr = struct();
significant.lower_corr.state = struct();

K = length(tests.higher.state); 
for k = 1:K
    significant.higher.state(k).psd = tests.higher.state(k).psd < alpha;
    significant.higher_corr.state(k).psd = tests.higher_corr.state(k).psd < alpha;
    significant.lower.state(k).psd = tests.lower.state(k).psd < alpha;
    significant.lower_corr.state(k).psd = tests.lower_corr.state(k).psd < alpha;
    significant.higher.state(k).coh = tests.higher.state(k).coh < alpha;
    significant.higher_corr.state(k).coh = tests.higher_corr.state(k).coh < alpha;
    significant.lower.state(k).coh = tests.lower.state(k).coh < alpha;
    significant.lower_corr.state(k).coh = tests.lower_corr.state(k).coh < alpha;
end

end