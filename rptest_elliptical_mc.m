% RPTEST Random projection test for spherical uniformity
%
%
%     This modification of Brian Lau's implementation has three changes
%       1) Removes the asymptotic / empirical comparisons of Cuesta-Albertos 2009, as I believe they're incorrect.
%       2) In their place, implements a bootstrap comparison based on the Kolmogorov-Smirnov test statistic.
%       3) Allowing for aspherical distributions, and directly testing these.
%
%     [pval,stat] = rptest(U,varargin)
%
%     INPUTS
%     U - [n x p] matrix, n samples with dimensionality p
%         the data should already be projected to the unit hypersphere
%
%     OPTIONAL
%     test -
%
%     OUTPUTS
%          pval - p-value for n_projections random projections (default n_projections=20)
%          stat - statistic, projections onto n_projections random p-vectors
%
%     METHOD
%         Compares two distributions
%             a) the distribution of angles found between data U projected onto n_projections random vectors
%             b) the distribution of angles found between a set of random vectors

%         Comparison is made by either
%             a) asymptotic assumption about the distribution derived from
%             b) empirical comparison with distribution of *nmc* projection angles
%             arising from a uniform distribution on the hypersphere, compared
%             using the two-sample Kolmogorov-Smirnov test (default).

%
%     REFERENCE
%     Cuesta-Albertos, JA et al (2009). On projection-based tests for
%       directional and compositional data. Stat Comput 19: 367-380
%     Cuesta-Albertos, JA et al (2007). A sharp form of the Cramer-Wold
%       theorem. J Theor Probab 20: 201-209
%
%     SEE ALSO
%     UniSphereTest, rp, rppdf, rpcdf

%     $ Copyright (C) 2014 Brian Lau http://www.subcortex.net/ $
%     The full license and most recent version of the code can be found at:
%     https://github.com/brian-lau/highdim
%
%     Modified to include elliptical prior and bootstrap test 
%     by Alex Vaughan 2015 - alex dot vaughan at gmail dot com
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.

%{

Given real data and a bunch of random vectors,
find the distribution of angles between data and those vectors.

EMPIRICAL :: For a random set of data, and another bunch of vectors
find the (prior) distributino of angles between fake data and those
vectors.

For each each vector you tested (note that they do not match..)
calculate the Kolmogorov-Smirnov test of distinct distributions. This gives
a p-value for each projection vector.

Combine the p-values in some useful way - here it's roughly min(p(:)*n_projections)
which doesn't seem to work.

CHANGES ::
- use the same projection vectors, since the distributions will be really
different.
- Project three samples onto the same set of vectors.
1) Data, 2) Random ellipse, 3) Another random ellipse.
- Compare the two random samples to generate a monte-carlo prior over KS
p-values, and test whether median(p_value) < 5% of the monte-carlo
p-values.  We could equivalently use the KS stat for this, I guess.

%}
function [final_p_value,stats,ks_stat_data,ks_stat_mc] = rptest_elliptical_mc(U,varargin)

import sphere.*

par = inputParser;
par.KeepUnmatched = false;
par.StructExpand  = true;
addRequired(par,    'U',                                @isnumeric);
addParameter(par,   'verbose',              1,          @isnumeric);
addParameter(par,   'do_plot',              1,          @isnumeric);
addParameter(par,   'elliptical',            0,        @isnumeric);
addParameter(par,   'n_bootstrap_dists',    20,         @isnumeric); % Complexity scales as sum(1:n_bootstrap_dists)
addParameter(par,   'n_projections',        100,        @isnumeric);
addParameter(par,   'prior',               'gaussian',  @ischar); % Complexity scales as sum(1:n_bootstrap_dists)
addParameter(par,   'title_suffix',        '',          @ischar); % 
parse(par,U,varargin{:});

verbose             = par.Results.verbose;
do_plot             = par.Results.do_plot;
elliptical          = par.Results.elliptical;
n_bootstrap_dists   = par.Results.n_bootstrap_dists;
n_projections       = par.Results.n_projections;
prior               = par.Results.prior;
title_suffix        = par.Results.title_suffix;

% Normalize to unit total variance.
%U = U / sqrt(sum(var(U)));
[n,p] = size(U);

if ~elliptical
    % Just use equal variance
    dim_sizes = ones(1,p)/p; % Sum to unit vector.
else
    % Normalize by variance of input vector
    %     dim_sizes = dim_sizes(:)';  % Pass input sizes, but ensure row vector.
    %     dim_sizes = dim_sizes/norm(dim_sizes);
    dim_sizes = var(U)/sum(norm(U));
end

U_orig = U;
U = sphere.spatialSign(U);

%% If dim_sizes isn't specified, just scale to unit vector.

try
    assert(norm(dim_sizes)-1 < 2*eps,'ERROR :: norm(dim_sizes) does not equal 1');
catch ME
    keyboard
end

%%
% Generate statistics for actual data (first element of Us)
% and for bootstrap distributions (second element of Us).
% For n_bootstrap_dists, we have 10 comparisons to real data,
% and 10+9+...+1 = 55 comparisons between fake data sets.  Across some set
% of n_projections vectors, this gives ~100

% (1) For n_projections     : Calculate projection angles of each data point onto a random projection vector.
% (2) For n_bootstrap_dists : Generate an (elliptical) random Gaussian as a bootstrap distribution, and repeat (1).
% (3) Calculate the KS statistic for the difference between the angle distributions in (data) vs. each (bootstrap).  Keep the median KS statistic as "median_ks_stat_data"
% (4) Calculate the KS statistic for the difference between the angle distributions in each (bootstrap) vs. each other (bootstrap). Keep the set of medians (ie., keep one value for each bootstrap sample) as "median_ks_stat_mc".
% (5) Compare the distribution of KS statistics from (3) and (4).  Larger KS statistics are larger deviations
%     so the probability that the data distribution are nonuniform is defined as
%     P(uniform) = mean( ks_median_data <= median_ks_stat_mc).

%%



Us = cell(n_bootstrap_dists+1,1);
Us{1} = U;
for i = 1:n_bootstrap_dists
    switch prior
        case 'gaussian'
            
            % Project Gaussian onto unit hypersphere
            Us{i+1} = randn(n,p);
            
            
            if elliptical
                Us{i+1} = zscore(Us{i+1});
                %Us{i+1} = bsxfun(@rdivide,Us{i+1}, sqrt(var(Us{i+1})) ); % Normalize to unit variance
                Us{i+1} = bsxfun(@times,  Us{i+1}, sqrt(dim_sizes)    ); %
            end
            
            %             figure; clf; hold on
            %             plot(dim_sizes,'g-o')
            %             plot(var(Us{i+1}),'r--x')
            %
            %             sum(dim_sizes)
            %             sum(var(Us{i+1}))

            Us{i+1} = spatialSign(Us{i+1});         
            
        case 'permutation'
            Us{i+1} = U_orig;
            for pp = 1:p
                Us{i+1}(:,pp) = U(randperm(n),pp);
            end
            Us{i+1} = spatialSign(Us{i+1});
    end
end


%%
stats = rp_elliptical(Us,n_projections,dim_sizes); % Vector of angles between data and n_projections random vectors.
% Calculate KS p-values for all sets of distributions / vectors.
% P-values for comparison of data to monte-carlo samples

% Generate K-S stat for projection of random 

% Calculate ks_stat_data of input data vs. (n_bootstrap_dists control datasets) * (n_projections projections)
ks_stat_data = [];
for i_U = 2:(n_bootstrap_dists+1)
    % For each projection
    for i_k = 1:n_projections
        [~,~,ks_stat_data(end+1)] = kstest2( stats{1}(:,i_k), stats{i_U}(:,i_k)); %#ok<AGROW>
    end
end

%%
%Calculate ks_stat_mc for monte-carlo comparisons of each control datasets to
% each other control dataset, across all random projections.  
% This works out to sum(1:n_bootstrap_dists) * n_projections comparisons.
%
% We take median p-value across projections (ie., the median p-value within dataset*datset comparison
% distribution and test whether median(ks_stat_data) is less than expected.

ks_stat_mc = nan(sum(1:n_bootstrap_dists-1),n_projections);
row = 0;
for i_U = 2:length(Us)
    for j_U = (i_U+1):length(Us) % Avoid self-comparison
        row = row+1;
        for i_k = 1:n_projections
            % This output is quantized to approximately 1/n because of the KS CDF.
            [~,~,ks_stat_mc(row,i_k)] = kstest2( stats{i_U}(:,i_k), stats{j_U}(:,i_k));
        end
    end
end

%%
% Calculate median ks_stat for real data projections,
% distribution of median ks_stats for control projections
% and final p-value from comparison of these distributions.
mean_ks_stat_data = mean(ks_stat_data,2);
mean_ks_stat_mc   = mean(ks_stat_mc,2);
final_p_value     = mean(mean_ks_stat_data < mean_ks_stat_mc); % Higher values are more rare.

% Optional outputs.

if verbose
    fprintf('Generation of p-values done in %.1f seconds\n',toc)
end

if do_plot

      
    if ~isempty(title_suffix)
        title_suffix = [' :: ' title_suffix];
    end
        
    if length(unique(dim_sizes)) == 1
        namedFigure(sprintf('RPTest : Median KS statistic (spherical)%s',title_suffix)); clf
    else
        namedFigure(sprintf('RPTest : Median KS statistic (elliptical)%s',title_suffix)); clf
    end
    

    %%
    subplot(2,1,1); cla; hold on
    data = unique([ks_stat_mc(:)]);%; ks_stat_data(:)]);
    bins = linspace(min(data),max(data),100);
    bins = [-inf;bins(:);inf];
    %histogram(ks_stat_mc(:),bins,  'Normalization','cdf','EdgeColor','k','FaceAlpha',0,'DisplayStyle','stairs','LineWidth',1.5);
    %histogram(ks_stat_data(:),bins,'Normalization','cdf','EdgeColor','y','FaceAlpha',0,'DisplayStyle','stairs','LineWidth',1.5);
    %xlim([0.01 0.07])
    subplot(2,1,1); cla; hold on
    cdf_fn = @(x,bins) cumsum(histcounts(x(:),bins)) / sum(histcounts(x(:),bins));
    stairs(bins(1:end-1),cdf_fn(ks_stat_mc,bins),'r','LineWidth',1.5)
    stairs(bins(1:end-1),cdf_fn(ks_stat_data,bins),'g','LineWidth',1.5)
    set(gca,'XScale','log')
    
    h(1) = vline(mean_ks_stat_data,'g');
    h(2) = vline(median(mean_ks_stat_mc),'r');
    set(h,'LineWidth',2)
    title(sprintf('p < %.3f',final_p_value));
    %xlim([0 max([median(median_ks_stat_mc) median_ks_stat_data]) * 1.5])
    %%
    subplot(2,1,2); cla; hold on
    %     data = unique([mean_ks_stat_mc(:)]);%; ks_stat_data(:)]);
    %     bins = linspace(min(data),max(data),40);
    %bins = [-inf;data(1:10:end);inf];
    histogram(mean_ks_stat_mc(:),30,'Normalization','probability','FaceColor','r','FaceAlpha',0.5);
    %histogram(mean_ks_stat_data(:),100,'Normalization','probability','FaceColor','g','FaceAlpha',0.5);
    h(1) = vline(mean_ks_stat_data,'g');
    h(2) = vline(median(mean_ks_stat_mc),'r');
    set(h,'LineWidth',2)
    title(sprintf('p < %.3f',final_p_value));
    %xlim([0 max([median(median_ks_stat_mc) median_ks_stat_data]) * 1.5])
    
end



end % END rptest_elliptical

function stats = rp_elliptical(Us,n_projections,dim_sizes)
% This calculates the RP test statistic for a set of random vectors (n_projections)
% drawn from a guassian distribution of a given shape (dim_sizes).

[n,p] = size(Us{1});

% Normalize dimensions
%projection_vectors =  bsxfun(@times,randn(n_projections,p),dim_sizes); % Incorrect
projection_vectors = randn(n_projections,p);
% if length(unique(dim_sizes)) > 1
%     projection_vectors = zscore(projection_vectors);
%         %projection_vectors = bsxfun(@rdivide,projection_vectors, sqrt(var(projection_vectors)) ); % Normalize to unit variance
%     projection_vectors = bsxfun(@times,  projection_vectors, sqrt(dim_sizes)    ); %
% end


u0 = sphere.spatialSign(projection_vectors);

stats = cell(size(Us));
for i_Us = 1:length(Us)
    U = Us{i_Us};
    [n,p] = size(U);
    % Generate n_projections Uniform random directions to project onto.
    stats{i_Us} = zeros(n,n_projections);
    % This seems like it should be equivalent to U*u0' - why isn't it?
    %for i = 1:n_projections
    %stats{i_Us}(:,i) = acos(U*u0(i,:)');
    %end
    stats{i_Us} = acos(U*u0');
end

end