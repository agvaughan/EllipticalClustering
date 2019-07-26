function [clusteriness, p, dists] = pairsClusterTest_elliptical(vectors, params)
% [pairsVal, p, dists] = pairsClusterTest(vectors [ params])
%
% Implement the PAIRS clustering test. PAIRS stands for Projection Angle
% Index of Response Similarity.
%
% ALTERATION FROM ORIGINAL CODE - now generates a null distribution that matches
% the dimensionality (ie., major eigenvectors) of the original sample.  That is,
% it now works over elliptical priors.
%
%
% Tests whether a dataset in n dimensions is likely to be spherically
% symmetric (i.e., randomly distributed on the surface of an n-1 sphere).
% There is no really good known way to do this that I could find. Here, we
% use nearest neighbor angles. That is, for each datapoint, we find the
% angle to its nearest neighbor. This yields as many angles as there were
% datapoints. We then generate spherically random data of the same size and
% dimensionality, and perform the same operations. This gives us a null
% distribution of nearest-neighbor angles. This data is very nearly normal.
% We can then simply use the random distribution to determine whether the
% distribution of angles is different for the real data than the random
% data. We compare the mean, to be maximally sensitive to small numbers of
% clustered vectors. The p-value returned is two-sided.
%
% To illustrate use, imagine we are trying to find whether the responses of
% neurons tend to fall into categories. We would first build a matrix, A,
% of size N x CT, N the number of neurons, C the number of conditions, and
% T the number of time points. We would then perform PCA down to nDims:
%
% coeff = princomp(A);
% coeff = coeff(:, 1:nDims);
%
% Finally, we would run PAIRS on this matrix:
%
% [pairsVal, p, dists] = pairsClusterTest(coeff);
%
%
% INPUTS:
%
% vectors -- the data. Should be nDatapoints x nDims
% params  -- an optional struct. It may have fields:
%   .nKNN           -- number of knn to work with.  Stats are derived from the average distance to these knn.
%   .distanceMetric -- distance metric to use for knn search.  Note that 'cosine' is returned as an angle
%   .nDims          -- if present, will truncate the data to params.nDims x nDatapoints
%   .nBootstraps    -- number of bootstraps to run when generating the null
%                   distribution. Default 10000.
%   .reseedRNG      -- whether to re-seed rng and set method to 'twister'.
%                   Default 1.
%   .minLength      -- minimum magnitude of vectors to include. Default 0, but
%                   will always remove zero-length vectors.
%   .showVectors    -- whether to show a 2-D projection of the normalized
%                   vectors (first 2 dims). Default 1.
%   .showHist       -- whether to show the histogram comparing the data and
%                   the Monte Carlo simulations. Default 1.
%   .histBins       -- number of bins to use in the histogram (if applicable).
%                   Default 40.
%   .dimSizes       -- If the original data is not ~spherical, PAIRS
%                   analysis is not valid. This is the case even when
%                   renormalized by variance, magnitude, z-scoring, PCA,
%                   etc.  Add dimSizes to scale the multinormal prior
%                   to the same sizes as the raw data.
%   .test_side           -- 'left','right','both' [default:both]
%   .prior          -- Build bootstrap samples from a ['gaussian'] or ['permutation'].
%                    Permutation simply permutes values in each dimension of the input vectors.

%
% OUTPUTS:
%
% pairsVal     -- 1 if perfectly clustered, 0 if perfectly random. Note
%                 that 'perfectly clustered' means only that each point has
%                 an identically angled neighbor, not that there is only a
%                 single cluster. May be negative if data is smoother than
%                 chance.
% p            -- p-value (two-sided).
% dists        -- a struct, contains the distributions of the nearest-neighbor angles.
%   .data      -- nDatapoints nearest-neighbor angles for the data.
%   .neighbors       -- nDatapoints vector of which point made the smallest
%                       angle with this point.
%   .bootstrap -- nBoostraps x nDatapoints, nearest-neighbor angles
%                       for each Monte Carlo run.
%
%   For .data and .bootstrap, the following are also defined as dists.X
%         dists.bootstrap_1st = dists.bootstrap(:,:,1);
%         dists.data_1st      = dists.data(:,1);
%
%         dists.bootstrap_mean = mean(dists.bootstrap,3);
%         dists.data_mean      = mean(dists.data,2);
%
%         dists.bootstrap_median = median(dists.bootstrap,3);
%         dists.data_median      = median(dists.data,2);
%
%         dists.bootstrap_nth = dists.bootstrap(:,:,end);
%         dists.data_nth      = dists.data(:,:,end);
%
%
%
% Copyright (c) Matt Kaufman 2013, 2014 - antimatt AT gmail DOT com
% Modified (c) Alex Vaughan 2014, 2015 - alex dot vaughan at gmail dot com
% Cold Spring Harbor Laboratory

% If used in published work, please cite:


% Wish to match Matlab's convention for inputs, but the transpose is easier
% and faster to compute on
vectors = vectors';

%% Defaults

% Return the distance to the 1st or last KNN, or the mean distance to all
% KNN

nKNN            = 1;
distanceMetric  = 'cosine';
nDims           = size(vectors, 1);
nBootstraps     = 10000;
reseedRNG       = 1;
minLength       = 0;
dimSizes        = ones(nDims,1);
do_normalize    = 1; % Normalize vectors?  Shouldn't be necessary.
prior           = 'gaussian'; % 'permutation' or 'gaussian'
test_side       = 'both'; % 'left','right' or 'both'
plot_as         = 'distance'; % similarity, distance, or angle.  Angle will fail if distanceMetric is not 'cosine'
title_suffix    = ''; % similarity, distance, or angle.  Angle will fail if distanceMetric is not 'cosine'

% Plotting
showVectors = 0;
showHist    = 0;
histBins    = 50;

% % assign args from params struct if present
if exist('params', 'var')
    warnBadAssigns(assignFromParamsStruct(params, who));
end

% Trim vectors if required
if nDims < size(vectors, 1)
    vectors = vectors(1:nDims, :);
elseif nDims > size(vectors, 1)
    warning('pairsClusterTest:tooFewDims', ...
        'Dimensionality of data is lower than requested for testing, using dimensionality of data');
end
dimSizes = dimSizes(:);

%% Normalize, screen vectors
nSamples = size(vectors,2);
% Get magnitudes
lengths = sqrt(sum(vectors .^ 2));

% Normalize
if do_normalize
    vectors = bsxfun(@rdivide, vectors, lengths);
end

% Remove vectors that had zero length or were shorter than minLength
vectors = vectors(:, ~isnan(vectors(1, :)) & lengths >= minLength);

%% Pre-allocate, initialize

nVecs = size(vectors, 2);
distsbootstrap = squeeze(NaN(nBootstraps,nVecs,nKNN));

if reseedRNG
    rng(now, 'twister');
end

%% Display vectors

if showVectors
    namedFigure('Vectors'); clf; hold on;
    xs = [zeros(1, nVecs); vectors(1, :); NaN(1, nVecs)];
    ys = [zeros(1, nVecs); vectors(2, :); NaN(1, nVecs)];
    plot(xs(:), ys(:), 'bo', 'LineWidth', 1);
    rectangle('Position', [-1 -1 2 2], 'Curvature', 1, 'EdgeColor', 'k', 'LineWidth', 1.5);
    axis square on    
end


%% Find real distribution of distances
% Note - for distanceMetric == cosine, 
% we get dist = 1-cos(theta), such that 1 is a large angle
% and 0 is a small angle. 
% The actual angle then is theta = acos(1-dist);

[minDistsDataI,minDistsData] = knnsearch(vectors',vectors','K',nKNN+1,'Distance',distanceMetric);
minDistsData = minDistsData(:,2:end);
minDistsDataI = minDistsDataI(:,2:end);


%% Generate null distribution of angles
for b = 1:nBootstraps
    
    switch prior
        case 'gaussian'
            % Choose spherically random vectors from a unit-scaled multinormal distribution
            rVecs = bsxfun(@times,randn(size(vectors)),sqrt(dimSizes));
            
            % Normalize total variance across dimensions
            rVecs = rVecs / sqrt(sum(var(rVecs')));
            
            % Normalize - this shouldn't matter because cosine/correlation distances, but we do it anyway.
            if do_normalize
                rVecs = bsxfun(@rdivide, rVecs, sqrt(sum(rVecs .^ 2)));
            end
            
        case 'permutation'
            rVecs = vectors;
            for ii = 1:nDims
                rVecs(ii,:) = rVecs(ii,randperm(nSamples));
            end
            
        case 'none'
            rVcs = vectors;
    end
    
    % Calculate nearest neighbor distances.
    [~,bootstrap_distances] = knnsearch(rVecs',rVecs','K',nKNN+1,'Distance',distanceMetric);
    bootstrap_distances = bootstrap_distances(:,2:end);
    
    % Get all dot products
    %dotProds = rVecs' * rVecs;
    %bootstrap_distances = pdist(rVecs',clusterinpagOptions.distanceMetric);
    % Disqualify diagonal (which will be all 1's)
    % dotProds = dotProds - 2 * speye(nVecs);
    % pdist: not necessary, as correlations are zero
    
    % Do product with nearest neighbor
    %maxProdsbootstrap(b, :) = max(dotProds);
    distsbootstrap(b,:, :) = bootstrap_distances;
end

%% Turn dot products into angles

figure; 
subplot(2,1,1)
hist(minDistsData)
subplot(2,1,2)
hist(acos(1-minDistsData))
set(gca,'Xtick',[pi/8,pi/6,pi/4,pi/3],'XTickLabel',{'pi/8','pi/6','pi/4','pi/3'})

%%
% Plot everything as similarity instead of distance?
switch plot_as
    case 'similarity'
        % numNeurons x KNN
        dists.data = 1 - minDistsData;
        % nBootstraps x numNeurons x KNN
        dists.bootstrap = 1 - distsbootstrap;
    case 'angle'
        assert(strcmp(distanceMetric,'cosine'))
        % Cosine angle conversion; minDistsData is 1-cos(theta),
        % so we have to do acos(1-minDistsData)
        disp('Converting cosine distance to angle')
        dists.data      = real(acos(1-minDistsData));
        dists.bootstrap = real(acos(1-distsbootstrap));
    case 'distance'
        % Return a distance measure.
        dists.data      = minDistsData;
        dists.bootstrap = distsbootstrap;
    otherwise
        error('plot_as must be either similarity, angle, or distance')
end

%Distance to the first/last/mean/median neighbor for each point
if nKNN > 1
    dists.bootstrap_1st = dists.bootstrap(:,:,1);
    dists.data_1st      = dists.data(:,1);
    
    dists.bootstrap_mean = mean(dists.bootstrap,3);
    dists.data_mean      = mean(dists.data,2);
    
    dists.bootstrap_median = median(dists.bootstrap,3);
    dists.data_median      = median(dists.data,2);
    
    dists.bootstrap_nth = dists.bootstrap(:,:,end);
    dists.data_nth      = dists.data(:,:,end);
    
    dists.bootstrap_mean_knn = squeeze(mean(mean(dists.bootstrap,1),2)); % vector of 1:knn mean responses.
else
    % For KNN=1, all these values are the same.
    [dists.bootstrap_1st,dists.bootstrap_mean,dists.bootstrap_median,dists.bootstrap_nth] = deal(dists.bootstrap);
    [dists.data_1st,dists.data_mean,dists.data_median,dists.data_nth] = deal(dists.data);
end
%IDs
dists.neighbors = minDistsDataI;


%% Compute clusteriness

bootMeans = mean(dists.bootstrap_mean, 2);
bootMean  = mean(bootMeans);
dataMean  = mean(dists.data_mean);

% A value in the range of [-1..1], where
% 0 is random clumpiness compared to a random vectors
% > 0 reflects limited distance between a point and its neearest neighbors (ie., clusteriness)
% < 0 reflects excess spacing betweeen points (ie., regular spacing)
clusteriness = (bootMean - dataMean) / bootMean;

%% Compute p-value using bootstrap

% Alternate ways of getting p-value:
% [~, p] = ttest2(dists.data, dists.bootstrap(:), 0.05, 'left'); % one-sided
% [~, p] = kstest2(dists.data, dists.bootstrap(:));

%fracLess = mean(bootMeans <= dataMean); %%% This is not reliable.
fracLess = mean(dists.bootstrap_mean(:) <= dataMean);

% switch test_side
%     case 'both'
%         % Compensation for two-tailed test
%         if fracLess < 0.5
%             p = 2 * fracLess;
%         else
%             p = 2 * (1 - fracLess);
%         end
%     case 'left'
%         p = fracLess;
%     case 'right'
%         p  = 1-fracLess;
% end
% 
%[~,p] = kstest2(dists.bootstrap_mean(:),dists.data_mean)

% This works fine, although is quite sensitive
p = ranksum(dists.bootstrap_mean(:),dists.data_mean,'tail',test_side);

 %p = kruskalwallis([dists.bootstrap_mean(:); dists.data_mean(:)],...
 %    [ones(size(dists.bootstrap_mean(:)));2*ones(size(dists.data_mean(:)))],...
 %    'off')
 
%p = kruskalwallis

%keyboard

% if showHist
%     if length(unique(dimSizes)) == 1
%         namedFigure(sprintf('ePAIRS :: Hists of Distance to Nearest Neighbor, %s %s (spherical)',distanceMetric,plot_as));
%     else
%         namedFigure(sprintf('ePAIRS :: Hists of Distance to Nearest Neighbor, %s %s (elliptical)',distanceMetric,plot_as));
%     end
%     clf; hold on
%     subplot(2,1,1);
%     hist(bootMeans,1000);
%     xlim([0 1]);
%     vline(bootMean,'r')
%         xlabel([upper(distanceMetric) ' ' upper(plot_as)])
% 
%     title({'Mean bootstrap',sprintf('p < %.3f (all)',mean(dists.bootstrap_mean(:) <= dataMean)),...
%         sprintf('p < %.3f (means)',fracLess),...
%         sprintf('Clusteriness %.3f',clusteriness)})
%     
%     
%     subplot(2,1,2);
%     hist(dists.data_mean,1000);
%     xlim([0 1]);
%     vline(dataMean,'g')
%     title('Mean actual')
%     xlabel([upper(distanceMetric) ' ' upper(plot_as)])
% 
%     
% end



%% Show histogram

if showHist
    
    if ~isempty(title_suffix)
        title_prefix = [' :: ' title_suffix];
    end
        
    
    if length(unique(dimSizes)) == 1
        namedFigure(sprintf('ePAIRS :: Histogram of Distance to %.0f Nearest Neighbor, %s %s (spherical)%s',nKNN,distanceMetric,plot_as,title_suffix));
    else
        namedFigure(sprintf('ePAIRS :: Histogram of Distance to %.0f Nearest Neighbor, %s %s (elliptical)%s',nKNN,distanceMetric,plot_as,title_suffix));
    end
    clf; hold on;
    
    % Histograms
    nBootBins = 100;
    if strcmp(distanceMetric,'cosine') && strcmp(plot_as,'angle')
        histMin = 0;
        histMax = pi/2;
    else
        histMin = max(0, 0.8 * min([dists.data_mean(:);dists.bootstrap_mean(:)]));
        histMax = 1;%1.2 * max([dists.data_mean(:);dists.bootstrap_mean(:)]);
    end
    bins     = linspace(histMin,histMax,histBins);
    bootBins = linspace(histMin,histMax, nBootBins);
    
    % Calcualate histograms
    nData = histc(dists.data_mean, bins);
    nBoot = histc(dists.bootstrap_mean(:), bootBins) / nBootstraps * nBootBins / histBins;
    
    % Histogram
    hb = bar(bins + diff(bins(1:2))/2, nData, 'BarWidth', 0.9, 'FaceColor', [0.5 0.9 0.5], 'LineStyle', 'none');
    set(get(hb, 'BaseLine'), 'LineStyle', 'none');
    
    switch plot_as
        case 'angle'
            set(gca,'Xtick',[pi/8,pi/6,pi/4,pi/3,pi/2],'XTickLabel',{'pi/8','pi/6','pi/4','pi/3','pi/2'})
    end
    
    % Plot smoothing spline over data
    if 1
        inds = find(nData > 0,1);
        data_fit = fit(bins',nData,'smoothingspline','SmoothingParam',0.9999);
        h = plot(data_fit);
        set(h,'LineWidth',3,'Color',[0.1 0.4 0.1]) % green
    end
    axis tight
    
    % Bootstrap
    plot(bootBins(1:end-1) + diff(bins(1:2))/2, nBoot(1:end-1), 'r-', 'LineWidth', 3);
    legend('Observed','bootstrap')
    
    h = vline(median(dists.data_mean),'g--');
    set(h,'Color',[0.1 0.4 0.1])
    vline(median(dists.bootstrap_mean(:)),'r--')
    
    theMax = max([nData(:); nBoot(:)]);
    theMax = ceil(theMax);
    
    if strcmp(distanceMetric,'cosine') && strcmp(plot_as,'angle')
        axParams.fontSize = 11;
        axParams.tickLocations = [0 pi/4 pi/2];
        axParams.tickLabelLocations = axParams.tickLocations;
        axParams.tickLabels = {'0', 'pi/4', 'pi/2'};
        axParams.axisLabel = ['NN ' plot_as];
        AxisMMC(0, pi/2, axParams);
        
        axParams.axisOrientation = 'v';
        axParams.tickLocations = [0 theMax];
        axParams.tickLabelLocations = axParams.tickLocations;
        axParams.tickLabels = {'0', num2str(theMax)};
        axParams.axisLabel = 'Count';
        axParams.axisOffset = -pi/40;
        AxisMMC(0, theMax, axParams);
    else
        axParams.fontSize = 11;
        axParams.tickLocations = [0 0.5 1];
        axParams.tickLabelLocations = axParams.tickLocations;
        axParams.axisLabel = ['NN ' plot_as];
        AxisMMC(0, 1, axParams);
        xlim([0 1])
        
        axParams.axisOrientation = 'v';
        axParams.tickLocations = [0 theMax];
        axParams.tickLabelLocations = axParams.tickLocations;
        axParams.tickLabels = {'0', num2str(theMax)};
        axParams.axisLabel = 'Count';
        % axParams.axisOffset = -pi/40;
        AxisMMC(0, theMax, axParams);
    end
    
    set(gca,'visible', 'on');
    if length(unique(dimSizes)) == 1
        h = title(sprintf('Average Distance to %.0f Nearest Neighbor(s), %s, p=%01.4f (spherical)%s',nKNN,distanceMetric,p,title_suffix));
    else
        h = title(sprintf('Average Distance to %.0f Nearest Neighbor(s), %s, p=%01.4f (elliptical)%s',nKNN,distanceMetric,p,title_suffix));
    end
    %set(h,'visible','on')
    axis tight
    
    xlabel([upper(distanceMetric) ' ' upper(plot_as)])    
    
    
end


%%

    
%%

function outParams = AxisMMC(start, fin, varargin)

% plots an axis / calibration
%
% usage: outParams = AxisMMC(start, fin, params)
%
% 'start' and 'fin' are the starting and ending values of the axis
% 'params' is an optional structure with one or more of the following fields
%         tickLocations    default =  [start fin]
%            tickLabels    default = {'start',  'fin'}
%    tickLabelLocations    default is the numerical values of the labels themselves
%             longTicks    default is ticks with labels are long
%           extraLength    default = 0.5500  long ticks are this proportion longer
%             axisLabel    default = '' (no label)
%            axisOffset    default = 0
%       axisOrientation    default = 'h' ('h' = horizontal, 'v' = vertical)
%                invert    default = 0
%            tickLength    default is 1/100 of total figure span
%       tickLabelOffset    default based on tickLength
%       axisLabelOffset    default based on tickLength
%         lineThickness    default = 1
%                 color    default = 'k'
%              fontSize    default = 8
%
% Note that you can specify all, some, or none of these, in any order
%
% 'outParams' returns the parameters used (usually some mix of supplied and default)
% This is convenient if you don't like what you see and wish to know what a good
% rough starting value is for a given field.


% ********* PARSE INPUTS *******************
Pfields = {}; ipf = 1; %these will collect the fieldnames
% Locations of tick marks
Pfields{ipf} = 'tickLocations'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'tickLocations')
    tickLocations = varargin{1}.tickLocations;
else
    tickLocations = [start, fin];
end

% Numerical labels for the ticks
Pfields{ipf} = 'tickLabels'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'tickLabels')
    tickLabels = varargin{1}.tickLabels;
else
    for i = 1:length(tickLocations)
        tickLabels{i} = sprintf('%g', tickLocations(i)); % defaults to values based on the tick locations
    end
end

% Locations of the numerical labels
Pfields{ipf} = 'tickLabelLocations'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'tickLabelLocations')
    tickLabelLocations = varargin{1}.tickLabelLocations;
else
    for i = 1:length(tickLabels)
        tickLabelLocations(i) = eval(tickLabels{i}); %defaults to the values specified by the labels themselves
    end
end
if length(tickLabelLocations) ~= length(tickLabels)
    disp('ERROR, tickLabelLocations not the same length as tickLabels');
    disp('USER overridden, defaults used');
    clear tickLabelLocations;
    for i = 1:length(tickLabels)
        tickLabelLocations(i) = eval(tickLabels{i}); %defaults to the values specified by the labels themselves
    end
end

% Any long ticks
Pfields{ipf} = 'longTicks'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'longTicks')
    longTicks = varargin{1}.longTicks;  % these are the locations (must be a subset of the above)
    if (min(ismember(longTicks,tickLocations))==0), disp('One or more long ticks doesnt exist'); end
else
    longTicks = tickLabelLocations;  % default is labels get long ticks
end

% Length of the long ticks
Pfields{ipf} = 'extraLength'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'extraLength')
    extraLength = varargin{1}.extraLength;  % long ticks are 'extraLength' times as long as standard ticks
else
    extraLength = 0.55;
end

% axis label (e.g. 'spikes/s')
Pfields{ipf} = 'axisLabel'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'axisLabel')
    axisLabel = varargin{1}.axisLabel;
else
    axisLabel = '';
end

% Axis offset (vertical for a horizontal axis, and vice versa)
Pfields{ipf} = 'axisOffset'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'axisOffset')
    axisOffset = varargin{1}.axisOffset;
else
    axisOffset = 0;
end

% choose horizontal or vertical axis
Pfields{ipf} = 'axisOrientation'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'axisOrientation')
    axisOrientation = varargin{1}.axisOrientation(1); % just keep the first letter ('h' for horizontal, 'v' for vertical)
else
    axisOrientation = 'h';  % horizontal is default
end
if axisOrientation == 'H', axisOrientation = 'h'; end  % accept upper or lowercase
if axisOrientation ~= 'h', axisOrientation = 'v'; end


% normal or inverted axis (inverted = top for horizontal, rhs for vertical)
Pfields{ipf} = 'invert'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'invert')
    invert = varargin{1}.invert; % just keep the first letter ('h' for horizontal, 'v' for vertical)
else
    invert = 0; % default is normal axis
end

% length of ticks
Pfields{ipf} = 'tickLength'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'tickLength')
    tickLength = varargin{1}.tickLength;
else
    axLim = axis;  % default values based on 'actual' axis size of figure
    if axisOrientation == 'h'
        tickLength = abs(axLim(4)-axLim(3))/100;
    else
        tickLength = abs(axLim(2)-axLim(1))/100;
    end
end
if invert == 1, tickLength = -tickLength; end  % make negative if axis is inverted

% offset of numerical tick labels from the ticks (vertical offset if using a horizontal axis)
Pfields{ipf} = 'tickLabelOffset'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'tickLabelOffset')
    tickLabelOffset = varargin{1}.tickLabelOffset;
else
    tickLabelOffset = tickLength/2;
end

% offset of axis label
Pfields{ipf} = 'axisLabelOffset'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'axisLabelOffset')
    axisLabelOffset = varargin{1}.axisLabelOffset;
else
    
    if axisOrientation == 'h', axisLabelOffset = tickLength*4;
    else axisLabelOffset = tickLength*4.5; end
end

% line thickness
Pfields{ipf} = 'lineThickness'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'lineThickness')
    lineThickness = varargin{1}.lineThickness;
else
    lineThickness = 1; % default thickness is 1
end

% color
Pfields{ipf} = 'color'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'color')
    color = varargin{1}.color;
else
    color = 'k'; % default color is black
end

% font size
Pfields{ipf} = 'fontSize'; ipf=ipf+1;
if ~isempty(varargin) && isfield(varargin{1}, 'fontSize')
    fontSize = varargin{1}.fontSize;
else
    fontSize = 8; % default fontsize is 8 points (numerical labels are 1 pt smaller)
end

% warn if there is an unrecognized field in the input parameter structure
if ~isempty(varargin)
    fnames = fieldnames(varargin{1});
    for i = 1:length(fnames)
        recognized = max(strcmp(fnames{i},Pfields));
        if recognized == 0, fprintf('fieldname %s not recognized\n',fnames{i}); end
    end
end
% ********** DONE PARSING INPUTS ***************


% DETERMINE APPOPRIATE ALIGNMENT FOR TEXT (based on axis orientation)
if axisOrientation == 'h';  % for horizontal axis
    LalignH = 'center';  % axis label alignment
    NalignH = 'center';  % numerical labels alignment
    if invert==0
        LalignV = 'top';
        NalignV = 'top';
    else
        LalignV = 'bottom';
        NalignV = 'bottom';
    end
else                        % for vertical axis
    LalignH = 'center';  % axis label alignment
    NalignV = 'middle';  % numerical labels alignment
    if invert==0
        LalignV = 'bottom';  % axis label alignment
        NalignH = 'right';
    else
        LalignV = 'top';
        NalignH = 'left';
    end
end


% PLOT AXIS LINE
% plot main line with any ending ticks as part of the same line
% (looks better in illustrator that way)
axisX = [start, fin];
axisY = axisOffset * [1, 1];
if ismember(start, tickLocations)
    tempLen = tickLength + tickLength*extraLength*ismember(start, longTicks);
    axisX = [start, axisX];
    axisY = [axisY(1)-tempLen,axisY];
end
if ismember(fin, tickLocations)
    tempLen = tickLength + tickLength*extraLength*ismember(fin, longTicks);
    axisX = [axisX, fin];
    axisY = [axisY, axisY(end)-tempLen];
end
if axisOrientation == 'h', h = plot(axisX, axisY); else h = plot(axisY, axisX); end
set(h,'color', color, 'lineWidth', lineThickness);

% PLOT TICKS
for i = 1:length(tickLocations)
    if ~ismember(tickLocations(i),[start, fin]) % these have already been plotted
        tempLen = tickLength + tickLength*extraLength*ismember(tickLocations(i), longTicks);
        tickX =  tickLocations(i)*[1 1];
        tickY = axisOffset + [0 -tempLen];
        if axisOrientation == 'h', h = plot(tickX, tickY); else h = plot(tickY, tickX); end
        set(h,'color', color, 'lineWidth', lineThickness);
    end
end

% PLOT NUMERICAL LABELS (presumably on the ticks)
tickLim = tickLength + tickLength*extraLength*~isempty(longTicks); % longest tick length
for i = 1:length(tickLabelLocations)
    x = tickLabelLocations(i);
    y = axisOffset - tickLim - tickLabelOffset;
    if axisOrientation == 'h', h = text(x, y, tickLabels{i}); else h = text(y, x, tickLabels{i}); end
    set(h,'HorizontalA', NalignH, 'VerticalA', NalignV, 'fontsize', fontSize-1, 'color', color);
end

% PLOT AXIS LABEL
x = (start+fin)/2;
y = axisOffset - tickLim - axisLabelOffset;
if axisOrientation == 'h', h = text(x, y, axisLabel); else h = text(y, x, axisLabel); end
set(h,'HorizontalA', LalignH, 'VerticalA', LalignV, 'fontsize', fontSize, 'color', color);
if axisOrientation == 'v', set(h,'rotation',90); end
% DONE PLOTTING


% make outParams structure (tells user what default choices were made)
for i = 1:length(Pfields)
    outParams.(Pfields{i}) = eval(Pfields{i});
end;


