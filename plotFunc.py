import numpy as np
from matplotlib import pyplot as pl
from itertools import cycle
from matplotlib.lines import Line2D
import matplotlib
from scipy import interpolate
import scipy
from env import *

def visualizeStagingProb(stagingProbLong, tsStages, plotTrajParams, expNameCurrModel):

  nrGroups = len(stagingProbLong)
  labels = ['t%d' % i for i in range(20)]
  colors = ['r', 'g', 'b', 'k', 'm', 'c']

  linestyles = cycle(['-', '--'])

  nrColors = max([x.shape[0] for x in stagingProbLong])
  colors = iter(pl.cm.rainbow(np.linspace(0, 1, nrColors)))
  colors = np.random.permutation([next(colors) for x in range(nrColors)])

  for g in range(nrGroups):
    fig = pl.figure()
    nrTimepts = stagingProbLong[g].shape[0]
    for t in range(nrTimepts):
      pl.plot(stagingProbLong[g][t,:],next(linestyles),
              color=colors[t],label=labels[t], linewidth=4.0, markersize=3)

    legend = pl.legend(#bbox_to_anchor=plotTrajParams['legendPos'],
                       loc='upper right',
                       ncol=min(nrTimepts,5))

    fig.suptitle('%s - stagingProb' % expNameCurrModel, fontsize=20)

    mng = pl.get_current_fig_manager()
    mng.resize(*(1800, 400))
    pl.show()

def adjustCurrFig():
  fig = matplotlib.pyplot.gcf()
  fig.set_size_inches(180.5, 100.5)

  mng = pl.get_current_fig_manager()
  maxSize = mng.window.maxsize()
  maxSize = (maxSize[0]/2.1, maxSize[1]/1.1)
  print(maxSize)
  mng.resize(*maxSize)

  pl.tight_layout(pad=50, w_pad=25, h_pad=25)

def plotDiffData(yData, xData, labels):
  nrParticipants = len(yData)
  nrBiomk = yData.shape[1]
  fig = pl.figure()
  nrRows = 5
  nrCols = 6

  for r in range(nrRows):
    for c in range(nrCols):
      b = r*nrCols+c
      if b < nrBiomk:
        print("biomk %d %s" % (b, labels[b]))
        ax = pl.subplot(nrRows, nrCols,b+1)
        ax.set_title('%d - %s' %(b, labels[b]))
        pl.scatter(xData[:,b], yData[:,b])

  adjustCurrFig()
  pl.show()

  return fig


def plotDiffPredData(yData, xData, diagB, x_pred, dXdT_pred, sigma_pred,
                     posteriorSamples, labels, plotTrajParams):
  '''
  Plot the differential equation fit.

  Parameters
  ----------
  yData - rate of change, estimated as the slope of the linear fit
  xData - avg biomk value
  diag - vector of diagnoses
  x_pred - model fit, x-axis values (biomk values)
  dXdT_pred - model fit, y-axis values (rate of change)
  sigma_pred - standard deviation of Gaussian Process (GP)  fit.
  posteriorSamples - samples from the posterior distributio of GP
  labels
  plotTrajParams

  Returns
  -------

  '''
  nrParticipants = len(yData)
  nrBiomk = yData.shape[0]
  fig = pl.figure()
  nrRows = plotTrajParams['nrRows']
  nrCols = plotTrajParams['nrCols']

  diagNrs = np.unique(diagB[0])
  nrDiags = diagNrs.shape[0]
  # print('diag', diag)
  # print('nrDiags', nrDiags)
  # print(asdads)
  modelCol = plotTrajParams['modelCol']
  plotObjs = []

  nrSamples = posteriorSamples.shape[0]

  for r in range(nrRows):
    for c in range(nrCols):
      b = r*nrCols+c
      if b < nrBiomk:
        print("biomk %d %s" % (b, labels[b]))
        ax = pl.subplot(nrRows, nrCols,b+1)
        ax.set_title('%d - %s' %(b, labels[b]))

        for s in range(nrSamples):
          pl.plot(x_pred[:,b], posteriorSamples[s,:,b])

        for diagNr in range(nrDiags):
          plotObjs.append(pl.scatter(xData[b][diagB[b] == diagNrs[diagNr]],
                                     yData[b][diagB[b] == diagNrs[diagNr]],
                                     color=plotTrajParams['diagColors'][diagNrs[diagNr]]))

        pl.fill(np.concatenate([x_pred[:,b], x_pred[::-1,b]]),
          np.concatenate([dXdT_pred[:,b] - 1.9600 * sigma_pred[:,b],
                         (dXdT_pred[:,b] + 1.9600 * sigma_pred[:,b])[::-1]]),
          alpha=.5, fc=modelCol, ec='None', label='95% confidence interval')
        plotObjs.append(pl.plot(x_pred[:,b],
          dXdT_pred[:,b], 'k-', linewidth=2, label=u'Prediction'))

        yMin = np.nanmin(yData[b])
        yMax = np.nanmax(yData[b])

        xMax = np.nanmax(xData[b])
        xMin = np.nanmin(xData[b])

        deltaYLim = (yMax - yMin)/5
        deltaXLim = (xMax - xMin)/5

        pl.plot([xMin, xMax], [0, 0], 'k-', linewidth=2)

      
        if c == 0:
          pl.ylabel('$dx/dt\ (slope\ of\ biomk value)$')
        if r == nrRows-1:
          pl.xlabel('$x\ (biomarker\ value)$')



        print("deltaYLim", deltaYLim)
        ax.set_ylim([yMin - deltaYLim,yMax + deltaYLim])
        ax.set_xlim([xMin - deltaXLim,xMax + deltaXLim])

  adjustCurrFig()
  legendLabels = [plotTrajParams['diagStr'][diagNr] for diagNr in diagNrs] + ['GP fit']

  fig.suptitle('%s - differential equation fit' % plotTrajParams['expName'], fontsize=20)
  pl.figlegend( plotObjs, legendLabels, loc = 'lower center', ncol=4, labelspacing=0. )

  fig.show()
  pl.pause(0.1)
  

  return fig


def plotLongData(yData, xData, labels):
  nrParticipants = len(yData)
  nrBiomk = yData[0].shape[1]
  fig = pl.figure()
  nrRows = 5
  nrCols = 6

  for r in range(nrRows):
    for c in range(nrCols):
      b = r*nrCols+c
      if b < nrBiomk:
        print("biomk %d %s" % (b, labels[b]))
        ax = pl.subplot(nrRows, nrCols,b+1)
        ax.set_title('%d - %s' %(b, labels[b]))
        for p in range(nrParticipants):
          pl.plot(xData[p], yData[p][:,b], '-')

  adjustCurrFig()
  pl.show()

  return fig

def plotSubfigGroupComp(resList, diagNrs, plotTrajParams, labels, groupLabels
  , plotOneStdLine=True, plotZeroStdLine=True):
  '''
  Plots trajectories across diagnostic groups (PCA vs tAD or PCA subtypes)
  for different brain ROIs

  Parameters
  ----------
  resList - list of dictionaries representing results from DEM ran on PCA, tAD or other diag. subgroups
            the results contain trajectries and trajectory samples, among other things.
  plotTrajParams - various plotting parameters
  labels - labels for brain ROIs
  groupLabels - labels for the diagnostic groups used

  Returns
  -------
  fig - figure handle

  '''

  font = {'family': 'normal',
    # 'weight': 'bold',
    'size': 12}

  matplotlib.rc('font', **font)

  nrGroups = len(resList)


  #print "ts",ts
  #print "xs",xs

  (nrSamples, nrPoints, nrBiomk) = resList[0]['xsSamples'].shape
  figSizeInch = (plotTrajParams['SubfigGroupCompWinSize'][0] / 100,
  plotTrajParams['SubfigGroupCompWinSize'][1] / 100)
  fig = pl.figure(42, figsize=figSizeInch)
  pl.clf()
  ax = fig.add_axes(plotTrajParams['axisPos'])
  print(plotTrajParams['axisPos'])
  nrRows = plotTrajParams['nrRows']
  nrCols = plotTrajParams['nrCols']
  legendEntries = []
  sampleLabels = ['%s samples' % x for x in groupLabels]
  ph = [0 for x in range(nrGroups)]
  phSamples = [0 for x in range(nrGroups)]

  colors = [plotTrajParams['diagColors'][d] for d in diagNrs]

  for r in range(nrRows):
    for c in range(nrCols):
      b = r*nrCols+c
      if b < nrBiomk:
        print("biomk %d %s" % (b, labels[b]))
        ax = pl.subplot(nrRows, nrCols,b+1)
        ax.set_title('%s' %(labels[b]))

        for g in range(nrGroups):
          ph[g] = pl.plot(resList[g]['ts'][:,b], resList[g]['xsZ'][:,b],
            color=colors[g], label=groupLabels[g], linewidth=2.0, zorder=3)

        for s in range(nrSamples):
          for g in range(nrGroups):
            if not resList[g]['badSamples'][s,b]:
              phSamples[g] = pl.plot(resList[g]['tsSamples'][s,:,b],
                resList[g]['xsSamples'][s,:,b],'--', color=colors[g], label=sampleLabels[g],
                alpha=0.2, zorder=2)

        if b == 0:
          for g in range(nrGroups):
            legendEntries.append(ph[g])
            legendEntries.append(phSamples[g])

        minTs = np.min([np.min(resList[g]['ts'][:,b]) for g in range(nrGroups)])
        maxTs = np.max([np.max(resList[g]['ts'][:,b]) for g in range(nrGroups)])
        if plotZeroStdLine:
          zeroStdHandle = pl.plot([minTs,maxTs], [0, 0], '--',color='0.5', label='0 std', zorder=1)
          legendEntries.append(zeroStdHandle)
        #print "xs[1,b] < xs[0,b]", xs[1,b] < xs[0,b]
        #print "ts[1,b] < ts[0,b]", ts[1,b] < ts[0,b]
        #if (xs[1,b] < xs[0,b] == ts[1,b] < ts[0,b]):
        #  oneStdLine = 1
        #else:
        if plotOneStdLine:
          oneStdLine = -1
          oneStdHandle = pl.plot([minTs,maxTs], [oneStdLine, oneStdLine],
            '--',color='k', label='1 std', zorder=1)
          legendEntries.append(oneStdHandle)

        # if c == 0:
        #   pl.ylabel('Z-score of biomarker')
        # if r == nrRows-1:
        #   pl.xlabel('Years since disease onset')

        fig.text(0.06, 0.5, 'Z-score of biomarker', va='center', rotation='vertical',
          fontsize=20, family='normal', usetex=True)
        fig.text(0.5, 0.02, 'Years since $t_0$',
          ha='center', fontsize=20, family='normal', usetex=True )

        tMin, tMax = plotTrajParams['xLim']
        #print tMin, tMax
        # pl.ylim(np.min(pcaXsSamples), np.max(pcaXsSamples))
        ax.set_xticks(range(tMin, tMax+1, 10))
        ax.set_yticks(range(-20, 20, 2))

        pl.ylim(-8,3)
        pl.xlim(tMin, tMax)

        # print(tMin, tMax, np.linspace(tMin, tMax, 10))
        # print(asds)

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + 0.5*(1- plotTrajParams['axisHeightChopRatio'])*box.height, 
                         box.width, box.height * plotTrajParams['axisHeightChopRatio']])

  #adjustCurrFig()

  h, labels = ax.get_legend_handles_labels()
  print(h, labels)
  # legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])

  legend = pl.figlegend(h[0:4] + h[-2:], labels[0:4] + labels[-2:], loc='upper center',
    ncol=6, labelspacing=0. )
  # set the linewidth of each legend object
  for legobj in legend.legendHandles:
    legobj.set_linewidth(4.0)


  # fig.suptitle('%s - integrated trajectories' % ' vs '.join(groupLabels))

  # mng = pl.get_current_fig_manager()
  # maxSize = mng.window.maxsize()
  # maxSize = (maxSize[0]/1.5, maxSize[1]/1)
  # print(maxSize)
  # mng.resize(*plotTrajParams['trajPcaAdMaxWinSize'])

  fig.show()

  return fig


def plotTrajSubfigWithData(ts, xs, tsSamples, xsSamples, badSamples, labels, plotTrajParams,
                           data, diag, stagingMaxLik, thresh=None):
  """
  plot trajectories with data spread over them, aligned by the maxLik stage

  Parameters
  ----------
  ts - timepoints
  xs - biomk values
  tsSamples - timepoints of trajectory samples
  xsSamples - biomk values of traj samples
  badSamples - boolean mask saying if the trajectory sample is bad (crosses the zero line)
  labels - biomk labels
  plotTrajParams - plotting parameters
  data - data which will be plotted on the traj, already conv to z-scores
  stagingMaxLik - maximum likelihood stage for each subject visit
  thresh - (optional parameter) time axis threshold, used to separate between diagnoses

  Returns
  -------
  fig - figure handle

  """
  if xsSamples is not None:
    (nrSamples, nrPoints, nrBiomk) = xsSamples.shape
  else:
    nrBiomk = data.shape[1]
    nrSamples = 0

  fig = pl.figure()
  nrRows = plotTrajParams['nrRows']
  nrCols = plotTrajParams['nrCols']

  diagNrs = np.unique(diag)

  for r in range(nrRows):
    for c in range(nrCols):
      b = r * nrCols + c
      if b < nrBiomk:
        print("biomk %d %s" % (b, labels[b]))
        ax = pl.subplot(nrRows, nrCols, b + 1)
        ax.set_title('%d - %s' % (b, labels[b]))

        pl.plot(ts[:, b], xs[:, b], 'b', label='predicted trajectory')

        for s in range(nrSamples):
          if not badSamples[s, b]:
            pl.plot(tsSamples[s, :, b], xsSamples[s, :, b], color='0.5', label='predicted trajectory')

        for d in range(len(diagNrs)):
          pl.scatter(stagingMaxLik[diag == diagNrs[d]], data[diag == diagNrs[d], b], s=20,
                     c=plotTrajParams['diagColors'][diagNrs[d]], label=plotTrajParams['diagStr'][diagNrs[d]])

        #yMin = np.min([np.min(data[:,b]), xsSamples[:,:, b].min()])
        #yMax = np.max([np.max(data[:,b]), xsSamples[:,:, b].max()])
        yMin = np.min(data[:,b])
        yMax = np.max(data[:,b])
        yMin -= (yMax - yMin)/2
        yMax += (yMax - yMin)/2

        pl.plot([np.min(ts[:, b]), np.max(ts[:, b])], [0, 0], '--', color='0.5', label='0 std')
        if xs[1, b] < xs[0, b] == ts[1, b] < ts[0, b]:
          oneStdLine = 1
        else:
          oneStdLine = -1

        pl.plot([np.min(ts[:, b]), np.max(ts[:, b])], [oneStdLine, oneStdLine], '--', color='r', label='1 std')

        # plot diagnosis threshold
        if thresh is not None:
          pl.plot([thresh, thresh], [yMin, yMax], '--', color='g', label='diagnosis thresh')

        if c == 0:
          pl.ylabel('$Z-score\ of\ biomarker$')
        if r == nrRows - 1:
          pl.xlabel('$relative\ time\ (years)$')

        tMin, tMax = plotTrajParams['xLim']
        # print tMin, tMax
        pl.xlim(tMin, tMax)
        pl.ylim(yMin, yMax)

  adjustCurrFig()
  lgd = pl.legend(loc='lower center', ncol=4+len(diagNrs), bbox_to_anchor = (0.5, -0.5))
  fig.suptitle('%s - integrated trajectories' % plotTrajParams['expName'], fontsize=20)
  pl.show()

  return fig


def plotTrajSubfig(ts, xs, tsSamples, xsSamples, badSamples, labels, plotTrajParams,
                   xLim=None, yLim=None, xToAlign=None):
  '''
  Plots subfigures with the integrated trajectories across all ROIs, one ROI per subfigure

  Parameters
  ----------
  ts - points on the time axis of shape (nrPoints, nrBiomk)
  xs - corresponding biomarker values of shape (nrPoints, nrBiomk)
  tsSamples - bootstrap samples: (nrSamples, nrPoints, nrBiomk) array of points on the time axis
  xsSamples - bootstrap samples: (nrSamples, nrPoints, nrBiomk) array of points on the time axis
  badSamples - binary mask saying which trajectory samples could not be aligned or integrated properly.
               these will not be plotted
  labels - list of names of brain ROIS
  plotTrajParams - various plotting parameters
  xLim
  yLim
  xToAlign - x-values that have been used to align the traj, usually patient values at baseline.

  Returns
  -------
  fig - figure handle

  '''

  #print "ts",ts
  #print "xs",xs

  font = {'family': 'normal',
    # 'weight': 'bold',
    'size': 16}

  matplotlib.rc('font', **font)

  (nrSamples, nrPoints, nrBiomk) = xsSamples.shape
  figSizeInch = (plotTrajParams['TrajSubfigWinSize'][0] / 100,
  plotTrajParams['TrajSubfigWinSize'][1] / 100)
  fig = pl.figure(1, figsize = figSizeInch)
  pl.clf()
  nrRows = plotTrajParams['nrRows']
  nrCols = plotTrajParams['nrCols']
  tMin, tMax = plotTrajParams['xLim']

  for r in range(nrRows):
    for c in range(nrCols):
      b = r*nrCols+c
      if b < nrBiomk:
        print("biomk %d %s" % (b, labels[b]))
        ax = pl.subplot(nrRows, nrCols,b+1)
        ax.set_title('%d - %s' % (b, labels[b]))

        minYs = []
        maxYs = []
        for s in range(nrSamples):
          if not badSamples[s,b]:
            pl.plot(tsSamples[s,:,b], xsSamples[s,:,b], '--', color='b',alpha=0.2, label='bootstrap sample')

            mask = np.logical_and(tMin < tsSamples[s, :, b], tsSamples[s, :, b] < tMax)
            minYs += [np.min(xsSamples[s, mask, b])]
            maxYs += [np.max(xsSamples[s, mask, b])]


        # temporary hack to remove horiz asymptote: ignore the first and last pair of points
        # need to fix it properly in integrateTraj func
        pl.plot(ts[1:-1,b], xs[1:-1,b], 'k', linewidth=4, label='predicted trajectory')

        #pl.ylim(np.min(xs), np.max(xs))
        pl.plot([tMin,tMax], [0, 0], '--',color='0.5', label='0 std')
        # if xs[1,b] < xs[0,b] == ts[1,b] < ts[0,b]:
        #   oneStdLine = 1
        # else:
        #   oneStdLine = -1
        #
        # pl.plot([np.min(ts[:,b]),np.max(ts[:,b])], [oneStdLine, oneStdLine], '--',color='r', label='1 std')

        # if xToAlign is not None:
        #   print('xToAlign', xToAlign)
        #   pl.plot([np.min(ts[:, b]), np.max(ts[:, b])], [xToAlign[b], xToAlign[b]], '-', color='r', label='0 std')

        if c == 0 and r == 1:
          pl.ylabel('Z-score relative to controls')
        if r == nrRows-1:
          pl.xlabel('Years since $t_0$')


        print('np.min(minYs)', np.min(minYs))
        print('np.max(maxYs)', np.max(maxYs))
        #print tMin, tMax
        ax.set_xlim(tMin, tMax)
        ax.set_ylim(np.min(minYs), np.max(maxYs))


  # adjustCurrFig()
  pl.tight_layout(w_pad=0.3, h_pad=0.3)
  # pl.tight_layout(w_pad=1, h_pad=1)
  # mng = pl.get_current_fig_manager()
  # maxSize = mng.window.maxsize()
  # maxSize = (maxSize[0]/4.5, maxSize[1]/1.3)
  # mng.resize(*plotTrajParams['trajSubfigMaxWinSize'])



  # fig.suptitle('%s - integrated trajectories' % plotTrajParams['expName'], fontsize=20)
  fig.show()

  # import pdb
  # pdb.set_trace()

  return fig

def plotTrajAlign(ts, xs, labels, plotTrajParams, xLim=None, yLim=None):
  '''
  Plots trajectories of multiple ROIs that have been aligned on the temporal axis.
  Parameters
  ----------
  ts - points on the time axis
  xs - biomarker values
  labels - labels of brain ROIs
  plotTrajParams - various parameters for plotting such as colors, axis limits, etc ..
  xLim - limits on x-axis as tuple
  yLim - limits on y-axis as tuple

  Returns
  -------
  fig - figure handle
  legend

  '''

  font = {'family': 'normal',
    # 'weight': 'bold',
    'size': 16}

  matplotlib.rc('font', **font)

  nrBiomk = xs.shape[1]
  # figSizeInch = (plotTrajParams['TrajAlignWinSize'][0] / 100,
  # plotTrajParams['TrajAlignWinSize'][1] / 100)
  fig = pl.figure(2)
  pl.clf()
  ax = pl.subplot(111)
  colors=list(iter(pl.cm.rainbow(np.linspace(0,1,nrBiomk))))
  markerCycleObj = cycle(['-', ':'])
  linestyles = [next(markerCycleObj) for _ in range(nrBiomk)]
  # ax = fig.add_axes(plotTrajParams['axisPos'])
  # ax = fig.add_axes()
  # fig.tight_layout()

  markers = []
  for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

  markers = cycle(markers)

  legendEntries = []

  tMin, tMax = plotTrajParams['xLim']
  if xLim is not None:
    pl.xlim(xLim[0], xLim[1])
  else:
    xLim = (tMin, tMax)
    pl.xlim(xLim[0], xLim[1])

  fs = [interpolate.interp1d(ts[1:-1, b], xs[1:-1, b], kind='linear',
    fill_value='extrapolate') for b in range(nrBiomk)]
  fPredxMax = [fs[b](xLim[1]) for b in range(nrBiomk)]
  legendOrderInd = np.argsort(fPredxMax)[::-1]
  invPerm = np.argsort(legendOrderInd)

  for b in legendOrderInd:
    print("biomk %d %s" % (b, labels[b]))

    # temporary hack to remove horiz asymptote: ignore the first and last pair of points
    # need to fix it properly in integrateTraj func
    lh = ax.plot(ts[1:-1,b], xs[1:-1,b], c=colors[b],linestyle=linestyles[b],
      #marker=markers.next(), 
      label=labels[b],
      linewidth=4.0, 
      markersize=3)
    legendEntries.append(lh)

  mask1 = ts > tMin 
  mask2 = ts < tMax
  xsPlotVals = xs[mask1 * mask2]
  xMin = np.nanmin(xsPlotVals)
  xMax = np.nanmax(xsPlotVals)

  # yMin = np.nanmin(ys[mask1 * mask2])
  # yMax = np.nanmax(ys[mask1 * mask2])

  delta = (xMax - xMin)/6
  if yLim is not None:
    pl.ylim(yLim[0], yLim[1])
    # print(adsa)
  else:
    yLim = xMin, xMax
    pl.ylim(yLim[0], yLim[1]) # should be xMin, xMax as for x-axis we have ts.

  ax.plot([0, 0], yLim, '--', color='0.7')

  pl.xlabel(plotTrajParams['xLabel'])
  pl.ylabel('Z-score relative to controls')

  box = ax.get_position()
  # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  # legend = pl.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), ncol=1)
  print(box.x0, box.y0, box.width, box.height )
  ax.set_position([1.2*box.x0, 1.1*box.y0, box.width, box.height * 0.9])
  legend = pl.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.3), ncol=4)


  # legend = pl.legend(  # legendEntries, labels,
  #   bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])

  # set the linewidth of each legend object
  for legobj in legend.legendHandles:
    legobj.set_linewidth(4.0)
  
  mng = pl.get_current_fig_manager()
  maxSize = mng.window.maxsize()
  maxSize = (maxSize[0]/4.5, maxSize[1]/1.3)
  mng.resize(*plotTrajParams['trajAlignMaxWinSize'])

  # fig.suptitle('%s progression' % plotTrajParams['expName'], fontsize=20)
  # fig.suptitle('%s progression' % plotTrajParams['expNameFull'])
  # print(adaws)

  return fig, legend


def plotTrajAlignHist(ts, xs, labels, plotTrajParams, diag, maxLikStages, xLim=None, yLim=None):
  '''
  Plots trajectories of multiple ROIs on same axis, along with patient staging histogram (overlayed).
  Parameters
  ----------
  ts - points on the time axis
  xs - biomarker values
  labels - labels of brain ROIs
  plotTrajParams - various parameters for plotting such as colors, axis limits, etc ..
  xLim - limits on x-axis as tuple
  yLim - limits on y-axis as tuple

  Returns
  -------
  fig - figure handle
  legend

  '''

  font = {'family': 'normal',
    # 'weight': 'bold',
    'size': 16}

  matplotlib.rc('font', **font)

  nrBiomk = xs.shape[1]
  # figSizeInch = (plotTrajParams['TrajAlignWinSize'][0] / 100,
  # plotTrajParams['TrajAlignWinSize'][1] / 100)
  fig = pl.figure(2)
  pl.clf()
  ax = pl.subplot(111)
  colors = list(iter(pl.cm.rainbow(np.linspace(0, 1, nrBiomk))))
  markerCycleObj = cycle(['-', ':'])
  linestyles = [next(markerCycleObj) for _ in range(nrBiomk)]
  # ax = fig.add_axes(plotTrajParams['axisPos'])
  # ax = fig.add_axes()
  # fig.tight_layout()

  markers = []
  for m in Line2D.markers:
    try:
      if len(m) == 1 and m != ' ':
        markers.append(m)
    except TypeError:
      pass

  markers = cycle(markers)

  legendEntries = []

  tMin = np.min(maxLikStages)
  tMax = np.max(maxLikStages)

  if xLim is not None:
    pl.xlim(xLim[0], xLim[1])
  else:
    xLim = (tMin, tMax)
    pl.xlim(xLim[0], xLim[1])

  fs = [interpolate.interp1d(ts[1:-1, b], xs[1:-1, b], kind='linear',
    fill_value='extrapolate') for b in range(nrBiomk)]
  fPredxMax = [fs[b](xLim[1]) for b in range(nrBiomk)]
  legendOrderInd = np.argsort(fPredxMax)[::-1]
  invPerm = np.argsort(legendOrderInd)

  for b in legendOrderInd:
    print("biomk %d %s" % (b, labels[b]))

    # temporary hack to remove horiz asymptote: ignore the first and last pair of points
    # need to fix it properly in integrateTraj func
    lh = ax.plot(ts[1:-1, b], xs[1:-1, b], c=colors[b], linestyle=linestyles[b],
      # marker=markers.next(),
      label=labels[b],
      linewidth=4.0,
      markersize=3)
    legendEntries.append(lh)

  mask1 = ts > tMin
  mask2 = ts < tMax
  xsPlotVals = xs[mask1 * mask2]
  xMin = np.nanmin(xsPlotVals)
  xMax = np.nanmax(xsPlotVals)

  # yMin = np.nanmin(ys[mask1 * mask2])
  # yMax = np.nanmax(ys[mask1 * mask2])

  if yLim is None:
    yLim = xMin, xMax # should be xMin, xMax as for x-axis we have ts instead.

  pl.ylim(yLim[0], yLim[1])
  ax.plot([0, 0], yLim, '--', color='0.7')

  ##### plot staging histograms #######

  # first fit Kernel density estimator on staging histograms
  kernelWidth = np.std(maxLikStages) * plotTrajParams['kernelWidthFactor'] # need to test this parameter by visualisation

  diagNrs = np.unique(diag)
  from sklearn.neighbors.kde import KernelDensity
  kdes = []
  kdeXs = np.linspace(xLim[0], xLim[1], num=100).reshape(-1, 1)
  for d in diagNrs:
    kdeCurr = KernelDensity(kernel = 'gaussian', bandwidth = kernelWidth).fit(
      maxLikStages[diag == d].reshape(-1,1))
    print(np.exp(kdeCurr.score_samples(kdeXs)))
    # print(asdas)
    scores = np.exp(kdeCurr.score_samples(kdeXs))
    scaledScores = scores - np.min(scores)/(np.max(scores) - np.min(scores))
    scaledScores = scaledScores*(yLim[1] - yLim[0])*2 + yLim[0]

    pl.fill_between(kdeXs.reshape(-1), yLim[0], scaledScores,
      facecolor=plotTrajParams['diagColors'][d], alpha=0.4)

    maxScore = np.max(scaledScores)+0.25
    maxInd = np.argmax(scaledScores)
    pl.text(kdeXs[maxInd]-1.5, maxScore, plotTrajParams['diagLabels'][d], color=plotTrajParams['diagColors'][d])


  pl.xlabel(plotTrajParams['xLabel'])
  pl.ylabel('Z-score relative to controls')

  box = ax.get_position()
  # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  # legend = pl.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), ncol=1)
  print(box.x0, box.y0, box.width, box.height)
  ax.set_position([1.2 * box.x0, 1.1 * box.y0, box.width, box.height * 0.9])
  legend = pl.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4)

  # legend = pl.legend(  # legendEntries, labels,
  #   bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])

  # set the linewidth of each legend object
  for legobj in legend.legendHandles:
    legobj.set_linewidth(4.0)

  mng = pl.get_current_fig_manager()
  maxSize = mng.window.maxsize()
  maxSize = (maxSize[0] / 4.5, maxSize[1] / 1.3)
  mng.resize(*plotTrajParams['trajAlignMaxWinSize'])

  # fig.suptitle('%s progression' % plotTrajParams['expName'], fontsize=20)
  # fig.suptitle('%s progression' % plotTrajParams['expNameFull'])
  # print(adaws)

  return fig, legend


def plotTrajAlignHistCog(ts, xs, labels, plotTrajParams, diag, maxLikStages, xLim=None, yLim=None):
  '''
  Plots trajectories of multiple ROIs on same axis, along with patient staging histogram (overlayed).
  Parameters
  ----------
  ts - points on the time axis
  xs - biomarker values
  labels - labels of brain ROIs
  plotTrajParams - various parameters for plotting such as colors, axis limits, etc ..
  xLim - limits on x-axis as tuple
  yLim - limits on y-axis as tuple

  Returns
  -------
  fig - figure handle
  legend

  '''

  font = {'family': 'normal',
    # 'weight': 'bold',
    'size': 12}

  matplotlib.rc('font', **font)

  nrBiomk = xs.shape[1]
  # figSizeInch = (plotTrajParams['TrajAlignWinSize'][0] / 100,
  # plotTrajParams['TrajAlignWinSize'][1] / 100)
  fig = pl.figure(2)
  pl.clf()
  ax = pl.subplot(111)
  colors = list(iter(pl.cm.rainbow(np.linspace(0, 1, nrBiomk))))
  markerCycleObj = cycle(['-', ':'])
  linestyles = [next(markerCycleObj) for _ in range(nrBiomk)]
  # ax = fig.add_axes(plotTrajParams['axisPos'])
  # ax = fig.add_axes()
  # fig.tight_layout()

  markers = []
  for m in Line2D.markers:
    try:
      if len(m) == 1 and m != ' ':
        markers.append(m)
    except TypeError:
      pass

  markers = cycle(markers)

  legendEntries = []

  tMin = np.min(maxLikStages)
  tMax = np.max(maxLikStages)

  if xLim is not None:
    pl.xlim(xLim[0], xLim[1])
  else:
    xLim = (tMin, tMax)
    pl.xlim(xLim[0], xLim[1])

  fs = [interpolate.interp1d(ts[1:-1, b], xs[1:-1, b], kind='linear',
    fill_value='extrapolate') for b in range(nrBiomk)]
  fPredxMax = [fs[b](xLim[1]) for b in range(nrBiomk)]
  legendOrderInd = np.argsort(fPredxMax)[::-1]
  invPerm = np.argsort(legendOrderInd)

  for b in legendOrderInd:
    print("biomk %d %s" % (b, labels[b]))

    # temporary hack to remove horiz asymptote: ignore the first and last pair of points
    # need to fix it properly in integrateTraj func
    lh = ax.plot(ts[1:-1, b], xs[1:-1, b], c=colors[b], linestyle=linestyles[b],
      # marker=markers.next(),
      label=labels[b],
      linewidth=4.0,
      markersize=3)
    legendEntries.append(lh)

  mask1 = ts > tMin
  mask2 = ts < tMax
  xsPlotVals = xs[mask1 * mask2]
  xMin = np.nanmin(xsPlotVals)
  xMax = np.nanmax(xsPlotVals)

  # yMin = np.nanmin(ys[mask1 * mask2])
  # yMax = np.nanmax(ys[mask1 * mask2])

  if yLim is None:
    yLim = xMin, xMax # should be xMin, xMax as for x-axis we have ts instead.

  pl.ylim(yLim[0], yLim[1])
  ax.plot([0, 0], yLim, '--', color='0.7')

  ##### plot staging histograms #######

  # first fit Kernel density estimator on staging histograms
  kernelWidth = np.std(maxLikStages) * plotTrajParams['kernelWidthFactor'] # need to test this parameter by visualisation

  diagNrs = np.unique(diag)
  from sklearn.neighbors.kde import KernelDensity
  kdes = []
  kdeXs = np.linspace(xLim[0], xLim[1], num=100).reshape(-1, 1)
  for d in diagNrs:
    kdeCurr = KernelDensity(kernel = 'gaussian', bandwidth = kernelWidth).fit(
      maxLikStages[diag == d].reshape(-1,1))
    print(np.exp(kdeCurr.score_samples(kdeXs)))
    # print(asdas)
    scores = np.exp(kdeCurr.score_samples(kdeXs))
    scaledScores = scores - np.min(scores)/(np.max(scores) - np.min(scores))
    scaledScores = scaledScores*(yLim[1] - yLim[0])*2 + yLim[0]

    pl.fill_between(kdeXs.reshape(-1), yLim[0], scaledScores,
      facecolor=plotTrajParams['diagColors'][d], alpha=0.4)

    maxScore = np.max(scaledScores)+0.25
    maxInd = np.argmax(scaledScores)
    from matplotlib.font_manager import FontProperties
    font0 = FontProperties()
    font0.set_weight('bold')
    pl.text(kdeXs[maxInd]-1.5, maxScore, plotTrajParams['diagLabels'][d], fontproperties=font0, color=plotTrajParams['diagColors'][d])


  pl.xlabel(plotTrajParams['xLabel'])
  pl.ylabel('Z-score relative to controls')

  box = ax.get_position()
  # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  # legend = pl.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), ncol=1)
  # print(box.x0, box.y0, box.width, box.height)
  ax.set_position([1.2 * box.x0, 1.0 * box.y0, box.width, box.height * 0.9])
  legend = pl.legend(loc='upper center', bbox_to_anchor=(0.45, 1.58), ncol=3)

  pl.subplots_adjust(top=0.67)

  # legend = pl.legend(  # legendEntries, labels,
  #   bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])

  # set the linewidth of each legend object
  for legobj in legend.legendHandles:
    legobj.set_linewidth(4.0)

  mng = pl.get_current_fig_manager()
  maxSize = mng.window.maxsize()
  maxSize = (maxSize[0] / 4.5, maxSize[1] / 1.3)
  mng.resize(*plotTrajParams['trajAlignMaxWinSize'])

  # fig.suptitle('%s progression' % plotTrajParams['expName'], fontsize=20)
  # fig.suptitle('%s progression' % plotTrajParams['expNameFull'])
  # print(adaws)

  return fig, legend

def plotStagingHist(maxLikStages, diag, plotTrajParams, expNameCurrModel):
  '''
  Plots staging histogram for all diagnostic groups in 'diag'

  Parameters
  ----------
  maxLikStages
  nrStages
  diag
  plotTrajParams
  expNameCurrModel - unique identifier for the current experiment

  Returns
  -------
  fig - figure handle
  legend - legend handle

  '''

  fig = pl.figure()
  diagNrs = np.unique(diag)
  print(plotTrajParams['diagLabels'])
  colors = [plotTrajParams['diagColors'][d] for d in diagNrs]
  legendEntries = [plotTrajParams['diagLabels'][d] for d in diagNrs]
  histObj = pl.hist([maxLikStages[diag == d] for d in diagNrs],
    bins=plotTrajParams['stagingHistNrBins'], color=colors, label=legendEntries)
  lgd = pl.legend(loc='upper right', ncol=plotTrajParams['legendCols'])
  fig.suptitle('%s staging' % expNameCurrModel, fontsize=20)

  axes = pl.gca()
  yLimCurr = axes.get_ylim()

  pl.ylim(yLimCurr[0], yLimCurr[1]+(yLimCurr[1]-yLimCurr[0])/5)

  fig.show()
  #print(adsa)
  return fig, lgd

def plotTrajAlignConfInt(ts, xs, tsSamples, xsSamples, badSamples, plotTrajParams, labels):
  '''
  Plots trajectories that have been alinged on the temporal axis, along with confidence interval.
  WARNING: One should use less than 5 biomk, otherwise plot becomes too crowded and unreadable.

  Parameters
  ----------
  ts - points on the time axis of shape (nrPoints, nrBiomk)
  xs - corresponding biomarker values of shape (nrPoints, nrBiomk)
  tsSamples - bootstrap samples: (nrSamples, nrPoints, nrBiomk) array of points on the time axis
  xsSamples - bootstrap samples: (nrSamples, nrPoints, nrBiomk) array of points on the time axis
  badSamples - binary mask saying which trajectory samples could not be aligned or integrated properly.
               these will not be plotted
  labels - list of names of brain ROIS
  plotTrajParams

  Returns
  -------

  fig - figure handle

  '''

  (nrSamples, nrPoints, nrBiomk) = xsSamples.shape
  fig = pl.figure()
  colors=iter(pl.cm.rainbow(np.linspace(0,1,nrBiomk)))
  linestyles = cycle(['-', '--', '-.', ':'])
  ax = fig.add_axes(plotTrajParams['axisPos'])

  markers = []
  legendEntries = []
  for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

  markers = cycle(markers)

  # calculate confidence intervals
  nrPointsToEval = xs.shape[0]

  tsNew = np.zeros((nrPointsToEval, nrBiomk))
  xsNew = np.zeros((nrPointsToEval, nrBiomk))
  xsNewSamples = np.zeros((nrSamples, nrPointsToEval))
  stdTraj = np.zeros((nrPointsToEval, nrBiomk))

  for b in range(nrBiomk):
    tsGoodSamples = tsSamples[np.logical_not(badSamples[:,b]),:,b]
    xsGoodSamples = xsSamples[np.logical_not(badSamples[:,b]),:,b]
    minTs = np.max(tsGoodSamples[:,-1])
    maxTs = np.min(tsGoodSamples[:,0])

    print(tsGoodSamples[:,0], tsGoodSamples[:,-1])
    print(minTs, maxTs)

    print(scipy.__version__)
    tsNew[:,b] = np.linspace(minTs, maxTs, nrPointsToEval)
    f = interpolate.interp1d(ts[:,b], xs[:,b], kind='linear', fill_value='extrapolate')
    #f = interpolate.InterpolatedUnivariateSpline(ts[:,b], xs[:,b], k=1)
    xsNew[:,b] = f(tsNew[:,b])

    #print xsNew[:,b]
    #print asdasd

    fSamples = []
    nrGoodSamples = tsGoodSamples.shape[0]
    for s in range(nrGoodSamples):
      fSample = interpolate.interp1d(tsGoodSamples[s,:], xsGoodSamples[s,:], kind='linear', fill_value='extrapolate')
      #fSample = interpolate.InterpolatedUnivariateSpline(tsGoodSamples[s,:], xsGoodSamples[s,:], k=1)
      xsNewSamples[s,:] = fSample(tsNew[:,b])

    stdTraj[:,b] = np.std(xsNewSamples, axis=0)

  for b in range(nrBiomk):
    print("biomk %d %s" % (b, labels[b]))

    currCol = next(colors)
    lh = ax.plot(ts[:,b], xs[:,b], c=currCol,linestyle=next(linestyles), 
      #marker=markers.next(), 
      label=labels[b], 
      linewidth=4.0, 
      markersize=3)
    legendEntries.append(lh)
          
    pl.fill(np.concatenate([ts[:,b], ts[::-1,b]]),
          np.concatenate([xs[:,b] - 1.9600 * stdTraj[:,b],
                         (xs[:,b] + 1.9600 * stdTraj[:,b])[::-1]]),
          alpha=.5, fc=currCol, ec='None')#, label='95% confidence interval')

 
    #for s in range(nrSamples):
    #  if not badSamples[s,b]:
    #    pl.plot(tsSamples[s,:,b], xsSamples[s,:,b], color=currCol, alpha=0.4, label='predicted trajectory')
 
  
  tMin, tMax = plotTrajParams['xLim']
  #print tMin, tMax
  pl.xlim(tMin, tMax)
  mask1 = ts > tMin 
  mask2 = ts < tMax
  #print mask1 * mask2, mask1.shape, mask2.shape, (mask1*mask2).shape
  xsPlotVals = xs[mask1 * mask2]
  xMin = np.min(xsPlotVals)
  xMax = np.max(xsPlotVals)
  ax.plot([0, 0], [xMin, xMax], '--', color='0.7')

  delta = (xMax - xMin)/6
  pl.ylim(xMin, xMax)
  pl.xlabel(plotTrajParams['xLabel'])
  pl.ylabel('Z-score relative to control values')

  legend = pl.legend(#legendEntries, labels,
    bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])

  # set the linewidth of each legend object
  #for legobj in legend.legendHandles:
  #  legobj.set_linewidth(4.0)
  
  mng = pl.get_current_fig_manager()
  maxSize = mng.window.maxsize()
  maxSize = (maxSize[0]/4.5, maxSize[1]/1.3)
  mng.resize(*plotTrajParams['trajAlignMaxWinSize'])

  fig.suptitle('%s progression' % plotTrajParams['expName'], fontsize=20)

  pl.show()
  

  return fig


