import os
import math
from scipy.stats import *
import pickle
from diffEqModel import *
from plotFunc import *
from aligners import *

import DisProgBuilder

class DEMBuilder(DisProgBuilder.DPMBuilder):

  def __init__(self, fittingFunc, aligner):
    self.aligner = aligner
    self.fittingFunc = fittingFunc

  def generate(self, dataIndices, expName, params):
    return DEM(dataIndices, self.fittingFunc, self.aligner, expName, params)

class DEM:
  def __init__(self, dataIndices, fittingFunc, aligner, expName, params):

    assert(params['data'].shape[0] == dataIndices.shape[0])
    assert(params['diag'].shape[0] == dataIndices.shape[0])
    assert (params['scanTimepts'].shape[0] == dataIndices.shape[0])
    assert (params['partCode'].shape[0] == dataIndices.shape[0])
    assert (params['ageAtScan'].shape[0] == dataIndices.shape[0])
    # print('--------------data indices all fine')

    self.data = params['data'][dataIndices, :]
    self.diag = params['diag'][dataIndices]
    self.scanTimepts = params['scanTimepts'][dataIndices]
    self.partCode = params['partCode'][dataIndices]
    self.ageAtScan = params['ageAtScan'][dataIndices]
    # print(dataIndices, self.data, params['data'])
    self.nrBiomk = self.data.shape[1]

    for b in range(self.nrBiomk):
      filterIndices = np.logical_not(np.in1d(self.diag, params['excludeID']))

      self.data = self.data[filterIndices, :]
      self.diag = self.diag[filterIndices]
      self.scanTimepts = self.scanTimepts[filterIndices]
      self.partCode = self.partCode[filterIndices]
      self.ageAtScan = self.ageAtScan[filterIndices]


    self.fittingFunc = fittingFunc
    self.aligner = aligner


    self.params = params
    self.expName = expName
    self.params['plotTrajParams']['expNameFull'] = expName
    self.outFolder = 'matfiles/%s' % expName

    trajAlignWinSize = self.params['plotTrajParams']['trajAlignMaxWinSize']

    self.longDataFigName = '%s/longData.png' % self.outFolder
    self.diffDataFigName = '%s/diffData.png' % self.outFolder
    self.diffDataPredFigName = '%s/diffDataPred.png' % self.outFolder
    self.diffDataPredAllFigName = '%s/diffDataPredAll.png' % self.outFolder
    self.trajSubplotsFigName = '%s/trajSubplots.png' % self.outFolder
    self.trajAlignCtlZ = '%s/trajAlignCtlZ.png' % self.outFolder
    self.trajAlign = '%s/trajAlign_%d_%d' % (self.outFolder, trajAlignWinSize[0], trajAlignWinSize[1])
    self.trajAlignConfInt = '%s/trajAlignConfInterval.png' % self.outFolder
    self.trajAlignData = '%s/trajAlignData.png' % self.outFolder
    self.stagingHistFigName = '%s/stagingHist.png' % self.outFolder

  def runStd(self, runPart):
    res = self.run(self.params['runPartStd'])
    # res = self.genPosteriorSamples(res) # update the res dict with the samples
    return res

  def run(self, runPart):

    gaussProcFitFile = '%s/gaussProc.npz' % self.outFolder
    alignerFile = '%s/alignerRes.npz' % self.outFolder
    os.system('mkdir -p %s' % self.outFolder)

    (longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan) \
      = createLongData(self.data,
                       self.diag,
                       self.scanTimepts,
                       self.partCode,
                       self.ageAtScan)

    plotTrajParams = self.params['plotTrajParams']
    # fig = plotLongData(longData, longAgeAtScan, params['labels'])
    # fig.savefig(self.longDataFigName, dpi=100)


    #print(self.data)
    #print(longData[0].shape)
    # fit line for each subject and find gradient
    (patDXdTdata, patAvgXdata, estimNoise, patDiagB) = calcDiffData(longData,
      longAgeAtScan, longDiag, longPartCode)

    # plot dx/dt over dx
    # fig = plotDiffData(patDXdTdata, patAvgXdata, self.params['labels'])
    # fig.savefig(self.diffDataFigName, dpi=100)


    # selBiomkInd = [1,2,3,4,5,6]
    # dXdTdata = dXdTdata[:,selBiomkInd]
    # avgXdata = avgXdata[:, selBiomkInd]

    # print(self.params['labels'])
    # print(dXdTdata[dXdTdata[:,2] > 250,2]) # remove patrCodes [8, 145] because they have abnormal increase in hippocampus.
    # print(avgXdata[dXdTdata[:,2] > 250,2])
    # print(list([longPartCode[x] for x in np.where(dXdTdata[:,2] > 250)[0]]))
    # print(adsa)

    #print(filterMask)
    #print(dXdTdata.shape, patDXdTdata.shape)
    if runPart[0] == 'R':
      (x_pred, dXdT_pred, sigma_pred, _, posteriorSamples) = self.fittingFunc(
        patDXdTdata, patAvgXdata, self.params)
      np.savez(gaussProcFitFile, x_pred=x_pred, dXdT_pred=dXdT_pred, sigma_pred=sigma_pred,
               posteriorSamples=posteriorSamples)
    # elif runPart[0] == 'L':
    else:
      npData = np.load(gaussProcFitFile)
      x_pred = npData['x_pred']
      dXdT_pred = npData['dXdT_pred']
      sigma_pred = npData['sigma_pred']
      posteriorSamples = npData['posteriorSamples']

    fig = plotDiffPredData(patDXdTdata, patAvgXdata, patDiagB, x_pred, dXdT_pred, sigma_pred,
                           posteriorSamples, self.params['labels'], plotTrajParams)
    fig.savefig(self.diffDataPredFigName, dpi=100)
    # pl.pause(1000)

    # take largest (non-zero dXdT) section and integrate trajectory
    (xPredNzSect, dXdTpredNzSect, tsNzSect, _, biomkFailList, success) = \
      integrateTrajAll(x_pred, dXdT_pred, patAvgXdata)

    # print('biomkFailList', biomkFailList)
    # print('xPredNzSect[:,9]', xPredNzSect[:,9])
    # print('dXdTpredNzSect[:,9]', dXdTpredNzSect[:,9])
    # print(adsa)

    if not success:
      raise AssertionError("Failed to integrate traj as the following biomkers could not be sectioned:"
                           , biomkFailList, [self.params['labels'][i] for i in biomkFailList],
                           "Try to remove the biomks as they probably doesn't have enough signal anyway")

    # compute mean/std of CTL data for doing z-scores
    ctlDiagInd = np.array(np.where(self.diag == CTL)[0])
    ctlData = self.data[ctlDiagInd, :]
    self.muCtlData = np.nanmean(ctlData, axis=0)
    self.sigmaCtlData = np.nanstd(ctlData, axis=0)
    self.estimNoiseZ = estimNoise * self.params['noiseZfactor'] / self.sigmaCtlData
    self.covMatNoiseZ = getCovMatFromNoise(self.estimNoiseZ)


    # print('self.diag', self.diag)
    # print('ctlData', ctlData)
    # print('estimNoise', estimNoise)
    # print(adsa)

    # compute z-scores
    xPredZ = (xPredNzSect - self.muCtlData[None, :]) / self.sigmaCtlData[None, :]
    self.xsNz = xPredNzSect
    self.xsZ = xPredZ

    if runPart[1] == 'R':
      (tsAlign, resAlign, xToAlign) = self.aligner.align(self, tsNzSect, xPredZ, longData, longDiag)

      tsSamples, xsSamples, badSamples, xToAlignSamples = \
        self.genPosteriorSamples(posteriorSamples, x_pred, longData, longDiag, patAvgXdata,
          xToAlign)

      self.ts = tsAlign
      (maxLikStages, _, _, _, _, _) = self.stageSubjectsData(self.data)
      alignerClassSep = AlignerMaxClassSep()
      (tsAlign, _, _) = alignerClassSep.align(tsAlign, xPredZ,
        maxLikStages, self.params['tsStages'], self.diag, self.params)
      maxLikStages = maxLikStages + (tsAlign[0,0] - self.ts[0,0])
      tsSamples = tsSamples + (tsAlign[0,0] - self.ts[0,0])


      savedData = dict(tsAlign=tsAlign, res=resAlign, xToAlign=xToAlign, maxLikStages=maxLikStages,
        tsSamples=tsSamples, xsSamples=xsSamples,
        badSamples=badSamples, xToAlignSamples=xToAlignSamples)
      with open(alignerFile, 'wb') as outfile:
        pickle.dump(savedData, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      with open(alignerFile, 'rb') as outfile:
        savedData = pickle.load(outfile)
      tsAlign = savedData['tsAlign']
      resAlign = savedData['res']
      xToAlign = savedData['xToAlign']
      maxLikStages = savedData['maxLikStages']
      tsSamples = savedData['tsSamples']
      xsSamples = savedData['xsSamples']
      badSamples = savedData['badSamples']
      xToAlignSamples = savedData['xToAlignSamples']

    self.ts = tsAlign

    print(self.params['labels'])
    print('self.ts[:,4]', self.ts[:,4])
    print('self.xsZ[:,4]', self.xsZ[:, 4])
    # print(adsa)

    nanFlag = np.isnan(self.xsZ).any() \
              or np.isnan(self.xsNz).any() \
              or np.isnan(self.ts).any() \
              or np.isnan(self.estimNoiseZ).any()\
              or np.isnan(self.covMatNoiseZ).any()

    if nanFlag:
      print("self.ts", self.ts, np.where(np.isnan(self.ts)))
      print("self.xsNz", self.xsNz, np.where(np.isnan(self.xsNz)))
      print("self.xsZ", self.xsZ, np.where(np.isnan(self.xsZ)))
      print('self.estimNoiseZ', self.estimNoiseZ)
      print('self.covMatNoiseZ', self.covMatNoiseZ)
      raise AssertionError("NaN values in ts, xsNz and xsZ")

    res = {'ts': tsAlign, 'xsZ': xPredZ, 'xsNz': xPredNzSect,
           'patDXdTdata': patDXdTdata, 'patAvgXdata': patAvgXdata, 'patDiagB': patDiagB,
           'x_pred': x_pred, 'dXdT_pred': dXdT_pred, 'sigma_pred': sigma_pred, 'posteriorSamples': posteriorSamples,
           'longData':longData, 'longDiag':longDiag, 'xToAlign': xToAlign, 'outFolder': self.outFolder,
           'maxLikStages':maxLikStages, 'tsSamples':tsSamples,
           'xsSamples':xsSamples, 'badSamples':badSamples, 'xToAlignSamples':xToAlignSamples}

    return res

  def genPosteriorSamples(self, posteriorSamples, x_pred, longData, longDiag, patAvgXdata, xToAlign):
    # integrate all the posterior samples

    nrSamples = posteriorSamples.shape[0]
    # print(nrSamples)
    (nrPointsToEval, nrBiomk) = x_pred.shape
    xPredNzSamples = np.zeros((nrSamples, nrPointsToEval, nrBiomk))
    tsNzSamples = np.zeros((nrSamples, nrPointsToEval, nrBiomk))
    plotFlagSamples = np.zeros((nrSamples, 1))
    badSamples = np.zeros((nrSamples, nrBiomk), bool)
    for s in range(nrSamples):
      (xPredNzSamples[s, :, :], dummy, tsNzSamples[s, :, :], badSamples[s,:], _, _) = integrateTrajAll(
        x_pred, posteriorSamples[s, :, :], patAvgXdata)

    xPredZSamples = (xPredNzSamples - self.muCtlData[None, None, :]) / self.sigmaCtlData[None, None, :]

    tsAlignBaseVisitSamples = np.zeros((nrSamples, nrPointsToEval, nrBiomk))
      # trajectories that couldn't be aligned properly to the desired X value, due to bad region being selected
    (xValShifts) = getXshiftsFromNoise(self.estimNoiseZ, nrSamples)
    alignerWithNoise = AlignerBaseVisitNoise()
    # print(tsNzSamples, xPredZSamples, xValShifts)
    badSamplesAlign = np.zeros((nrSamples, nrBiomk), bool)
    # xToAlignSamples = np.zeros((nrSamples, nrBiomk), float)
    xToAlignSamples = xToAlign[None,:] + xValShifts # add noise to original alignment xs
    for s in range(nrSamples):
      # (tsAlignBaseVisitSamples[s, :, :], badSamplesAlign[s, :], xToAlignSamples[s,:]) = alignerWithNoise.alignNoise(
      #   tsNzSamples[s, :, :], xPredZSamples[s, :, :], longData, longDiag, self.params['anchorID'], self.muCtlData, self.sigmaCtlData,
      #   xValShifts[s, :])
      (tsAlignBaseVisitSamples[s, :, :], badSamplesAlign[s, :], _) = \
        alignerWithNoise.alignTrajXVal(tsNzSamples[s, :, :], xPredZSamples[s, :, :], xToAlignSamples[s,:])

    badSamples = np.logical_or(badSamples, badSamplesAlign)
    #print(tsAlignBaseVisitSamples[1, 1:10, 1], xPredZSamples[1, 1:10, 1])
    #print(tsAlignBaseVisitSamples.shape, badSamples.shape)


    # print('xToAlignSamples', np.mean(xToAlignSamples,0)[3])
    # print('xValShifts', np.mean(xValShifts,0)[3])
    # zeroInd = np.argmin(np.abs(res['ts'][:,3]))
    # print(res['xsZ'].shape, res['ts'].shape)
    # print('xsZ at origin', res['xsZ'][zeroInd, 3], res['xsZ'][zeroInd-10:zeroInd+10, 3])
    # print(asda)

    return tsAlignBaseVisitSamples, xPredZSamples, badSamples, xToAlignSamples

  def stageSubjects(self, indices):
    # stage subjects, whose data is stored in params['data'], selected by the given indices
    assert(indices.shape[0] == self.params['data'].shape[0])
    data = self.params['data'][indices, :]
    return self.stageSubjectsData(data)

  def getDataZ(self, data):
    dataZ = (data - self.muCtlData[None, :]) / self.sigmaCtlData[None, :]
    return dataZ

  def stageSubjectsData(self, data):
    # stage subjects based on likelihood of the data given the subject-specific time shift,
    # assuming gaussian noise of residual

    ts = self.ts
    xs = self.xsZ
    dataZ = self.getDataZ(data)
    (nrPat, nrBiomk) = data.shape
    maxLikStages = np.zeros(nrPat)
    tsStages = self.params['tsStages']
    nrStages = tsStages.shape[0]
    stagingLogLikPS = np.zeros((nrPat, nrStages))
    stagingLik = np.zeros((nrPat, nrStages))
    stagingProb = np.zeros((nrPat, nrStages))

    # fs = [UnivariateSpline(ts[:,b], xs[:,b], s=0) for b in range(nrBiomk)]
    fs = [interpolate.interp1d(ts[:, b], xs[:, b], kind='linear', fill_value='extrapolate') for b in range(nrBiomk)]

    if np.isnan(dataZ).any():
      ''' if data contains nans then staging is done for every patient individually'''
      meanCurrS = np.zeros((nrStages,nrBiomk))
      for s, stage in enumerate(tsStages):
        meanCurrS[s,:] = [fs[b](stage) for b in range(nrBiomk)]

      # print(np.sum(np.sum(np.isnan(dataZ),1)==nrBiomk ))
      # print(dataZ.shape)
      # print(asdsad)

      # print('# of subj with full data', np.sum(np.sum(np.isnan(dataZ),1) == 0))
      # print(sda)

      for p in range(nrPat):
        nnInd = np.logical_not(np.isnan(dataZ[p,:]))
        # print('dataZ[p,:]', dataZ[p,:])
        assert np.sum(nnInd) >= 1
        for s, stage in enumerate(tsStages):
          if any([math.isnan(x) for x in meanCurrS[s,:]]):
            raise AssertionError("Trajectory interpolated values contain NaNs, check if ts, xs are ok")

          stagingLogLikPS[p, s] = multivariate_normal.logpdf(dataZ[p,nnInd], meanCurrS[s,nnInd], self.covMatNoiseZ[nnInd,nnInd])
          stagingLik[p, s] = multivariate_normal.pdf(dataZ[p,nnInd], meanCurrS[s,nnInd], self.covMatNoiseZ[nnInd,nnInd])
          #print(meanCurrS)

    else:
      for s, stage in enumerate(tsStages):
        meanCurrS = [fs[b](stage) for b in range(nrBiomk)]
        #print(meanCurrS)
        if any([math.isnan(x) for x in meanCurrS]):
          raise AssertionError("Trajectory interpolated values contain NaNs, check if ts, xs are ok")
        # func = lambda x: multivariate_normal.pdf(x, meanCurrS, self.covMatNoiseZ)
        # stagingLik[p,s] = np.apply_along_axis(func,)
        stagingLogLikPS[:, s] = multivariate_normal.logpdf(dataZ, meanCurrS, self.covMatNoiseZ)
        stagingLik[:, s] = multivariate_normal.pdf(dataZ, meanCurrS, self.covMatNoiseZ)

    stagingPost = stagingLogLikPS
    if self.params['stagingInformPrior']:
      # set informative prior only for controls, because they otherwise might
      # be sent to -inf time due to flat trajectories before onset
      logStagingPriorPS = multivariate_normal.logpdf(tsStages, -3,0.4)

      print(self.diag.shape)
      print(stagingPost.shape)
      stagingPost[self.diag == CTL,:] = stagingLogLikPS[self.diag == CTL] + logStagingPriorPS

    print('stagingLogLikPS[:,100]\n\n', stagingLogLikPS[:,100])
    print('stagingPost[:,100]\n\n', stagingPost[:,100])
    # print(asda)

    maxStagesIndex = np.argmax(stagingPost, axis=1)
    maxLikStages = tsStages[maxStagesIndex]

    for s, stage in enumerate(tsStages):
      expDiffs = np.power(np.e,stagingPost - stagingPost[:, s][:, None])
      stagingProb[:,s] = np.divide(1, np.sum(expDiffs, axis=1))

    # sumStagingLik = np.sum(stagingLik, axis=1)
    # zeroIndices = np.nonzero(sumStagingLik == 0)
    # stagingLikNoZero = copy.deepcopy(stagingLik)
    # stagingLikNoZero[zeroIndices,:] = 1/nrStages
    # sumStagingLikNoZero = np.sum(stagingLikNoZero, axis=1)
    # stagingProb2 = stagingLikNoZero / sumStagingLikNoZero[:, None]
    # print(stagingProb - stagingProb2)

    if np.any(np.isnan(stagingProb)):
      print(stagingLogLikPS[np.isnan(stagingProb)])
      raise AssertionError("stagingProb is NaN")


    otherParams = None
    return maxLikStages, maxStagesIndex, stagingProb, stagingProb, tsStages, otherParams

  def plotTrajectories(self, res):

    plotTrajParams = self.params['plotTrajParams']


    # fig = plotDiffPredData(res['patDXdTdata'], res['patAvgXdata'], res['patDiag'], res['x_pred'], res['dXdT_pred'], res['sigma_pred'],
    #                        res['posteriorSamples'], self.params['labels'], plotTrajParams)
    # fig.savefig(self.diffDataPredFigName, dpi=100)

    plotTrajParams['xLim'] = (-10, 20)
    fig = plotTrajSubfig(res['ts'], res['xsZ'], res['tsSamples'], res['xsSamples'], res['badSamples'],
                         self.params['labels'], plotTrajParams, xToAlign=res['xToAlign'])
    fig.savefig(self.trajSubplotsFigName, dpi=100)

    plotTrajParams['xLim'] = (-10, 20)
    plotTrajParams['xLabel'] = 'Years since $t_0$'

    # fig, lgd = plotTrajAlign(res['ts'], res['xsZ'], self.params['labels'], plotTrajParams,
    #   xLim = plotTrajParams['trajSubfigXlim'], yLim=plotTrajParams['trajSubfigYlim'])
    # fig.show()
    # fig.savefig(self.trajAlign, bbox_extra_artists=(lgd,), bbox_inches='tight')

    # (maxLikStages, _, _, _, _, _) = self.stageSubjectsData(self.data)
    plotTrajParams['xLabel'] = 'Years since $t_0$'
    if self.expName.startswith('mri'):
      fig, lgd = plotTrajAlignHist(res['ts'], res['xsZ'], self.params['labels'], plotTrajParams,
        self.diag, res['maxLikStages'], xLim = plotTrajParams['trajSubfigXlim'], yLim=plotTrajParams['trajSubfigYlim'])
    elif self.expName.startswith('cog'):
      fig, lgd = plotTrajAlignHistCog(res['ts'], res['xsZ'], self.params['labels'], plotTrajParams,
         self.diag, res['maxLikStages'], xLim=plotTrajParams['trajSubfigXlim'],
           yLim=plotTrajParams['trajSubfigYlim'])

    fig.show()
    fig.savefig(self.trajAlign + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig(self.trajAlign + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

    # plotTrajParams['xLim'] = (-20, 10)
    # fig = plotTrajSubfigWithData(res['ts'], res['xsZ'], None, None, None,
    #      self.params['labels'], plotTrajParams,
    #      self.getDataZ(self.data), self.diag, maxLikStages, thresh=0)
    # fig.savefig(self.trajAlignData, dpi=100)

  def plotTrajSummary(self, res):
    outFolder = 'matfiles/%s' % self.expName
    trajAlignBaseVisit = '%s/trajAlignBaseVisit.png' % self.outFolder

    plotTrajParams = self.params['plotTrajParams']

    plotTrajParams['xLim'] = (-20, 10)
    plotTrajParams['xLabel'] = 'Years since average biomarker value at baseline'
    plotTrajParams['expName'] = self.expName
    fig = plotTrajAlign(res['ts'],res['xsZ'], self.params['labels'], plotTrajParams)
    fig.savefig(self.trajAlignBaseVisit, dpi=100)
