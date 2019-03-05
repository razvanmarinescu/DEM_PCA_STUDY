import scipy.io as sio
import sys

# from DisProgBuilder import *
from evaluationFramework import *
from DEM import *
from aux import *
import numpy as np
import math

minTstaging = -20
maxTstaging = 30
nrStages = 100
params = {'tsStages' : np.linspace(minTstaging, maxTstaging, num=nrStages)}
plotTrajParams = {'diagStr' : ['CTL', 'aMC', 'sMC'],
                  'diagColors' : ['g', 'y', 'r'],
                  'xLim' : (-20, 30) }

def main(runIndex, nrProcesses, modelToRun):
  # corrected for TIV, age and gender in function processDIAN
  matData = sio.loadmat('../data/DIAN/DIANselectedBiomk.mat')

  np.random.seed(8)

  expName = 'dian'

  # remove some biomk which have no signal as DEM picks up the wrong direction
  selectedBiomk = np.array([x for x in range(matData['selectedData'].shape[1]) if x not in [3,10]])

  diag = np.array(np.squeeze(matData['diag']))
  data = matData['selectedData'][:, selectedBiomk]
  labels = matData['selectedLabels']
  labels = [labels[i][0][0] for i in range(len(labels)) if i in selectedBiomk]
  scanTimepts = np.squeeze(matData['scanTimepoints'])
  partCode = np.squeeze(matData['participantCode'])
  ageAtScan = np.squeeze(matData['ageAtScan'])
  years2onset = np.squeeze(matData['years2onset'])

  # filter rows containing NaNs
  notNanIndices = ~np.isnan(data).any(axis=1)

  params['data'] = data[notNanIndices,:]
  params['diag'] = diag[notNanIndices]
  params['labels'] = labels
  params['scanTimepts'] = scanTimepts[notNanIndices]
  params['partCode'] = partCode[notNanIndices]
  params['ageAtScan'] = ageAtScan[notNanIndices]
  params['years2onset'] = years2onset[notNanIndices]
  params['lengthScaleFactors'] = np.ones(len(selectedBiomk))

  nrBiomk = params['data'].shape[1]
  biomkProgDir = matData['biomkProgDir'][selectedBiomk]

  params['data'] = makeAllSameProgDir(params['data'], biomkProgDir, uniformDir)

  nrRows = int(np.sqrt(nrBiomk) * 0.95)
  nrCols = int(np.ceil(float(nrBiomk) / nrRows))
  assert (nrRows * nrCols >= nrBiomk)

  plotTrajParams['modelCol'] = 'r'  # orange
  plotTrajParams['axisPos'] = [0.06, 0.06, 0.9, 0.6]
  plotTrajParams['legendPos'] = (0.5, 1.5)
  plotTrajParams['legendPosSubplotsPcaAd'] = (0.5, 0)
  plotTrajParams['legendCols'] = 3
  plotTrajParams['nrRows'] = nrRows
  plotTrajParams['nrCols'] = nrCols
  plotTrajParams['trajAlignMaxWinSize'] = (900, 850)
  plotTrajParams['trajPcaAdMaxWinSize'] = (1200, 900)
  plotTrajParams['axisHeightChopRatio'] = 0.8
  plotTrajParams['expName'] = expName

  params['plotTrajParams'] = plotTrajParams

  params['runPartMain'] = ['R', 'R', 'I', 'R']  # [gaussian fit, aligner, plot, staging]
  params['runPartStaging'] = ['R', 'R', 'R']

  params['masterProject'] = runIndex == 0

  if params['masterProject']:
    params['runPartMain'] = ['L', 'L', 'I', 'I']
    params['runPartStaging'] = ['L', 'L', 'I']

  runAllExpFunc = runAllExpDIAN
  modelNames, res = runModels(params, expName, modelToRun, runAllExpFunc)

  printResDIAN(modelNames, res)

def runAllExpDIAN(params, expName, dpmBuilder):
  """ runs all experiments"""

  res = {}
  params['excludeID'] = [sNC]
  params['excludeXvalidID'] = [sNC]
  params['excludeStaging'] = [CTL, sNC]
  params['anchorID'] = sMC

  dpmObj, dpmRes = runStdDPM(params, expName, dpmBuilder, params['runPartMain'])
  res['upEqStagesPerc'], res['pFUgrBLAll']= evalStaging(dpmBuilder, expName, params)

  inclusionIDs = [aMC, sMC]
  res['convPredStats'] = stagesCorrWithYearsToOnset(dpmBuilder, expName, params, dpmObj, dpmRes, inclusionIDs)

  print(res)
  return res

def stagesCorrWithYearsToOnset(dpmBuilder, expName, params, dpmObj, dpmRes, inclusionIDs):

  dataIndices = np.arange(0, len(params['diag']), 1)
  (maxLikStages, maxStagesIndex, stagingProb, stagingLik, tsStages) = dpmObj.stageSubjects(dataIndices)

  inclusionIndices = np.in1d(params['diag'], inclusionIDs)

  varMat = np.zeros((maxLikStages[inclusionIndices].shape[0],2), float)
  varMat[:,0] = maxLikStages[inclusionIndices]
  varMat[:,1] = params['years2onset'][inclusionIndices]

  rho, pVal = scipy.stats.spearmanr(varMat)

  lower, upper = detSpearmanConfInt(rho, np.sum(inclusionIndices))

  return rho, (upper  - lower)/2

def detSpearmanConfInt(r, num):
  stderr = 1.0 / math.sqrt(num - 3)
  delta = 1.96 * stderr
  lower = math.tanh(math.atanh(r) - delta)
  upper = math.tanh(math.atanh(r) + delta)

  return lower, upper

if __name__ == '__main__':
  runIndex = int(sys.argv[1])
  nrProcesses = int(sys.argv[2])
  modelToRun = int(sys.argv[3])
  main(runIndex, nrProcesses, modelToRun)