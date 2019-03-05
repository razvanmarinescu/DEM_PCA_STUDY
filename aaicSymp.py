import scipy.io as sio
import sys
import pickle

from DisProgBuilder import *
from evaluationFramework import *
from DEM import *
from aux import *
import csv

minTstaging = -30
maxTstaging = 20
nrStages = 60
params = {'tsStages' : np.linspace(minTstaging, maxTstaging, num=nrStages)}
plotTrajParams = {'diagStr' : ['CTL', 'EMCI', 'LMCI', 'AD', 'SMC'],
                  'diagColors' : ['g', 'y', 'r'],
                  'xLim' : (-20, 30) }

CTL = 1
EMCI = 2
LMCI = 3
AD = 4
SMC = 5

def parseDiag(diagStr):
  if diagStr == 'CN':
    d = CTL
  elif diagStr == 'EMCI':
    d = EMCI
  elif diagStr == 'LMCI':
    d = LMCI
  elif diagStr == 'AD':
    d = AD
  elif diagStr == 'SMC':
    d = SMC
  else:
    raise ValueError('diag not recognised')

  return d

def parseScanTmp(scanStr):
  if scanStr == 'bl':
    timept = 0
  elif scanStr[0] == 'm':
    timept = int(scanStr[1:])/6
  else:
    raise ValueError('scan timepoint not recognised')

  return timept

def computeAge(ageStr, scanStr):
  if scanStr == 'bl':
    dt = 0 # time that passed since bl (in years)
  elif scanStr[0] == 'm':
    dt = float(scanStr[1:])/12
  else:
    raise ValueError('scan timepoint not recognised')

  return float(ageStr) + dt

def parseData(datafile):
  '''
  Parses data and returns it in dict params.
  Warning: params is a global which will be modified by this function.

  Parameters
  ----------
  datafile

  Returns
  -------

  '''
  with open(datafile, 'r') as csvfile:
    reader = list(csv.reader(csvfile, delimiter=',', quotechar='"'))

    labels = ['FDG', 'AV45', 'ABETA_MEDIAN', 'PTAU_MEDIAN', 'TAU_MEDIAN', 'ADAS11',
              'ADAS13', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting',
              'MOCA', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal',
              'Fusiform', 'MidTemp']
    labelsAll = reader[0]
    selectedBiomkInd = [[i for i in range(len(labelsAll)) if labelsAll[i] == l][0]
                        for l in labels ]
    diagInd = [i for i in range(len(labelsAll)) if labelsAll[i] == 'DX_bl'][0]
    ageInd = [i for i in range(len(labelsAll)) if labelsAll[i] == 'AGE'][0]

    nrBiomk = len(selectedBiomkInd)
    print('nrBiomk', nrBiomk, 'selectedBiomkInd', selectedBiomkInd)
    assert(len(selectedBiomkInd) == len(labels))
    reader = reader[1:]
    nrSubjCross = len(reader)
    data = np.zeros((nrSubjCross, nrBiomk), float)
    diag = np.zeros(nrSubjCross, int)
    scanTimepts = np.zeros(nrSubjCross, int)
    partCode = np.zeros(nrSubjCross, int)
    ageAtScan = np.zeros(nrSubjCross, float)
    s = 0
    # labels = [labelsAll[i] for i in selectedBiomkInd]
    #
    # print('labelsAll', labelsAll)

    for s in range(nrSubjCross):
      # parse data from biomarkers

      try:
        for b in range(nrBiomk):
          # print(reader[s][0])
          # print('selectedBiomkInd[b]', selectedBiomkInd[b])
          # print('data[s, selectedBiomkInd[b]]', reader[s][selectedBiomkInd[b]])
          data[s, b] = float(reader[s][selectedBiomkInd[b]])

        # parse diag
        diag[s] = parseDiag(reader[s][diagInd])

        scanTimepts[s] = parseScanTmp(reader[s][1])
        partCode[s] = int(reader[s][0])
        ageAtScan[s] = computeAge(reader[s][ageInd], reader[s][1])
      except ValueError:
        print('error:', reader[s])
        break

      s += 1

    data[data == -4] = np.nan

    print(list(zip(list(np.nanmean(data[diag == CTL],0)), list(np.nanmean(data[diag == AD],0)), range(nrBiomk),
                   labels)))

    # remove subjects that have more than 1/4 of biomk values missing
    filterInd = (np.sum(np.isnan(data),1) < nrBiomk/2)
    print(data.shape, data[filterInd,:].shape)
    # print(adsa)
    data = data[filterInd,:]
    diag = diag[filterInd]
    scanTimepts = scanTimepts[filterInd]
    partCode = partCode[filterInd]
    ageAtScan = ageAtScan[filterInd]

    # print(np.sum(np.sum(np.isnan(data), 1) == nrBiomk))

    params['data'] = data
    params['diag'] = diag
    params['labels'] = labels
    params['scanTimepts'] = scanTimepts
    params['partCode'] = partCode
    params['ageAtScan'] = ageAtScan
    params['lengthScaleFactors'] = np.ones(nrBiomk)


def main(runIndex, nrProcesses, modelToRun):

  # changes the global variable params
  parseData('../data/DDMPAD/adni_adnimerge.csv')

  np.random.seed(7)

  expName = 'adniSymp'

  params['runIndex'] = runIndex
  params['nrProcesses'] = nrProcesses
  params['modelToRun'] = modelToRun

  nrBiomk = len(params['labels'])

  print(nrBiomk)
  biomkProgDir = np.zeros(nrBiomk)
  #biomkProgDir[[0,2,3,9,12,13]] = INCR
  biomkProgDir[[1,3,4,5,6,10,12]] = INCR
  biomkProgDir[[0,2,7,8,9,11,13,14,15,16,17]] = DECR

  selectedBiomk = [i for i in range(nrBiomk) if i != 11]
  params['data'] = params['data'][:,selectedBiomk]
  params['labels'] = [params['labels'][i] for i in selectedBiomk]
  biomkProgDir = biomkProgDir[selectedBiomk]

  params['data'] = makeAllSameProgDir(params['data'], biomkProgDir, uniformDir)

  nrRows = int(np.sqrt(nrBiomk) * 0.95)
  nrCols = int(np.ceil(float(nrBiomk) / nrRows))
  assert(nrRows * nrCols >= nrBiomk)

  ctlDiagInd = params['diag'] == CTL
  ctlData = params['data'][ctlDiagInd, :]
  muCtlData = np.nanmean(ctlData, axis = 0)
  sigmaCtlData = np.nanstd(ctlData, axis = 0)


  # dataZ = (params['data'] - muCtlData[None, :]) / sigmaCtlData[None, :]
  #
  # unqDiags = np.unique(params['diag'])
  # nrDiags = len(unqDiags)
  # nrBiomk = params['data'].shape[1]
  # meanDiags = np.zeros((nrBiomk, 5))
  # for d in range(nrDiags):
  #   meanDiags[:, d] = np.nanmean(dataZ[params['diag'] == (d+1),:], axis = 0)
  #
  # print(params['labels'])
  # print('meanDiags', meanDiags)
  # print(ads)

  plotTrajParams['modelCol'] = 'r' # red
  plotTrajParams['xLim'] = [-20, 30]
  plotTrajParams['axisPos'] = [0.06, 0.1, 0.9, 0.75]
  plotTrajParams['legendPos'] = (1.2, 0.5)
  plotTrajParams['legendPosSubplotsPcaAd'] = (0.5, 0)
  plotTrajParams['legendCols'] = 4
  plotTrajParams['nrRows'] = nrRows
  plotTrajParams['nrCols'] = nrCols
  plotTrajParams['trajAlignMaxWinSize'] = (900, 700)
  plotTrajParams['trajPcaAdMaxWinSize'] = (1200, 500)
  plotTrajParams['axisHeightChopRatio'] = 0.8
  plotTrajParams['expName'] = expName
  plotTrajParams['diagColors'] = ['g', 'b', 'y', 'r', 'm']
  plotTrajParams['trajSubfigXlim'] = [-40,30]
  plotTrajParams['trajSubfigYlim'] = [-30, 5]
  params['plotTrajParams'] = plotTrajParams

  params['runPartStd'] = ['L', 'R']  #[GP fit, traj alignment]
  params['runPartMain'] = ['R', 'R', 'R']  # [mainPart, plot, stage]
  params['runPartStaging'] = ['L', 'L', 'R']

  params['masterProcess'] = runIndex == 0

  if params['masterProcess']:
    params['runPartMain'] = ['L', 'L', 'I', 'I']
    params['runPartStaging'] = ['R', 'R', 'L']

  runAllExpFunc = runAllExpADNI
  modelNames, res = runModels(params, expName, modelToRun, runAllExpFunc)



def runAllExpADNI(params, expName, dpmBuilder):
  """ runs all experiments"""

  res = {}

  # params['patientID'] = AD
  params['excludeID'] = -1
  params['excludeXvalidID'] = -1
  params['excludeStaging'] = [-1]
  params['anchorID'] = AD

  # run if this is the master   process or nrProcesses is 1
  unluckyProc = (np.mod(params['currModel'] - 1, params['nrProcesses']) == params['runIndex'] - 1)
  unluckyOrNoParallel = unluckyProc or (params['nrProcesses'] == 1) or params['masterProcess']

  if unluckyOrNoParallel:
    dpmObj, res['std'] = runStdDPM(params, expName, dpmBuilder, params['runPartMain'])

  # res['cogCorr'] = crossValidAndCorrCog(dpmBuilder, expName, params)

  # print(res)

  return res

if __name__ == '__main__':
  runIndex = int(sys.argv[1])
  nrProcesses = int(sys.argv[2])
  modelToRun = int(sys.argv[3])
  main(runIndex, nrProcesses, modelToRun)