import numpy as np
import sys
sys.path.append('/Users/nfirth/Desktop/UCL/Models/EBM/EBM/')
import fileReaders
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from sklearn import gaussian_process
from sklearn import preprocessing
from sklearn import covariance
import cPickle as pickle
import mixtureModelling

plt.rcParams.update({'figure.max_open_warning': 0})


def formatLongData(data, labels, longInf, diseaseLabel=None):
    # Get patient ids which have multiple time points
    # and then create an array mask with it
    pIds = np.unique(longInf[longInf[:, 2] > 1, 0])
    dataMask = np.in1d(longInf[:, 0], pIds)

    # Apply mask to data
    newLongInf = longInf[dataMask]
    newData = data[dataMask]
    newLabels = labels[dataMask].astype(int)

    # Get specific disease
    if(diseaseLabel is not None):
        newLongInf = newLongInf[newLabels == diseaseLabel]
        newData = newData[newLabels == diseaseLabel]
        newLabels = newLabels[newLabels == diseaseLabel]
    return newData, newLabels, newLongInf


def getBiomarkerDxDt(data, longInf, biomarkerNum, bmName, ax=None,
                     minSeqLen=2, figIdx=0):
    X = []
    y = []
    nVisits = []
    pids = np.unique(longInf[:, 0])
    if(ax is not None):
        ax[figIdx].set_xlabel('Years from baseline')
        ax[figIdx].set_ylabel('Raw score')
    for idx in pids:
        pData = data[longInf[:, 0] == idx, biomarkerNum]
        pTimes = longInf[longInf[:, 0] == idx, 3]
        for j in xrange(pData.shape[0]-1, -1, -1):
            if(np.isnan(pData[j]) or np.isinf(pData[j])):
                pData = np.delete(pData, j)
                pTimes = np.delete(pTimes, j)
        if(pData.shape[0] < minSeqLen):
            continue
        pTimes = [float((x-pTimes[0]).days)/365 for x in pTimes]
        m, c = np.polyfit(pTimes, pData, 1)
        X.append(np.mean(pData))
        y.append(m)
        nVisits.append(len(pData))
        if(ax is not None):
            ax[figIdx].plot(pTimes, pData)
    return np.array(X), np.array(y), ax, np.array(nVisits)


def removeOutliers(X, y, outliersFraction):
    clf = covariance.EllipticEnvelope(contamination=outliersFraction)

    fitData = np.vstack((X, y)).T
    fitData = preprocessing.StandardScaler().fit_transform(fitData)
    try:
        clf.fit(fitData)
        outMask = clf.decision_function(fitData)
        threshold = stats.scoreatpercentile(outMask, 100*outliersFraction)
        outMask = (outMask > threshold).flatten()
    except:
        outMask = np.ones(fitData.shape[0], dtype=bool)
    return X[outMask], y[outMask], outMask


def removeDuplicateXPoints(X, y):
    gpX, gpy = X.copy(), y.copy()
    idxs = np.argsort(gpX)

    gpX = gpX[idxs]
    gpy = gpy[idxs]

    for i in xrange(idxs.shape[0]-1, 0, -1):
        if(gpX[i] == gpX[i-1]):
            gpX = np.delete(gpX, i)
            gpy = np.delete(gpy, i)
    return gpX, gpy


def plotGPR(linSpace, yPred, MSE, X, y, nVisits, outMask, bmName, ax, idx=1):
    sigma = np.sqrt(MSE)
    # print y.shape, nVisits.shape, len(outMask)
    ax[idx].scatter(X, y, label="Data", s=np.power(nVisits, 2),
                    marker='o')
    ax[idx].scatter(X[outMask == 0], y[outMask == 0],
                    s=np.power(nVisits[outMask == 0], 2),
                    marker='o', label='Outliers', color='r')
    ax[idx].plot(linSpace, yPred, 'r--')
    ax[idx].fill(np.concatenate([linSpace, linSpace[::-1]]),
                 np.concatenate([yPred - 1.9600 * sigma,
                                (yPred + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
    # nVisits = np.unique(nVisits)
    # for i in xrange(nVisits.shape[0]):
    #     ax[idx].scatter(None, None, s=np.power(nVisits[i], 2),
    #                     label='{0} vists'.format(nVisits[i]))
    ax[idx].legend(loc='best', fontsize=12)
    ax[idx].set_ylabel('d({0})/dt'.format(bmName))
    ax[idx].set_xlabel('{0} Score'.format(bmName))
    return ax


def integrateTraj(linSpace, yPred, meanVal):
    intDomSpace = linSpace.flatten()
    if(np.sum(yPred > 0) > np.sum(yPred < 0)):
        intDomSpace = intDomSpace[yPred > 0]
        intDomY = yPred[yPred > 0]
    else:
        intDomSpace = intDomSpace[yPred < 0]
        intDomY = yPred[yPred < 0]

    intDomY = np.delete(intDomY, intDomY.shape[0]-1)
    intDomSpaceDiff = np.diff(intDomSpace, 1)
    intDomSpace = np.delete(intDomSpace, intDomSpace.shape[0]-1)

    intDomDt = np.divide(intDomSpaceDiff, intDomY)
    intDomT = np.cumsum(intDomDt, axis=0)

    if(intDomT[0] < 0):
        intDomT = intDomT-1*np.min(intDomT)
    for i in xrange(1, intDomT.shape[0]):
        if(np.abs(intDomT[i] - intDomT[i-1]) > 5):
            break
    intDomT = intDomT[:i]
    intDomSpace = intDomSpace[:i]
    cutoff = intDomSpace.shape[0]-1
    while(intDomSpace[cutoff] > meanVal):
        cutoff -= 1

    intDomT = intDomT[:cutoff]
    intDomSpace = intDomSpace[:cutoff]
    intDomT = intDomT - intDomT.min()

    return intDomT, intDomSpace


def plotTraj(t, x, bmName, ax, idx=2):
    ax[idx].plot(t, x, '-', linewidth=2,)
    ax[idx].set_xlabel('Time (years) from average baseline value')
    ax[idx].set_ylabel('{0} score'.format(bmName))
    # ax[idx].set_title('{0} integrated trajectory'.format(bmName))
    return ax


def main():
    data, labels, longInf, headers = fileReaders.readLongitudinalData(pca=True)
    crossData, crossLabels, crossHeaders = fileReaders.pickleFileReader()

    removeList = ['PAL', 'Addition  ', 'Shape dis.'] # 'MMSE', 'SRMT (w)', 'SRMT (f)', 'Naming',
                  # 'Fragmented letters', 'Object decision', 'Shape dis.',
                  # 'Max. b.', 'A (time) cancellation', 'Max. f.', 'Digit span f. ',
                  # 'Digit span b.', 'Dot counting (n correct)']
    headers = list(headers)
    data = data[:, [headers.index(x) for x in headers if x not in removeList]]
    for x in removeList:
        headers.remove(x)

    crossHeaders = list(crossHeaders)
    crossData = crossData[:, [crossHeaders.index(x) for x in crossHeaders if x not in removeList]]
    for x in removeList:
        crossHeaders.remove(x)

    dMeans = np.nanmean(crossData[crossLabels == 1], axis=0)
    means = np.nanmean(crossData[crossLabels == 0], axis=0)
    stds = np.nanstd(crossData[crossLabels == 0], axis=0)

    newData, newLabels, newLongInf = formatLongData(data, labels, longInf,
                                                    diseaseLabel=1)
    combiFig, combiAx = plt.subplots(figsize=(10, 10))
    for bmNum in xrange(data.shape[1]):
        fig, ax = plt.subplots(3, figsize=(14, 10))
        fig.suptitle(headers[bmNum].strip(), fontsize=15)
        allX, allY, ax, nVisits = getBiomarkerDxDt(newData, newLongInf,
                                                   bmNum, headers[bmNum].strip(), ax)
        X, y, outMask = removeOutliers(allX, allY, 0.05)
        X, y = removeDuplicateXPoints(X, y)

        lower, upper = np.abs(1/np.max(X)), np.abs(1/(np.min(X)+1e-6))
        if(lower > upper):
            lower, upper = upper, lower
        gpr = gaussian_process.GaussianProcess(regr='constant',
                                               corr='squared_exponential',
                                               theta0=1/np.abs(np.mean(X)),
                                               thetaL=lower,
                                               thetaU=upper,
                                               nugget=3.5, optimizer='Welch',
                                               normalize=False)
        gpr.fit(X.reshape(-1, 1), y)
        linSpace = np.atleast_2d(np.linspace(np.min(X), np.max(X), 100)).T
        yPred, MSE = gpr.predict(linSpace.reshape(-1, 1), eval_MSE=True)

        # sigma = np.sqrt(MSE)
        # top = yPred - 1.9600 * sigma
        # bottom = (yPred + 1.9600 * sigma)[::-1]

        ax = plotGPR(linSpace, yPred, MSE, allX, allY, nVisits,
                     outMask, headers[bmNum].strip(), ax)

        t, x = integrateTraj(linSpace, yPred, dMeans[bmNum])
        ax = plotTraj(t, x, headers[bmNum].strip(), ax)
        # t, x = integrateTraj(linSpace, top, dMeans[bmNum])
        # ax = plotTraj(t, x, headers[bmNum].strip(), ax)
        # t, x = integrateTraj(linSpace, bottom, dMeans[bmNum])
        # ax = plotTraj(t, x, headers[bmNum].strip(), ax)

        x = (x-means[bmNum])/stds[bmNum]
        if(x.sum() < 0):
            x *= -1
        combiAx.plot(t, x, '-', linewidth=2,
                     label='{0}'.format(headers[bmNum].strip()),
                     color=cm.rainbow(float(bmNum)/data.shape[1]))

        fig.savefig('/Users/nfirth/Desktop/PCAFigures/DEM/{0}_DEM_MM.png'.format(headers[bmNum].strip()))
    combiAx.legend(loc='best', fontsize=12)
    combiAx.grid(True)
    combiAx.set_xlabel('Time (years) from average baseline value')
    combiAx.set_ylabel('z-Score w.r.t. control population')
    combiFig.savefig('/Users/nfirth/Desktop/PCAFigures/DEM/DEM_MM.png')
    # combiFig.show()
    # plt.show()
if __name__ == '__main__':
    main()
