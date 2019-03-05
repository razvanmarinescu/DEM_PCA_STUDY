import os
import subprocess
import sys
from socket import gethostname

# Run as: python launchProc.py <firstInstanceNr> <lastInstanceNr> <nrProcesses> <experimentToRun> <launcherScript>


import argparse

parser = argparse.ArgumentParser(description='Launches processes and waits for them until finished')
parser.add_argument('--firstInstance', dest='firstInstance', type=int, 
                    default=1,help='index of first run instance/process')

parser.add_argument('--lastInstance', dest='lastInstance', type=int,
                   default=8,help='index of last run instance/process')

parser.add_argument('--nrProc', dest='nrProc', type=int,
                   default=8,help='# of processes')

parser.add_argument('--firstModel', dest='firstModel', type=int,
                   help='index of first experiment to run')

parser.add_argument('--lastModel', dest='lastModel', type=int,
                   help='index of last experiment to run')

parser.add_argument('--models', dest='models',
                   help='list of models to run')

parser.add_argument('--launcherScript', dest='launcherScript', 
                   help='launcher script, such as adniEBM, mriLarge')

parser.add_argument('--cluster', action="store_true", help='set this if the program should run on the CMIC cluster')

parser.add_argument('--timeLimit', dest='timeLimit',
                   help='timeLimit for the job in hours')

parser.add_argument('--printOnly', action="store_true", help='only print experiment to be run, not actualy run it')

args = parser.parse_args()

print(args)
print(args.firstInstance)
print(args.launcherScript)

MEM_LIMIT = 7.9 # in GB
REPO_DIR = '/home/rmarines/phd/mres/diffEqModel'
OUTPUT_DIR = '%s/clusterOutput' % REPO_DIR

hostName = gethostname()

if args.cluster:
  WALL_TIME_LIMIT_HOURS = int(args.timeLimit)
  WALL_TIME_LIMIT_MIN = 15

def getQsubTextOneLine(instanceNr, nrProc, modelNr, launcherScript):
  # if there's an error about tty, add & in the last parameter
  runCmd = "cd %s; /share/apps/python-3.5.1/bin/python3 %s %d %d %d " % (REPO_DIR, launcherScript, instanceNr, nrProc, modelNr)
  jobName = "i%d_m%d_%s" % (instanceNr, modelNr, launcherScript)
  qsubCmd = 'qsub -S /bin/bash -l tmem=%.1fG -l h_vmem=%.1fG -l h_rt=%d:%d:0 -N %s -j y -wd %s ' % (MEM_LIMIT, MEM_LIMIT, WALL_TIME_LIMIT_HOURS, WALL_TIME_LIMIT_MIN, jobName, OUTPUT_DIR)
  
  cmd = 'echo "%s" | %s' % (runCmd, qsubCmd) # echo the matlab cmd then pipe it to qsub
  return cmd

pList = []
#instanceIndices = [2,3,4,5,6,7,8];
instanceIndices = range(args.firstInstance, args.lastInstance+1)
quitFlag = 1

if not args.cluster:
  # if (args.firstModel is not None) and (args.lastModel is not None):
  modelIndices = range(args.firstModel, args.lastModel+1)
  for m in modelIndices:
    for i in instanceIndices:
      cmdArgs = [str(x) for x in ['python3', args.launcherScript, i, args.nrProc, m]]
      print(cmdArgs)
      p = subprocess.Popen(cmdArgs)
      pList.append(p)

  nrProcs = len(pList)
  for i in range(nrProcs):
    p = pList.pop()
    print(p)
    p.wait()

    print("---------------------->finished")
else:
  modelIndices = range(args.firstModel, args.lastModel+1)
  for m in modelIndices:
    for i in instanceIndices: 
      cmd = getQsubTextOneLine(i, args.nrProc, m, args.launcherScript)
      print(cmd)
      if not args.printOnly:
        os.system(cmd)
