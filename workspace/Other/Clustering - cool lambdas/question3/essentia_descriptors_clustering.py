from __future__ import print_function
from essentia.standard import MusicExtractor, YamlOutput
from essentia import Pool
from argparse import ArgumentParser
import numpy as np
import os
import json
import fnmatch
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.vq import vq, kmeans, whiten

DESCRIPTORS = [
    'lowlevel.barkbands_crest.mean',
    'lowlevel.barkbands_flatness_db.mean',
    'lowlevel.barkbands_kurtosis.mean',
    'lowlevel.barkbands_skewness.mean',
    'lowlevel.barkbands_spread.mean',
    'lowlevel.dissonance.mean',
    'lowlevel.erbbands_crest.mean',
    'lowlevel.erbbands_flatness_db.mean',
    'lowlevel.erbbands_kurtosis.mean',
    'lowlevel.erbbands_skewness.mean',
    'lowlevel.erbbands_spread.mean',
    'lowlevel.hfc.mean',
    'lowlevel.loudness_ebu128.momentary.mean',
    'lowlevel.loudness_ebu128.short_term.mean',
    'lowlevel.melbands_crest.mean',
    'lowlevel.melbands_flatness_db.mean',
    'lowlevel.melbands_kurtosis.mean',
    'lowlevel.melbands_skewness.mean',
    'lowlevel.melbands_spread.mean',
    'lowlevel.pitch_salience.mean',
    'lowlevel.silence_rate_20dB.mean',
    'lowlevel.silence_rate_30dB.mean',
    'lowlevel.silence_rate_60dB.mean',
    'lowlevel.spectral_centroid.mean',
    'lowlevel.spectral_complexity.mean',
    'lowlevel.spectral_decrease.mean',
    'lowlevel.spectral_energy.mean',
    'lowlevel.spectral_energyband_high.mean',
    'lowlevel.spectral_energyband_low.mean',
    'lowlevel.spectral_energyband_middle_high.mean',
    'lowlevel.spectral_energyband_middle_low.mean',
    'lowlevel.spectral_entropy.mean',
    'lowlevel.spectral_flux.mean',
    'lowlevel.spectral_kurtosis.mean',
    'lowlevel.spectral_rms.mean',
    'lowlevel.spectral_rolloff.mean',
    'lowlevel.spectral_skewness.mean',
    'lowlevel.spectral_spread.mean',
    'lowlevel.spectral_strongpeak.mean',
    'lowlevel.zerocrossingrate.mean',
    'rhythm.beats_loudness.mean',
    'tonal.chords_strength.mean',
    'tonal.hpcp_crest.mean',
    'tonal.hpcp_entropy.mean',
    'lowlevel.barkbands.mean',
    'lowlevel.erbbands.mean',
    'lowlevel.gfcc.mean',
    'lowlevel.melbands.mean',
    'lowlevel.melbands128.mean',
    'lowlevel.mfcc.mean',
    'lowlevel.spectral_contrast_coeffs.mean',
    'lowlevel.spectral_contrast_valleys.mean',
    'rhythm.beats_loudness_band_ratio.mean',
    'tonal.hpcp.mean'
]


def descriptorPairScatterPlot(inputDir, x_descriptor, y_descriptor, anotOn = 0):
  """
  This function does a scatter plot of the chosen feature pairs for all the sounds in the 
  directory inputDir.
  Additionally, you can annotate the sound id on the scatter plot by setting anotOn = 1

  Input:
    inputDir (string): path to the directory where the sound samples and descriptors are present
    x_descriptor (string): name of the descriptor for the X axis.
    y_descriptor (string): name of the descriptor for the Y axis.
    anotOn (int): Set this flag to 1 to annotate the scatter points with the sound id. (Default = 0)
    
  Output:
    scatter plot of the chosen pair of descriptors for all the sounds in the directory inputDir
  """
  #dataDetails = fetchDataDetails(inputDir, '.mp3')
  dataDetails = fetchDataDetails(inputDir, '_trimmed.wav')
  colors = ['r', 'g', 'c', 'b', 'k', 'm', 'y', 'gray', 'lime', 'deeppink']
  plt.figure()
  legArray = []
  catArray = []
  for ii, category in enumerate(dataDetails.keys()):
    catArray.append(category)
    for soundId in dataDetails[category].keys():
      descSound = dataDetails[category][soundId]
      x_cord = descSound[x_descriptor]
      y_cord = descSound[y_descriptor]

      plt.scatter(x_cord,y_cord, c = colors[ii], s=200, alpha=0.75)
      if anotOn==1:
         plt.annotate(soundId, xy=(x_cord, y_cord), xytext=(x_cord, y_cord))
    
    circ = Line2D([0], [0], linestyle="none", marker="o", alpha=0.75, markersize=10, markerfacecolor=colors[ii])
    legArray.append(circ)
  
  plt.ylabel(y_descriptor, fontsize =16)
  plt.xlabel(x_descriptor, fontsize =16)
  plt.legend(legArray, catArray ,numpoints=1,bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(catArray), mode="expand", borderaxespad=0.)
  plt.show()


def clusterSounds(targetDir, nCluster = -1, descriptors=[]):
  """
  This function clusters all the sounds in targetDir using kmeans clustering.
  
  Input:
    targetDir (string): Directory where sound descriptors are stored (all the sounds in this 
                        directory will be used for clustering)
    nCluster (int): Number of clusters to be used for kmeans clustering.
    descriptors (list): List of names of the descriptors to use.

  Output:
    Prints the class of each cluster (computed by a majority vote), number of sounds in each 
    cluster and information (sound-id, sound-class and classification decision) of the sounds 
    in each cluster. Optionally, you can uncomment the return statement to return the same data.
  """
  
  dataDetails = fetchDataDetails(targetDir, '_trimmed.wav')
  
  ftrArr = []
  infoArr = []
  
  if nCluster ==-1:
    nCluster = len(dataDetails.keys())
  for cname in dataDetails.keys():
    #iterating over sounds
    for sname in dataDetails[cname].keys():
      ftr = dataDetails[cname][sname]
      ftrArr.append([ftr[d] for d in descriptors])
      infoArr.append([sname, cname])
  
  ftrArr = np.array(ftrArr)
  infoArr = np.array(infoArr)
  
  ftrArrWhite = whiten(ftrArr)
  centroids, distortion = kmeans(ftrArrWhite, nCluster)
  clusResults = -1*np.ones(ftrArrWhite.shape[0])
  
  for ii in range(ftrArrWhite.shape[0]):
    diff = centroids - ftrArrWhite[ii,:]
    diff = np.sum(np.power(diff,2), axis = 1)
    indMin = np.argmin(diff)
    clusResults[ii] = indMin
  
  ClusterOut = []
  classCluster = []
  globalDecisions = []  
  for ii in range(nCluster):
    ind = np.where(clusResults==ii)[0]
    freqCnt = []
    for elem in infoArr[ind,1]:
      freqCnt.append(infoArr[ind,1].tolist().count(elem))
    indMax = np.argmax(freqCnt)
    classCluster.append(infoArr[ind,1][indMax])
    
    print("\n(Cluster: " + str(ii) + ") Using majority voting as a criterion this cluster belongs to " + 
          "class: " + classCluster[-1])
    print ("Number of sounds in this cluster are: " + str(len(ind)))
    decisions = []
    for jj in ind:
        if infoArr[jj,1] == classCluster[-1]:
            decisions.append(1)
        else:
            decisions.append(0)
    globalDecisions.extend(decisions)
    print ("sound-id, sound-class, classification decision")
    ClusterOut.append(np.hstack((infoArr[ind],np.array([decisions]).T)))
    print (ClusterOut[-1])
  globalDecisions = np.array(globalDecisions)
  totalSounds = len(globalDecisions)
  nIncorrectClassified = len(np.where(globalDecisions==0)[0])
  print("Out of %d sounds, %d sounds are incorrectly classified considering that one cluster should "
        "ideally contain sounds from only a single class"%(totalSounds, nIncorrectClassified))
  print("You obtain a classification (based on obtained clusters and majority voting) accuracy "
         "of %.2f percentage"%round(float(100.0*float(totalSounds-nIncorrectClassified)/totalSounds),2))
  # return ClusterOut


def fetchDataDetails(inputDir, audioFileSuffix):
  """Search audio files in a directory and compute their descriptors."""
  dataDetails = {}
  extractor = MusicExtractor()
  for path, dname, fnames  in os.walk(inputDir):
    for fname in fnames:
      if audioFileSuffix in fname.lower():
        remain, rname, cname, sname = path.split('/')[:-3], path.split('/')[-3], path.split('/')[-2], path.split('/')[-1]
        if cname not in dataDetails:
          dataDetails[cname]={}
        audio_file = os.path.join('/'.join(remain), rname, cname, sname, fname)
        json_path = os.path.splitext(audio_file)[0] + '_musicextractor.json'
        if not os.path.exists(json_path):
            try:
                poolStats, _ = extractor(audio_file)
                json_data = poolToDict(poolStats)
                with open(json_path, 'w') as json_file:
                    json.dump(json_data, json_file, indent=4)
            except RuntimeError:
                print(f'Could not extract features for {audio_file}.')
        else:
            with open(json_path, 'r') as json_file:
                json_data = json.load(json_file)
        dataDetails[cname][sname] = json_data
  return dataDetails


def poolToDict(pool):
    conv = lambda v: v.tolist() if type(v) is np.ndarray else v
    return {k: conv(pool[k]) for k in pool.descriptorNames()}


if __name__ == '__main__':
    parser = ArgumentParser(description = """
Analyzes all audio files found (recursively) in a folder using MusicExtractor.
""")

    parser.add_argument('-d', '--dir', help='input directory', required=True)
    parser.add_argument('--profile', help='MusicExtractor profile', required=False)
    parser.add_argument('-x', help='Y-axis descriptor', required=True)
    parser.add_argument('-y', help='X-axis descriptor', required=True)
    args = parser.parse_args()

# Uncomment to plot all the mean descriptors against themselves:
#    for d in DESCRIPTORS:
#        descriptorPairScatterPlot(args.dir, d, d, anotOn = 0)

clusterSounds(
        targetDir=args.d,
        nCluster=10,
        #descriptors=['lowlevel.erbbands_crest.mean', 'rhythm.beats_loudness.mean']
        #descriptors=['lowlevel.pitch_salience.mean', 'lowlevel.silence_rate_60dB.mean']
        #descriptors=['lowlevel.erbbands_flatness_db.mean', 'lowlevel.erbbands_skewness.mean', 'lowlevel.pitch_salience.mean']
        #descriptors=['lowlevel.erbbands_flatness_db.mean', 'lowlevel.erbbands_skewness.mean']
        #descriptors=['lowlevel.erbbands_flatness_db.mean', 'lowlevel.erbbands_skewness.mean', 'lowlevel.spectral_decrease.mean']
        #descriptors=['lowlevel.loudness_ebu128.momentary.mean', 'lowlevel.silence_rate_60dB.mean', 'lowlevel.erbbands_skewness.mean'] # 57%
        descriptors=['lowlevel.loudness_ebu128.momentary.mean', 'lowlevel.silence_rate_60dB.mean', 'lowlevel.erbbands_skewness.mean', 'lowlevel.erbbands_flatness_db.mean', 'lowlevel.erbbands_skewness.mean', 'lowlevel.spectral_decrease.mean'] # %61
)
