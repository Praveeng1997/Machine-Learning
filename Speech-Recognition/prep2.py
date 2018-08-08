from __future__ import absolute_import
from __future__ import division

import os
import re
import hashlib
import glob
import pdb

import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc


import librosa




def get_mfcc2(audio):
  s,record = wav.read(audio)
  train_inputs = mfcc(record,samplerate = s)
  train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
  return train_inputs

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape



def pad_sequences(sequences, maxlen=None, dtype=np.float32, padding='post', truncating='post', value=0.):
   
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.
        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths


  




def which_set(filename, validation_percentage, testing_percentage):
  MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
  """Determines which data partition the file should belong to.
  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


def getSparseTensor(arr):
  idx  = np.where(arr != 0.0)
  return tf.SparseTensor(np.vstack(idx).T, arr[idx], arr.shape)


def assign_files_and_labels(rootdir):
  sets = {'training':0,'testing':1,'validation':2}
  inputs_with_labels = [[],[],[]]
  subdirlist = glob.glob(rootdir+"/*/")
  for subdir in subdirlist:
    for files in glob.glob(subdir+"/*.wav"):
        label = files.split("/")[-2]        # The Label is obtained from the path
        index = sets[which_set(files,10,10)]
        inputs_with_labels[index].append([files,label])
  return inputs_with_labels


def process_text(line):
  line = line.replace('\n','')
  line = line.replace('_background_noise_','noise')
  line = line.replace('.','')
  line = line.replace(' ','  ')
  line = line.lower()
  line = line.split(' ')
  return line

def modify_text(targets,SPACE_TOKEN = '<space>',SPACE_INDEX = 0,FIRST_INDEX = ord('a') - 1):   
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])
    # Transform char into index
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])
    return targets
    
    
    
    
    
def get_mfcc(audio):
    y,s = librosa.load(audio)
    train_inputs = librosa.feature.mfcc(y = y,sr = s,n_mfcc = 20)
    train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
    return train_inputs
    
    
    

def process_input(input_files):
    inputs = [[],[],[]]
    targets = [[],[],[]]
    j =1
    for i in range(len(input_files)):
      j = 0
      for audio,text in input_files[i]:
        print i,j
        j+=1
        inputs[i].append(get_mfcc(audio))
        targets[i].append(modify_text(process_text(text)))
    return inputs,targets
