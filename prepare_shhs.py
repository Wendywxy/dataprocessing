import os
import numpy as np

import argparse
import glob
import math
import ntpath

import shutil
import urllib
# import urllib2

from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from mne.io import concatenate_raws, read_raw_edf
import dhedfreader
import xml.etree.ElementTree as ET

from scipy import signal
###############################
EPOCH_SEC_SIZE = 30

def main():
    #for SHHS1 START
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=r'..\..\EEG Dataset\SHHS\polysomnography\edfs\shhs1',
                        help="File path to the PSG files.")
    parser.add_argument("--ann_dir", type=str, default=r'..\..\EEG Dataset\SHHS\polysomnography\annotations-events-profusion\shhs1',
                        help="File path to the annotation files.")
    parser.add_argument("--output_dir", type=str, default=r'..\output_npz_new2\shhs1',
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG C4-A1",
                        help="The selected channel")
    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        

    ids = pd.read_csv('shhs1-ids.txt', header=None, names=['a'])
    ids = ids['a'].values.tolist()
    #for SHHS1 END
    '''
    #for SHHS2 START
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=r'..\..\EEG Dataset\SHHS\polysomnography\edfs\shhs2',
                        help="File path to the PSG files.")
    parser.add_argument("--ann_dir", type=str, default=r'..\..\EEG Dataset\SHHS\polysomnography\annotations-events-profusion\shhs2',
                        help="File path to the annotation files.")
    parser.add_argument("--output_dir", type=str, default=r'..\output_npz_new2\shhs2',
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG C4-A1",
                        help="The selected channel")
    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        

    ids = pd.read_csv('shhs2-ids.txt', header=None, names=['a'])
    ids = ids['a'].values.tolist()
    #for SHHS2 END

    print("ids=",ids)
    edf_fnames = [os.path.join(args.data_dir, i + ".edf") for i in ids]
    ann_fnames = [os.path.join(args.ann_dir,  i + "-profusion.xml") for i in ids]
    edf_fnames.sort()
    ann_fnames.sort()

    edf_fnames = np.asarray(edf_fnames)
    ann_fnames = np.asarray(ann_fnames)



    for file_id in range(len(edf_fnames)):
        if os.path.exists(os.path.join(args.output_dir, edf_fnames[file_id].split('\\')[-1])[:-4]+".npz"):
            continue
        print("file id=",file_id)

        raw = read_raw_edf(edf_fnames[file_id], preload=True, stim_channel=None, verbose=None)
        sampling_rate = raw.info['sfreq']
        print("sample rate =",sampling_rate)
        ch_type = args.select_ch.split(" ")[0]
        select_ch = [s for s in raw.info["ch_names"] if ch_type in s][0]
        raw_ch_df = raw.to_data_frame(scalings=sampling_rate)[select_ch]
        '''
        if sampling_rate == 250:#downsample shhs2 by two steps
            raw_ch_df = signal.resample(raw_ch_df.values,int(len(raw_ch_df)*175/sampling_rate))
            raw_ch_df = signal.resample(raw_ch_df,int(len(raw_ch_df)*100/175))
        else:
            raw_ch_df = signal.resample(raw_ch_df.values,int(len(raw_ch_df)*100/sampling_rate))
        '''
        #down sample to 100hz
        raw_ch_df = signal.resample(raw_ch_df.values, int(len(raw_ch_df) * 100 / sampling_rate))



        raw_ch_df = pd.Series(raw_ch_df)
        raw_ch_df = raw_ch_df.to_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))
        print("len of raw ch df",raw_ch_df.shape)

    ###################################################
        labels = []
        # Read annotation and its header
        t = ET.parse(ann_fnames[file_id])
        r = t.getroot()
        faulty_File = 0
        for i in range(len(r[4])):
            lbl = int(r[4][i].text)
            if lbl == 4:  # make stages N3, N4 same as N3
                labels.append(3)
            elif lbl == 5:  # Assign label 4 for REM stage
                labels.append(4)
            else:
                labels.append(lbl)
            if lbl > 5:  # some files may contain labels > 5 BUT not the selected ones.
                faulty_File = 1

        if faulty_File == 1:
            print( "============================== Faulty file ==================")
            continue

        labels = np.asarray(labels)

        # Remove movement and unknown stages if any
        raw_ch = raw_ch_df.values
        print(raw_ch.shape)

        # Verify that we can split into 30-s epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * 100) != 0:
            raise Exception("Something wrong")
        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * 100)
        print("num of epochs",n_epochs)

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)

        print(x.shape)
        print(y.shape)
        assert len(x) == len(y)

        # Select on sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != 0)[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        print("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        print("Data after selection: {}, {}".format(x.shape, y.shape))

        # Saving as numpy files
        filename = os.path.basename(edf_fnames[file_id]).replace(".edf",  ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": 100
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)
        print(" ---------- Done this file ---------")



if __name__ == "__main__":
    main()
