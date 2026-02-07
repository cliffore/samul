import os
import numpy as np
import re
import torch
import datetime
from collections import defaultdict
from pathlib import Path
import json
import traceback
import shutil
from utils import *



def average_common_ids(t1, t2):
    """
    Compute average weights for IDs that appear in both tensors.
    Works if inputs are torch.Tensors or numpy arrays.
    """
    # Convert to torch if numpy
    if isinstance(t1, np.ndarray):
        t1 = torch.from_numpy(t1)
    if isinstance(t2, np.ndarray):
        t2 = torch.from_numpy(t2)

    # Flatten to (N,2)
    t1 = t1.reshape(-1, 2)
    t2 = t2.reshape(-1, 2)

    ids1, vals1 = t1[:, 0].long(), t1[:, 1].float()
    ids2, vals2 = t2[:, 0].long(), t2[:, 1].float()

    # Find common IDs
    common_ids = torch.tensor(list(set(ids1.tolist()) & set(ids2.tolist())), dtype=torch.long)

    results = []
    for cid in common_ids:
        # All weights for this id in t1 and t2
        w1 = vals1[ids1 == cid]
        w2 = vals2[ids2 == cid]
        all_w = torch.cat([w1, w2])
        avg = all_w.mean()
        results.append([cid.item(), avg.item()])

    if results:
        return torch.tensor(results, dtype=torch.float32)
    else:
        return torch.empty((0, 2), dtype=torch.float32)


def average_all_ids(t1, t2):
    """
    Average weights per concept id across BOTH inputs (union of ids).
    If an id appears in only one input, its average is computed from that input alone.

    Args:
        t1, t2: torch.Tensor or numpy.ndarray of shape (N,2) or (1,N,2)
                Column 0 = concept id, Column 1 = weight
    Returns:
        torch.Tensor of shape (K, 2) with [id, avg_weight], sorted by id ascending.
    """
    # Convert numpy -> torch if needed
    if isinstance(t1, np.ndarray):
        t1 = torch.from_numpy(t1)
    if isinstance(t2, np.ndarray):
        t2 = torch.from_numpy(t2)

    # Handle empty inputs gracefully
    if t1.numel() == 0 and t2.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32)

    # Flatten to (N,2)
    t1 = t1.reshape(-1, 2) if t1.numel() else torch.empty((0, 2))
    t2 = t2.reshape(-1, 2) if t2.numel() else torch.empty((0, 2))

    # Concatenate rows from both
    cat = torch.cat([t1, t2], dim=0)

    # Split ids/vals and make sure dtypes are right
    ids  = cat[:, 0].to(torch.int64)
    vals = cat[:, 1].to(torch.float32)

    # Group by id using unique + inverse index
    unique_ids, inv = torch.unique(ids, return_inverse=True)

    sums   = torch.zeros(unique_ids.size(0), dtype=torch.float32, device=cat.device)
    sums.index_add_(0, inv, vals)

    counts = torch.bincount(inv, minlength=unique_ids.size(0)).to(torch.float32)

    avgs = sums / counts.clamp_min(1)

    # Pack as (K,2): [id, avg], sort by id
    out = torch.stack([unique_ids.to(torch.float32), avgs], dim=1)
    out = out[torch.argsort(out[:, 0])]

    return out


def average_all_ids_2(t1, t2):
    """
    Average weights per concept id across BOTH inputs (union of ids).
    If an id appears in only one input, its average is computed as if
    the other input had a value of 0 for that id.

    Args:
        t1, t2: torch.Tensor or numpy.ndarray of shape (N,2) or (1,N,2)
                Column 0 = concept id, Column 1 = weight
    Returns:
        torch.Tensor of shape (K, 2) with [id, avg_weight], sorted by id ascending.
    """
    # Convert numpy -> torch if needed
    if isinstance(t1, np.ndarray):
        t1 = torch.from_numpy(t1)
    if isinstance(t2, np.ndarray):
        t2 = torch.from_numpy(t2)

    # Handle empty inputs gracefully
    if t1.numel() == 0 and t2.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32)

    # Flatten to (N,2)
    t1 = t1.reshape(-1, 2) if t1.numel() else torch.empty((0, 2))
    t2 = t2.reshape(-1, 2) if t2.numel() else torch.empty((0, 2))

    # Concatenate rows from both
    cat = torch.cat([t1, t2], dim=0)

    # Split ids/vals and make sure dtypes are right
    ids  = cat[:, 0].to(torch.int64)
    vals = cat[:, 1].to(torch.float32)

    # Group by id using unique + inverse index
    unique_ids, inv = torch.unique(ids, return_inverse=True)

    sums = torch.zeros(unique_ids.size(0), dtype=torch.float32, device=cat.device)
    sums.index_add_(0, inv, vals)

    # Always divide by 2 (treating missing values as 0)
    avgs = sums / 2.0

    # Pack as (K,2): [id, avg], sort by id
    out = torch.stack([unique_ids.to(torch.float32), avgs], dim=1)
    out = out[torch.argsort(out[:, 0])]

    return out

def generalized_mean_ids(t1, t2, p: float = 1.0):
    """
    Compute the generalized mean (power mean) of weights grouped by concept id
    across the union of ids in both tensors.

    Args:
        t1, t2: torch.Tensor or numpy.ndarray of shape (N,2) or (1,N,2)
                Column 0 = concept id, Column 1 = weight
        p: float, order of the generalized mean.
           p=1 -> arithmetic mean
           p=0 -> geometric mean
           p=-1 -> harmonic mean
           p>1 -> higher-order means, etc.

    Returns:
        torch.Tensor of shape (K,2) with [id, generalized_mean],
        sorted by id.
    """
    # Convert numpy -> torch if needed
    if isinstance(t1, np.ndarray):
        t1 = torch.from_numpy(t1)
    if isinstance(t2, np.ndarray):
        t2 = torch.from_numpy(t2)

    # Flatten to (N,2)
    t1 = t1.reshape(-1, 2)
    t2 = t2.reshape(-1, 2)

    # Concatenate both
    cat = torch.cat([t1, t2], dim=0)

    ids = cat[:, 0].long()
    vals = cat[:, 1].float()

    unique_ids, inv = torch.unique(ids, return_inverse=True)

    results = []
    for i, cid in enumerate(unique_ids):
        x = vals[inv == i]

        if p == 0:  # geometric mean
            gm = torch.exp(torch.mean(torch.log(x)))
            results.append([cid.item(), gm.item()])
        else:
            gm = (torch.mean(x ** p)) ** (1.0 / p)
            results.append([cid.item(), gm.item()])

    out = torch.tensor(results, dtype=torch.float32)
    out = out[torch.argsort(out[:, 0])]  # sort by ID
    return out


def calculate_averages(thisProcess, log_file, averages_folders, activations_folder, concept_pos_in_file, languages, inspect, inspect_concepts, inspect_folder, activation_function, average_function):

    try:

        if activation_function == '':
            out_dir = averages_folders
            in_dir = activations_folder
        else:
            out_dir = averages_folders + '/' + activation_function
            in_dir = activations_folder + '/' + activation_function

        out_dir_rep = out_dir

        # cycle through activations/activation types folders
        for act_dir in Path(in_dir).iterdir():
            if act_dir.is_dir():
                print(act_dir)
                out_dir = out_dir_rep + '/' + act_dir.name
                if os.path.exists(out_dir):
                    msg = f"⚠️ Folder already exists — deleting: {out_dir}"
                    print(msg); update_log(log_file, thisProcess + ": " + msg)
                    shutil.rmtree(out_dir)
                os.makedirs(out_dir, exist_ok=True)
                update_log(log_file, thisProcess + f": ✅ New folder created: {out_dir}")

                filenameArr = []

                for l in languages:
                    
                    in_dir2 = in_dir + '/' + act_dir.name

                    for filep in Path(in_dir2).iterdir():
                        
                        file = filep.name

                        if '.DS_Store' in file:
                            do_nothing = 1
                        elif '-' + l + '--' in file:
                            ffile = in_dir2 + '/' + file
                            ta = [l,ffile]
                            filenameArr.append(ta)
                        elif '-' + languages[0] + '--' not in file and '-' + languages[1] + '--' not in file:
                            ffile = in_dir2 + '/' + file
                            ta = ['en',ffile]
                            if ta not in filenameArr:
                                filenameArr.append(ta)

                print('loaded all filenames to array')

                eng = [item for item in filenameArr if item[0] == 'en']
                lang1 = [item for item in filenameArr if item[0] == languages[0]]
                lang2 = [item for item in filenameArr if item[0] == languages[1]]


                fec = 0
                for feX in eng:

                    fe = feX[1]
                    fec += 1
                    print(fec)
                    concept1 = fe.split('--')[concept_pos_in_file]

                    t1 = np.load(fe)

                    for f1X in lang1:

                        f1 = f1X[1]
                        concept2 = f1.split('--')[concept_pos_in_file]

                        same = False
                        #print(concept1 + " <> " + concept2)
                        if concept1 == concept2:
                            same = True

                        if same:

                            t2 = np.load(f1)

                            newFileName = f1.replace(".npy", "--avg-" + average_function + "--" + languages[0] + "-v-en.npy").replace(in_dir, out_dir_rep)

                            if Path(newFileName).exists():
                                print("File exists")
                            else:
                                if average_function == 'intersection_mean':
                                    avg_tensor = average_common_ids(t1,t2)
                                elif average_function == 'full_mean':
                                    avg_tensor = average_all_ids(t1,t2)
                                elif average_function == 'full_mean_2':
                                    avg_tensor = average_all_ids_2(t1,t2)
                                elif average_function == 'harmonic_mean':
                                    avg_tensor = generalized_mean_ids(t1, t2, p=-1) # harmonic mean
                                elif average_function == 'geometric_mean':
                                    avg_tensor = generalized_mean_ids(t1, t2, p=0) # geometric mean

                                np.save(newFileName, avg_tensor.numpy())

                                if inspect:
                                    for inspect_concept in inspect_concepts:
                                        ifilename = inspect_folder + '/inspect--' + inspect_concept + '.txt'
                                        ifilename2 = inspect_folder + '/inspect--tensors.csv'

                                        if inspect_concept == f1.split('--')[3]:
                                            with open(ifilename, 'a') as file:
                                                file.write("\n")
                                                file.write("=========================================================")
                                                file.write("\n")
                                                file.write("average tensor for " + activation_function + " (" + newFileName + "): " + "\n" )
                                                for snp in avg_tensor:
                                                    '''
                                                    if 'e+' in str(snp):
                                                        ccpt = int(str(snp).split(', ')[0].replace('tensor(','').replace('[',''))  
                                                    else:
                                                        ccpt = str(snp).split(', ')[0].replace('tensor(','').replace('.0000','')
                                                    actval = str(snp).split(', ')[1].replace(')','')
                                                    '''
                                                    val1, val2 = snp.tolist()
                                                    file.write(str(int(val1)).strip() + ', ' + str(int(val2)).strip())
                                                    file.write("\n")

                                            sourceCorpusLang = newFileName.split('/')[-1].split('--')[1]
                                            append_tensor_to_file_as_csv(ifilename2, 'average', sourceCorpusLang, inspect_concept, act_dir.name, activation_function, average_function, avg_tensor, '', '', '')

                    for f1X in lang2:

                        f1 = f1X[1]
                        concept2 = f1.split('--')[concept_pos_in_file]

                        same = False
                        if concept1 == concept2:
                            same = True

                        if same:

                            t2 = np.load(f1)

                            newFileName = f1.replace(".npy", "--avg-" + average_function + "--" + languages[1] + "-v-en.npy").replace(in_dir, out_dir_rep)
                            if Path(newFileName).exists():
                                print("File exists")
                            else:
                                if average_function == 'intersection_mean':
                                    avg_tensor = average_common_ids(t1,t2)
                                elif average_function == 'full_mean':
                                    avg_tensor = average_all_ids(t1,t2)
                                elif average_function == 'full_mean_2':
                                    avg_tensor = average_all_ids_2(t1,t2)
                                elif average_function == 'harmonic_mean':
                                    avg_tensor = generalized_mean_ids(t1, t2, p=-1) # harmonic mean
                                elif average_function == 'geometric_mean':
                                    avg_tensor = generalized_mean_ids(t1, t2, p=0) # geometric mean

                                np.save(newFileName, avg_tensor.numpy())

                                if inspect:
                                    for inspect_concept in inspect_concepts:
                                        ifilename = inspect_folder + '/inspect--' + inspect_concept + '.txt'

                                        if inspect_concept == f1.split('--')[3]:
                                            with open(ifilename, 'a') as file:
                                                file.write("\n")
                                                file.write("=========================================================")
                                                file.write("\n")
                                                file.write("average tensor for agg_top1 (" + newFileName + "): " + "\n" )
                                                for snp in avg_tensor:
                                                    val1, val2 = snp.tolist()
                                                    file.write(str(int(val1)).strip() + ', ' + str(int(val2)).strip())
                                                    #ccpt = str(snp).split('. ')[0]
                                                    #actval = str(snp).split('. ')[1]
                                                    #file.write(ccpt.strip() + ', ' + actval.strip())
                                                    file.write("\n")
                                            sourceCorpusLang = newFileName.split('/')[-1].split('--')[1]
                                            append_tensor_to_file_as_csv(ifilename2, 'average', sourceCorpusLang, inspect_concept, act_dir.name, activation_function, average_function, avg_tensor, '', '', '')


        update_log(log_file, thisProcess + ": end")
        print("Done.")

    except Exception as e:
        update_log(log_file, thisProcess + ": Exception = " + str(e))
        update_log(log_file, thisProcess + ": Exception = " + traceback.format_exc())
        print(e)
