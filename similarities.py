
import os
import re
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import euclidean
import torch
from pathlib import Path
from utils import *
import random
import random
from collections import defaultdict




def scan_folder_for_type(folder_path: str, file_extension: str) -> list[Path]:
    """
    Recursively scan folders until files of the specified type are found.
    
    Args:
        folder_path: Starting folder path
        file_extension: File extension to search for (e.g., '.txt', '.py')
    
    Returns:
        List of matching file paths
    """
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        print(f"Invalid folder: {folder_path}")
        return []
    
    # Normalize extension to include the dot
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension
    
    matching_files = []
    
    # Get folder contents, sorted so files come before folders
    contents = sorted(folder.iterdir(), key=lambda x: (x.is_dir(), x.name))
    
    if not contents:
        return []
    
    # Check if first item is a file of the target type
    first_item = contents[0]
    
    if first_item.is_file() and first_item.suffix.lower() == file_extension.lower():
        # Found target files - collect all matching files in this folder
        for item in contents:
            if item.is_file() and item.suffix.lower() == file_extension.lower():
                matching_files.append(item)
                #print(f"Found: {item}")
    else:
        # No matching files at this level - recurse into subfolders
        for item in contents:
            if item.is_dir():
                matching_files.extend(scan_folder_for_type(item, file_extension))
    
    return matching_files




def tensor_to_dense(tensor, vocab):
    dense = torch.zeros(len(vocab))
    id_to_index = {cid: i for i, cid in enumerate(vocab)}
    for row in tensor:
        cid = int(row[0].item())
        weight = row[1].item()
        dense[id_to_index[cid]] = weight
    return dense


def resize_vector(vec, target_length):
    if len(vec) == target_length:
        return vec
    elif len(vec) > target_length:
        return vec[:target_length]
    else:
        return np.pad(vec, (0, target_length - len(vec)))
    

def calculate_similarities(thisProcess, log_file, ground_truth_file, langs, concept_pos_in_file, similarities_folder, averages_folder, activations_folder, data_start_file, binary_target, average_function):    

    data_folder = averages_folder

    ground_truth = {}
    df_gt = pd.read_csv(ground_truth_file, header=1, names=["file", "concept1", "concept2", "value"])

    for _, row in df_gt.iterrows():
        c1 = str(row['concept1']).lower()
        c2 = str(row['concept2']).lower()
        ground_truth[(c1, c2)] = row["value"]
        ground_truth[(c2, c1)] = row["value"]  # make symmetric`

    ground_truth_length = int(len(ground_truth)/2)
    print(ground_truth_length)

    update_log(log_file, thisProcess + " loaded ground truth data from " + ground_truth_file)

    cnt = 0
    
    languages = langs
    #languages.append('avg')

    # for translated tensors
    for lang in languages:
        l = '-' + lang + '--'
        print(l)
        dataFolders = []
        if Path(data_folder).is_dir():
            npy_results = scan_folder_for_type(data_folder, ".npy")

            for npy in npy_results:
                thisPath = str(Path(npy).parent)
                if thisPath not in dataFolders:
                    dataFolders.append(thisPath)
                    
        for folder in dataFolders:

            concept_tensors = {}
            cct = 0
            fileForParameters = ""
            for fname in os.listdir(folder):
                if fname.endswith(".npy") and l.lower() in fname.lower() and average_function.lower() in fname.lower():
                    concept = str(fname.split('--')[concept_pos_in_file]).lower()
                    tensor = np.load(os.path.join(folder, fname))
                    concept_tensors[concept] = tensor

                    if cct == 0:
                        cct += 1
                        fileForParameters = fname

            update_log(log_file, thisProcess + " Loaded concept tensors")


            activation_method = 'unknown'

            if '--max-per-concept--' in fileForParameters:
                activation_method = 'max-per-concept'
            elif '--mean-per-concept--' in fileForParameters:
                activation_method = 'mean-per-concept'
            elif '--sum-per-concept--' in fileForParameters:
                activation_method = 'sum-per-concept'
            elif '--top-1-per-token--' in fileForParameters:
                activation_method = 'top-1-per-token'
            elif 'resid_post_mlp' in fileForParameters:
                activation_method = 'resid_post_mlp'

            sparsity = 'unknown'
            if 'average_l0_' in fileForParameters:
                sparsity = fileForParameters.split('average_l0_')[1].split('--')[0].replace('.csv', '')

            layer = 'unknown'
            if '__width_' in fileForParameters:
                layer = fileForParameters.split('layer_')[1].split('__width_16')[0]
            else:
                layer = fileForParameters.split('layer_')[1].split('-')[0]

            width = 'unknown'
            if '__width_' in fileForParameters:
                width = fileForParameters.split('__width_')[1].split('__average_l0')[0]
            elif '_v5_' in fileForParameters:
                width = fileForParameters.split('_v5_')[1].split('--')[0]


            results = []
            cnt = 0
            cnt_negs = 0
            trial_count = 10

            if binary_target:

                all_combinations = list(combinations(concept_tensors.items(), 2))

                # Group by first key
                groups = defaultdict(list)
                for combo in all_combinations:
                    groups[combo[0][0]].append(combo)

                # Shuffle within each group
                for key in groups:
                    random.shuffle(groups[key])

                # Interleave: round-robin through groups in random order
                keys = list(groups.keys())
                random.shuffle(keys)
                queues = {k: groups[k] for k in keys}

                all_combinations = []
                while queues:
                    for key in list(queues.keys()):
                        if queues[key]:
                            all_combinations.append(queues[key].pop(0))
                        else:
                            del queues[key]

                for (concept1, tensor1), (concept2, tensor2) in all_combinations:

                    if ground_truth.get((concept1, concept2),0) == 1:

                        cnt += 1
                        if cnt % 100 == 0:
                            print(cnt)

                        #if tensor1.shape != tensor2.shape:
                        #    target_len = max(len(t) for t in concept_tensors.values())
                        #    tensor1 = resize_vector(tensor1, target_len)
                        #    tensor2 = resize_vector(tensor2, target_len)
                        
                        # Suppose you have two tensors: tensorA and tensorB
                        # Build combined vocabulary
                        vocab = sorted(set(tensor1[:, 0].tolist()) | set(tensor2[:, 0].tolist()))

                        denseA = tensor_to_dense(tensor1, vocab)
                        denseB = tensor_to_dense(tensor2, vocab)


                        # Cosine similarity
                        cosine = torch.nn.functional.cosine_similarity(denseA.unsqueeze(0), denseB.unsqueeze(0)).item()

                        dist = cosine
                        #dist = euclidean(tensor1, tensor2)
                        c1 = concept1
                        c2 = concept2 
                        gt = "1"
                        results.append([c1, c2, dist, gt])

                    else:

                        #if cnt_negs < ground_truth_length * trial_count:
                        cnt += 1
                        if cnt % 100 == 0:
                            print(cnt)

                        #if tensor1.shape != tensor2.shape:
                        #    target_len = max(len(t) for t in concept_tensors.values())
                        #    tensor1 = resize_vector(tensor1, target_len)
                        #    tensor2 = resize_vector(tensor2, target_len)
                        
                        # Suppose you have two tensors: tensorA and tensorB
                        # Build combined vocabulary
                        vocab = sorted(set(tensor1[:, 0].tolist()) | set(tensor2[:, 0].tolist()))

                        denseA = tensor_to_dense(tensor1, vocab)
                        denseB = tensor_to_dense(tensor2, vocab)

                        # Cosine similarity
                        cosine = torch.nn.functional.cosine_similarity(denseA.unsqueeze(0), denseB.unsqueeze(0)).item()

                        dist = cosine
                        #dist = euclidean(tensor1, tensor2)
                        c1 = concept1
                        c2 = concept2
                        gt = "0"
                        results.append([c1, c2, dist, gt])

                        #cnt_negs += 1

            else:

                for (c1, c2) in ground_truth.keys():
                    
                    if c1 in concept_tensors and c2 in concept_tensors:
                        tensor1 = concept_tensors[c1]
                        tensor2 = concept_tensors[c2]
                        
                        cnt += 1
                        if cnt % 100 == 0:
                            print(cnt)

                        #if tensor1.shape != tensor2.shape:
                        #    target_len = max(len(t) for t in concept_tensors.values())
                        #    tensor1 = resize_vector(tensor1, target_len)
                        #    tensor2 = resize_vector(tensor2, target_len)
                        
                        # Suppose you have two tensors: tensorA and tensorB
                        # Build combined vocabulary
                        vocab = sorted(set(tensor1[:, 0].tolist()) | set(tensor2[:, 0].tolist()))

                        denseA = tensor_to_dense(tensor1, vocab)
                        denseB = tensor_to_dense(tensor2, vocab)

                        # Cosine similarity
                        cosine = torch.nn.functional.cosine_similarity(denseA.unsqueeze(0), denseB.unsqueeze(0)).item()

                        dist = cosine
                        #dist = euclidean(tensor1, tensor2)
                        gt = ground_truth.get((c1, c2), None)
                        results.append([c1, c2, dist, gt])


            ex_format = 'similarities'
            output_filename = os.path.join(similarities_folder, f"{ex_format}--{average_function}--{lang}--{activation_method}--layer_{layer}--width_{width}--sparsity_{sparsity}.csv")
            # Write to CSV
            output_df = pd.DataFrame(results, columns=["concept1", "concept2", "similarity", "ground_truth"])
            output_df.to_csv(output_filename, index=False)

            print(f"Saved average similarities and averages ground truth to: {output_filename}")
            update_log(log_file, thisProcess + f"Saved average similarities and averages ground truth to: {output_filename}")


    # for english only
    data_folder = activations_folder
    l = '-en--'
    
    print(l)
    dataFolders = []
    if Path(data_folder).is_dir():
        npy_results = scan_folder_for_type(data_folder, ".npy")

        for npy in npy_results:
            thisPath = str(Path(npy).parent)
            if thisPath not in dataFolders:
                dataFolders.append(thisPath)
                
    for folder in dataFolders:

        concept_tensors = {}
        cct = 0
        fileForParameters = ""
        for fname in os.listdir(folder):
            if fname.endswith(".npy") and l in fname:
                concept = str(fname.split('--')[concept_pos_in_file]).lower()
                tensor = np.load(os.path.join(folder, fname))
                concept_tensors[concept] = tensor

                if cct == 0:
                    cct += 1
                    fileForParameters = fname

        update_log(log_file, thisProcess + " Loaded concept tensors")


        activation_method = 'unknown'

        if 'max-per-concept' in fileForParameters:
            activation_method = 'max-per-concept'
        elif 'mean-per-concept' in fileForParameters:
            activation_method = 'mean-per-concept'
        elif 'sum-per-concept' in fileForParameters:
            activation_method = 'sum-per-concept'
        elif 'top-1-per-token' in fileForParameters:
            activation_method = 'top-1-per-token'
        elif 'resid_post_mlp' in fileForParameters:
            activation_method = 'resid_post_mlp'

        sparsity = 'unknown'
        if 'average_l0_' in fileForParameters:
            sparsity = fileForParameters.split('average_l0_')[1].split('--')[0].replace('.csv', '')

        layer = 'unknown'
        if '__width_' in fileForParameters:
            layer = fileForParameters.split('layer_')[1].split('__width_16')[0]
        else:
            layer = fileForParameters.split('layer_')[1].split('-')[0]

        width = 'unknown'
        if '__width_' in fileForParameters:
            width = fileForParameters.split('__width_')[1].split('__average_l0')[0]
        elif '_v5_' in fileForParameters:
            width = fileForParameters.split('_v5_')[1].split('--')[0]


        results = []
        cnt = 0
        cnt_negs = 0

        if binary_target:

            all_combinations = list(combinations(concept_tensors.items(), 2))

            # Group by first key
            groups = defaultdict(list)
            for combo in all_combinations:
                groups[combo[0][0]].append(combo)

            # Shuffle within each group
            for key in groups:
                random.shuffle(groups[key])

            # Interleave: round-robin through groups in random order
            keys = list(groups.keys())
            random.shuffle(keys)
            queues = {k: groups[k] for k in keys}

            all_combinations = []
            while queues:
                for key in list(queues.keys()):
                    if queues[key]:
                        all_combinations.append(queues[key].pop(0))
                    else:
                        del queues[key]


            for (concept1, tensor1), (concept2, tensor2) in all_combinations:

                if ground_truth.get((concept1, concept2),0) == 1:

                    cnt += 1
                    if cnt % 100 == 0:
                        print(cnt)

                    #if tensor1.shape != tensor2.shape:
                    #    target_len = max(len(t) for t in concept_tensors.values())
                    #    tensor1 = resize_vector(tensor1, target_len)
                    #    tensor2 = resize_vector(tensor2, target_len)
                    
                    # Suppose you have two tensors: tensorA and tensorB
                    # Build combined vocabulary
                    vocab = sorted(set(tensor1[:, 0].tolist()) | set(tensor2[:, 0].tolist()))

                    denseA = tensor_to_dense(tensor1, vocab)
                    denseB = tensor_to_dense(tensor2, vocab)


                    # Cosine similarity
                    cosine = torch.nn.functional.cosine_similarity(denseA.unsqueeze(0), denseB.unsqueeze(0)).item()

                    dist = cosine
                    #dist = euclidean(tensor1, tensor2)
                    c1 = concept1 
                    c2 = concept2 
                    gt = "1"
                    results.append([c1, c2, dist, gt])

                else:

                    #if cnt_negs < ground_truth_length * trial_count:
                    cnt += 1
                    if cnt % 100 == 0:
                        print(cnt)

                    #if tensor1.shape != tensor2.shape:
                    #    target_len = max(len(t) for t in concept_tensors.values())
                    #    tensor1 = resize_vector(tensor1, target_len)
                    #    tensor2 = resize_vector(tensor2, target_len)
                    
                    # Suppose you have two tensors: tensorA and tensorB
                    # Build combined vocabulary
                    vocab = sorted(set(tensor1[:, 0].tolist()) | set(tensor2[:, 0].tolist()))

                    denseA = tensor_to_dense(tensor1, vocab)
                    denseB = tensor_to_dense(tensor2, vocab)

                    # Cosine similarity
                    cosine = torch.nn.functional.cosine_similarity(denseA.unsqueeze(0), denseB.unsqueeze(0)).item()

                    dist = cosine
                    #dist = euclidean(tensor1, tensor2)
                    c1 = concept1
                    c2 = concept2
                    gt = "0"
                    results.append([c1, c2, dist, gt])

                    #cnt_negs += 1

        else:

            for (c1, c2) in ground_truth.keys():
                
                if c1 in concept_tensors and c2 in concept_tensors:
                    tensor1 = concept_tensors[c1]
                    tensor2 = concept_tensors[c2]
                    
                    cnt += 1
                    if cnt % 100 == 0:
                        print(cnt)

                    #if tensor1.shape != tensor2.shape:
                    #    target_len = max(len(t) for t in concept_tensors.values())
                    #    tensor1 = resize_vector(tensor1, target_len)
                    #    tensor2 = resize_vector(tensor2, target_len)
                    
                    # Suppose you have two tensors: tensorA and tensorB
                    # Build combined vocabulary
                    vocab = sorted(set(tensor1[:, 0].tolist()) | set(tensor2[:, 0].tolist()))

                    denseA = tensor_to_dense(tensor1, vocab)
                    denseB = tensor_to_dense(tensor2, vocab)

                    # Cosine similarity
                    cosine = torch.nn.functional.cosine_similarity(denseA.unsqueeze(0), denseB.unsqueeze(0)).item()

                    dist = cosine
                    #dist = euclidean(tensor1, tensor2)
                    gt = ground_truth.get((c1, c2), None)
                    results.append([c1, c2, dist, gt])


        ex_format = 'similarities'
        output_filename = os.path.join(similarities_folder, f"{ex_format}-{l}{activation_method}--layer_{layer}--width_{width}--sparsity_{sparsity}.csv")
        # Write to CSV
        output_df = pd.DataFrame(results, columns=["concept1", "concept2", "similarity", "ground_truth"])
        output_df.to_csv(output_filename, index=False)

        print(f"Saved average similarities and averages ground truth to: {output_filename}")
        update_log(log_file, thisProcess + f"Saved average similarities and averages ground truth to: {output_filename}")



    # for translated language only
    for lang in langs:

        if lang != 'avg':

            data_folder = activations_folder
            l = '-' + lang + '--'
            
            print(l)
            dataFolders = []
            if Path(data_folder).is_dir():
                npy_results = scan_folder_for_type(data_folder, ".npy")

                for npy in npy_results:
                    thisPath = str(Path(npy).parent)
                    if thisPath not in dataFolders:
                        dataFolders.append(thisPath)
                        
            for folder in dataFolders:

                concept_tensors = {}
                cct = 0
                fileForParameters = ""
                for fname in os.listdir(folder):
                    if fname.endswith(".npy") and l in fname:
                        concept = str(fname.split('--')[concept_pos_in_file]).lower()
                        tensor = np.load(os.path.join(folder, fname))
                        concept_tensors[concept] = tensor

                        if cct == 0:
                            cct += 1
                            fileForParameters = fname

                update_log(log_file, thisProcess + " Loaded concept tensors")


                activation_method = 'unknown'

                if 'max-per-concept' in fileForParameters:
                    activation_method = 'max-per-concept'
                elif 'mean-per-concept' in fileForParameters:
                    activation_method = 'mean-per-concept'
                elif 'sum-per-concept' in fileForParameters:
                    activation_method = 'sum-per-concept'
                elif 'top-1-per-token' in fileForParameters:
                    activation_method = 'top-1-per-token'
                elif 'resid_post_mlp' in fileForParameters:
                    activation_method = 'resid_post_mlp'

                sparsity = 'unknown'
                if 'average_l0_' in fileForParameters:
                    sparsity = fileForParameters.split('average_l0_')[1].split('--')[0].replace('.csv', '')

                layer = 'unknown'
                if '__width_' in fileForParameters:
                    layer = fileForParameters.split('layer_')[1].split('__width_16')[0]
                else:
                    layer = fileForParameters.split('layer_')[1].split('-')[0]

                width = 'unknown'
                if '__width_' in fileForParameters:
                    width = fileForParameters.split('__width_')[1].split('__average_l0')[0]
                elif '_v5_' in fileForParameters:
                    width = fileForParameters.split('_v5_')[1].split('--')[0]


                results = []
                cnt = 0
                cnt_negs = 0

                if binary_target:

                    all_combinations = list(combinations(concept_tensors.items(), 2))

                    # Group by first key
                    groups = defaultdict(list)
                    for combo in all_combinations:
                        groups[combo[0][0]].append(combo)

                    # Shuffle within each group
                    for key in groups:
                        random.shuffle(groups[key])

                    # Interleave: round-robin through groups in random order
                    keys = list(groups.keys())
                    random.shuffle(keys)
                    queues = {k: groups[k] for k in keys}

                    all_combinations = []
                    while queues:
                        for key in list(queues.keys()):
                            if queues[key]:
                                all_combinations.append(queues[key].pop(0))
                            else:
                                del queues[key]


                    for (concept1, tensor1), (concept2, tensor2) in all_combinations:

                        if ground_truth.get((concept1, concept2),0) == 1:

                            cnt += 1
                            if cnt % 100 == 0:
                                print(cnt)

                            #if tensor1.shape != tensor2.shape:
                            #    target_len = max(len(t) for t in concept_tensors.values())
                            #    tensor1 = resize_vector(tensor1, target_len)
                            #    tensor2 = resize_vector(tensor2, target_len)
                            
                            # Suppose you have two tensors: tensorA and tensorB
                            # Build combined vocabulary
                            vocab = sorted(set(tensor1[:, 0].tolist()) | set(tensor2[:, 0].tolist()))

                            denseA = tensor_to_dense(tensor1, vocab)
                            denseB = tensor_to_dense(tensor2, vocab)


                            # Cosine similarity
                            cosine = torch.nn.functional.cosine_similarity(denseA.unsqueeze(0), denseB.unsqueeze(0)).item()

                            dist = cosine
                            #dist = euclidean(tensor1, tensor2)
                            c1 = concept1 
                            c2 = concept2 
                            gt = "1"
                            results.append([c1, c2, dist, gt])

                        else:

                            #if cnt_negs < ground_truth_length * trial_count:
                            cnt += 1
                            if cnt % 100 == 0:
                                print(cnt)

                            #if tensor1.shape != tensor2.shape:
                            #    target_len = max(len(t) for t in concept_tensors.values())
                            #    tensor1 = resize_vector(tensor1, target_len)
                            #    tensor2 = resize_vector(tensor2, target_len)
                            
                            # Suppose you have two tensors: tensorA and tensorB
                            # Build combined vocabulary
                            vocab = sorted(set(tensor1[:, 0].tolist()) | set(tensor2[:, 0].tolist()))

                            denseA = tensor_to_dense(tensor1, vocab)
                            denseB = tensor_to_dense(tensor2, vocab)

                            # Cosine similarity
                            cosine = torch.nn.functional.cosine_similarity(denseA.unsqueeze(0), denseB.unsqueeze(0)).item()

                            dist = cosine
                            #dist = euclidean(tensor1, tensor2)
                            c1 = concept1
                            c2 = concept2
                            gt = "0"
                            results.append([c1, c2, dist, gt])

                            #cnt_negs += 1

                else:

                    for (c1, c2) in ground_truth.keys():
                        
                        if c1 in concept_tensors and c2 in concept_tensors:
                            tensor1 = concept_tensors[c1]
                            tensor2 = concept_tensors[c2]
                            
                            cnt += 1
                            if cnt % 100 == 0:
                                print(cnt)

                            #if tensor1.shape != tensor2.shape:
                            #    target_len = max(len(t) for t in concept_tensors.values())
                            #    tensor1 = resize_vector(tensor1, target_len)
                            #    tensor2 = resize_vector(tensor2, target_len)
                            
                            # Suppose you have two tensors: tensorA and tensorB
                            # Build combined vocabulary
                            vocab = sorted(set(tensor1[:, 0].tolist()) | set(tensor2[:, 0].tolist()))

                            denseA = tensor_to_dense(tensor1, vocab)
                            denseB = tensor_to_dense(tensor2, vocab)

                            # Cosine similarity
                            cosine = torch.nn.functional.cosine_similarity(denseA.unsqueeze(0), denseB.unsqueeze(0)).item()

                            dist = cosine
                            #dist = euclidean(tensor1, tensor2)
                            gt = ground_truth.get((c1, c2), None)
                            results.append([c1, c2, dist, gt])


                ex_format = 'similarities'
                output_filename = os.path.join(similarities_folder, f"{ex_format}-{l}{activation_method}--layer_{layer}--width_{width}--sparsity_{sparsity}.csv")
                # Write to CSV
                output_df = pd.DataFrame(results, columns=["concept1", "concept2", "similarity", "ground_truth"])
                output_df.to_csv(output_filename, index=False)

                print(f"Saved average similarities and averages ground truth to: {output_filename}")
                update_log(log_file, thisProcess + f"Saved average similarities and averages ground truth to: {output_filename}")
