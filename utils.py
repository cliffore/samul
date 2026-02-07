
import datetime
from pathlib import Path
import numpy as np
import datetime
from pathlib import Path
import pandas as pd


def create_log(thisProcess, log_file):
    print(log_file + " doesn't exist.")
    print(f"Attempting to write to: {log_file}")
    print(f"Parent exists: {Path(log_file).parent.exists()}")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #line = f"[{timestamp}] {thisProcess + ": start\n"}\n"
    line = "[" + timestamp + " " + thisProcess + ": start\n\n"
    with open(log_file, 'w') as file:
        file.write(line)


def update_log(file_path: str, message: str):
    """Append a timestamped message to a log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(line)


def is_binary_array(x: np.ndarray, tol: float = 0.0) -> bool:
    """
    True if all finite values are in {0,1} (within tol).
    """
    if x.size == 0:
        return False
    x = x[np.isfinite(x)]
    if x.size == 0:
        return False
    uniq = np.unique(x)
    if tol == 0.0:
        return set(uniq.tolist()).issubset({0.0, 1.0})
    return all((abs(v-0.0) <= tol) or (abs(v-1.0) <= tol) for v in uniq)


def createTensorFile(filename):
    write_text = "tensorValType, sourceCorpusLang, concept, layerLoc, activationScheme, averageAlg, elementCount, conceptId, activationVal, conceptCompareDiff, distance, target\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(write_text)


def append_tensor_to_file_as_csv(filename, tensorValType, sourceCorpusLang, concept, layerLoc, activationScheme, averageAlg, tensor, conceptCompareDiff, distance, target):

    write_text = ""
    ct = 0

    if len(tensor) == 0:
        write_text = write_text + tensorValType + ',' + sourceCorpusLang + ',' + str(concept) + ',' + layerLoc.replace('__params.npz','') + ',' + activationScheme  + ',' + averageAlg + ',,,,' + conceptCompareDiff + ',' + str(distance) + ',' +  str(target) + '\n'
    else:
        for t in tensor:
            ct += 1
            # need to cleaup the data by removing various characters.
            nt = str(t).replace('(','').replace('[','').replace(')','').replace(',','').replace(']','').replace('tensor','').split(' ')
            if 'e+' in str(nt[0]):
                conceptId = str(int(float(str(nt[0])))).replace('tensor(','').replace('.0000','').replace(',','')
            else:
                conceptId = str(nt[0]).replace('.0000','').replace('.','').replace('tensor(','').replace(',','')
            if 'e+' in str(nt[len(nt)-1]) :
                activationValue = str(float(str(nt[len(nt)-1]))).replace(')','')
            else:
                activationValue = str(nt[len(nt)-1]).replace(')','')
            write_text = write_text + tensorValType + ',' + sourceCorpusLang + ',' + str(concept) + ',' + layerLoc.replace('__params.npz','') + ',' + activationScheme  + ',' + averageAlg + ',' + str(ct) + ',' + conceptId + ',' +  activationValue + ',' + conceptCompareDiff + ',' + str(distance) + ',' +  str(target) + '\n'
            #print(write_text)

    with open(filename, "a", encoding="utf-8") as f:
        f.write(write_text)


def lower_case_attribute(currentFilename, columnName, newFilename):
    print('start')
    df = pd.read_csv(currentFilename)

    for index, row in df.iterrows():
        df.at[index, columnName] = row[columnName].lower()

    df.to_csv(newFilename, index=False)


def read_npy(filename):
    t1 = np.load(filename)

    for t in t1:
        print(t)


def print_tensor_nicely(t):
    for row in t:
        print(f"[{int(row[0])}, {row[1]:.9f}]")

def return_tensor_nicely(t):
    r = ""
    for row in t:
        r = r + f"[{int(row[0])}, {row[1]:.9f}]" + "\n"
    return r

# reformat a column in a csv file to be lowercase

#source_data_file = '/Users/cliff/Documents/PhD/dev/samul_v2/source_data/ontology_classes_summary_full-fr.csv'
#new_source_data_file = '/Users/cliff/Documents/PhD/dev/samul_v2/source_data/ontology_classes_summary_full-fr-2.csv'
#lower_case_attribute(source_data_file, 'concept', new_source_data_file)


# read a npy file tensor and print to screen

#folder = '/Users/Shared/Documents/PhD/dev/year3/pramantha/samul_v1/ex1/activations_oaei/top-1-per-token/layer_2__width_16k__average_l0_13__params.npz'
#filename = folder + '/' + 'google_gemma-2-2b--ontology_classes_summary_full_2-en--467--cmt-chairman--layer_2__width_16k__average_l0_13--agg_top1.npy'
#read_npy(filename)


# folder = '/Users/Shared/Documents/PhD/dev/year2/pramantha/millms_old9 copy/data-processing/experiments/ex_32/data'
# summary--9--confOf-Conference--layer_11-width_16k-average_l0_22-params.npz.npy
# summary-fr--9--confOf-Conference--layer_11-width_16k-average_l0_22-params.npz.npy - 1 extra token?! 
# weighted_avg--fve--summary--9--confOf-Conference--layer_11-width_16k-average_l0_22-params.npz.npy

# folder = '/Users/Shared/Documents/PhD/dev/year2/pramantha/millms_old9 copy/data-processing/experiments/ex_69/data'
# summary-zh--9--confOf-Conference--layer_11-width_16k-average_l0_22-params.npz.npy
# weighted_avg--fve--summary--9--confOf-Conference--layer_11-width_16k-average_l0_22-params.npz.npy


# cmt-chairman	conference-chair	0.109621279


