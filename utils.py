
import datetime
from pathlib import Path
import numpy as np
import datetime
from pathlib import Path
import pandas as pd
import os

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


def renameFiles(folder, from_str, to_str):
    for filename in os.listdir(folder):
        if from_str in filename:
            old_path = os.path.join(folder, filename)
            new_filename = filename.replace(from_str, to_str)
            new_path = os.path.join(folder, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")


