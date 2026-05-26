from pathlib import Path
import csv
from utils import *
import traceback
from similarities import *
from averages import *


def doCosine(tensor1, tensor2):

    vocab = sorted(set(tensor1[:, 0].tolist()) | set(tensor2[:, 0].tolist()))
    denseA = tensor_to_dense(tensor1, vocab)
    denseB = tensor_to_dense(tensor2, vocab)
    cosine = torch.nn.functional.cosine_similarity(denseA.unsqueeze(0), denseB.unsqueeze(0)).item()

    return cosine

def calculate_inspect(thisProcess, log_file, data_start_file, translations_folder, ground_truth_file, activations_folder, averages_folder, similarities_folder, inspect_concepts, inspect_folder, languages):

    try:

        print("started Inspect")

        mainOut = ("\n"*4) 
        separator = ("\n"*4) + ("+"*30) + ("+"*30) +  ("\n"*4)
        
        textOriginal = []
        textTrans1 = []
        textTrans2 = []
        groundTruth = []
        baseActivations = {}
        averages = {}

        # get text
        mainOut = mainOut + "Source text:\n"
        with open(data_start_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for line_number, row in enumerate(reader, start=1):
                concept = row[0]
                for ci in inspect_concepts:
                    if concept.lower() == ci.lower():
                        print(row)
                        mainOut = mainOut + str(row) + "\n"
                        textOriginal.append(row)


        mainOut = mainOut + separator
        mainOut = mainOut + "Translated text:\n"
        lang = 0
        for filep in Path(translations_folder).iterdir():
            lang += 1
            file = translations_folder + '/' + filep.name
            with open(file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for line_number, row in enumerate(reader, start=1):
                    concept = row[0]
                    for ci in inspect_concepts:
                        if concept.lower() == ci.lower():
                            print(row)
                            mainOut = mainOut + str(row) + "\n"
                            if lang == 1:
                                textTrans1.append(row)
                            else:
                                textTrans2.append(row)
                mainOut = mainOut + "\n"

        mainOut = mainOut + separator

        # get ground truth relations
        mainOut = mainOut + "Ground truth similarity scores:\n"
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for line_number, row in enumerate(reader, start=1):
                concept = row[1]
                for ci in inspect_concepts:
                    if concept.lower() == ci.lower():
                        print(row)
                        mainOut = mainOut + str(row) + "\n"
                        groundTruth.append(row)
                concept2 = row[2]
                for ci in inspect_concepts:
                    if concept2.lower() == ci.lower():
                        print(row)
                        mainOut = mainOut + str(row) + "\n"
                        groundTruth.append(row)


        mainOut = mainOut + separator


        # get similarities
        mainOut = mainOut + "Similarity scores:\n"
        for filep in Path(similarities_folder).iterdir():
            with open(similarities_folder + '/' + filep.name, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for line_number, row in enumerate(reader, start=1):
                    for ci in inspect_concepts:
                        if ci.lower() == row[0].lower():
                            print(row)
                            mainOut = mainOut + filep.name + "," + str(row).replace('[','').replace(']','') + "\n"
                        if ci.lower() == row[1].lower():
                            print(filep.name)
                            print(row)
                            mainOut = mainOut + filep.name + "," + str(row).replace('[','').replace(']','') + "\n"


        mainOut = mainOut + separator


        # get activations
        mainOut = mainOut + "Base activations:\n"
        npy_results = scan_folder_for_type(activations_folder, ".npy")

        for ci in inspect_concepts:
            for npy in npy_results:
                thisPath = str(Path(npy))
                thisFile = thisPath.split('/')[-1]
                if '--' + ci.lower() + '--' in thisFile.lower():
                    print(thisFile)
                    mainOut = mainOut + thisFile + "\n"
                    t1 = np.load(thisPath)
                    tensor1 = torch.tensor(t1)
                    r = return_tensor_nicely(tensor1)
                    mainOut = mainOut + r + "\n"
                    baseActivations[thisFile] = tensor1


        mainOut = mainOut + separator

        # get averages
        mainOut = mainOut + "Averages of activations:\n"
        npy_results = scan_folder_for_type(averages_folder, ".npy")

        for ci in inspect_concepts:
            for npy in npy_results:
                thisPath = str(Path(npy))
                thisFile = thisPath.split('/')[-1]
                if '--' + ci.lower() + '--' in thisFile.lower():
                    print(thisFile)
                    mainOut = mainOut + thisFile + "\n"
                    t1 = np.load(thisPath)
                    tensor1 = torch.tensor(t1)
                    r = return_tensor_nicely(tensor1)
                    mainOut = mainOut + r + "\n"
                    averages[thisFile] = tensor1

 
        
        # validate calculations

        for l in languages:

            for i in baseActivations:

                if '-' + l + '--' in i:
                    concep = i.split('--')[3]
                    params = i.split('--')[4]
                    print(i)
                    mainOut = mainOut + "Compare average of" + i + "\n"
                    #print_tensor_nicely(baseActivations[i])
                    for k in baseActivations:
                        if '-en--' in k and '--' + concep + '--' in k and '--' + params + '--' in k:
                            print(k)
                            mainOut = mainOut + "... and  " + k + "\n"
                            #print_tensor_nicely(baseActivations[k])
                            #print_tensor_nicely(average_all_ids(baseActivations[i], baseActivations[k]))

                            for ai in averages:
                                if '-' + l + '--' in ai and '--' + concep + '--' in ai and '--' + params + '--' in ai:
                                    print(ai)
                                    #print_tensor_nicely(averages[ai])
                                    result = torch.equal(average_all_ids(baseActivations[i], baseActivations[k]), averages[ai]) 
                                    print(result)
                                    mainOut = mainOut + "... with  " + ai + "\n"
                                    mainOut = mainOut + "... is " + str(result) + "\n"

        for i in groundTruth:
                print(i)
                for l in languages:
                    for k in averages:
                        if '-' + l + '--' in k and '--' + i[1].lower() + '--' in k:
                            print(k)
                            concep = k.split('--')[3]
                            params = k.split('--')[4]
                            for j in averages:
                                if '-' + l + '--' in j and '--' + i[2].lower() + '--' in j and '--' + params + '--' in j:
                                    print(j)
                                    print(doCosine(averages[k], averages[j]))

                for k in baseActivations:
                    if '-en--' in k and '--' + i[1].lower() + '--' in k:
                        print(k)
                        concep = k.split('--')[3]
                        params = k.split('--')[4]
                        for j in baseActivations:
                            if '-en--' in j and '--' + i[2].lower() + '--' in j and '--' + params + '--' in j:
                                print(j)
                                print(doCosine(baseActivations[k], baseActivations[j]))



        outFile = inspect_folder + "/inspect-report.txt"
        with open(outFile, "w", encoding="utf-8") as f:
            f.write(mainOut)

        update_log(log_file, thisProcess + " saved data file to " + outFile)


        update_log(log_file, thisProcess + ": end")
        print("Done.")

    except Exception as e:
        update_log(log_file, thisProcess + ": Exception = " + str(e))
        update_log(log_file, thisProcess + ": Exception = " + traceback.format_exc())
        print("ERROR:", e)




