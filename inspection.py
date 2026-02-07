from pathlib import Path
import csv
from utils import *
import traceback
from similarities import *



def calculate_inspect(thisProcess, log_file, data_start_file, translations_folder, ground_truth_file, activations_folder, averages_folder, similarities_folder, inspect_concepts, inspect_folder):

    try:

        print("started Inspect")

        mainOut = ("\n"*4) 

        separator = ("\n"*4) + ("+"*30) + ("+"*30) +  ("\n"*4)

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


        mainOut = mainOut + separator
        mainOut = mainOut + "Translated text:\n"
        for filep in Path(translations_folder).iterdir():
                        
            file = translations_folder + '/' + filep.name
            with open(file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for line_number, row in enumerate(reader, start=1):
                    concept = row[0]
                    for ci in inspect_concepts:
                        if concept.lower() == ci.lower():
                            print(row)
                            mainOut = mainOut + str(row) + "\n"
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
                concept2 = row[2]
                for ci in inspect_concepts:
                    if concept2.lower() == ci.lower():
                        print(row)
                        mainOut = mainOut + str(row) + "\n"


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
                    indices = torch.argsort(tensor1[:, 1], descending=True)
                    sorted_t = tensor1[indices]
                    r = return_tensor_nicely(sorted_t)
                    mainOut = mainOut + r + "\n"


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
                    indices = torch.argsort(tensor1[:, 1], descending=True)
                    sorted_t = tensor1[indices]
                    r = return_tensor_nicely(sorted_t)
                    mainOut = mainOut + r + "\n"


        outFile = inspect_folder + "/inspect-report.txt"
        with open(outFile, "w", encoding="utf-8") as f:
            f.write(mainOut)

        update_log(log_file, thisProcess + " saved data file to " + outFile)


        # get concept metadata

        





        update_log(log_file, thisProcess + ": end")
        print("Done.")

    except Exception as e:
        update_log(log_file, thisProcess + ": Exception = " + str(e))
        update_log(log_file, thisProcess + ": Exception = " + traceback.format_exc())
        print("ERROR:", e)

