from pathlib import Path
import csv


# =======================================
# is the source file in the right format?
# =======================================

def validate_source_file(source_file):
    test = 'data source file'
    column_count = 2
    invalid_rows = []

    with open(source_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        cn = 0
        for line_number, row in enumerate(reader, start=1):
            # Check if each row has exactly 2 columns
            if len(row) != column_count:
                invalid_rows.append((line_number, row))
            cn += 1

    returnValue = ""
    if invalid_rows:
        msg = "Test: " + test + ": Found " + str(len(invalid_rows)) + " invalid rows:\n"
        print(msg)
        returnValue = returnValue + msg
        for line_number, row in invalid_rows:
            print(f"Line {line_number}: {row}")
    else:
        msg = "Test: " + test + ": All rows have exactly " + str(column_count) + " columns."
        print(msg)
        returnValue = returnValue + msg
        returnValue = returnValue + ": row count = " + str(cn)


    return returnValue




# =============================================
# is the ground truth file in the right format?
# =============================================
def validate_ground_truth_file(ground_truth_source_file):
    test = 'ground truth source file'
    column_count = 4
    invalid_rows = []

    with open(ground_truth_source_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        cn = 0
        for line_number, row in enumerate(reader, start=1):
            # Check if each row has exactly 2 columns
            if len(row) != column_count:
                invalid_rows.append((line_number, row))
            cn += 1

    returnValue = ""
    if invalid_rows:
        msg = "Test: " + test + ": Found " + str(len(invalid_rows)) + " invalid rows\n"
        print(msg)
        returnValue =  returnValue + msg
        for line_number, row in invalid_rows:
            print(f"Line {line_number}: {row}")
    else:
        msg = "Test: " + test + ": All rows have exactly " + str(column_count) + " columns."
        print(msg)
        returnValue =  returnValue + msg
        returnValue = returnValue + ": row count = " + str(cn)

    return returnValue



# =============================================
# for each concept in the source files (data and 
# ground truth), make sure they all match and 
# there are no gaps
# =============================================



# =============================================
# for each concept and language, are the numbers
# of activations accurate, e.g. top1 should be 
# the smallest?
# =============================================


# =============================================
# for each concept, language and average type, 
# do the numbers match, e.g. for intersection 
# there are the same number of average concepts 
# as the concepts that intersect in the sources
# =============================================


# =============================================
# for each concept, language and average type, 
# do the averages look right? take a couple of 
# examples and re-calculate to check
# =============================================


