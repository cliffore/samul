import json
from utils import *
import os
import shutil
from validation import *
from translation import *
from activations import *
from averages import *
from correlations import *
from similarities import *
from inspection import *


'''

This program runs a series of experiment steps to anaylyse text inputs via LLMs and Sparse Autoencoders.
Current constraints:
    - Base language is in a single file of csv format: Concept, Text
    - Ground truth file csv format is Source File, Concept 1, Concept 2, Similarity Value
    - The target language during development was English and is hard-coded to be this way
    - Two non-original languages are supported, the target non-Engoish languages during development were French and Chinese
    - The development environment was MacOS 15, but any Python environment should be supported, however some minor adjustments may be 
        necessary, e.g. folder strng names and/or Torch device (MPS was used for development)
    - huggingface usage of gemma scope requires a login and download of credentials to copy the models to a local environment
    - The type of activation calculation ('max-per-concept','mean-per-concept', 'sum-per-concept' and 'top-1-per-token') is set in 
        the config variable activation_function
    - The type of average calculations ('intersection_mean', 'full_mean', 'harmonic_mean' and 'geometric_mean') is set in 
        the config variable average_function
    - If testing a small corpus, be careful not to feed a large ground truth file because there may be a mismatch between the ground truth 
        counts and the number in the activations dataset leading to a crash. The best way around this is to include only ground truth 
        values that are in the corpus

    
Code Version 1.0
Latest Change: 04/01/2026

'''


# load config
config_file = "config.json"
with open(config_file, "r") as f:
    config = json.load(f)

process_validation = config["process_validation"]
process_translations = config["process_translations"]
process_activations = config["process_activations"]
process_averages = config["process_averages"]
process_similarities = config["process_similarities"]
process_correlations = config["process_correlations"]
experiment_folder = config["experiment_folder"]
experiment_id = config["experiment_id"]
log_file = config["log_file"]
delete_existing = config["delete_existing"]
inspect_bool = config["inspect"]
data_source_folder = config["data_source_folder"]
corpus = config["corpus"]
data_start_file = config["data_start_file"]
ground_truth_file = config["ground_truth_file"]
inspect_concepts = config["inspect_concepts"]
languages = config["languages"]
translated_source_folder = config["translated_source_folder"] + '_' + corpus
activations_source_folder = config["activations_source_folder"] + '_' + corpus
activations_layer_list = config["activations_layer_list"]
s_autoencoder = config["sparse_autoencoder"]
filerepo = config["filerepo"]
layer_locations = config[filerepo]
layer_sparsity_type = config["layer_sparsity_type"]
concept_pos_in_file = config["concept_pos_in_file"]
averages_source_folder = config["averages_source_folder"] + '_' + corpus
multi_row_concept_match = config["multi_row_concept_match"]
similarities_source_folder = config["similarities_source_folder"] + '_' + corpus
activation_function = config["activation_function"]
average_function = config["average_function"]
ground_truth_file = experiment_folder + '/' + data_source_folder + '/' + ground_truth_file
this_exp_folder = experiment_folder + '/ex' + experiment_id
inspect_folder = experiment_folder + '/ex' + experiment_id + '/inspect'
source_data_folder = experiment_folder + '/' + data_source_folder
translations_folder = experiment_folder + '/ex' + experiment_id + '/' + translated_source_folder
activations_folder = experiment_folder + '/ex' + experiment_id + '/' + activations_source_folder
averages_folder = experiment_folder + '/ex' + experiment_id + '/' + averages_source_folder
similarities_folder = experiment_folder + '/ex' + experiment_id + '/' + similarities_source_folder



# ============================= create experiment folder =============================M
if delete_existing:
    if os.path.exists(this_exp_folder):
        shutil.rmtree(this_exp_folder)
 
# create experiment folder
if not os.path.exists(this_exp_folder):
    os.makedirs(this_exp_folder)


# ============================= create log =============================
log_file = this_exp_folder + '/' + log_file
create_log('samul', log_file)


update_log(log_file, 'process_validation=' + str(process_validation))
update_log(log_file, 'process_translations=' + str(process_translations))
update_log(log_file, 'process_activations=' + str(process_activations))
update_log(log_file, 'process_averages=' + str(process_averages))
update_log(log_file, 'process_similarities=' + str(process_similarities))
update_log(log_file, 'process_correlations=' + str(process_correlations))
update_log(log_file, 'experiment_folder=' + experiment_folder)
update_log(log_file, 'experiment_id=' + str(experiment_id))
update_log(log_file, 'log_file=' + log_file)
update_log(log_file, 'delete_existing=' + str(delete_existing))
update_log(log_file, 'inspect_folder=' + inspect_folder)
update_log(log_file, 'data_source_folder=' + data_source_folder)
update_log(log_file, 'corpus=' + corpus)
update_log(log_file, 'data_start_file=' + data_start_file)
update_log(log_file, 'ground_truth_file=' + ground_truth_file)
update_log(log_file, 'inspect_concepts=' + str(inspect_concepts))
update_log(log_file, 'languages=' + str(languages))
update_log(log_file, 'filerepo=' + str(filerepo))
update_log(log_file, 'layer_sparsity_type=' + str(layer_sparsity_type))
update_log(log_file, 'translated_source_folder=' + translated_source_folder) 
update_log(log_file, 'activations_source_folder=' + activations_source_folder) 
update_log(log_file, 'activations_layer_list=' + str(activations_layer_list))
update_log(log_file, 'sparse_autoencoder=' + s_autoencoder)
update_log(log_file, 'layer_locations=' + str(layer_locations))
update_log(log_file, 'concept_pos_in_file=' + str(concept_pos_in_file))
update_log(log_file, 'averages_source_folder=' + averages_source_folder)
update_log(log_file, 'multi_row_concept_match=' + str(multi_row_concept_match))
update_log(log_file, 'similarities_source_folder=' + similarities_source_folder) 
update_log(log_file, 'activation_function=' + activation_function)
update_log(log_file, 'average_function=' + average_function)
update_log(log_file, 'ground_truth_file=' + ground_truth_file) 
update_log(log_file, 'this_exp_folder=' + this_exp_folder) 
update_log(log_file, 'inspect_folder=' + inspect_folder)
update_log(log_file, 'source_data_folder=' + source_data_folder) 
update_log(log_file, 'translations_folder=' + translations_folder) 
update_log(log_file, 'activations_folder=' + activations_folder) 
update_log(log_file, 'averages_folder=' + averages_folder) 
update_log(log_file, 'similarities_folder=' + similarities_folder) 



# ============================= validate data files =============================
if process_validation:

    thisProcess = "validation"

    msg = "starting process = " + thisProcess
    update_log(log_file, msg)


    data_source_file = source_data_folder + '/' + data_start_file
    returnVal = validate_source_file(data_source_file)
    update_log(log_file, returnVal)


    ground_truth_source_file = ground_truth_file
    returnVal = validate_ground_truth_file(ground_truth_source_file)
    update_log(log_file, returnVal)




# ============================= translations =============================
if process_translations:

    thisProcess = "translation"
    # ============================= update log =============================
    msg = "starting process = " + thisProcess
    update_log(log_file, msg)


    # ============================= create translated folder =============================
    if delete_existing:
        if os.path.exists(translations_folder):
            shutil.rmtree(translations_folder)

    if not os.path.exists(translations_folder):
        os.makedirs(translations_folder)

    for lang in languages:
        translate_from_source(lang, data_source_file, translations_folder, inspect_folder, log_file, delete_existing, inspect_bool, inspect_concepts)




# ============================= activations =============================
if process_activations:

    thisProcess = "activations"
    # ============================= update log =============================
    msg = "starting process = " + thisProcess
    update_log(log_file, msg)


    # ============================= create activations folder =============================
    if delete_existing:
        if os.path.exists(activations_folder):
            shutil.rmtree(activations_folder)

    if not os.path.exists(activations_folder):
        os.makedirs(activations_folder)

    # original language file
    calculate_activations(thisProcess, log_file, s_autoencoder, activations_layer_list, layer_locations, data_source_file, translations_folder, activations_folder, delete_existing, languages, inspect_bool, inspect_concepts, inspect_folder, activation_function, filerepo, layer_sparsity_type)

    # translated language files
    for file in Path(translations_folder).iterdir():
        if '.DS_Store' not in file.name:
            fullFile = translations_folder + '/' + file.name
            calculate_activations(thisProcess, log_file, s_autoencoder, activations_layer_list, layer_locations, fullFile, translations_folder, activations_folder, delete_existing, languages, inspect_bool, inspect_concepts, inspect_folder, activation_function, filerepo, layer_sparsity_type)




# ============================= averages =============================
if process_averages:

    thisProcess = "averages"
    # ============================= update log =============================
    msg = "starting process = " + thisProcess
    update_log(log_file, msg)


    # ============================= create averages folder =============================
    if delete_existing:
        if os.path.exists(averages_folder):
            shutil.rmtree(averages_folder)

    if not os.path.exists(averages_folder):
        os.makedirs(averages_folder)

    if s_autoencoder == 'openai/gpt2':
        activation_function = ''
    
    calculate_averages(thisProcess, log_file, averages_folder, activations_folder, concept_pos_in_file, languages, inspect_bool, inspect_concepts, inspect_folder, activation_function, average_function)


# ============================= differences =============================
if process_similarities:

    thisProcess = "similarities"
    # ============================= update log =============================
    msg = "starting process = " + thisProcess
    update_log(log_file, msg)


    # ============================= create differences folder =============================
    if delete_existing:
        if os.path.exists(similarities_folder):
            shutil.rmtree(similarities_folder)

    if not os.path.exists(similarities_folder):
        os.makedirs(similarities_folder)

    df = pd.read_csv(ground_truth_file, encoding="utf-8")
    gt = pd.to_numeric(df["value"], errors="coerce").to_numpy()
    mask = np.isfinite(gt)
    gt = gt[mask]

    binary_target = False
    if is_binary_array(gt, tol=0.0):
        binary_target = True
        msg = "Ground truth variable is binary. Using the cartesian method of difference calculation."
        update_log(log_file, msg)

    else:
        msg = "Ground truth variable is continuous. Using the non-cartesian method of difference calculation."
        update_log(log_file, msg)
    
    calculate_similarities(thisProcess, log_file, ground_truth_file, languages, concept_pos_in_file, inspect_bool, inspect_concepts, inspect_folder, similarities_folder, averages_folder, activations_folder, data_start_file, binary_target, average_function)


# ============================= correlations =============================
if process_correlations:

    thisProcess = "correlations"
    # ============================= update log =============================
    msg = "starting process = " + thisProcess
    update_log(log_file, msg)

    calculate_correlations(thisProcess, log_file, similarities_folder, this_exp_folder)


# ============================= inspect =============================
if inspect_bool:

    thisProcess = "inspect"
    # ============================= update log =============================
    msg = "starting process = " + thisProcess
    update_log(log_file, msg)


    if delete_existing:
        if os.path.exists(inspect_folder):
            shutil.rmtree(inspect_folder)

    # create experiment folder
    if not os.path.exists(inspect_folder):
        os.makedirs(inspect_folder)


    calculate_inspect(thisProcess, log_file, source_data_folder + '/' + data_start_file, translations_folder, ground_truth_file, activations_folder, averages_folder, similarities_folder, inspect_concepts, inspect_folder)

