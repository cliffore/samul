from googletrans import Translator
import pandas as pd
import asyncio
import traceback
from utils import *
from pathlib import Path


def translate_from_source(lang, source_data_file, translated_folder, inspect_folder, log_file, delete_existing, inspect, inspect_concepts):

    try:

        async def translate_word(word, lang):
            translator = Translator()
            try:
                translation = await translator.translate(word, src="en", dest=lang)
                return translation.text
            except Exception as e:
                msg = 'ERROR >> ' + word + ' FAILED'
                print(msg)
                update_log(log_file,  "translation error: " + msg)
                return 'ERROR >> ' + word + ' FAILED'


        df_prompts = pd.read_csv(source_data_file, header=0, names=['concept', 'prompt'])

        # Ensure order by resetting index
        df_prompts = df_prompts.reset_index()

        # Extract prompts and concepts in their original order
        concepts = df_prompts['concept'].tolist()
        prompts = df_prompts['prompt'].tolist()


        cn = 0
        output_t = "concept,prompt\n"

        for p in prompts:

            print(str(cn) + " ;; " + concepts[cn] + " ;; " + p)

            translated_word = asyncio.run(translate_word(p, lang))
            print(translated_word)
            if translated_word[0:9] != 'ERROR >> ':
                output_t = output_t + concepts[cn] + "," + translated_word.replace(","," ") + "\n"

                if inspect:
                    for inspect_concept in inspect_concepts:
                        #ifilename = inspect_folder + '/inspect--' + inspect_concept + '.txt'
                        ifilename2 = inspect_folder + '/inspect--prompts.txt'
                        
                        if inspect_concept == concepts[cn]:
                            #with open(ifilename, 'a') as file:
                            #    file.write("concept value (orig): " + p + "\n")
                            #    file.write("concept value (" + lang + "): " + translated_word + "\n")

                            with open(ifilename2, 'a') as file:
                                file.write("concept: " + str(inspect_concept) + ", value (orig): " + p + "\n")
                                file.write("concept: " + str(inspect_concept) + ", value (" + lang + "): " + translated_word + "\n")


            cn = cn + 1

            outfile = translated_folder + '/' + Path(source_data_file).name.replace(".csv", "-" + lang + ".csv")
            with open(outfile, 'w') as file:
                file.write(output_t)
            update_log(log_file, "Saved file as " + outfile)
            update_log(log_file, "Translated " + str(cn) + " prompts")


    except Exception as e:
        update_log(log_file, "Exception = " + str(e))
        update_log(log_file, "Exception = " + traceback.format_exc())
        print(e)
