#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
import datetime
import json
import shutil
import traceback
from pathlib import Path
from utils import *
import blobfile as bf
import transformer_lens
import sparse_autoencoder


# ----------------------------
# Tunables
# ----------------------------
DEBUG_BUDGET = 8  # total times we allow printing per run
DEBUG_FACTOR = 2.0  # how far above the 99.5th percentile to flag
MAX_LENGTH = 256
BATCH_SIZE = 32
SAE_ROWS_PER_CHUNK = 8192  # safety; with B=32,T=256, BT=8192 → one chunk

# ----------------------------
# SAE module
# ----------------------------
class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        acts = torch.relu(pre_acts) * (pre_acts > self.threshold)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec


def sae_forward_chunked(hidden_bt: torch.Tensor, sae: JumpReLUSAE, rows_per_chunk: int = SAE_ROWS_PER_CHUNK):
    """hidden_bt: (B,T,D) → returns (sae_acts,recon) in same (B,T,*) shapes."""
    B, T, D = hidden_bt.shape
    X = hidden_bt.reshape(-1, D)  # (BT, D)
    parts = []
    with torch.inference_mode():
        for s in range(0, X.size(0), rows_per_chunk):
            Xi = X[s:s+rows_per_chunk]
            parts.append(sae.encode(Xi))
        sae_acts = torch.cat(parts, dim=0).reshape(B, T, -1)

        Y = sae_acts.reshape(-1, sae_acts.size(-1))
        rparts = []
        for s in range(0, Y.size(0), rows_per_chunk):
            Yi = Y[s:s+rows_per_chunk]
            rparts.append(sae.decode(Yi))
        recon = torch.cat(rparts, dim=0).reshape(B, T, D)
    return sae_acts, recon


def calculate_activations(thisProcess, log_file, this_autoencoder, layer_list, layer_locations, data_file, translated_source_folder, activations_source_folder, delete_existing, languages, inspect, inspect_concepts, inspect_folder, activation_function, filerepo, layer_sparsity_type):

    try:

        if this_autoencoder == "google/gemma-2-2b":

            print("Torch version:", torch.__version__)
            print("MPS available:", torch.backends.mps.is_available())
            print("MPS built:", torch.backends.mps.is_built())

            layer_to_use = layer_list
            locations_to_use = []
            main_layer_array = []

            for l in layer_to_use:
                for ull in layer_locations:
                    #for ull in layer_locations:
                    if ull["layer"] == l:
                        ta = (l, ull["type"], ull["value"])
                        main_layer_array.append(ta)

            print(main_layer_array)
            update_log(log_file, "main_layer_array: " + str(main_layer_array))


            # ---- Device / Model / Tokenizer (load once) ----
            device = torch.device("mps" if torch.backends.mps.is_available()
                                    else "cuda" if torch.cuda.is_available()
                                    else "cpu")

            sa_model = "google/gemma-2-2b"
            model = AutoModelForCausalLM.from_pretrained(
                sa_model,
                device_map=None,
                torch_dtype=torch.float32,  # MPS-safe
            ).to(device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

            # Make behavior consistent across languages and batches
            if hasattr(tokenizer, "padding_side"):
                tokenizer.padding_side = "left"   # or "right" if you prefer, just keep it fixed


            # ---- Pick SAE weights once ----
            api = HfApi()
            files = api.list_repo_files(filerepo)

            SAECount = 0
            for layer_array in main_layer_array:
                this_layer = layer_array[0]
                this_layer_type = layer_array[1]
                this_location = layer_array[2]

                if layer_sparsity_type == this_layer_type:
                    sae_file = ""
                    for fil in files:
                        if "layer_" in fil and "width_16k" in fil:
                            file_layer = fil.split("layer_")[1].split("/")[0]
                            if str(file_layer) == str(this_layer):
                                l0 = int(fil.split("average_l0_")[1].split("/")[0])
                                if l0 == this_location:
                                    sae_file = fil
                                    SAECount += 1

                                    update_log(log_file, thisProcess + ": sae file = " + sae_file)
                                    out_dir = activations_source_folder + '/' + activation_function + '/' + sae_file.replace('/','__')

                                    if os.path.exists(out_dir):
                                        update_log(log_file, thisProcess + f": ✅ Already exists: {out_dir}")
                                    else:
                                        os.makedirs(out_dir, exist_ok=True)
                                        update_log(log_file, thisProcess + f": ✅ New folder created: {out_dir}")


                                    path_to_params = hf_hub_download(repo_id=filerepo, filename=sae_file, force_download=False)
                                    params = np.load(path_to_params)
                                    pt_params = {k: torch.from_numpy(v).to(torch.float32) for k, v in params.items()}

                                    d_model = int(pt_params["W_enc"].shape[0])
                                    d_sae   = int(pt_params["W_enc"].shape[1])
                                    sae = JumpReLUSAE(d_model, d_sae)
                                    _ = sae.load_state_dict(pt_params)
                                    sae.to(device)
                                    sae.eval()

                                    sae_file_tag = sae_file.replace("/", "__")
                                    
                                    cl = 0
                                    # ---- Process each CSV ONCE in batches (no per-prompt nested loop) ----
                                    csv_path = data_file
                                        
                                    cl += 1
                                    update_log(log_file, thisProcess + ": started processing " + str(csv_path))

                                    df = pd.read_csv(csv_path, header=0, names=['concept', 'prompt'])
                                    prompts  = df['prompt'].astype(str).tolist()
                                    concepts = df['concept'].astype(str).tolist()

                                    # filter empty prompts
                                    keep = [i for i, p in enumerate(prompts) if p.strip() != ""]
                                    if len(keep) != len(prompts):
                                        prompts  = [prompts[i]  for i in keep]
                                        concepts = [concepts[i] for i in keep]

                                    file_stub = Path(csv_path).name.replace(".csv", "")

                                    saved = 0
                                    with torch.inference_mode():
                                        # Ask model to return hidden states (avoid hooks entirely)
                                        model.config.output_hidden_states = True

                                        batches = len(prompts)/BATCH_SIZE

                                        cs = 0
                                        for s in range(0, len(prompts), BATCH_SIZE):
                                            cs += 1
                                            
                                            batch_prompts  = prompts[s:s + BATCH_SIZE]
                                            batch_concepts = concepts[s:s + BATCH_SIZE]

                                            print(str(cs) + ' / ' + str(batches) + " ...")

                                            enc = tokenizer(
                                                batch_prompts,
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                                max_length=MAX_LENGTH,
                                                add_special_tokens=True,
                                            )
                                            enc = {k: v.to(device) for k, v in enc.items()}
                                            attn_cpu = enc["attention_mask"].to("cpu")

                                            out = model(**enc)
                                            hs  = out.hidden_states
                                            # residual after layer i is typically hs[i+1]
                                            idx = this_layer + 1
                                            if idx >= len(hs):
                                                raise RuntimeError(f"Requested hidden_states[{idx}] but only {len(hs)} states available.")
                                            hidden = hs[idx].to(torch.float32)  # (B,T,D)

                                            # SAE forward (chunk-safe)
                                            sae_acts, _ = sae_forward_chunked(hidden, sae, rows_per_chunk=SAE_ROWS_PER_CHUNK)
                                            B, T = sae_acts.shape[:2]

                                            for b in range(B):
                                                if batch_concepts[b] == 'conference-workshop':
                                                    stopIt = True

                                                # Build mask on CPU to avoid device mismatch
                                                m = attn_cpu[b].to(torch.bool)              # CPU mask: 1 = real token, 0 = pad

                                                # Drop only the first non-pad token (BOS position)
                                                nonpad = torch.nonzero(m, as_tuple=False).squeeze(1)
                                                if nonpad.numel():
                                                    first = int(nonpad[0])
                                                    m[first] = False

                                                if not m.any():
                                                    # nothing to save
                                                    paired_np = np.empty((0, 2), dtype=np.float32)
                                                else:
                                                    # (T_real, d_sae) on CPU
                                                    sae_bt = sae_acts[b].to("cpu")[m].to(torch.float32)

                                                    
                                                    if activation_function == 'top-1-per-token':

                                                        # ---------- A) Per-token top-1 (what you had) ----------
                                                        vals_top1, inds_top1 = torch.max(sae_bt, dim=-1)                    # (T_real,)
                                                        paired_top1 = torch.stack((inds_top1.to(torch.float32), vals_top1), dim=1)
                                                        paired_np = paired_top1.numpy()
                                                                                    
                                                    if activation_function == 'max-per-concept':

                                                        # ---------- B) Global "max over tokens" per concept ----------
                                                        feat_max, _ = torch.max(sae_bt, dim=0)                               # (d_sae,)
                                                        nz_mask = feat_max > 0
                                                        ids_max = torch.nonzero(nz_mask, as_tuple=False).squeeze(1)
                                                        vals_max = feat_max[nz_mask]
                                                        paired_max = torch.stack((ids_max.to(torch.float32), vals_max), dim=1)
                                                        paired_np = paired_max.numpy()

                                                    if activation_function == 'sum-per-concept':

                                                        # ---------- C) Global "sum over tokens" per concept ----------
                                                        feat_sum = sae_bt.sum(dim=0)                                         # (d_sae,)
                                                        nz_mask_sum = feat_sum > 0
                                                        ids_sum = torch.nonzero(nz_mask_sum, as_tuple=False).squeeze(1)
                                                        vals_sum = feat_sum[nz_mask_sum]
                                                        paired_sum = torch.stack((ids_sum.to(torch.float32), vals_sum), dim=1)
                                                        paired_np = paired_sum.numpy()
                                                    
                                                    if activation_function == 'mean-per-concept':

                                                        # ---------- D) Mean over positions where the concept fired ----------
                                                        fired = (sae_bt > 0).to(torch.int32)                                 # (T_real, d_sae)
                                                        counts = fired.sum(dim=0)                                            # (d_sae,)
                                                        mean_pos = feat_sum / counts.clamp_min(1)
                                                        nz_mask_mean = counts > 0
                                                        ids_mean = torch.nonzero(nz_mask_mean, as_tuple=False).squeeze(1)
                                                        vals_mean = mean_pos[nz_mask_mean]
                                                        paired_meanpos = torch.stack((ids_mean.to(torch.float32), vals_mean), dim=1)
                                                        paired_np = paired_meanpos.numpy()
                                                        counts_np = counts[nz_mask_mean].cpu().numpy().astype(np.int32)


                                                # ---------- Save ----------
                                                concept = str(batch_concepts[b]).replace("/", "__")
                                                if '-' + languages[0] in file_stub:
                                                    nf = file_stub
                                                elif '-' + languages[1] in file_stub:
                                                    nf = file_stub
                                                else:
                                                    nf = file_stub + '-en'
                                        
                                                base = f"{sa_model.replace('/','_')}--{nf}--{s+b}--{concept}--{sae_file_tag.replace('__params.npz','')}"

                                                # Also save aggregates (separate files or a single NPZ)

                                                np.save(os.path.join(out_dir, base + "--" +  activation_function+ ".npy"), paired_np)

                                                if inspect:
                                                    
                                                    ifilename2 = inspect_folder + '/inspect--tensors.csv'
                                                    for inspect_concept in inspect_concepts:
                                                        ifilename = inspect_folder + '/inspect--' + inspect_concept + '.txt'

                                                        if inspect_concept == concept:

                                                            with open(ifilename, 'a') as file:

                                                                file.write("\n")
                                                                file.write("=========================================================")
                                                                file.write("\n")

                                                                file.write("tensor for " + activation_function + " (" + nf + " >> " + sae_file_tag + "): " + "\n") 
                                                                for snp in paired_np:
                                                                    if '. ' in str(snp):
                                                                        ccpt = str(snp).split('. ')[0]
                                                                        actval = str(snp).split('. ')[1]
                                                                        file.write(ccpt.strip() + ', ' + actval.strip())
                                                                        file.write("\n")
                                                                    else:
                                                                        print(str(snp))
                                                                        ccpt = int(float(str(snp).split(' ')[0].replace('[','')))
                                                                        actval = str(snp).split(' ')[1].replace(']','')
                                                                        file.write(str(ccpt) + ', ' + str(actval))
                                                                        file.write("\n")
                                                                file.write("=========================================================")
                                                                file.write("\n")
                                                            
                                                                #tensorValType, sourceCorpusLang, concept, layerLoc, activationScheme, averageAlg
                                                                append_tensor_to_file_as_csv(ifilename2, 'sae_act', nf, inspect_concept, sae_file_tag, activation_function, '', paired_np, '', '', '')

                                                saved += 1

                                                if saved % 200 == 0:
                                                    update_log(log_file, thisProcess + f": saved {saved}/{len(prompts)} prompts...")
                
                                    update_log(log_file, thisProcess + f": finished processing {csv_path}, wrote {saved} tensors")
        
            if SAECount == 0:
                update_log(log_file, thisProcess + "Could not find a suitable SAE file for the requested layer.")

        elif this_autoencoder == "openai/gpt2":

            # -------- Config --------
            PROMPT = "If you still hit the same error after upgrading, double-check that your shell is using the same venv"
            LOCATION = "resid_post_mlp"   # mlp_post_act | resid_delta_attn | resid_post_attn | resid_delta_mlp | resid_post_mlp
            THRESH = 1e-6                 # activation threshold for "active"

            # -------- Device selection --------
            if torch.backends.mps.is_available():
                device = torch.device("mps")   # Apple Silicon
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            print("Using device:", device)

            # -------- Load GPT-2 --------
            model = transformer_lens.HookedTransformer.from_pretrained(
                "gpt2", center_writing_weights=False, device=str(device)
            )


            for layer in range(12):

                if layer in layer_list:
                    # Map LOCATION → TransformerLens hook name
                    hook_name = {
                        "mlp_post_act":   f"blocks.{layer}.mlp.hook_post",
                        "resid_delta_attn": f"blocks.{layer}.hook_attn_out",
                        "resid_post_attn":  f"blocks.{layer}.hook_resid_mid",
                        "resid_delta_mlp":  f"blocks.{layer}.hook_mlp_out",
                        "resid_post_mlp":   f"blocks.{layer}.hook_resid_post",
                    }[LOCATION]


                    # -------- Load SAE --------

                    #latent_sparsity = 'v5_32k'
                    latent_sparsity = 'v5_128k'

                    if latent_sparsity == 'v5_128k':
                        sae_path = sparse_autoencoder.paths.v5_128k(LOCATION, layer)

                    if latent_sparsity == 'v5_32k':
                        sae_path = sparse_autoencoder.paths.v5_32k(LOCATION, layer)

                    with bf.BlobFile(sae_path, "rb") as f:
                        state_dict = torch.load(f, map_location=device)

                    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict).to(device)
                    autoencoder.eval()

                    # Get decoder weight matrix
                    W_dec = getattr(autoencoder, "W_dec", None)
                    if W_dec is None and hasattr(autoencoder, "decoder"):
                        W_dec = autoencoder.decoder.weight
                    assert W_dec is not None, "Could not find decoder weights."

                    # L2 norm of each decoder column (‖W_dec[:,k]‖₂)
                    dec_col_norms = torch.norm(W_dec, dim=0)           # (d_latent,)

                    p = 0
                    out_dir = activations_source_folder + '/' + sae_path.replace("az://openaipublic/sparse-autoencoder/","").replace('/','-').replace(".pt","")
                    os.makedirs(out_dir, exist_ok=True)

                    file_stub = Path(data_file).name.replace(".csv", "")

                    with open(data_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()        
                        for line in lines:
                            p += 1
                            if p > 1:
                                
                                concept = line.split(',')[0]
                                prompt = line.split(',')[1]

                                # -------- Forward pass --------
                                tokens = model.to_tokens(prompt)
                                with torch.no_grad():
                                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

                                input_acts = cache[hook_name].to(device)

                                # remove BOS token
                                input_acts = input_acts[1:]

                                # -------- Encode --------
                                with torch.no_grad():
                                    latents, info = autoencoder.encode(input_acts)   # (seq_len, d_latent)

                                z = latents
                                l0_per_token = (z>0).sum(dim=-1)
                                l0 = l0_per_token.float().mean().item()
                                sl0 = str(l0)
                                #print("L0 (avg active SAE) features per token)=" + sl0)
                                spartity = sl0

                                # -------- Compute mean contribution magnitude --------
                                with torch.no_grad():
                                    # contribution magnitude per token-feature
                                    contrib_mag = torch.abs(latents) * dec_col_norms   # (seq_len, d_latent)
                                    mean_contrib_mag = contrib_mag.mean(dim=0)         # (d_latent,)


                                # -------- Filter nonzero features --------
                                nonzero_mask = mean_contrib_mag > THRESH
                                inds = nonzero_mask.nonzero(as_tuple=False).flatten()   # (n_nonzero,)
                                values = mean_contrib_mag[inds]                        # (n_nonzero,)

                                # -------- Pack into a single tensor --------
                                # Shape: (n_nonzero, 2), where [:,0]=feature IDs, [:,1]=weights
                                feature_tensor = torch.stack((inds, values), dim=1)

                                #print(f"\nPrompt: {PROMPT}")
                                #print(f"Layer {LOCATION}  | hook: {hook_name}")
                                #print(f"Output tensor shape: {feature_tensor.shape}")
                                #print(feature_tensor[:20])  # show first 20 rows

                                if '-' + languages[0] in file_stub:
                                    nf = file_stub
                                elif '-' + languages[1] in file_stub:
                                    nf = file_stub
                                else:
                                    nf = file_stub + '-en'
                        
                                f_sae = sae_path.replace("az://openaipublic/sparse-autoencoder/","").replace(".pt","").split('/')[0]
                                f_conf = sae_path.replace("az://openaipublic/sparse-autoencoder/","").replace(".pt","").split('/')[1]
                                f_layer = sae_path.replace("az://openaipublic/sparse-autoencoder/","").replace(".pt","").split('/')[3]
                                fileOut = f_sae + '--' + nf + '--' + str(p) + '--' + concept.lower() + '--layer_' + f_layer + '-' + f_conf + '--sparsity_' + spartity
                                np_tensor = feature_tensor.cpu().numpy()  # Move to CPU if necessary
                                np.save(os.path.join(out_dir, fileOut + ".npy"), np_tensor)
                                #print('saved ' + out_dir + '/' + fileOut)
                                print(p)

        else:
            update_log(log_file, thisProcess + ": sparse autoencoder not set correctly")

        update_log(log_file, thisProcess + ": end")
        print("Done.")

    except Exception as e:
        update_log(log_file, thisProcess + ": Exception = " + str(e))
        update_log(log_file, thisProcess + ": Exception = " + traceback.format_exc())
        print(e)
        print(traceback.format_exc())
