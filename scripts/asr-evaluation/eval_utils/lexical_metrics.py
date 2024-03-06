from jiwer import wer, mer, wil, cer, process_words, process_characters
import jiwer
import pandas as pd
import librosa
import os

transf_all = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.Strip()
]) 

transf_lc = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.ReduceToListOfListOfWords()
]) 

transf_blanks = jiwer.Compose([
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords()
]) 

transf_punc = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords()
])
#transf_chars = jiwer.Compose([
#    jiwer.ReduceToListOfListOfChars()
#])



def prepare_refs_hyps(df_eval_input, ref_col, hyp_col, norm):
    # filter out hypotheses which are empty and corresponding references
    audio_paths = df_eval_input['audiopath_local'].tolist()
    # get masking vector for non-empty hypotheses
    non_empty_hyps = df_eval_input[hyp_col].notnull()
    # filter out non-empty hypotheses from dataframe
    df_eval_input = df_eval_input[non_empty_hyps]
    
    # retrieve non-empty hypotheses and references    
    ref = df_eval_input[ref_col]
    hyp = df_eval_input[hyp_col]
    #print ("refs: ", ref)
    #print ("hyps: ", hyp)
    
    if len(ref) != len(hyp):
        print("Warning: number of references and hypotheses does not match")
        print("Generating metrics for common subset of references and hypotheses")
        # TODO consider returning None or raising an exception
        # Naive approach: cut the longer list to the length of the shorter one
        #ref = ref[:min(len(ref), len(hyp))]
        #hyp = hyp[:min(len(ref), len(hyp))]

        # More sophisticated approach: find common subset of references and hypotheses
        # Generate mask from df_eval_input where ref and hyp are not empty
        df_eval_input["mask"] = df_eval_input[ref_col].notnull() & df_eval_input[hyp_col].notnull()
        ref = df_eval_input[ref_col][df_eval_input["mask"]].astype(str).tolist() 
        #print("refs filtered to match hyps availability: ", ref)
        # Apply mask to df_eval_input
        hyp = df_eval_input[hyp_col][df_eval_input["mask"]].astype(str).tolist()
        #print("hyps filtered to match hyps availability: ", hyp)

        audio_paths = df_eval_input['audiopath_local'][df_eval_input["mask"]].tolist()

        # Remove mask column
        df_eval_input = df_eval_input.drop(columns=["mask"])
    
    
    ids = []
    for i in range(len(audio_paths)):
        ids.append(os.path.basename(audio_paths[i]))

    # Count matching elements
    if norm == "all":
        ref=transf_all(ref)
        hyp=transf_all(hyp)
    elif norm == "lowercase":
        ref=transf_lc(ref)
        hyp=transf_lc(hyp)
    elif norm == "blanks":
        ref=transf_blanks(ref)
        hyp=transf_blanks(hyp)
    elif norm == "punct":
        ref=transf_punc(ref)
        hyp=transf_punc(hyp)
    else:
        ref=ref
        hyp=hyp
    
    print("Ref post-proceessed: ", ref)
    print("Hyp post-proceessed: ", hyp)

    # Calculate metrics for the whole dataset
    return ref, hyp, ids, audio_paths

def get_lexical_metrics_per_sample(df_eval_input, dataset, subset, split, system_codename, ref_type, norm)->pd.DataFrame:
    print("Calculating metrics for individual sentences for dataset:\nDataset: {}\nSubset: {}\nSplit: {}\nSystem: {}\nRef_type: {}\nNormalization: {}\n".format(dataset, subset, split, system_codename, ref_type, norm))
    # assume that the input dataframe   
    # a. was prefiltered accordingly to the proper business logic e.g. only test set, only specific subset, etc.
    # b. has the following columns: ref_col, hyp_col
    ref_col = "ref_" + ref_type
    hyp_col = "hyp_" + system_codename

    ref, hyp, ids, audio_paths = prepare_refs_hyps(df_eval_input, ref_col, hyp_col, norm)
    
    # output columns
    df_results_header = ["dataset", "subset", "split", "ref_type", "eval_norm", "system", "id", "audio_duration", "WIL", "MER", "WER", "CER"]
    
    result=[]

    for index in range(len(ids)):
        # calculate WER, CER, etc. for each sample
        # TODO consider adding more metrics e.g. TER, PER, etc.
        # TODO consider adding more metadata e.g. audio duration, etc.
        # calculate audio_duration
        audio_path = audio_paths[index]
        audio_duration = round(librosa.get_duration(path=audio_path),2)
        print("Audio duration: ", audio_duration)

        id = ids[index]
        print("ID: ", id)
        ref_single = ref[index]
        print("Ref: ", ref_single)
        hyp_single = hyp[index]
        print("Hyp: ", hyp_single)

        output_words = jiwer.process_words(ref_single, hyp_single)
        wer = round(output_words.wer * 100 ,2)
        mer = round(output_words.mer * 100 ,2)
        wil = round(output_words.wil * 100 ,2)
        print(jiwer.visualize_alignment(output_words))

        output_chars = jiwer.process_characters(ref_single, hyp_single)
        cer = round(output_chars.cer * 100, 2)

        print(jiwer.visualize_alignment(output_chars))

        # TODO add more metrics e.g. TER, PER, etc.
        result.append([dataset, subset, split, ref_type, norm, system_codename, id, audio_duration, wil, mer, wer, cer])

    df_results = pd.DataFrame(result, columns=df_results_header)
    return df_results

    
def get_lexical_metrics_per_dataset(df_eval_input, dataset, subset, split, system_codename, ref_type, norm)->pd.DataFrame:

    # TODO consider moving to config
    # TODO consider standardizing the names of transformations
    # TODO consider splitting into specific metrics
    # TODO consider generating multiple metrics for all norm types
    # https://jitsi.github.io/jiwer/reference/transformations/


    print("Calculating metrics for whole dataset:\nDataset: {}\nSubset: {}\nSplit: {}\nSystem: {}\nRef_type: {}\nNormalization: {}\n".format(dataset, subset, split, system_codename, ref_type, norm))
    # assume that the input dataframe   
    # a. was prefiltered accordingly to the proper business logic e.g. only test set, only specific subset, etc.
    # b. has the following columns: ref_col, hyp_col
    ref_col = "ref_" + ref_type
    hyp_col = "hyp_" + system_codename

    ref, hyp, ids, audio_paths = prepare_refs_hyps(df_eval_input, ref_col, hyp_col, norm)
    
    # output columns
    df_results_header=["dataset", "subset", "split", "samples", "ref_type", "eval_norm", "system", "SER", "WIL", "MER", "WER", "CER"]
    # dataset  - name of the dataset
    # test cases - number of test cases
    # reference - type of source reference from the dataset (original, manually verified, normalized, etc.)
    # eval norm - type of normalization applied to the reference and hypothesis using jiwer
    # system - codename of the ASR system
    
    result=[]
    
    match_sents = sum(r == h for r, h in zip(ref, hyp))
    print("match_sents: ", match_sents)
    ser = round((1 - (match_sents / len(ref))) * 100,2)

    output_words = jiwer.process_words(ref, hyp)
    wer = round(output_words.wer * 100 ,2)
    mer = round(output_words.mer * 100 ,2)
    wil = round(output_words.wil * 100 ,2)

    output_chars = jiwer.process_characters(ref, hyp)
    cer = round(output_chars.cer * 100, 2)
    
    print("SER: ", ser)
    print("WER: ", wer)
    print("CER: ", cer)
    print("MER: ", mer)
    print("WIL: ", wil)
    # TODO add more metrics e.g. TER, PER, etc.
    result.append([dataset, subset, split, len(ref), ref_type, norm, system_codename, ser, wil, mer, wer, cer])
    
    df_results = pd.DataFrame(result, columns=df_results_header)
    
    return df_results


def get_lexical_metrics_all_norm_types(df_eval_input, dataset_codename, system_codename, ref_type)-> pd.DataFrame:
    df_results = pd.DataFrame([])

    for norm in postnorm_types:
        print("Norm:" + norm)
        df_single_result = get_lexical_metrics_per_dataset(df_eval_input, dataset_codename, system_codename, ref_type, norm) 
        df_results = df_results.append(df_single_result, ignore_index=True)
    
    return(df_results)


def get_lexical_metrics_per_dataset_all_ref_types(df_eval_input, dataset_codename, system_codename, norm)-> pd.DataFrame:
    df_results = pd.DataFrame([])

    ref_cols = [col for col in df_eval_input.columns if col.startswith('ref')]
    for ref_col in ref_cols:
        print("ref_col" + ref_col)
        ref_type = ref_col.split("_")[1]
        df_single_result = get_lexical_metrics_per_dataset(df_eval_input, dataset_codename, system_codename, ref_type, norm) 
        df_results = df_results.append(df_single_result, ignore_index=True)
    
    return(df_results)

def get_lexical_metrics_per_dataset_all_systems(df_eval_input, dataset_codename, ref_type, norm)-> pd.DataFrame:
    df_results = pd.DataFrame([])

    hyp_cols =  [col for col in df_eval_input.columns if col.startswith('hyp')]
    for hyp_col in hyp_cols:
        print("hyp_col" + hyp_col)
        system_codename = hyp_col.split("_")[1]
        df_single_result = get_lexical_metrics_per_dataset(df_eval_input, dataset_codename, system_codename, ref_type, norm) 
        df_results = df_results.append(df_single_result, ignore_index=True)
    
    return(df_results)

def get_lexical_metrics_per_sample_all_ref_types(df_eval_input, dataset_codename, system_codename, norm)-> pd.DataFrame:
    df_results = pd.DataFrame([])

    ref_cols = [col for col in df_eval_input.columns if col.startswith('ref')]
    for ref_col in ref_cols:
        print("ref_col" + ref_col)
        ref_type = ref_col.split("_")[1]
        df_single_result = get_lexical_metrics_per_sample(df_eval_input, dataset_codename, system_codename, ref_type, norm) 
        df_results = df_results.append(df_single_result, ignore_index=True)
    
    return(df_results)