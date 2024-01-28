import jiwer
import pandas as pd

def get_lexical_metrics(df_eval_input, test_set_name, system_codename, ref_type, norm)->pd.DataFrame:
    # TODO consider moving to config
    # TODO consider standardizing the names of transformations
    # TODO consider splitting into specific metrics
    # TODO consider generating multiple metrics for all norm types

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

    transf_chars = jiwer.Compose([
        jiwer.ReduceToListOfListOfChars()
    ])

    print("get_lexical_metrics:\nDataset: {}System: {}\nRef_type: {}\nNormalization: {}\n".format(test_set_name, system_codename, ref_type, norm))
    # assume that the input dataframe   
    # a. was prefiltered accordingly to the proper business logic e.g. only test set, only specific subset, etc.
    # b. has the following columns: ref_col, hyp_col
    ref_col = "ref_" + ref_type
    hyp_col = "hyp_" + system_codename

    ref = df_eval_input[ref_col].dropna().astype(str).tolist()
    #print ("refs: ", ref)
    
    hyp = df_eval_input[hyp_col].dropna().astype(str).tolist()
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
        print("refs filtered to match hyps availability: ", ref)
        # Apply mask to df_eval_input
        hyp = df_eval_input[hyp_col][df_eval_input["mask"]].astype(str).tolist()
        print("hyps filtered to match hyps availability: ", hyp)

    # output columns
    df_results_header=["dataset", "samples", "ref_type", "eval_norm", "system", "SER", "WIL", "MER", "WER", "CER"]
    # dataset  - name of the dataset
    # test cases - number of test cases
    # reference - type of source reference from the dataset (original, manually verified, normalized, etc.)
    # eval norm - type of normalization applied to the reference and hypothesis using jiwer
    # system - codename of the ASR system
    
    result=[]
    #print("hyp_col:" + hyp_col)

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

    # Calculate metrics
    match_sents = sum(r == h for r, h in zip(ref, hyp))
    print("match_sents: ", match_sents)
    ser = round((1 - (match_sents / len(ref))) * 100,2)
    wer = round(jiwer.wer(ref, hyp) * 100 ,2)
    cer = round(jiwer.cer(ref, hyp) * 100, 2)
 
    mer = round(jiwer.mer(ref, hyp) * 100 ,2)
    wil = round(jiwer.wil(ref, hyp) * 100 ,2)

    
    print("SER: ", ser)
    print("WER: ", wer)
    print("CER: ", cer)
    print("MER: ", mer)
    print("WIL: ", wil)
    # TODO add more metrics e.g. TER, PER, etc.
    
    result.append([test_set_name, len(ref), ref_type, norm, system_codename, ser, wil, mer, wer, cer])
    
    df_results = pd.DataFrame(result, columns=df_results_header)
    
    return df_results


def get_lexical_metrics_all_norm_types(df_eval_input, test_set_name, system_codename, ref_type)-> pd.DataFrame:
    df_results = pd.DataFrame([])

    for norm in postnorm_types:
        print("Norm:" + norm)
        df_single_result = get_lexical_metrics(df_eval_input, test_set_name, system_codename, ref_type, norm) 
        df_results = df_results.append(df_single_result, ignore_index=True)
    
    return(df_results)


def get_lexical_metrics_all_ref_types(df_eval_input, test_set_name, system_codename, norm)-> pd.DataFrame:
    df_results = pd.DataFrame([])

    ref_cols = [col for col in df_eval_input.columns if col.startswith('ref')]
    for ref_col in ref_cols:
        print("ref_col" + ref_col)
        ref_type = ref_col.split("_")[1]
        df_single_result = get_lexical_metrics(df_eval_input, test_set_name, system_codename, ref_type, norm) 
        df_results = df_results.append(df_single_result, ignore_index=True)
    
    return(df_results)

def get_lexical_metrics_all_systems(df_eval_input, test_set_name, ref_type, norm)-> pd.DataFrame:
    df_results = pd.DataFrame([])

    hyp_cols =  [col for col in df_eval_input.columns if col.startswith('hyp')]
    for hyp_col in hyp_cols:
        print("hyp_col" + hyp_col)
        system_codename = hyp_col.split("_")[1]
        df_single_result = get_lexical_metrics(df_eval_input, test_set_name, system_codename, ref_type, norm) 
        df_results = df_results.append(df_single_result, ignore_index=True)
    
    return(df_results)

