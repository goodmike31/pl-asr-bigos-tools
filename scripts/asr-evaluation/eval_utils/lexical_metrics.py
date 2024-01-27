import jiwer
import pandas as pd

postnorm_types=["none", "lower_case","blanks","punct", "all"]

transf_all = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
]) 

transf_lc = jiwer.Compose([
    jiwer.ToLowerCase()
]) 

transf_blanks = jiwer.Compose([
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces()
]) 

transf_punc = jiwer.Compose([
    jiwer.RemovePunctuation()
])

# TODO consider splitting into specific metrics
def get_lexical_metrics(df_eval_input, test_set_name, system_codename, ref_type, norm)->pd.DataFrame:

    # assume that the input dataframe 
    # a. was prefiltered accordingly to the proper business logic e.g. only test set, only specific subset, etc.
    # b. has the following columns: ref_col, hyp_col
    ref_col = "ref_" + ref_type
    hyp_col = "hyp_" + system_codename

    ref = df_eval_input[ref_col].tolist()

    hyp = df_eval_input[hyp_col].tolist()

    assert len(ref) == len(hyp)

    # output columns
    df_results_header=["dataset", "samples", "ref_type", "eval_norm", "system", "SER", "WIL", "MER", "WER", "CER"]
    # dataset  - name of the dataset
    # test cases - number of test cases
    # reference - type of source reference from the dataset (original, manually verified, normalized, etc.)
    # eval norm - type of normalization applied to the reference and hypothesis using jiwer
    # system - codename of the ASR system

    df_results = pd.DataFrame([], columns=df_results_header)
    
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
    
    # Calculate metrics
    match_sents = sum(r == h for r, h in zip(ref, hyp))

    ser = round((1 - (match_sents / len(ref))) * 100,2)
    wer = round(jiwer.wer(ref, hyp) * 100 ,2) 
    cer = round(jiwer.cer(ref, hyp) * 100 ,2)
    mer = round(jiwer.mer(ref, hyp) * 100 ,2)
    wil = round(jiwer.wil(ref, hyp) * 100 ,2)
    
    print("SER: ", ser)
    print("WER: ", wer)
    print("CER: ", cer)
    print("MER: ", mer)
    print("WIL: ", wil)
    # TODO add more metrics e.g. TER, PER, etc.
    
    result.append([test_set_name, len(ref), norm, ref_type, system_codename, ser, wil, mer, wer, cer])
    
    df_results = pd.DataFrame(result, columns=df_results_header)
    return df_results


def get_lexical_metrics_all_norm_types(df_eval_input, test_set_name, system_codename, ref_type)-> pd.DataFrame:
    df_results_header=["dataset", "samples", "ref_type", "eval_norm", "system", "SER", "WIL", "MER", "WER", "CER"]
    df_results = pd.DataFrame([], columns=df_results_header)

    for norm in postnorm_types:
        print("Norm:" + norm)
        df__single_result = get_lexical_metrics(df_eval_input, test_set_name, system_codename, ref_type, norm) 
        df_results = df_results.append(df__single_result, ignore_index=True)

def get_lexical_metrics_all_ref_types(df_eval_input, test_set_name, system_codename, norm)-> pd.DataFrame:
    df_results_header=["dataset", "samples", "ref_type", "eval_norm", "system", "SER", "WIL", "MER", "WER", "CER"]
    df_results = pd.DataFrame([], columns=df_results_header)

    ref_cols = [col for col in df_eval_input.columns if col.startswith('ref')]
    for ref_col in ref_cols:
        print("ref_col" + ref_col)
        ref_type = ref_col.split("_")[1]
        df__single_result = get_lexical_metrics(df_eval_input, test_set_name, system_codename, ref_type, norm) 
        df_results = df_results.append(df__single_result, ignore_index=True)

def get_lexical_metrics_all_systems(df_eval_input, test_set_name, ref_type, norm)-> pd.DataFrame:
    df_results_header=["dataset", "samples", "ref_type", "eval_norm", "system", "SER", "WIL", "MER", "WER", "CER"]
    df_results = pd.DataFrame([], columns=df_results_header)

    hyp_cols =  [col for col in df_eval_input.columns if col.startswith('hyp')]
    for hyp_col in hyp_cols:
        print("hyp_col" + hyp_col)
        system_codename = hyp_col.split("_")[1]
        df__single_result = get_lexical_metrics(df_eval_input, test_set_name, system_codename, ref_type, norm) 
        df_results = df_results.append(df__single_result, ignore_index=True)

