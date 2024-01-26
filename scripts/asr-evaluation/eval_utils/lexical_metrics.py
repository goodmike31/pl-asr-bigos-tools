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

# TODO consider splitting into functions for each metric
def calculate_lexical_metrics(df_eval_input, test_set_name)->pd.DataFrame:

    # assume that the input dataframe 
    # a. was prefiltered accordingly to the proper business logic e.g. only test set, only specific subset, etc.
    
    # b. has the following columns: ref_reftypename, hyp_systemcodename
    ref_cols = [col for col in df_eval_input.columns if col.startswith('ref')]
    hyp_cols =  [col for col in df_eval_input.columns if col.startswith('hyp')]
    if len(ref_cols) == 0 or len(hyp_cols) == 0:
        print("ERROR: No ref or hyp columns found in the input dataframe")
        return None
    
    # c. has the same number of rows for each ref and hyp column
    assert len(ref_cols) == len(hyp_cols)

    # output columns
    df_results_header=["dataset", "test cases", "ref type", "eval norm", "system", "SER", "WIL", "MER", "WER", "CER"]
    # dataset  - name of the dataset
    # test cases - number of test cases
    # reference - type of source reference from the dataset (original, manually verified, normalized, etc.)
    # eval norm - type of normalization applied to the reference and hypothesis using jiwer
    # system - codename of the ASR system

    df_results = pd.DataFrame([], columns=df_results_header)

    
    result=[]
    for norm in postnorm_types:
        #print("Norm:" + norm)
        for ref_col in ref_cols:
            #print("ref_col:" + ref_col)
            for hyp_col in hyp_cols:
                #print("hyp_col:" + hyp_col)
                ref = df_eval_input[ref_col].tolist()

                hyp = df_eval_input[hyp_col].tolist()

                assert len(ref) == len(hyp)

                ref_type=ref_col.split("_")[1]
                system = hyp_col.split("_")[1]
                
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
                ser = jiwer.sentence_error_rate(ref, hyp)
                wil = jiwer.word_error_rate(ref, hyp)
                mer = jiwer.match_error_rate(ref, hyp)
                wer = jiwer.wer(ref, hyp)
                cer = jiwer.cer(ref, hyp)
                
                #print("SER: ", ser)
                #print("WER: ", wer)
                #print("CER: ", cer)
                
                result.append([test_set_name, len(ref), norm, ref_type, system, ser, wil, mer, wer, cer])
    
    df_results = pd.DataFrame(result, columns=df_results_header)
    return df_results