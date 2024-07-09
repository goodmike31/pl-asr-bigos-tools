import sys
import os
import configparser
import pandas as pd
import argparse
import pandas as pd
import jiwer


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


def calculate_eval_metrics(test_set_name, df_in)->pd.DataFrame:

    df_results_header=["dataset", "test cases", "normalization", "reference", "system", "variant", "SER", "WIL", "MER", "WER","CER"]
    df_results = pd.DataFrame([], columns=df_results_header)
    ref_cols = [col for col in df_in.columns if col.startswith('ref')]
    hyp_cols =  [col for col in df_in.columns if col.startswith('hyp')]

    result=[]
    for norm in postnorm_types:
        #print("Norm:" + norm)
        for ref_col in ref_cols:
            #print("ref_col:" + ref_col)
            for hyp_col in hyp_cols:
                #print("hyp_col:" + hyp_col)
                ref = df_in[ref_col].tolist()

                hyp = df_in[hyp_col].tolist()

                assert len(ref) == len(hyp)

                ref_type=ref_col.split("_")[1]
                system = hyp_col.split("_")[1]
                variant = hyp_col.split("_")[2]
                
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
                
                size=len(ref)
                match_sents = sum(r == h for r, h in zip(ref, hyp))

                ser = round((1 - (match_sents / len(ref))) * 100,2)
                wer = round(jiwer.wer(ref, hyp) * 100 ,2) 
                cer = round(jiwer.cer(ref, hyp) * 100 ,2)
                mer = round(jiwer.mer(ref, hyp) * 100 ,2)
                wil = round(jiwer.wil(ref, hyp) * 100 ,2)

                result=[test_set_name, size, norm, ref_type, system, variant, ser, wil, mer, wer,cer]
                #print(result)
                df_line=pd.DataFrame([result], columns=df_results_header)
                df_results=pd.concat([df_results,df_line], axis=0, ignore_index=True)

    return(df_results)

def format_eval_results(df_results, view):
    if(view == "normalization_all_original_ref"):
        filtered_data = df_results.query('normalization == "all" and reference == "original"')
        print(filtered_data)

    elif(view == "WER_norm_verif"):
        filtered_data = df_results.query('normalization == "all" and reference == "verified"')
    else:
        print("Unknown view!")
        sys.exit(2)
        
    #if(filtered_data.empty):
    #    print("Cannot generate view - empty result")
    #    sys.exit(3)
    #else:
        # Create a pivot table with 'System' and 'Variant' as columns and 'WER'/'CER' as rows
        #pivot_table = filtered_data.pivot_table(index=['system', 'variant'], columns=['dataset', 'normalization'], values=['WER'])
        #transposed_data = pivot_table.T

        #print(transposed_data)
    return