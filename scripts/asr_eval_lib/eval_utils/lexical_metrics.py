from jiwer import wer, mer, wil, cer, process_words, process_characters
import jiwer
import pandas as pd
import librosa
import os

# Define transformations for text normalization
transf_all = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.Strip()
]) 

transf_lc = jiwer.Compose([
    jiwer.ToLowerCase(),
]) 

transf_blanks = jiwer.Compose([
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
]) 

transf_punc = jiwer.Compose([
    jiwer.RemovePunctuation()
])
#transf_chars = jiwer.Compose([
#    jiwer.ReduceToListOfListOfChars()
#])

# Function to replace words
def replace_words(sentence, replacement_dict):
    """
    Replace words in a sentence based on a provided dictionary.
    
    Args:
        sentence (str): Input sentence where words need to be replaced.
        replacement_dict (dict): Dictionary mapping words to their replacements.
        
    Returns:
        str: Sentence with words replaced according to the dictionary.
    """
    print("Lexicon - Words in: ", sentence)
  
    words = sentence.split()
    
    replaced_sentence = ' '.join([replacement_dict.get(word, word) for word in words])
    print("Lexicon - Words out: ", replaced_sentence)

    return replaced_sentence

def remove_tags(sentence, tags=["<unk>", "<silence>", "trunc"]):
    """
    Remove specified tags from a sentence, both standalone and embedded in words.
    
    Args:
        sentence (str): Input sentence containing tags to be removed.
        tags (list): List of tag strings to remove from the sentence. 
                    Default: ["<unk>", "<silence>", "trunc"]
        
    Returns:
        str: Sentence with all tags removed.
    """
    print("Tags - Words in: ", sentence)
    words = sentence.split()
    
    # remove stand alone tags
    # e.g. "this is the example of <unk>" becomes "this is the example of"
    without_stand_alone_tags = [word for word in words if not any(tag.lower() == word.lower() for tag in tags)]

    # remove word if it contains any of the artifacts (regardless of the case and adjacent punctuation and characters)
    # e.g. "this is the example of sente_trunc" becomes "this is the example of"
    # e.g. "trunc_this is the example" becomes "is the example"
    # e.g. "this is the example of <unk>" becomes "this is the example of"
    without_glued_tags = ' '.join([word for word in without_stand_alone_tags if not any(tag.lower() in word.lower() for tag in tags)])
    print("Tags - Words out: ", without_glued_tags)

    return without_glued_tags

def prepare_refs_hyps(df_eval_input, ref_col, hyp_col, norm, norm_lexicon=None):
    """
    Prepare reference and hypothesis pairs for evaluation by applying normalization
    and filtering out invalid entries.
    
    Args:
        df_eval_input (pandas.DataFrame): DataFrame containing reference and hypothesis columns.
        ref_col (str): Name of the reference column.
        hyp_col (str): Name of the hypothesis column.
        norm (str): Type of normalization to apply ('none', 'lowercase', 'blanks', 
                   'punct', 'tags', 'dict', or 'all').
        norm_lexicon (dict, optional): Dictionary for lexicon-based normalization. Default is None.
        
    Returns:
        tuple: (references, hypotheses, ids, audio_paths) - Lists containing the processed
               reference and hypothesis texts, sample IDs, and paths to audio files.
    """
    # get masking vector for non-empty hypotheses
    non_empty_hyps = df_eval_input[hyp_col].notnull()
    # filter out non-empty hypotheses from dataframe
    df_eval_input = df_eval_input[non_empty_hyps]
    print("Number of non-empty hypotheses: ", len(df_eval_input))
    
    # remove hypothesis with values EMPTY or INVALID
    # TODO - move filtering logic to config
    df_eval_input = df_eval_input[df_eval_input[hyp_col] != "EMPTY"]
    df_eval_input = df_eval_input[df_eval_input[hyp_col] != "INVALID"]
    print("Number of non-empty hypotheses after filtering out EMPTY and INVALID: ", len(df_eval_input))

    # retrieve non-empty hypotheses and references    
    ref = df_eval_input[ref_col].tolist()
    hyp = df_eval_input[hyp_col].tolist()
    print ("refs len: ", len(ref))
    print ("hyps len: ", len(hyp))
    
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

        # Remove mask column
        df_eval_input = df_eval_input.drop(columns=["mask"])
    
    
    # make sure values passed to jiwer are strings
    ref = [str(i) for i in ref]
    hyp = [str(i) for i in hyp]

    if norm == "none":
        ref=ref
        hyp=hyp
    elif norm == "lowercase":
        ref=transf_lc(ref)
        hyp=transf_lc(hyp)
    elif norm == "blanks":
        ref=transf_blanks(ref)
        hyp=transf_blanks(hyp)
    elif norm == "punct":
        ref=transf_punc(ref)
        hyp=transf_punc(hyp)
    elif norm == "tags":
        ref = transf_blanks([remove_tags(str(sentence)) for sentence in ref])
        hyp = transf_blanks([remove_tags(str(sentence)) for sentence in hyp])
        
    elif norm == "dict":
        # TODO - add dictionary based normalization
        # Apply the function to both lists
        if (norm_lexicon is not None):
            print("Normalizing using lexicon")
            print("Lexicon: ", norm_lexicon)
            ref = transf_blanks([replace_words(str(sentence), norm_lexicon) for sentence in ref])
            hyp = transf_blanks([replace_words(str(sentence), norm_lexicon) for sentence in hyp])
    elif norm == "all":
        ref=transf_all(ref)
        hyp=transf_all(hyp)
        ref = transf_blanks([remove_tags(str(sentence)) for sentence in ref])
        hyp = transf_blanks([remove_tags(str(sentence)) for sentence in hyp])
        
        if (norm_lexicon is not None):
            ref = transf_blanks([replace_words(str(sentence), norm_lexicon) for sentence in ref])
            hyp = transf_blanks([replace_words(str(sentence), norm_lexicon) for sentence in hyp])
    else:
        print("Normalization type not recognized. Please choose one of the following: none, lowercase, blanks, punct, dict, all.")
        exit(1)
        
    print ("refs len after normalization: ", len(ref))
    print ("hyps len after normalization: ", len(hyp))

    # combine into dataframe
    df_eval_input[ref_col] = ref
    df_eval_input[hyp_col] = hyp
    # eliminate rows with empty references equal to ""
    df_eval_input = df_eval_input[df_eval_input[ref_col] != ""]
    # eliminate rows with empty hypotheses equal to ""
    df_eval_input = df_eval_input[df_eval_input[hyp_col] != ""]
        
    ref = df_eval_input[ref_col].tolist()
    hyp = df_eval_input[hyp_col].tolist()
    
    assert(len(ref) == len(hyp))
   
    print("Number of non-empty references: ", len(ref))
    print("Number of non-empty hypotheses: ", len(hyp))

    audio_paths = df_eval_input['audiopath_local'].tolist()
    ids = []
    for i in range(len(audio_paths)):
        ids.append(os.path.basename(audio_paths[i]))


    # Calculate metrics for the whole dataset
    return ref, hyp, ids, audio_paths

def get_lexical_metrics_per_sample(df_eval_input, dataset, subset, split, system_codename, ref_type, norm, norm_lexicon = None)->pd.DataFrame:
    """
    Calculate speech recognition metrics for each individual sample.
    
    Args:
        df_eval_input (pandas.DataFrame): DataFrame with reference and hypothesis data.
        dataset (str): Name of the dataset.
        subset (str): Subset of the dataset.
        split (str): Data split (e.g., 'train', 'test', 'val').
        system_codename (str): Identifier for the ASR system.
        ref_type (str): Type of reference (e.g., 'original', 'normalized').
        norm (str): Normalization type applied.
        norm_lexicon (dict, optional): Dictionary for lexicon-based normalization. Default is None.
        
    Returns:
        pandas.DataFrame: DataFrame with speech metrics for each sample, including
                         Word Information Lost (WIL), Match Error Rate (MER), 
                         Word Error Rate (WER), and Character Error Rate (CER).
    """
    print("Calculating metrics for individual sentences for dataset:\nDataset: {}\nSubset: {}\nSplit: {}\nSystem: {}\nRef_type: {}\nNormalization: {}\n".format(dataset, subset, split, system_codename, ref_type, norm))
    # assume that the input dataframe   
    # a. was prefiltered accordingly to the proper business logic e.g. only test set, only specific subset, etc.
    # b. has the following columns: ref_col, hyp_col
    print("Norm lexicon: ", norm_lexicon)

    ref_col = "ref_" + ref_type
    hyp_col = "hyp_" + system_codename

    if (norm == "dict"):
        ref, hyp, ids, audio_paths = prepare_refs_hyps(df_eval_input, ref_col, hyp_col, norm, norm_lexicon)
    else:
        ref, hyp, ids, audio_paths = prepare_refs_hyps(df_eval_input, ref_col, hyp_col, norm)
    
    # output columns
    df_results_header = ["dataset", "subset", "split", "ref_type", "norm_type", "system", "id", "ref", "hyp", "audio_duration", "WIL", "MER", "WER", "CER"]
    #print (df_results_header)
    result=[]

    for index in range(len(ids)):
        # calculate WER, CER, etc. for each sample
        # TODO consider adding more metrics e.g. TER, PER, etc.
        # TODO consider adding more metadata e.g. audio duration, etc.
        # calculate audio_duration
        audio_path = audio_paths[index]
        if librosa.__version__ < "0.10.0":
            audio_duration = round(librosa.get_duration(path=audio_path),2)
        else:
            audio_duration = round(librosa.get_duration(filename=audio_path),2)
        #print("Audio duration: ", audio_duration)

        id = ids[index]
        #print("ID: ", id)
        ref_single = ref[index]
        #print("Ref: ", ref_single)
        hyp_single = hyp[index]
        #print("Hyp: ", hyp_single)

        output_words = jiwer.process_words(ref_single, hyp_single)
        wer = round(output_words.wer * 100 ,2)
        mer = round(output_words.mer * 100 ,2)
        wil = round(output_words.wil * 100 ,2)
        #print(jiwer.visualize_alignment(output_words))

        output_chars = jiwer.process_characters(ref_single, hyp_single)
        cer = round(output_chars.cer * 100, 2)

        #print(jiwer.visualize_alignment(output_chars))

        # TODO add more metrics e.g. TER, PER, etc.
        result.append([dataset, subset, split, ref_type, norm, system_codename, id, ref_single, hyp_single, audio_duration, wil, mer, wer, cer])

    df_results = pd.DataFrame(result, columns=df_results_header)
    return df_results

    
def get_lexical_metrics_per_dataset(df_eval_input, dataset, subset, split, system_codename, ref_type, norm, norm_lexicon)->pd.DataFrame:
    """
    Calculate aggregated speech recognition metrics across an entire dataset.
    
    Args:
        df_eval_input (pandas.DataFrame): DataFrame with reference and hypothesis data.
        dataset (str): Name of the dataset.
        subset (str): Subset of the dataset.
        split (str): Data split (e.g., 'train', 'test', 'val').
        system_codename (str): Identifier for the ASR system.
        ref_type (str): Type of reference (e.g., 'original', 'normalized').
        norm (str): Normalization type applied.
        norm_lexicon (dict, optional): Dictionary for lexicon-based normalization.
        
    Returns:
        pandas.DataFrame: DataFrame with aggregated metrics including Sentence Error Rate (SER),
                         Word Information Lost (WIL), Match Error Rate (MER), Word Error Rate (WER),
                         and Character Error Rate (CER).
    """
    # TODO consider moving to config
    # TODO consider standardizing the names of transformations
    # TODO consider splitting into specific metrics
    # TODO consider generating multiple metrics for all norm types
    # https://jitsi.github.io/jiwer/reference/transformations/


    print("Calculating metrics for whole dataset:\nDataset: {}\nSubset: {}\nSplit: {}\nSystem: {}\nRef_type: {}\nNormalization: {}\n".format(dataset, subset, split, system_codename, ref_type, norm))
    # assume that the input dataframe   
    # a. was prefiltered accordingly to the proper business logic e.g. only test set, only specific subset, etc.
    # b. has the following columns: ref_col, hyp_col
    print("Norm lexicon: ", norm_lexicon)

    ref_col = "ref_" + ref_type
    hyp_col = "hyp_" + system_codename


    if (norm_lexicon is not None):
        ref, hyp, ids, audio_paths = prepare_refs_hyps(df_eval_input, ref_col, hyp_col, norm, norm_lexicon)
    else:
        ref, hyp, ids, audio_paths = prepare_refs_hyps(df_eval_input, ref_col, hyp_col, norm)

    # output columns
    df_results_header=["dataset", "subset", "split", "samples", "ref_type", "norm_type", "system", "SER", "WIL", "MER", "WER", "CER"]
    # dataset  - name of the dataset
    # test cases - number of test cases
    # reference - type of source reference from the dataset (original, manually verified, normalized, etc.)
    # norm_type - type of normalization applied to the reference and hypothesis using jiwer
    # system - codename of the ASR system
    
    result=[]
    
    match_sents = sum(r == h for r, h in zip(ref, hyp))
    #print("match_sents: ", match_sents)
    ser = round((1 - (match_sents / len(ref))) * 100,2)

    output_words = jiwer.process_words(ref, hyp)
    wer = round(output_words.wer * 100 ,2)
    mer = round(output_words.mer * 100 ,2)
    wil = round(output_words.wil * 100 ,2)

    output_chars = jiwer.process_characters(ref, hyp)
    cer = round(output_chars.cer * 100, 2)
    
    #print("SER: ", ser)
    #print("WER: ", wer)
    #print("CER: ", cer)
    #print("MER: ", mer)
    #print("WIL: ", wil)
    # TODO add more metrics e.g. TER, PER, etc.
    result.append([dataset, subset, split, len(ref), ref_type, norm, system_codename, ser, wil, mer, wer, cer])
    
    df_results = pd.DataFrame(result, columns=df_results_header)
    
    return df_results

def get_lexical_metrics_per_dataset_all_ref_types(df_eval_input, dataset_codename, system_codename, norm, norm_lexicon = None)-> pd.DataFrame:
    """
    Calculate metrics across all reference types for a given dataset and ASR system.
    
    Args:
        df_eval_input (pandas.DataFrame): DataFrame with reference and hypothesis data.
        dataset_codename (str): Name of the dataset.
        system_codename (str): Identifier for the ASR system.
        norm (str): Normalization type applied.
        norm_lexicon (dict, optional): Dictionary for lexicon-based normalization. Default is None.
        
    Returns:
        pandas.DataFrame: Combined DataFrame with metrics for all reference types.
    """
    df_results = pd.DataFrame([])

    ref_cols = [col for col in df_eval_input.columns if col.startswith('ref')]
    for ref_col in ref_cols:
        print("ref_col" + ref_col)
        ref_type = ref_col.split("_")[1]
        df_single_result = get_lexical_metrics_per_dataset(df_eval_input, dataset_codename, system_codename, ref_type, norm, norm_lexicon) 
        df_results = df_results.append(df_single_result, ignore_index=True)
    
    return(df_results)

def get_lexical_metrics_per_dataset_all_systems(df_eval_input, dataset_codename, ref_type, norm, norm_lexicon = None)-> pd.DataFrame:
    """
    Calculate metrics for all ASR systems on a given dataset and reference type.
    
    Args:
        df_eval_input (pandas.DataFrame): DataFrame with reference and hypothesis data.
        dataset_codename (str): Name of the dataset.
        ref_type (str): Type of reference (e.g., 'original', 'normalized').
        norm (str): Normalization type applied.
        norm_lexicon (dict, optional): Dictionary for lexicon-based normalization. Default is None.
        
    Returns:
        pandas.DataFrame: Combined DataFrame with metrics for all ASR systems.
    """
    df_results = pd.DataFrame([])

    hyp_cols =  [col for col in df_eval_input.columns if col.startswith('hyp')]
    for hyp_col in hyp_cols:
        print("hyp_col" + hyp_col)
        system_codename = hyp_col.split("_")[1]
        df_single_result = get_lexical_metrics_per_dataset(df_eval_input, dataset_codename, system_codename, ref_type, norm, norm_lexicon) 
        df_results = df_results.append(df_single_result, ignore_index=True)
    
    return(df_results)

def get_lexical_metrics_per_sample_all_ref_types(df_eval_input, dataset_codename, system_codename, norm)-> pd.DataFrame:
    """
    Calculate per-sample metrics across all reference types for a given dataset and ASR system.
    
    Args:
        df_eval_input (pandas.DataFrame): DataFrame with reference and hypothesis data.
        dataset_codename (str): Name of the dataset.
        system_codename (str): Identifier for the ASR system.
        norm (str): Normalization type applied.
        
    Returns:
        pandas.DataFrame: Combined DataFrame with per-sample metrics for all reference types.
    """
    df_results = pd.DataFrame([])

    ref_cols = [col for col in df_eval_input.columns if col.startswith('ref')]
    for ref_col in ref_cols:
        print("ref_col" + ref_col)
        ref_type = ref_col.split("_")[1]
        df_single_result = get_lexical_metrics_per_sample(df_eval_input, dataset_codename, system_codename, ref_type, norm) 
        df_results = df_results.append(df_single_result, ignore_index=True)
    
    return(df_results)