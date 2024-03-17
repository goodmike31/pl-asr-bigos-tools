import pandas as pd
import os
import requests

def download_tsv_from_google_sheet(sheet_url):
    # Modify the Google Sheet URL to export it as TSV
    tsv_url = sheet_url.replace('/edit#gid=', '/export?format=tsv&gid=')
    
    # Send a GET request to download the TSV file
    response = requests.get(tsv_url)
    response.encoding = 'utf-8'

    # Check if the request was successful
    if response.status_code == 200:
        # Read the TSV content into a pandas DataFrame
        from io import StringIO
        tsv_content = StringIO(response.text)
        df = pd.read_csv(tsv_content, sep='\t', encoding='utf-8')
        return df
    else:
        print("Failed to download the TSV file.")
        return None
    
def get_meta_header_tts():
    # TODO move to config
    bigos_common_header =  "audioname split dataset speaker_id samplingrate_orig sampling_rate ref_orig audiopath_bigos".split()
    tts_header = "prompt_id tts_engine voice_name".split()
    df_header = bigos_common_header + tts_header

    return(df_header)