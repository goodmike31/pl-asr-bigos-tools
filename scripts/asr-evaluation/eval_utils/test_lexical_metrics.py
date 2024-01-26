import pytest
import pandas as pd
import os
from pandas.testing import assert_frame_equal


from lexical_metrics import calculate_lexical_metrics

@pytest.fixture
def input_file(tmpdir):
    # Create a temporary input file
    file_path = tmpdir.join("input_file.csv")
    content = "ref_col1,ref_col2,hyp_col1,hyp_col2\nref1,ref2,hyp1,hyp2\n"
    file_path.write(content)
    return str(file_path)

@pytest.fixture
def dataset(tmpdir):
    # Create a temporary dataset INI file
    file_path = tmpdir.join("dataset.ini")
    content = "[dataset]\nparam1=value1\nparam2=value2\n"
    file_path.write(content)
    return str(file_path)

@pytest.fixture
def output_file(tmpdir):
    # Create a temporary output file path
    file_path = tmpdir.join("output_file.tsv")
    return str(file_path)

# Function to be tested
def create_dataframe():
    return pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

# Test function
def test_create_dataframe():
    expected_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    result_df = create_dataframe()
    assert_frame_equal(result_df, expected_df)
    
def test_calculate_lexical_metrics(input_file, dataset, output_file):
    # Call the function being tested
    df_results = calculate_lexical_metrics(input_file, dataset, output_file)

    # Assert that the output file was created
    assert os.path.isfile(output_file)

    # Assert that the DataFrame was created and has the expected structure
    assert isinstance(df_results, pd.DataFrame)
    expected_columns = ["dataset", "test cases", "ref type", "eval norm", "system", "SER", "WIL", "MER", "WER", "CER"]
    assert df_results.columns.tolist() == expected_columns

    # Assert that the DataFrame has the correct number of rows
    assert len(df_results) == 20  # 5 normalization types * 2 ref columns * 2 hyp columns

    # Assert that the values in the DataFrame are within the expected range
    assert df_results["SER"].min() >= 0
    assert df_results["SER"].max() <= 1
    assert df_results["WIL"].min() >= 0
    assert df_results["WIL"].max() <= 1
    assert df_results["MER"].min() >= 0
    assert df_results["MER"].max() <= 1
    assert df_results["WER"].min() >= 0
    assert df_results["WER"].max() <= 1
    assert df_results["CER"].min() >= 0
    assert df_results["CER"].max() <= 1

    # Add more specific assertions as needed for your test case

def test_calculate_lexical_metrics_missing_arguments():
    with pytest.raises(SystemExit):
        calculate_lexical_metrics(None, None, None)

# Add more test cases as needed