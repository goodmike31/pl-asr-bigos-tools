import pytest, requests
from bs4 import BeautifulSoup
from const import BIGOS_SUBSETS, BIGOS_SPLITS

_HOMEPAGE = 'https://huggingface.co/datasets/michaljunczyk/pl-asr-bigos-v2'

_BIGOS_SUBSETS = set(BIGOS_SUBSETS)
_BIGOS_SPLITS = set(BIGOS_SPLITS)


@pytest.mark.parametrize('subset', _BIGOS_SUBSETS)
def test_if_subset_uploaded(subset):
    """
    Checks whether subset is uploaded to the repository on Hugging Face.

    Parameters
    ----------
    subset : str
        Name of a subset taken from _BIGOS_SUBSETS.
    """
    response = requests.get(f'{_HOMEPAGE}/tree/main/data/{subset}')
    assert response.status_code == 200, \
        f'{subset} subset not uploaded to Hugging Face'


def test_for_extra_subsets():
    """Checks if there are unexpected subsets uploaded to the repository on Hugging Face."""
    response = requests.get(f'{_HOMEPAGE}/tree/main/data')
    soup = BeautifulSoup(response.text, 'html.parser')
    subsets = {s.string for s in soup.find_all(name='span', attrs={'class': 'truncate'})}
    assert len(subsets) <= len(_BIGOS_SUBSETS) and subsets == _BIGOS_SUBSETS, \
        f'Additional subsets found on Hugging Face: {subsets - _BIGOS_SUBSETS}'


@pytest.mark.parametrize(
    'subset,split,frmt', 
    [tuple([subset, split, frmt]) for subset in _BIGOS_SUBSETS 
     for split in _BIGOS_SPLITS 
     for frmt in ['tar.gz', 'tsv']]
)
def test_if_file_exists(subset, split, frmt):
    """
    Checks whether all expected files exist for a subset.

    Parameters
    ----------
    subset : str
        Name of a subset taken from _BIGOS_SUBSETS.
    split : str
        Name of a split taken from _BIGOS_SPLITS.
    frmt : str
        Format of a file, i.e. tar.gz or tsv.
    """
    response = requests.get(f'{_HOMEPAGE}/blob/main/data/{subset}/{split}.{frmt}')
    assert response.status_code == 200, \
        f'{subset}/{split}.{frmt} file missing'


@pytest.mark.parametrize('subset', _BIGOS_SUBSETS)
def test_for_extra_files(subset):
    """
    Checks if there are unexpected files uploaded to the repository on Hugging Face for a subset.

    Parameters
    ----------
    subset : str
        Name of a subset taken from _BIGOS_SUBSETS.
    """
    expected_files = {f'{split}{frmt}' for split in _BIGOS_SPLITS for frmt in ['.tar.gz', '.tsv']}
    response = requests.get(f'{_HOMEPAGE}/tree/main/data/{subset}')
    soup = BeautifulSoup(response.text, 'html.parser')
    files = {f.string for f in soup.find_all(name='span', attrs={'class': 'truncate'})}
    assert len(files) <= len(expected_files) or files == expected_files, \
        f'Subset {subset} contains additional files: {files - expected_files}'