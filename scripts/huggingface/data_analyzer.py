from datasets import load_dataset
from tqdm import tqdm 
import json, re, string
from langdetect import detect
from collections import Counter
from huggingface.const import BIGOS_SUBSETS, BIGOS_SPLITS


class DataAnalyzer:
    def __init__(
        self, 
        dataset: str = 'amu-cai/pl-asr-bigos-v2', 
        subsets: list = BIGOS_SUBSETS, 
        splits: list = BIGOS_SPLITS,
    ) -> None:
        self.dataset_name = dataset
        self.subsets = subsets
        self.splits = splits
        self.output = self._create_output_dict()

    def _create_output_dict(self) -> dict:
        """Creates an output dictionary with initial values."""
        output = {}
        for subset in self.subsets if len(self.subsets) == 1 else self.subsets + ['all']:
            output[subset] = {}
            if 'train' in self.splits:
                output[subset]['train'] = {
                    'noSamples': 0, 'noEmptyReferences': 0, 'noDifferentLanguage': 0, 
                    'noDoubleWhitespaces': 0, 'noWithoutPunctuation': 0, 'punctuationCounter': Counter(),
                }
                if subset != 'all':
                    output[subset]['train'].update({'emptyReference': [], 'differentLanguage': [],})
            if 'validation' in self.splits:
                output[subset]['validation'] = {
                    'noSamples': 0, 'noEmptyReferences': 0, 'noDifferentLanguage': 0, 
                    'noDoubleWhitespaces': 0, 'noWithoutPunctuation': 0, 'punctuationCounter': Counter(),
                }
                if subset != 'all':
                    output[subset]['validation'].update({'emptyReference': [], 'differentLanguage': [],})
            if 'test' in self.splits:
                output[subset]['test'] = {
                    'noSamples': 0, 'noNonEmptyReferences': 0,
                }
            if len(self.splits) > 1:
                output[subset]['all'] = { 'noSamples': 0 }
        return output

    def _is_empty(self, reference: str) -> bool:
        """
        Checks whether reference is empty.

        Parameters
        ----------
        reference : str
            Reference sentence of which the content is to be examined.

        Returns
        -------
        True if reference sentence is empty, False otherwise.
        """
        return reference.strip() == ''
    
    def _in_polish(self, reference: str) -> bool:
        """
        Checks whether reference is written in Polish language.

        Parameters
        ----------
        reference : str
            Reference sentence of which language is to be detected.

        Returns
        -------
        True if reference sentence is most likely written in Polish language, False otherwise.
        """
        if not self._is_empty(reference):
            return detect(reference) == 'pl'
        
    def _contains_double_white_spaces(self, reference: str) -> bool:
        """
        Checks whether reference contains double white spaces.

        Parameters
        ----------
        reference : str
            Reference sentence in which double whitespaces are to be found.

        Returns
        -------
        True if reference sentence contains double whitespaces, False otherwise.
        """
        return '  ' in reference

    def _contains_punctuation(self, reference: str) -> bool:
        """
        Checks whether reference contains any punctuation marks.
        
        Parameters
        ----------
        reference : str
            Reference sentence in which double whitespaces are to be found.

        Returns
        -------
        True if reference sentence contains any punctuation marks, False otherwise.
        """
        return reference.translate(str.maketrans('', '', string.punctuation)) != reference

    def _find_punctuation(self, reference: str) -> list:
        """
        Returns list of all punctuation marks in reference.

        Parameters
        ----------
        reference : str
            Reference sentence in which punctuation marks are to be found.

        Returns
        -------
        List of all punctuation marks in reference sentence.
        """
        return re.findall('[^\w\s]', reference)

    def _analyze_sample(self, subset: str, split: str, sample: dict) -> None:
        """
        Updates statistics for selected sample.

        Parameters
        ----------
        subset : str
            Name of the currently processed subset.
        split : str
            Name of the currently processed split.
        sample : dict
            Currently processed sample. Available keys: ['file_id', 'dataset_id', 'ref_orig', 'audio'].
        """
        self.output[subset][split]['noSamples'] += 1
        if split == 'test':
            if not self._is_empty(sample['ref_orig']):
                self.output[subset][split]['noNonEmptyReferences'] += 1
        else:
            if self._is_empty(sample['ref_orig']):
                self.output[subset][split]['noEmptyReferences'] += 1
                self.output[subset][split]['emptyReference'].append(sample['file_id'])

            if not self._in_polish(sample['ref_orig']):
                self.output[subset][split]['noDifferentLanguage'] += 1
                self.output[subset][split]['differentLanguage'].append(sample['file_id'])

            if self._contains_double_white_spaces(sample['ref_orig']):
                self.output[subset][split]['noDoubleWhitespaces'] += 1

            if not self._contains_punctuation(sample['ref_orig']):
                self.output[subset][split]['noWithoutPunctuation'] += 1
            else:
                self.output[subset][split]['punctuationCounter'].update(self._find_punctuation(sample['ref_orig']))

    def _analyze_split(self, subset: str, split: str) -> None:
        """
        Loads dataset based on selected subset and split, and iterates over samples analyzing each of them.

        Parameters
        ----------
        subset : str
            Name of the currently processed subset.
        split : str
            Name of the currently processed split.
        """
        dataset = load_dataset(self.dataset_name, subset, split=split, streaming=True)
        for sample in tqdm(dataset):
            self._analyze_sample(subset=subset, split=split, sample=sample)

    def _analyze_subset(self, subset: str) -> None:
        """
        Iterates over selected splits and updates statistics for them.

        Parameters
        ----------
        subset : str
            Name of the currently processed subset.
        """
        for split in self.splits:
            self._analyze_split(subset=subset, split=split)
            if len(self.splits) > 1:
                self.output[subset]['all']['noSamples'] += self.output[subset][split]['noSamples']
            if len(self.subsets) > 1:
                for field in self.output['all'][split]:
                    self.output['all'][split][field] += self.output[subset][split][field]
            if len(self.splits) > 1 and len(self.subsets) > 1:
                self.output['all']['all']['noSamples'] += self.output[subset][split]['noSamples']

    def analyze(self) -> dict:
        """
        Iterates over selected subsets and returns a dictionary containing statistics for them. Available
        keys depend on subsets and splits.

        Keys
        ----
        noSamples : int
            Total number of samples in dataset, subset, or split. Available in all cases.
        noEmptyReferences : int
            Number of empty reference sentences. Available for 'train' and 'validation' split.
        noDifferentLanguage : int
            Number of reference sentences most likely written in language different than Polish. Available 
            for 'train' and 'validation' split.
        noDoubleWhitespaces : int
            Number of reference sentences containing double white spaces. Available for 'train' and
            'validation' split.
        noWithoutPunctuation : int
            Number of reference sentences without punctuation marks. Available for 'train' and 'validation'
            split.
        punctuationCounter : collections.Counter()
            Counter of punctuation marks in reference sentences. Available for 'train' and 'validation'
            split.
        emptyReference : list
            List of file IDs for which the reference sentence is empty. Available for 'train' and
            'validation' split. Not available for 'all' subset.
        differentLanguage : list
            List of file IDs for which language different than Polish was detected. Available for 'train'
            and 'validation' split. Not available for 'all' subset.
        noNonEmptyReferences : int
            Number of non-empty reference sentences. Available for 'test' split.
        """
        for subset in self.subsets:
            self._analyze_subset(subset=subset)
        return self.output

    def save_output(self, filename: str) -> None:
        """
        Saves output as a JSON file.

        Parameters
        ----------
        filename : str
            Name of the output file.
        """
        json.dump(self.output, open(filename, 'w'))