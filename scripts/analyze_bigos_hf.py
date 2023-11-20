from datasets import load_dataset, DatasetDict
from tqdm import tqdm 
import json, re, string
from langdetect import detect
from collections import Counter


BIGOS_SUBSETS = [
    # 'fair-mls-20', 'mailabs-corpus_librivox-19', 'mozilla-common_voice_15-23', 
    # 'pwr-shortwords-unk', 'pwr-maleset-unk',
     'pwr-viu-unk', 
    # 'pwr-azon_read-20', 'pwr-azon_spont-20', 'pjatk-clarin_studio-15', 
    # 'pjatk-clarin_mobile-15', 'google-fleurs-22', 
    'polyai-minds14-21'
]

BIGOS_SPLITS = ['train', 'validation', 'test']




class DataAnalyzer:
    def __init__(
        self, dataset: str = 'michaljunczyk/pl-asr-bigos-v2', subsets: list = BIGOS_SUBSETS, splits: list = BIGOS_SPLITS,
    ) -> None:
        self.dataset_name = dataset
        self.subsets = subsets
        self.splits = splits
        self.output = self._create_output_dict()

    def _create_output_dict(self) -> dict:
        output = {}
        for subset in self.subsets if len(self.subsets) == 1 else self.subsets + ['all']:
            output[subset] = {}
            if 'train' in self.splits:
                output[subset]['train'] = {
                    'noSamples': 0, 'noEmptyReferences': 0, 'noDifferentLanguage': 0, 
                    'noDoubleWhitespaces': 0, 'noWithoutPunctuation': 0, 'punctuationCounter': Counter(),
                    #  'emptyReference': [], 'differentLanguage': [],
                }
                if subset != 'all':
                    output[subset]['train'].update({'emptyReference': [], 'differentLanguage': [],})
            if 'validation' in self.splits:
                output[subset]['validation'] = {
                    'noSamples': 0, 'noEmptyReferences': 0, 'noDifferentLanguage': 0, 
                    'noDoubleWhitespaces': 0, 'noWithoutPunctuation': 0, 'punctuationCounter': Counter(),
                    #  'emptyReference': [], 'differentLanguage': [],
                }
                if subset != 'all':
                    output[subset]['validation'].update({'emptyReference': [], 'differentLanguage': [],})
            if 'test' in self.splits:
                output[subset]['test'] = {
                    'noSamples': 0, 'noNotEmptyReferences': 0,
                }
            if len(self.splits) > 1:
                output[subset]['all'] = { 'noSamples': 0 }
        return output

    def _is_empty(self, reference: str) -> bool:
        return reference.strip() == ''
    
    def _in_polish(self, reference: str) -> bool:
        if not self._is_empty(reference):
            return detect(reference) == 'pl'
        
    def _contains_double_white_spaces(self, reference: str) -> bool:
        return '  ' in reference

    def _contains_punctuation(self, reference: str) -> bool:
        return reference.translate(str.maketrans('', '', string.punctuation)) != reference

    def _find_punctuation(self, reference: str) -> list:
        return re.findall('[^\w\s]', reference)


    def _analyze_sample(self, subset: str, split: str, sample: dict) -> None:
        self.output[subset][split]['noSamples'] += 1

        if split == 'test':
            if not self._is_empty(sample['ref_orig']):
                self.output[subset][split]['noNotEmptyReferences'] += 1
        
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
        dataset = load_dataset(self.dataset_name, subset, split=split, streaming=True)
        for sample in tqdm(dataset):
            self._analyze_sample(subset=subset, split=split, sample=sample)


    def analyze_subset(self, subset: str) -> None:
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
        for subset in self.subsets:
            self.analyze_subset(subset=subset)
        return self.output


    def save_output(self, filename: str) -> None:
        json.dump(self.output, open(filename, 'w'))