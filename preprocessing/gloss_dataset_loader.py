import csv
import os
from typing import Optional, List
from pydantic import BaseModel
import fsspec
# HF tokenizers
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel, Unigram
from tokenizers.trainers import WordLevelTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

csv_file_name = "GKSL3k_original.csv"
file_path = f"data/{csv_file_name}"

class GlossSchema(BaseModel):
    dataset: str
    video_num: Optional[int]=None   # null 기본값
    question: bool=False            # FALSE(기본값), TRUE
    gloss_tokens: List[str]         # 토큰을 나누어 리스트로 저장하려면 List[str]
    korean: str

class DataParser:
    def parse_data(self, file_path: str) -> List[GlossSchema]:
        parsed = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None) # 첫 줄 스킵
            for data in reader:
                question = (data[3] or "").strip().upper()
                is_question = (question == "TRUE")

                # 토큰 리스트 저장할 때
                gloss_tokens = (data[4] or "").strip().split() if data[4] else []
                if is_question:
                    gloss_tokens.append("<Q>") # 질문 특수토큰 추가

                # gloss_tokens = (data[4] or "").strip()
                # if is_question: gloss_tokens += "<Q>"

                schema = GlossSchema(
                    dataset=data[0].strip(),
                    video_num=int(data[1].strip()) if data[1].isdigit() else None,
                    question=question,
                    gloss_tokens=gloss_tokens,
                    korean=(data[5] or "").strip()
                )
                parsed.append(schema)

            return parsed

# Tokenizer
class TokenizerManager:
    def __init__(
            self,
            examples: List, # List[GlossSchema]
            save_ksl_path: str = "ksl_gloss_wordlevel.json",
            save_ko_path: str = "ko_unigram.json",
            vocab_min_freq: int = 1,
            vocab_size: int = 16000,
            specials: Optional[List[str]] = None,
    ):
        self.examples = examples
        self.vocab_min_freq=vocab_min_freq
        self.vocab_size=vocab_size
        self.specials=specials or ["<pad>", "<unk>", "<bos>", "<eos>", "<Q>"]
        self.save_ksl_path=save_ksl_path
        self.save_ko_path=save_ko_path

        self.gloss_tok: Optional[HFTokenizer] = self.train_gloss_wordlevel_tokenizer()
        self.ko_tok: Optional[HFTokenizer] = self.train_ko_unigram_tokenizer()

    @staticmethod
    def save_tokenizer(tok: HFTokenizer, path: str):
        if path.startswith("gs://"):
            with fsspec.open(path, "w") as f:
                f.write(tok.to_str())
        else:
            tok.save(path)

    @staticmethod
    def load_tokenizer(path: str) -> HFTokenizer:
        if path.startswith("gs://"):
            with fsspec.open(path, "r") as f:
                return HFTokenizer.from_str(f.read())
        else:
            return HFTokenizer.from_file(path)

    def train_gloss_wordlevel_tokenizer(self) -> HFTokenizer:
        """examples: List[GlossSchema]"""
        tok = HFTokenizer(WordLevel(unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=self.specials,
            min_frequency=self.vocab_min_freq
        )
        def gloss_corpus_iter():
            for ex in self.examples: # examples : GlossSchema
                yield " ".join(ex.gloss_tokens)

        tok.train_from_iterator(gloss_corpus_iter(), trainer=trainer)
        self.save_tokenizer(tok, self.save_ksl_path)
        self.gloss_tok = tok
        return tok

    def train_ko_unigram_tokenizer(self) -> HFTokenizer:
        tok = HFTokenizer(Unigram())
        trainer = UnigramTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.specials
        )
        def ko_corpus_iter():
            for ex in self.examples:
                if ex.korean:
                    yield ex.korean
        tok.train_from_iterator(ko_corpus_iter(), trainer=trainer)

        bos_id = tok.token_to_id("<bos>")
        eos_id = tok.token_to_id("<eos>")
        if bos_id is not None and eos_id is not None:
            from tokenizers.processors import TemplateProcessing
            tok.post_processor = TemplateProcessing(
                single="<bos> $A <eos>",
                pair="<bos> $A <eos> <bos> $B <eos>",
                special_tokens = [("<bos>",bos_id), ("<eos>", eos_id)],
            )
        self.save_tokenizer(tok, self.save_ko_path)
        self.ko_tok = tok
        return tok