from transformers import M2M100Tokenizer
from transformers.configuration_utils import PretrainedConfig
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from typing import List, Dict
from network_proto import SingleRecord, TranslationRequest, TranslationResponse
import onnxruntime as ort
import torch


device = "cpu"
ort_provider = "CPUExecutionProvider"
if torch.cuda.is_available():
    device = "cuda"
    ort_provider = "CUDAExecutionProvider"


class ModelFiles:
    def __init__(self, encoder: str, decoder: str, config: str, model_dir: str) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.model_dir = model_dir


MODEL_DIR_PREFIX = "./"
ENCODER_FILE = MODEL_DIR_PREFIX + "model/encoder_model.onnx"
DECODER_FILE = MODEL_DIR_PREFIX + "model/decoder_model.onnx"
CONFIG_JSON = MODEL_DIR_PREFIX + "model/config.json"
MODEL_DIR = MODEL_DIR_PREFIX + "model"

MODEL_FILES = ModelFiles(ENCODER_FILE, DECODER_FILE, CONFIG_JSON, MODEL_DIR)


class InferenceEngine:
    """
    A translate inference engine, implemented via onnxruntime.
    It is much faster than pytorch.

    Usage example:
      infer = InferenceEngine(model_file)
      result = infer.batch_forward(input_ids, attention_mask, target_language_id)
    """

    def __init__(self, model_files: ModelFiles) -> None:
        encoder = ort.InferenceSession(model_files.encoder, providers=[
                                       ort_provider])
        decoder = ort.InferenceSession(model_files.decoder, providers=[
            ort_provider])
        config = PretrainedConfig.from_json_file(model_files.config)
        self.model = ORTModelForSeq2SeqLM(
            encoder_session=encoder, decoder_session=decoder, config=config, model_save_dir=model_files.model_dir).to(device)

    def batch_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, lang_id: int):
        result = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, forced_bos_token_id=lang_id)
        return result


class Translator:
    """
    A translator, which supports multilingual and batch process.

    Usage example:
      translator = Translator(model_file)
      result = translator.translate(source_language, target_language, records)

    Args:
      source_language: str
      target_language: str
      records: a list of SingleRecord, like:
      [{"id": "123", "text": "Hello, world!"}, {"id": "456", "text": "Good morning."}]
    """

    def __init__(self, model_files: ModelFiles) -> None:
        self.infer = InferenceEngine(model_files)
        self.tokenizers_dict = {}

    def _preprocess(self, source_language: str, target_language: str, records: List[SingleRecord]) -> List:
        lang = source_language + target_language
        if lang not in self.tokenizers_dict:
            tokenizer = M2M100Tokenizer.from_pretrained(
                'facebook/m2m100_418M', src_lang=source_language, tgt_lang=target_language)
            self.tokenizers_dict[lang] = tokenizer

        id_list = []
        text_list = []
        for single_record in records:
            id_list.append(single_record.id)
            text_list.append(single_record.text)
        tensors = self.tokenizers_dict[lang](
            text_list, return_tensors="pt", padding=True, return_attention_mask=True)

        lang_id = self.tokenizers_dict[lang].get_lang_id(target_language)

        return [id_list, tensors["input_ids"], tensors["attention_mask"], lang_id]

    def _postprocess(self, source_language: str, target_language: str, generated_tensor: torch.Tensor, id_list: List[str]) -> TranslationResponse:
        lang = source_language + target_language
        generated_tensor = generated_tensor.to("cpu")
        text_list = self.tokenizers_dict[lang].batch_decode(
            generated_tensor, skip_special_tokens=True)
        result_list = []
        for cur_id, cur_text in zip(id_list, text_list):
            result_list.append(SingleRecord(id=cur_id, text=cur_text))
        return TranslationResponse(result=result_list)

    def _batch_infer(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, lang_id: int) -> torch.Tensor:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        generated_tensor = self.infer.batch_forward(
            input_ids, attention_mask, lang_id)
        generated_tensor = generated_tensor.to("cpu")
        return generated_tensor

    def translate(self, source_language: str, target_language: str, records: List[SingleRecord]) -> TranslationResponse:
        pre_result = self._preprocess(
            source_language, target_language, records)
        generated_tensor = self._batch_infer(
            pre_result[1], pre_result[2], pre_result[3])
        post_result = self._postprocess(
            source_language, target_language, generated_tensor, pre_result[0])
        return post_result


global_translator = Translator(MODEL_FILES)


def translate(req: TranslationRequest) -> TranslationResponse:
    return global_translator.translate(source_language=req.payload.fromLang, target_language=req.payload.toLang, records=req.payload.records)


if __name__ == "__main__":
    from_lang = "en"
    to_lang = "ja"
    text = "Life is like a box of chocolates."
    text2 = "Hello, world"
    record1 = SingleRecord(id="123", text=text)
    record2 = SingleRecord(id="456", text=text2)
    records_list = [record1, record2]

    test_result = global_translator.translate(from_lang, to_lang, records_list)
    print(test_result)
