import argparse
import logging
import config
import openpyxl
import torch
import pandas as pd
import re
from numpy import result_type
from transformers import EncoderDecoderModel, BertTokenizer
from robertamodel import RoBERTaModel

parser = argparse.ArgumentParser(description=config.TEST_DESCRIPTION)

parser.add_argument("--chat", default=False, help=config.CHAT_HELP)
parser.add_argument("--test", default=False, help=config.TEST_HELP)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = parser.parse_args()
logging.info(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(config.ROBERTA_PATH)
model = RoBERTaModel()
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
model.to(device)

if __name__ == "__main__":
    if args.chat:
        while True:
            sentence = input("user > ")
            if sentence == "q":
                break
            input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=False), device=device).unsqueeze(0)
            generated = tokenizer.decode(model.generative(input_ids)[0])
            generated = generated[:generated.find("[SEP]")]
            output = re.sub('[^.-?-가-힣ㄱ-ㅎㅏ-ㅣ]',' ',generated)
            print("bot > " + output.lstrip())

    if args.test:
        test = pd.read_excel(config.TEST_DATA)
        test_Q = test['Question'].to_list()
        result = []
        for sentence in test_Q:
            input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=False), device=device).unsqueeze(0)
            generated = tokenizer.decode(model.generative(input_ids)[0])
            generated = generated[:generated.find("[SEP]")]
            output = re.sub('[^.-?-가-힣ㄱ-ㅎㅏ-ㅣ]',' ',generated)
            result.append(output)
        test['chatbot'] = result
        test.to_csv(config.TEST_RESULT_CSV)