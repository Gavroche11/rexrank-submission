import pandas as pd
import json
from f1chexbert import F1CheXbert
import torch

result = json.load(open('/home/data1/workspace/bih1122/llava_test_outputs/v2.0-old/mc_test_output.json'))

mimic_result = result[:3401]
chexpert_result = result[3401:]

mimic_result_df = pd.DataFrame(mimic_result)
mimic_result_df = mimic_result_df[['question_id', 'answer', 'text']]
mimic_result_df = mimic_result_df.rename(columns={'question_id': 'object_id', 'answer': 'gt', 'text': 'pred'})
mimic_result_df['gt'] = mimic_result_df['gt'].apply(lambda x: x[0])

mimic_result_df.to_csv('/home/data1/workspace/bih1122/llava_test_outputs/v2.0-old/mc_test_outputs_mimic.csv', index=False)

# Load CheXbert model
model = F1CheXbert()

# Evaluate F1 score using CheXbert
accuracy, accuracy_not_averaged, class_report, class_report_5 = model(mimic_result_df['gt'], mimic_result_df['pred'])

print(class_report)