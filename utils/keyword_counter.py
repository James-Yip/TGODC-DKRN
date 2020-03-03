"""function to count the total non-repeated keywords in the dataset
"""

import os

# load keyword files of train, valid and test sets
def load_file(file_name):
  with open(file_name) as f:
    data = f.readlines()
  return data
keywords_dict = {
    stage: load_file(os.path.join('../tx_data', stage, 'keywords_vocab.txt'))
    for stage in ['train','valid','test']
}

# count keywords
merged_keywords_list = []
for keywords in keywords_dict.values():
  merged_keywords_list.extend(keywords)
print('length of merged keywords:', len(merged_keywords_list))
print('length of merged keywords (remove dumplicated keywords):', len(set(merged_keywords_list)))

# check whether the merged keywords are same as the keywords in training set
merged_keywords_set = set(merged_keywords_list)
train_keywords_set = set(keywords_dict['train'])
if merged_keywords_set.symmetric_difference(merged_keywords_set) == set():
  print('The merged keywords are same as the training set keywords')
