import pandas as pd
import json

del_set = pd.read_csv("../cleaned_benchmark_dataset/PAR3-annotated_init_bench.csv") #benchmark_dataset_all_src_tgt.csv") #sampled_bench.csv")
print(del_set.shape)
# Load template file
with open("template/template_baseline.txt", "r") as f:
    template = f.read()

with open("template/template_stepwise_v2.txt", "r") as f:
    template_stepwise = f.read()

# Load QA files
#with open("template/QA1.txt", "r") as f:
 #   QA1 = f.read()

#with open("template/QA2.txt", "r") as f:
 #   QA2 = f.read()

with open("template/QA_final.txt", "r") as f:
    QA = f.read()

#with open("template/QA_stepwisw_final.txt", "r") as f:
 #   QA_stepwise = f.read()
# Process the DataFrame
'''
df_qa1 = [template.format(
    source=row["src"],
    translation=row["tgt"],
    questions=QA1
) for index, row in del_set.iterrows()]
del_set["QA1"] = df_qa1

df_qa2 = [template.format(
    source=row["src"],
    translation=row["tgt"],
    questions=QA2
) for index, row in del_set.iterrows()]
del_set["QA2"] = df_qa2
del_set[["src", "tgt", "QA1", "pair", "model", "dataset"]].to_csv("del_dataset/del_set_with_QA1.csv", index=False)
del_set[["src", "tgt", "QA2", "pair", "model", "dataset"]].to_csv("del_dataset/del_set_with_QA2.csv", index=False)

df_qa = [template_stepwise.format(
    source=row["src"],
    translation=row["tgt"],
    questions=QA
) for index, row in del_set.iterrows()]
del_set["QA"] = df_qa
del_set[["src", "tgt", "QA", "pair", "model", "dataset"]].to_csv("final_set/final_set_with_QA_stepwise.csv", index=False)
'''

df_qa = [template.format(
    source=row["src"],
    translation=row["tgt"],
    questions=QA
) for index, row in del_set.iterrows()]
del_set["QA"] = df_qa
del_set[["src", "tgt", "QA", "pair", "model", "dataset"]].to_csv("final_set/PAR3-final_set_with_QA.csv", index=False)

df_qa = [template_stepwise.format(
    source=row["src"],
    translation=row["tgt"],
    questions=QA
) for index, row in del_set.iterrows()]
del_set["QA"] = df_qa
del_set[["src", "tgt", "QA", "pair", "model", "dataset"]].to_csv("final_set/PAR3-final_set_with_plevel_stepv2.csv", index=False)


